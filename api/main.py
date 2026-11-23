from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="MiCreditoYA - Motor de Credito 'Si o Si'",
    description="API REST para evaluacion crediticia dinamica en Pesos Colombianos (COP) con Arbol de Regresion y politica de Microcredito.",
    version="2.0.0",
)

# Permitir peticiones desde el archivo index.html local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo entrenado al iniciar el servidor
modelo = joblib.load("modelo_arbol.pkl")

# Factor de escala de paridad: 1 unidad del dataset = $1.000 COP en ingresos
# y 1 unidad de LoanAmount (en miles) = $1.000.000 COP
ESCALA_INGRESO_COP = 1000.0
ESCALA_PRESTAMO_COP = 1_000_000.0
MONTO_MICROCREDITO_COP = 800_000.0


def formato_cop(valor: float) -> str:
    return f"${valor:,.0f}".replace(",", ".") + " COP"


class SolicitudCredito(BaseModel):
    Nombre: str = Field(..., min_length=1, description="Nombre del solicitante")
    Ocupacion: str = Field(..., min_length=1, description="Ocupacion del solicitante")
    MontoSolicitado: float = Field(..., gt=0, description="Monto deseado en COP")
    ApplicantIncome: float = Field(..., gt=0, description="Ingreso mensual en COP")
    CoapplicantIncome: float = Field(0.0, ge=0, description="Ingreso del codeudor en COP (opcional)")
    Dependents: int = Field(..., ge=0, le=3, description="Personas a cargo (0 a 3)")
    Loan_Amount_Term: float = Field(36.0, gt=0, description="Plazo en meses")
    Credit_History: int = Field(..., ge=0, le=1, description="1 = Historial positivo, 0 = Mora")
    Property_Area: str = Field(..., description="Urban, Semiurban o Rural")


@app.post("/predict")
def predecir_credito(datos: SolicitudCredito):
    nombre = datos.Nombre.strip().upper()
    ocupacion = datos.Ocupacion.strip().upper()

    # 1. Conversion de COP a la escala del dataset (Feature Engineering interno)
    ingreso_escala = datos.ApplicantIncome / ESCALA_INGRESO_COP
    codeudor_escala = datos.CoapplicantIncome / ESCALA_INGRESO_COP
    total_income_escala = ingreso_escala + codeudor_escala
    total_income_cop = datos.ApplicantIncome + datos.CoapplicantIncome

    area_rural = 1 if datos.Property_Area == "Rural" else 0
    area_semiurban = 1 if datos.Property_Area == "Semiurban" else 0
    area_urban = 1 if datos.Property_Area == "Urban" else 0

    entrada = pd.DataFrame([{
        "ApplicantIncome": ingreso_escala,
        "Dependents": datos.Dependents,
        "Loan_Amount_Term": datos.Loan_Amount_Term,
        "Credit_History": datos.Credit_History,
        "TotalIncome": total_income_escala,
        "Area_Rural": area_rural,
        "Area_Semiurban": area_semiurban,
        "Area_Urban": area_urban,
    }])

    # 2. Prediccion del Arbol de Decision convertida a COP
    prediccion_arbol_cop = float(modelo.predict(entrada)[0]) * ESCALA_PRESTAMO_COP

    # Tope de capacidad de pago real: cuota maxima = 30% del ingreso mensual total
    tope_capacidad_cop = total_income_cop * 0.30 * datos.Loan_Amount_Term
    cupo_maximo_cop = round(min(prediccion_arbol_cop, tope_capacidad_cop), -4)

    # 3. Wrapper Logico (Politica de retencion 'Si o Si')
    if datos.Credit_History == 1 and cupo_maximo_cop >= 1_000_000:
        # Si pidio menos o igual a su cupo maximo, le aprobamos exactamente lo que pidio
        if datos.MontoSolicitado <= cupo_maximo_cop:
            monto_aprobado_cop = round(datos.MontoSolicitado, -3)
            mensaje = (
                f"{nombre}, ¡felicidades! Sabiendo que eres {ocupacion}, analizamos tu solicitud "
                f"y te aprobamos exactamente los {formato_cop(monto_aprobado_cop)} que pediste "
                f"(tienes un cupo máximo disponible de hasta {formato_cop(cupo_maximo_cop)})."
            )
        else:
            monto_aprobado_cop = cupo_maximo_cop
            mensaje = (
                f"{nombre}, revisamos tu solicitud de {formato_cop(datos.MontoSolicitado)}. "
                f"Sabiendo que eres {ocupacion} y cuidando tu capacidad de pago, hoy te aprobamos "
                f"tu monto máximo seguro de {formato_cop(monto_aprobado_cop)}."
            )

        cuota_estimada_cop = round(monto_aprobado_cop / datos.Loan_Amount_Term, 0)
        relacion_cuota_ingreso = round((cuota_estimada_cop / total_income_cop) * 100, 2)

        return {
            "tipo_aprobacion": "CREDITO_PRINCIPAL",
            "monto_aprobado_cop": monto_aprobado_cop,
            "cupo_maximo_cop": cupo_maximo_cop,
            "cuota_estimada_cop": cuota_estimada_cop,
            "relacion_cuota_ingreso": relacion_cuota_ingreso,
            "mensaje": mensaje,
        }
    else:
        # Regla 2: Ruta de Mitigacion / Microcredito de Retencion
        plazo_micro = min(datos.Loan_Amount_Term, 12)
        cuota_micro_cop = round(MONTO_MICROCREDITO_COP / plazo_micro, 0)
        relacion_cuota_ingreso = round((cuota_micro_cop / total_income_cop) * 100, 2)

        mensaje = (
            f"{nombre}, tu crédito principal por {formato_cop(datos.MontoSolicitado)} está en estudio. "
            f"Sabiendo que eres {ocupacion}, queremos que construyas historial con nosotros hoy mismo, "
            f"por eso te aprobamos un Microcrédito rotativo de {formato_cop(MONTO_MICROCREDITO_COP)}."
        )

        return {
            "tipo_aprobacion": "MICROCREDITO_RETENCION",
            "monto_aprobado_cop": MONTO_MICROCREDITO_COP,
            "cupo_maximo_cop": MONTO_MICROCREDITO_COP,
            "cuota_estimada_cop": cuota_micro_cop,
            "relacion_cuota_ingreso": relacion_cuota_ingreso,
            "mensaje": mensaje,
        }
