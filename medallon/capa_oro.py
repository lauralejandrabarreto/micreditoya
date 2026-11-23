"""
CAPA ORO — Arquitectura Medallón · Proyecto MiCreditoYA (Loan Prediction)

Toma los datos de la capa Plata y genera tablas resumen listas para consumo
(indicadores para el banco) y la tabla de evidencias "Antes (Bronce) vs Después (Plata)".

Uso (desde la carpeta medallon/):
    python capa_oro.py

Requisitos: pandas, scikit-learn, joblib, fastapi, httpx, openpyxl
Lee:
    bronce/loan_train_bronce.csv   (train.csv de Kaggle, 614 x 13)
    bronce/DATA_SET_bronce.xlsx    (test.csv de Kaggle, 367 x 12)
    plata/loan_clasificacion_limpio.csv, plata/loan_test_plata.csv
    ../api/main.py y ../api/modelo_arbol.pkl (el mismo motor de la demo)
Escribe:
    oro/*.csv, oro/DATA_SET_oro.xlsx, evidencias/antes_vs_despues.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
MVP = os.path.join(AQUI, "..", "api")
ESCALA_PRESTAMO_COP = 1_000_000  # misma escala que usa main.py (1 unidad de LoanAmount = $1.000.000)

def ruta(*p):
    return os.path.join(AQUI, *p)

# ------------------------------------------------------------------ Lectura
bronce_train = pd.read_csv(ruta("bronce", "loan_train_bronce.csv"))
bronce_test = pd.read_excel(ruta("bronce", "DATA_SET_bronce.xlsx"))
plata = pd.read_csv(ruta("plata", "loan_clasificacion_limpio.csv"))
plata_test = pd.read_csv(ruta("plata", "loan_test_plata.csv"))

consultas = []  # registro de cada consulta para mostrarla como evidencia

def zona(df):
    return np.select([df.Area_Rural == 1, df.Area_Semiurban == 1], ["Rural", "Semiurbana"], "Urbana")

# ------------------------------------------------------------------ ORO 1
# Tasa histórica de aprobación por historial crediticio
etiqueta_hist = {1: "Positivo", 0: "Negativo", -1: "Sin registro"}
oro_hist = (plata.groupby("Credit_History")
            .agg(solicitudes=("Loan_Status", "size"),
                 aprobadas=("Loan_Status", "sum"))
            .reset_index())
oro_hist["tasa_aprobacion_pct"] = (oro_hist.aprobadas / oro_hist.solicitudes * 100).round(1)
oro_hist.insert(1, "historial", oro_hist.Credit_History.map(etiqueta_hist))
oro_hist = oro_hist.sort_values("Credit_History", ascending=False)
consultas.append(("oro_tasa_aprobacion_historial",
                  "plata.groupby('Credit_History').agg(solicitudes=size, aprobadas=sum(Loan_Status)); tasa = aprobadas/solicitudes"))

# ------------------------------------------------------------------ ORO 2
# Monto promedio solicitado por zona
p = plata.assign(zona=zona(plata))
oro_zona = (p.groupby("zona")
            .agg(solicitudes=("LoanAmount", "size"),
                 monto_promedio_miles=("LoanAmount", "mean"),
                 monto_mediano_miles=("LoanAmount", "median"),
                 tasa_aprobacion_pct=("Loan_Status", "mean"))
            .reset_index())
oro_zona["tasa_aprobacion_pct"] = (oro_zona.tasa_aprobacion_pct * 100).round(1)
oro_zona["monto_promedio_cop_escala_mvp"] = (oro_zona.monto_promedio_miles * ESCALA_PRESTAMO_COP).round(-3)
oro_zona[["monto_promedio_miles", "monto_mediano_miles"]] = oro_zona[["monto_promedio_miles", "monto_mediano_miles"]].round(1)
consultas.append(("oro_monto_promedio_zona",
                  "plata.groupby(zona).agg(mean(LoanAmount), median(LoanAmount), mean(Loan_Status))"))

# ------------------------------------------------------------------ ORO 3
# Distribución de la relación cuota-ingreso
cortes = [0, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf]
nombres = ["0-5 %", "5-10 %", "10-15 %", "15-20 %", "20-30 %", "más de 30 %"]
p["rango_cuota_ingreso"] = pd.cut(p.Relacion_cuota_ingreso, cortes, labels=nombres, right=True)
oro_rel = (p.groupby("rango_cuota_ingreso", observed=False)
           .agg(solicitudes=("Loan_Status", "size"), tasa_aprobacion_pct=("Loan_Status", "mean"))
           .reset_index())
oro_rel["porcentaje_solicitudes"] = (oro_rel.solicitudes / len(p) * 100).round(1)
oro_rel["tasa_aprobacion_pct"] = (oro_rel.tasa_aprobacion_pct * 100).round(1)
consultas.append(("oro_distribucion_cuota_ingreso",
                  "pd.cut(Relacion_cuota_ingreso, [0,5,10,15,20,30,inf] %) -> groupby(rango).agg(size, mean(Loan_Status))"))

# ------------------------------------------------------------------ ORO 4
# Evaluación de las 367 solicitudes nuevas (test.csv) con el mismo motor de la demo
sys.path.insert(0, MVP)
cwd = os.getcwd()
os.chdir(MVP)  # main.py carga modelo_arbol.pkl con ruta relativa
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
os.chdir(cwd)
cliente = TestClient(main.app)

filas = []
for i, r in plata_test.iterrows():
    ingreso_cop = float(r.ApplicantIncome) * main.ESCALA_INGRESO_COP
    codeudor_cop = float(r.CoapplicantIncome) * main.ESCALA_INGRESO_COP
    if ingreso_cop <= 0:  # la API exige ingreso > 0; el ingreso del codeudor pasa a ser el ingreso base
        ingreso_cop, codeudor_cop = codeudor_cop, 0.0
    payload = {
        "Nombre": f"Solicitud {i + 1}", "Ocupacion": "No registrada",
        "MontoSolicitado": float(r.LoanAmount) * ESCALA_PRESTAMO_COP,
        "ApplicantIncome": ingreso_cop, "CoapplicantIncome": codeudor_cop,
        "Dependents": int(r.Dependents), "Loan_Amount_Term": float(r.Loan_Amount_Term),
        "Credit_History": int(r.Credit_History),
        "Property_Area": "Rural" if r.Area_Rural == 1 else ("Semiurban" if r.Area_Semiurban == 1 else "Urban"),
    }
    res = cliente.post("/predict", json=payload).json()
    filas.append({
        "solicitud": i + 1,
        "zona": {"Rural": "Rural", "Semiurban": "Semiurbana", "Urban": "Urbana"}[payload["Property_Area"]],
        "historial": int(r.Credit_History),
        "ingreso_total_cop": ingreso_cop + codeudor_cop,
        "monto_solicitado_cop": payload["MontoSolicitado"],
        "plazo_meses": payload["Loan_Amount_Term"],
        "tipo_aprobacion": res["tipo_aprobacion"],
        "cupo_maximo_cop": res["cupo_maximo_cop"],
        "monto_aprobado_cop": res["monto_aprobado_cop"],
        "cuota_estimada_cop": res["cuota_estimada_cop"],
        "relacion_cuota_ingreso_pct": res["relacion_cuota_ingreso"],
        "monto_ajustado": res["tipo_aprobacion"] == "CREDITO_PRINCIPAL" and res["monto_aprobado_cop"] < payload["MontoSolicitado"],
    })
oro_scoring = pd.DataFrame(filas)
consultas.append(("oro_scoring_solicitudes_nuevas",
                  "Para cada fila de plata/loan_test_plata.csv: POST /predict (main.py) -> tipo, cupo, monto aprobado, cuota"))

# ------------------------------------------------------------------ ORO 5
# KPIs de operación calculables con los datos disponibles
s = oro_scoring
princ = s[s.tipo_aprobacion == "CREDITO_PRINCIPAL"]
kpis = [
    ("Solicitudes evaluadas", len(s), "solicitudes"),
    ("Crédito principal", round(len(princ) / len(s) * 100, 1), "% de solicitudes"),
    ("Microcrédito de retención", round((len(s) - len(princ)) / len(s) * 100, 1), "% de solicitudes"),
    ("Créditos principales con monto ajustado al cupo", round(princ.monto_ajustado.mean() * 100, 1), "% de créditos principales"),
    ("Monto promedio aprobado (crédito principal)", round(princ.monto_aprobado_cop.mean(), -3), "COP"),
    ("Cuota sobre ingreso promedio (crédito principal)", round(princ.relacion_cuota_ingreso_pct.mean(), 1), "%"),
    ("Monto total aprobado", round(s.monto_aprobado_cop.sum(), -3), "COP"),
]
oro_kpis = pd.DataFrame(kpis, columns=["indicador", "valor", "unidad"]).astype({"valor": object})
oro_kpis["valor"] = [f"{v:,.0f}" if u in ("COP", "solicitudes") else f"{v:.1f}" for _, v, u in kpis]
oro_monto_zona_scoring = (s.groupby(["zona", "tipo_aprobacion"])
                          .agg(solicitudes=("solicitud", "size"), monto_promedio_aprobado_cop=("monto_aprobado_cop", "mean"))
                          .reset_index())
oro_monto_zona_scoring["monto_promedio_aprobado_cop"] = oro_monto_zona_scoring.monto_promedio_aprobado_cop.round(-3)
consultas.append(("oro_kpis", "Indicadores calculados sobre oro_scoring_solicitudes_nuevas"))
consultas.append(("oro_desembolso_por_zona", "oro_scoring.groupby([zona, tipo_aprobacion]).agg(size, mean(monto_aprobado_cop))"))

# ------------------------------------------------------------------ Evidencia: Antes vs Después
tecnica = {
    "Gender": "Moda", "Married": "Moda", "Dependents": "Moda + '3+' convertido a 3",
    "Self_Employed": "Moda", "Loan_Amount_Term": "Moda (360 meses)",
    "LoanAmount": "Mediana (por educación y empleo en train; mediana global en test)",
    "Credit_History": "Train: categoría propia −1 «sin registro» · Test: moda (1)",
}
filas_ev = []
for col, tec in tecnica.items():
    filas_ev.append({
        "columna": col,
        "nulos_bronce_train": int(bronce_train[col].isna().sum()),
        "nulos_bronce_test": int(bronce_test[col].isna().sum()) if col in bronce_test else None,
        "tecnica": tec,
        "nulos_plata_train": int(plata[col].isna().sum()),
        "nulos_plata_test": int(plata_test[col].isna().sum()),
    })
antes_despues = pd.DataFrame(filas_ev)

ingreso_total_b = bronce_train.ApplicantIncome + bronce_train.CoapplicantIncome
asimetria = pd.DataFrame([
    {"variable": "Ingreso total", "asimetria_bronce": round(ingreso_total_b.skew(), 2), "asimetria_plata_log": round(plata.TotalIncome_log.skew(), 2)},
    {"variable": "Monto del préstamo", "asimetria_bronce": round(bronce_train.LoanAmount.skew(), 2), "asimetria_plata_log": round(plata.LoanAmount_log.skew(), 2)},
])

capas = pd.DataFrame([
    ("Bronce", "loan_train_bronce.csv (train.csv Kaggle)", *bronce_train.shape),
    ("Bronce", "DATA_SET_bronce.xlsx (test.csv Kaggle)", *bronce_test.shape),
    ("Plata", "loan_clasificacion_limpio.csv", *plata.shape),
    ("Plata", "loan_regresion_limpio.csv", *pd.read_csv(ruta("plata", "loan_regresion_limpio.csv")).shape),
    ("Plata", "loan_test_plata.csv", *plata_test.shape),
    ("Oro", "oro_tasa_aprobacion_historial", *oro_hist.shape),
    ("Oro", "oro_monto_promedio_zona", *oro_zona.shape),
    ("Oro", "oro_distribucion_cuota_ingreso", *oro_rel.shape),
    ("Oro", "oro_scoring_solicitudes_nuevas", *oro_scoring.shape),
    ("Oro", "oro_desembolso_por_zona", *oro_monto_zona_scoring.shape),
    ("Oro", "oro_kpis", *oro_kpis.shape),
], columns=["capa", "entidad", "registros", "columnas"])

# ------------------------------------------------------------------ Escritura
tablas = {
    "oro_tasa_aprobacion_historial": oro_hist,
    "oro_monto_promedio_zona": oro_zona,
    "oro_distribucion_cuota_ingreso": oro_rel,
    "oro_scoring_solicitudes_nuevas": oro_scoring,
    "oro_desembolso_por_zona": oro_monto_zona_scoring,
    "oro_kpis": oro_kpis,
}
for nombre, df in tablas.items():
    df.to_csv(ruta("oro", nombre + ".csv"), index=False)
antes_despues.to_csv(ruta("evidencias", "antes_vs_despues.csv"), index=False)
asimetria.to_csv(ruta("evidencias", "asimetria_ingresos.csv"), index=False)
capas.to_csv(ruta("evidencias", "inventario_capas.csv"), index=False)

with pd.ExcelWriter(ruta("oro", "DATA_SET_oro.xlsx")) as xl:
    capas.to_excel(xl, "inventario_capas", index=False)
    for nombre, df in tablas.items():
        df.to_excel(xl, nombre.replace("oro_", "")[:31], index=False)
    antes_despues.to_excel(xl, "antes_vs_despues", index=False)
    asimetria.to_excel(xl, "asimetria", index=False)
    pd.DataFrame(consultas, columns=["tabla", "consulta"]).to_excel(xl, "consultas", index=False)

# ------------------------------------------------------------------ Salida en consola (evidencia)
pd.set_option("display.width", 160)
for nombre, df in [("INVENTARIO DE CAPAS", capas), ("ANTES vs DESPUÉS", antes_despues),
                   ("ASIMETRÍA", asimetria)] + [(k.upper(), v) for k, v in tablas.items() if k != "oro_scoring_solicitudes_nuevas"]:
    print(f"\n=== {nombre} ===")
    print(df.to_string(index=False))
print(f"\nScoring: {len(oro_scoring)} solicitudes evaluadas -> oro/oro_scoring_solicitudes_nuevas.csv")
