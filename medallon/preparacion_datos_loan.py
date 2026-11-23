"""
Preparación de datos — Loan Prediction Problem Dataset (train.csv, 614 x 13)
Uso:  python preparacion_datos_loan.py train.csv

Genera dos archivos listos para modelar:
  - loan_clasificacion_limpio.csv  -> objetivo Loan_Status (614 filas)
  - loan_regresion_limpio.csv      -> objetivo LoanAmount (592 filas, sin imputar el objetivo)

Nota metodológica: este script deja los datos listos para EDA y para entregar.
Al entrenar, las imputaciones (moda/mediana) deben ajustarse SOLO con el conjunto
de entrenamiento (sklearn Pipeline + ColumnTransformer) para no filtrar información
del conjunto de prueba (data leakage).
"""
import sys
import numpy as np
import pandas as pd

ruta = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
df = pd.read_csv(ruta)

# 1. Quitar el identificador: no aporta información predictiva
df = df.drop(columns=["Loan_ID"])

# 2. Corregir tipos: "3+" en Dependents pasa a número
df["Dependents"] = df["Dependents"].replace("3+", "3")

# 3. Nulos en categóricas pequeñas (<= 5,2 %): moda
for col in ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term"]:
    df[col] = df[col].fillna(df[col].mode()[0])
df["Dependents"] = df["Dependents"].astype(int)

# 4. Credit_History (8,1 % nulos): NO se imputa con 1 ni con 0.
#    Los nulos se aprueban al 74 %, distinto de 0 (8 %) y de 1 (80 %),
#    así que se conservan como categoría propia: -1 = "sin dato".
df["Credit_History"] = df["Credit_History"].fillna(-1).astype(int)

# 5. Variables derivadas (ingeniería de características)
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])          # corrige asimetría (skew 6,5)

# 6. Codificación de categóricas
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, columns=["Property_Area"], prefix="Area", dtype=int)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# ---- Archivo de REGRESIÓN: el objetivo LoanAmount no se imputa (se eliminan 22 filas)
reg = df.dropna(subset=["LoanAmount"]).copy()
reg["LoanAmount_log"] = np.log1p(reg["LoanAmount"])
reg = reg.drop(columns=["Loan_Status"])  # evita usar la decisión final como predictor
reg.to_csv("loan_regresion_limpio.csv", index=False)

# ---- Archivo de CLASIFICACIÓN: LoanAmount es predictor -> mediana por grupo
df["LoanAmount"] = df.groupby(["Education", "Self_Employed"])["LoanAmount"] \
                     .transform(lambda s: s.fillna(s.median()))
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
df["Cuota_mensual"] = df["LoanAmount"] / df["Loan_Amount_Term"]          # EMI aproximada (miles)
df["Relacion_cuota_ingreso"] = df["Cuota_mensual"] * 1000 / df["TotalIncome"]
df.to_csv("loan_clasificacion_limpio.csv", index=False)

print("Clasificación:", df.shape, "| nulos:", int(df.isna().sum().sum()))
print("Regresión:   ", reg.shape, "| nulos:", int(reg.isna().sum().sum()))
