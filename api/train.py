import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Cargar el dataset procesado en capa Plata
df = pd.read_csv("loan_clasificacion_limpio.csv")

features = [
    "ApplicantIncome",
    "Dependents",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "Area_Rural",
    "Area_Semiurban",
    "Area_Urban",
]
target = "LoanAmount"

X = df[features]
y = df[target]

# 2. Division para validar rendimiento del arbol
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hiperparametros controlados para evitar sobreajuste
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"MAE en prueba: {mean_absolute_error(y_test, preds):.2f} (en miles)")
print(f"R2 en prueba:  {r2_score(y_test, preds):.4f}")

# 3. Reentrenar con el 100% de los datos para el MVP y exportar
model.fit(X, y)
joblib.dump(model, "modelo_arbol.pkl")
print("Modelo entrenado y guardado exitosamente como 'modelo_arbol.pkl'")
