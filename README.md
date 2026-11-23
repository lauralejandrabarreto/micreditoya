# MiCreditoYA

Simulador de crédito basado en un **árbol de decisión de regresión**, construido sobre una **arquitectura Medallón** (Bronce → Plata → Oro) con el dataset [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) de Kaggle.

Proyecto académico · Tema: Google data y modelo supervisado (árboles de decisión).

## Ver en línea

- **Sitio del proyecto:** `https://lauralejandrabarreto.github.io/micreditoya/` (GitHub Pages)
- **Presentación:** carpeta `presentacion/`
- **Simulador:** carpeta `simulador/` (consulta la API publicada en Render)

## Cómo funciona

1. El cliente diligencia la solicitud en el **simulador** (HTML y JavaScript).
2. La **API** en FastAPI (`api/main.py`) recibe los datos en el endpoint `/predict`.
3. El **árbol de regresión** (`api/modelo_arbol.pkl`) estima el monto.
4. La **regla de negocio** limita el monto para que la cuota no supere el 30 % del ingreso. Con historial positivo y cupo de al menos $1.000.000 se ofrece crédito principal; si no, un microcrédito de $800.000.

## Arquitectura Medallón

| Capa | Carpeta | Contenido |
|---|---|---|
| Bronce | `medallon/bronce/` | `loan_train_bronce.csv` (614 × 13) y `DATA_SET_bronce.xlsx` (test, 367 × 12), sin modificaciones |
| Plata | `medallon/plata/` | `loan_clasificacion_limpio.csv` (614 × 19), `loan_regresion_limpio.csv` (592 × 16), `loan_test_plata.csv` (367 × 18), sin nulos |
| Oro | `medallon/oro/` | 6 tablas de indicadores y `DATA_SET_oro.xlsx` con consultas y evidencias |

Scripts:
- `medallon/preparacion_datos_loan.py`: Bronce → Plata.
- `medallon/capa_oro.py`: Plata → Oro. Evalúa las 367 solicitudes de test con la misma API.
- `api/train.py`: entrena el árbol a partir de la capa Plata y guarda `modelo_arbol.pkl`.

## Resultados del modelo

- `DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)`, 80 % entrenamiento / 20 % prueba.
- MAE en prueba: 39,5 (miles) · R² en prueba: 0,18.
- Variables con más peso: ingreso total (64,4 %) e ingreso del solicitante (25,5 %).

## Estructura del repositorio

```
├── index.html              Página de inicio (GitHub Pages)
├── presentacion/           Presentación de la sustentación (HTML y PDF)
├── simulador/              Página del simulador y config.js con la dirección de la API
├── api/                    API FastAPI, modelo entrenado, entrenamiento y Dockerfile
├── medallon/               Capas Bronce, Plata y Oro, evidencias y scripts
└── render.yaml             Configuración para publicar la API en Render
```

## Ejecutar en el computador

```bash
cd api
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload
```

Después se abre `simulador/index.html` en el navegador. Al abrirlo como archivo local, el simulador usa la API en `http://127.0.0.1:8000`.

Para regenerar la capa Oro:

```bash
python3 -m pip install -r api/requirements.txt openpyxl httpx
python3 medallon/capa_oro.py
```

## Limitaciones

- El R² de 0,18 indica que el monto depende de factores que el dataset no registra.
- El dataset no declara moneda; la conversión a pesos colombianos es un supuesto (ingreso × $1.000; monto × $1.000.000).
- El historial crediticio se simula en el simulador; no hay datos reales de pago.
