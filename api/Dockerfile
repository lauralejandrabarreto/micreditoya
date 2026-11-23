# Imagen de la API de MiCreditoYA (sirve para Render y para Google Cloud Run)
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py modelo_arbol.pkl ./
# Render y Cloud Run indican el puerto en la variable PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
