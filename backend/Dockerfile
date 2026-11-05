FROM python:3.11-slim

#Para evitar prompts interactivos durante las instalaciones de los paquetes
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

#Instalar SWI-Prolog y otras herramientas basicas para la compilacion
RUN apt-get update && apt-get install -y --no-install-recommends \
    swi-prolog \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#El codigo se montara via volumen en desarrollo
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
