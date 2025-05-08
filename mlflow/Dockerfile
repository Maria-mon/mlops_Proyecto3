FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /mlflow

# Instalar dependencias básicas del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requerimientos
COPY requirements.txt ./

# Instalar requerimientos de Python
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

# Exponer el puerto de MLflow
EXPOSE 5000

# Comando por defecto para iniciar el servidor
CMD ["python","-m","mlflow", "server","--backend-store-uri", "mysql+pymysql://mlflow_user:mlflow_pass@mysql/mlflow_db", "--default-artifact-root", "s3://mlflow/", "--host", "0.0.0.0", "--port", "5000"]

