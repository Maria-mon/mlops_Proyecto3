FROM python:3.10-slim

# Instala dependencias
RUN apt-get update && \
    apt-get install -y curl iputils-ping && \
    pip install --no-cache-dir locust && \
    apt-get clean

# Crear usuario no root (opcional para seguridad)
RUN useradd -m locustuser
USER locustuser

WORKDIR /mnt/locust

