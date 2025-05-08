FROM locustio/locust:2.22.0

# Copia el archivo de pruebas al contenedor
COPY locustfile.py /locustfile.py

# Directorio de trabajo dentro del contenedor
WORKDIR /

# Comando por defecto si no se sobreescribe
CMD ["locust", "-f", "locustfile.py", "--host=http://inference-api:8000"]

