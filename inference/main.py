import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ─── Configuración ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "diabetes_model")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")  # ahora configurable

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ─── Carga del modelo ───────────────────────────────────────────────────────────
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    raise RuntimeError(f"Error cargando modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}': {e}")

app = FastAPI(title="Inference API")

# ─── Métricas Prometheus ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total de peticiones a /predict",
    ["method", "status"]
)
LATENCY = Histogram(
    "inference_latency_seconds",
    "Latencia de inferencia",
    ["method"]
)

# ─── Schema de entrada ───────────────────────────────────────────────────────────
class InputData(BaseModel):
    age: float
    race: str
    gender: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int
    metformin: str
    insulin: str

    class Config:
        extra = "forbid"

# ─── Endpoint /predict ──────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    # Parche: crea las dos columnas que el pipeline espera
    input_df['encounter_id'] = 0
    input_df['patient_nbr']   = 0
    
    with LATENCY.labels(method="predict").time():
        try:
            preds = model.predict(input_df)
            REQUEST_COUNT.labels(method="predict", status="200").inc()
            return {"prediction": int(preds[0])}
        except Exception as e:
            REQUEST_COUNT.labels(method="predict", status="500").inc()
            raise HTTPException(status_code=500, detail=f"Inferencia fallida: {e}")

# ─── Endpoint /metrics ──────────────────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

