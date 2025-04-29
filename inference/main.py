from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Cargar el modelo desde MLflow



