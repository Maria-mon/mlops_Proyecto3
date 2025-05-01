import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Parámetros por defecto para las tareas
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Configuración de MLflow desde variables de entorno
MLFLOW_URI      = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
S3_BUCKET       = os.getenv('MLFLOW_S3_BUCKET',  'mlflow-artifacts')
EXPERIMENT_NAME = 'diabetes_readmission'
MODEL_NAME      = 'diabetes_model'

# Inicializar MLflow y asegurar existencia del experimento
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()
if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    client.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=f"s3://{S3_BUCKET}/{EXPERIMENT_NAME}"
    )
mlflow.set_experiment(EXPERIMENT_NAME)

# Función auxiliar para preparar datos y crear preprocesador
def _prepare_data():
    hook   = PostgresHook('postgres_default')
    engine = hook.get_sqlalchemy_engine()
    df     = pd.read_sql('SELECT * FROM clean_data', engine)
    X      = df.drop('readmitted', axis=1)
    y      = (df['readmitted'] != 'NO').astype(int)
    # Detectar columnas numéricas y categóricas
    numeric_cols     = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # Construir preprocesador evitando one-hot de alta cardinalidad
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ], remainder='drop')
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

# Función genérica de entrenamiento
def _train_model(model, run_name, **kwargs):
    (X_train, X_val, y_train, y_val), preprocessor = _prepare_data()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    with mlflow.start_run(run_name=run_name):
        # Registrar parámetros básicos
        mlflow.log_param('model_type', run_name)
        # Entrenar y evaluar
        pipeline.fit(X_train, y_train)
        val_acc = pipeline.score(X_val, y_val)
        mlflow.log_metric('val_accuracy', val_acc)
        # Registrar pipeline completo
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path='model',
            registered_model_name=MODEL_NAME
        )

# Tarea RandomForest
def train_random_forest(**kwargs):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    _train_model(rf, 'RandomForest')

# Tarea LogisticRegression
def train_logistic_regression(**kwargs):
    lr = LogisticRegression(max_iter=200)
    _train_model(lr, 'LogisticRegression')

def select_and_promote(**kwargs):
    client = MlflowClient()
    # Recoge todas las versiones registradas del modelo
    all_versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    if not all_versions:
        raise ValueError(f"No hay versiones registradas de {MODEL_NAME}")

    # Elige la versión con el número más alto
    latest = max(all_versions, key=lambda v: int(v.version))
    latest_version = latest.version

    # Promueve la más reciente a Production (archivando la anterior)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage='Production',
        archive_existing_versions=True
    )
    print(f"Promovida la versión {latest_version} de '{MODEL_NAME}' a Production.")

# Definición del DAG
dag = DAG(
    'training_and_selection',
    default_args=default_args,
    description='Entrena dos modelos y selecciona el mejor en MLflow',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 27),
    catchup=False,
    tags=['training', 'mlflow']
)

t1 = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    dag=dag
)
t2 = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_logistic_regression,
    dag=dag
)
t3 = PythonOperator(
    task_id='select_and_promote',
    python_callable=select_and_promote,
    dag=dag
)

t1 >> t2 >> t3

