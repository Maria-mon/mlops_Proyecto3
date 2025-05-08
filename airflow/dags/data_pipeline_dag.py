from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import os
import requests
import pandas as pd


def download_dataset(**kwargs):
    data_root = '/opt/airflow/datasets/Diabetes'
    os.makedirs(data_root, exist_ok=True)
    data_filepath = os.path.join(data_root, 'Diabetes.csv')
    if not os.path.isfile(data_filepath):
        url = (
            'https://docs.google.com/uc?export=download'
            '&confirm=&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
        )
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(data_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        print(f"Dataset already exists at {data_filepath}")


def extract_to_raw(**kwargs):
    data_path = '/opt/airflow/datasets/Diabetes/Diabetes.csv'
    df = pd.read_csv(data_path)
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    df.to_sql('raw_data', engine, if_exists='replace', index=False)


def transform_to_clean(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    df = pd.read_sql('SELECT * FROM raw_data', engine)

    # 1) Drop duplicates and critical nulls
    df = df.drop_duplicates().dropna(subset=['race', 'gender', 'age'])

    # 2) Select only a manageable set of features
    keep_columns = [
        'encounter_id', 'patient_nbr',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'race', 'gender', 'age', 'readmitted'
    ]
    df = df[keep_columns]

    # 3) One-hot encode key categorical columns
    cat_cols = ['race', 'gender', 'age']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4) Write to clean_data table
    df.to_sql('clean_data', engine, if_exists='replace', index=False)


# Default task arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='data_pipeline',
    default_args=default_args,
    description='Pipeline diario: descarga, extracción y limpieza del dataset de diabetes',
    schedule_interval='@daily',
    start_date=datetime(2025, 4, 25),
    catchup=False,
    tags=['diabetes', 'etl']
) as dag:

    download_task = PythonOperator(
        task_id='download_dataset',
        python_callable=download_dataset
    )

    extract_task = PythonOperator(
        task_id='extract_to_raw',
        python_callable=extract_to_raw
    )

    transform_task = PythonOperator(
        task_id='transform_to_clean',
        python_callable=transform_to_clean
    )

    download_task >> extract_task >> transform_task




