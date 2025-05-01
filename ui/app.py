import os
import streamlit as st
import requests
import pandas as pd

# Configuración de la API y modelo
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Configuración de la página
st.set_page_config(page_title="Diabetes Readmission Predictor", layout="centered")

# Título e información de modelo
st.title("Predicción de Readmisión por Diabetes")
st.markdown(
    f"**Modelo usado:** {MODEL_NAME} (Stage: {MODEL_STAGE})"
)
st.markdown(
    "Completa los campos y haz click en **Predecir** para saber si habrá readmisión."
)

# --- Formulario de entrada ---
with st.form("predict_form"):
    # Numéricos
    age                   = st.number_input("Age",                  min_value=0.0,   max_value=120.0, value=50.0)
    time_in_hospital      = st.number_input("Time in Hospital (days)", min_value=0, max_value=365,   value=5)
    num_lab_procedures    = st.number_input("Num Lab Procedures",     min_value=0,     max_value=200,  value=40)
    num_procedures        = st.number_input("Num Procedures",         min_value=0,     max_value=50,   value=1)
    num_medications       = st.number_input("Num Medications",        min_value=0,     max_value=100,  value=10)
    number_outpatient     = st.number_input("Number Outpatient Visits", min_value=0,   max_value=50,   value=0)
    number_emergency      = st.number_input("Number Emergency Visits",  min_value=0,   max_value=50,   value=0)
    number_inpatient      = st.number_input("Number Inpatient Visits",  min_value=0,   max_value=50,   value=0)
    number_diagnoses      = st.number_input("Number of Diagnoses",     min_value=0,     max_value=20,   value=5)

    # Categóricos
    race                  = st.selectbox("Race",                  ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
    gender                = st.selectbox("Gender",                ["Male", "Female"])
    admission_type_id     = st.selectbox("Admission Type",        [1, 2, 3, 4, 5, 6, 7, 8, 9])
    discharge_disposition_id = st.selectbox("Discharge Disposition", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    admission_source_id   = st.selectbox("Admission Source",      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    diag_1                = st.text_input("Primary Diagnosis (diag_1)", "")
    diag_2                = st.text_input("Secondary Diagnosis (diag_2)", "")
    diag_3                = st.text_input("Tertiary Diagnosis (diag_3)", "")

    # Medicamentos
    metformin             = st.selectbox("Metformin",             ["Yes", "No", "Steady", "Up", "Down"])
    insulin               = st.selectbox("Insulin",               ["Yes", "No", "Steady", "Up", "Down"])

    submitted = st.form_submit_button("Predecir")

# --- Llamada a la API y presentación de resultados ---
if submitted:
    payload = {
        "age": age,
        "race": race,
        "gender": gender,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "metformin": metformin,
        "insulin": insulin
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        pred = data.get("prediction")
        model_used = data.get("model_name", f"{MODEL_NAME} ({MODEL_STAGE})")
        st.success(f"Predicción: {pred} (1 = Readmitirá, 0 = No readmitirá)")
        st.info(f"Modelo usado: {model_used}")
    except requests.exceptions.RequestException as err:
        st.error(f"Error al llamar a la API: {err}")
    except KeyError:
        st.error("Respuesta inesperada de la API.")

