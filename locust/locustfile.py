from locust import HttpUser, task, between
from requests.exceptions import RequestException

class InferenceUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def send_prediction(self):
        payload = {
            "age": 50,
            "race": "Caucasian",
            "gender": "Female",
            "admission_type_id": 1,
            "discharge_disposition_id": 1,
            "admission_source_id": 7,
            "time_in_hospital": 3,
            "num_lab_procedures": 41,
            "num_procedures": 0,
            "num_medications": 13,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "diag_1": "428",
            "diag_2": "414",
            "diag_3": "786",
            "number_diagnoses": 9,
            "metformin": "No",
            "insulin": "Steady"
        }

        try:
            response = self.client.post("/predict", json=payload, timeout=10)
            if response.status_code != 200:
                print(f"❌ Error {response.status_code}: {response.text}")
            else:
                print(f"✅ Predicción correcta: {response.text}")
        except RequestException as e:
            print(f"❌ Excepción de conexión: {str(e)}")
