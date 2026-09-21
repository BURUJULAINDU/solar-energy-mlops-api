from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib, json
from pathlib import Path
import pandas as pd

app = FastAPI(title="Solar Energy Consumption Prediction API")
MODEL_PATH = Path("solar_model.pkl")
INFO_PATH = Path("model_info.json")

class SolarInput(BaseModel):
    solar_irradiance_w_m2: float
    temperature_c: float
    humidity_percent: float
    energy_generated_kwh: float
    hour: int
    day_of_week: int
    month: int

def get_info():
    return json.loads(INFO_PATH.read_text())

@app.get("/")
def home():
    info = get_info()
    return {
        "message": "Solar Energy Consumption Prediction API is running",
        "model_version": info["model_version"],
        "training_rows": info["training_rows"]
    }

@app.get("/model-info")
def model_info():
    return get_info()

@app.post("/predict")
def predict(solar: SolarInput):
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([solar.model_dump()])
    value = float(model.predict(row[get_info()["features"]])[0])
    info = get_info()
    return {
        "predicted_energy_consumption_kwh": value,
        "model_version": info["model_version"],
        "training_rows": info["training_rows"]
    }
