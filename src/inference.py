# src/inference.py

import json
import numpy as np
import joblib

def init():
    global model
    model = joblib.load("model.pkl")  # Azure ML platziert das Modell hier

def run(data):
    try:
        inputs = json.loads(data)["data"]
        inputs = np.array(inputs)
        preds = model.predict(inputs)
        return preds.tolist()
    except Exception as e:
        return {"error": str(e)}
