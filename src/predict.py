import os
import joblib
import numpy as np

MODEL_PATH = os.path.join("models", "modelo_logistico.pkl")

# Cargar modelo
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

# Nuevo paciente (ejemplo)
import pandas as pd

# Usa las mismas columnas que en train.py
nuevo_paciente = pd.DataFrame([[52, 1, 240, 130, 150, 0, 1]],
                              columns=["edad", "sexo", "colesterol", "presion", "frecuencia", "azucar", "angina"])

pred_binario = model.predict(nuevo_paciente)[0]
pred_porcentaje = model.predict_proba(nuevo_paciente)[0,1] * 100


print("Predicción binaria:", pred_binario)
print("Probabilidad de enfermedad (%):", round(pred_porcentaje,2))
