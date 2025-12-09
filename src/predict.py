# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Rutas
# -----------------------------
MODEL_PATH = os.path.join("models", "modelo_logistico.pkl")

# -----------------------------
# 1. Cargar modelo
# -----------------------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]   # ← columnas exactas usadas en entrenamiento

print("Modelo cargado correctamente.")
print("Columnas requeridas:", features)

# -----------------------------
# 2. Crear un nuevo paciente (EJEMPLO REAL)
#    * Debe tener las MISMAS columnas que features
# -----------------------------
nuevo_paciente = pd.DataFrame([[
    52,   # edad
    1,    # sexo
    1,    # dolor_pecho
    130,  # presion_reposo
    240,  # colesterol
    0,    # glucosa_ayunas_gt120
    1,    # ecg_reposo
    150,  # frecuencia_max
    0,    # angina_ejercicio
    1.2,  # depresion_st
    2,    # pendiente_st
    0,    # vasos_fluoroscopia
    3     # thal
]],
columns=features)

# -----------------------------
# 3. Realizar predicción
# -----------------------------
pred_binario = model.predict(nuevo_paciente)[0]
pred_porcentaje = model.predict_proba(nuevo_paciente)[0, 1] * 100

# -----------------------------
# 4. Mostrar resultados
# -----------------------------
print("\n=== RESULTADOS ===")
print("Predicción binaria (0=sano, 1=enfermo):", pred_binario)
print("Probabilidad de enfermedad (%):", round(pred_porcentaje, 2))
