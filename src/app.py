from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd
import numpy as np

# Flask busca templates/ y static/ en la ruta superior al archivo src/
app = Flask(__name__, template_folder=os.path.join("..", "templates"), static_folder=os.path.join("..", "static"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_logistico.pkl")

model = None
features = None

# Cargar el bundle/modelo (admite dos formatos: dict con "model" y "features", o objeto directo)
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle and "features" in bundle:
        model = bundle["model"]
        features = bundle["features"]
    else:
        # Si tu archivo contiene solo el estimator, pide al usuario que indique features en el código
        model = bundle
        features = None
        print("Modelo cargado pero no se encontró la lista 'features' en el pickle. Edita src/app.py para indicar las features.")
    print("Modelo cargado desde:", MODEL_PATH)
else:
    print(f"ERROR: modelo no encontrado en {MODEL_PATH}. Coloca models/modelo_logistico.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/features", methods=["GET"])
def get_features():
    if not features:
        return jsonify({"error": "No hay lista 'features' en el modelo. Edita el backend para definirla."}), 500
    # Devuelve la lista de nombres de columnas tal como está en el pipeline
    return jsonify({"features": features})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado en el servidor."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibió JSON."}), 400

    # Aceptamos dos formatos:
    # 1) {"features": {"edad": 52, "sexo": 1, ...}}
    # 2) {"features": [52, 1, ...]}   (orden según features list)
    payload = data.get("features")
    if payload is None:
        return jsonify({"error": "Falta la clave 'features' en el JSON."}), 400

    # Si recibimos objeto (mapa), lo convertimos a DataFrame en el orden correcto
    try:
        if isinstance(payload, dict):
            if not features:
                return jsonify({"error": "El modelo no incluye la lista 'features' y se envió un mapa. Ajusta el backend."}), 500
            # Aseguramos que todas las features requeridas estén presentes en el mapa
            missing = [f for f in features if f not in payload]
            if missing:
                return jsonify({"error": f"Faltan las siguientes columnas: {missing}"}), 400
            row = [payload[f] for f in features]
            X = pd.DataFrame([row], columns=features)
        elif isinstance(payload, list):
            # lista: asumimos que el orden coincide con features
            if features and len(payload) != len(features):
                return jsonify({"error": f"El número de valores ({len(payload)}) no coincide con el número de features ({len(features)})."}), 400
            if features:
                X = pd.DataFrame([payload], columns=features)
            else:
                # Si no tenemos features, generamos columnas genéricas
                cols = [f"f{i}" for i in range(len(payload))]
                X = pd.DataFrame([payload], columns=cols)
        else:
            return jsonify({"error": "Formato de 'features' no válido. Debe ser dict o lista."}), 400

        # Convertir a numpy si el pipeline lo requiere
        # Ejecutar predicción
        pred = model.predict(X)
        result = {"prediction": int(pred[0])}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # si es binario, proba[:,1] es la prob. de clase positiva
            if proba.shape[1] >= 2:
                result["probability"] = float(proba[0, 1])
            else:
                result["probability"] = float(proba[0, 0])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error durante la inferencia: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)