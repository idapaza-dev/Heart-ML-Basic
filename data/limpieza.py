# -*- coding: utf-8 -*-
"""
Script: limpieza.py
Autor: Ivan (con asistencia de Rocky IA)
Descripción:
  - Lee cleveland.csv descargado de GitHub sin nombres de columna.
  - Asigna nombres oficiales del dataset Cleveland (UCI).
  - Limpia NaN, convierte tipos y crea la etiqueta binaria.
  - Guarda dos CSV limpios en carpeta /data:
        * cleveland_clean.csv  (inglés)
        * cleveland_clean_es.csv (español)
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
INPUT = DATA_DIR / "cleveland.csv"
OUTPUT_EN = DATA_DIR / "cleveland_clean.csv"
OUTPUT_ES = DATA_DIR / "cleveland_clean_es.csv"

# Nombres oficiales del dataset Cleveland (14 columnas)
columns_en = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

columns_es_map = {
    "age": "edad",
    "sex": "sexo",
    "cp": "dolor_pecho",
    "trestbps": "presion_reposo",
    "chol": "colesterol",
    "fbs": "glucosa_ayunas_gt120",
    "restecg": "ecg_reposo",
    "thalach": "frecuencia_max",
    "exang": "angina_ejercicio",
    "oldpeak": "depresion_st",
    "slope": "pendiente_st",
    "ca": "vasos_fluoroscopia",
    "thal": "thal",
    "num": "diagnostico"
}

def main():
    # 1) Leer CSV sin encabezados
    df = pd.read_csv(INPUT, header=None, names=columns_en)

    # 2) Reemplazar "?" por NaN y convertir a numérico
    df = df.replace("?", pd.NA)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Imputación simple
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())

    # 4) Crear etiqueta binaria
    df["target"] = (df["num"] > 0).astype(int)

    # 5) Guardar versión inglesa (oficial)
    df.to_csv(OUTPUT_EN, index=False)

    # 6) Guardar versión en español
    df_es = df.rename(columns=columns_es_map)
    df_es = df_es.rename(columns={"target": "enfermedad"})
    df_es.to_csv(OUTPUT_ES, index=False)

    print("✔ Limpieza completa")
    print(f"✔ Guardado EN -> {OUTPUT_EN}")
    print(f"✔ Guardado ES -> {OUTPUT_ES}")
    print(df.head())


if __name__ == "__main__":
    main()
