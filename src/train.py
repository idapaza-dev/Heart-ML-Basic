import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Rutas
# -----------------------------
DATA_PATH = os.path.join("data", "pacientes_prueba.csv")
MODEL_PATH = os.path.join("models", "modelo_logistico.pkl")
REPORT_PATH = os.path.join("models", "reporte_clasificacion.txt")

# -----------------------------
# 1. Cargar dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Features y target
# Aquí debes poner TODAS las columnas que quieras usar como entrada.
# Ejemplo con 13 columnas clínicas + hábitos:
feature_cols = [
    "edad", "sexo", "colesterol", "presion", "frecuencia", "glucosa", "angina",
    "tabaquismo", "alcohol", "actividad_fisica", "historial_familiar",
    "imc", "diabetes"
]
X = df[feature_cols]
y = df["enfermedad"]  # Etiqueta objetivo

# -----------------------------
# 3. Gráfico de conteo de clases
# -----------------------------
conteo = y.value_counts().sort_index()
labels = ["Saludable (0)", "Enfermo (1)"]
plt.bar(labels, conteo.values, color=["green", "red"])
plt.title("Cantidad de pacientes por clase")
plt.ylabel("Número de personas")
plt.savefig("plots/conteo_clases.png")
plt.close()

# -----------------------------
# 4. Split train/test (80/20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # ← 20% prueba, 80% entrenamiento
    stratify=y,
    random_state=42
)

# -----------------------------
# 5. Pipeline: escalado + regresión logística
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

# -----------------------------
# 6. Evaluación
# -----------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Reporte de clasificación:\n", report)

# Guardar reporte en archivo de texto
with open(REPORT_PATH, "w") as f:
    f.write("Accuracy: " + str(accuracy) + "\n\n")
    f.write("Reporte de clasificación:\n")
    f.write(report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.xticks([0,1], ["0", "1"])
plt.yticks([0,1], ["0", "1"])
plt.savefig("plots/matriz_confusion.png")
plt.close()

# -----------------------------
# 7. Guardar modelo
# -----------------------------
joblib.dump({"model": pipeline, "features": feature_cols}, MODEL_PATH)
print("Modelo guardado en:", MODEL_PATH)
print("Reporte guardado en:", REPORT_PATH)
