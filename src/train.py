import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV

# -----------------------------
# Rutas
# -----------------------------
DATA_PATH = os.path.join("data", "cleveland_clean_es.csv")
MODEL_PATH = os.path.join("models", "modelo_logistico.pkl")
REPORT_PATH = os.path.join("models", "reporte_clasificacion.txt")

# -----------------------------
# 1. Cargar dataset real
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. Features y target
# Columnas reales del Cleveland dataset
# -----------------------------
feature_cols = [
    "edad",
    "sexo",
    "dolor_pecho",
    "presion_reposo",
    "colesterol",
    "glucosa_ayunas_gt120",
    "ecg_reposo",
    "frecuencia_max",
    "angina_ejercicio",
    "depresion_st",
    "pendiente_st",
    "vasos_fluoroscopia",
    "thal"
]

# X = datos de entrada
X = df[feature_cols]

# y = etiqueta objetivo (0 = sano, 1 = riesgo/enfermedad)
y = df["enfermedad"]

# -----------------------------
# 3. Gráfico balance de clases
# -----------------------------
conteo = y.value_counts().sort_index()
plt.bar(["Sanos (0)", "Enfermos (1)"], conteo.values)
plt.title("Distribución de clases")
plt.ylabel("Pacientes")
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/clases.png")
plt.close()

# -----------------------------
# 4. Dividir datos Train/Test
# Recomiendo 70/30 para dataset pequeño (~300)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# -----------------------------
# 5. Pipeline con regularización L2
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        C=0.7
    ))
])

# -----------------------------
# 6. Validación cruzada
# -----------------------------
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print("Accuracy promedio CV:", cv_scores.mean())

# -----------------------------
# 7. Entrenar modelo base
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# 8. Calibración probabilística
# -----------------------------
calibrated_model = CalibratedClassifierCV(pipeline, cv=5)
calibrated_model.fit(X_train, y_train)

# -----------------------------
# 9. Predicciones
# -----------------------------
y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy final:", accuracy)
print(report)

# Guardar reporte
with open(REPORT_PATH, "w") as f:
    f.write("Accuracy: " + str(accuracy) + "\n\n")
    f.write("Cross Validation (5 folds): " + str(cv_scores.mean()) + "\n\n")
    f.write(report)

# -----------------------------
# 10. Matriz de confusión
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.savefig("plots/matriz_confusion.png")
plt.close()

# -----------------------------
# 11. Curva ROC + AUC
# -----------------------------
roc_auc = roc_auc_score(y_test, y_proba)
print("AUC:", roc_auc)

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC")
plt.savefig("plots/roc_curve.png")
plt.close()

# -----------------------------
# 12. Guardar modelo final
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump({
    "model": calibrated_model,
    "features": feature_cols
}, MODEL_PATH)

print("\nModelo guardado en:", MODEL_PATH)
print("Reporte guardado en:", REPORT_PATH)

