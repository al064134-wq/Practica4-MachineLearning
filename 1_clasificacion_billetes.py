# Práctica 4 - Ejemplo 1: Red neuronal de RECONOCIMIENTO DE PATRONES (clasificación)
# Problema real (industria bancaria): autenticación de billetes (genuino vs. falso)
# Dataset: UCI Banknote Authentication (1372 muestras, 4 características de la transformada wavelet)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"
cols = ["varianza", "asimetria", "curtosis", "entropia", "clase"]
try:
    df = pd.read_csv(URL, header=None, names=cols)
except Exception:
    df = pd.read_csv("banknote_authentication.csv", header=None, names=cols)
print("Dataset:", df.shape, "| clases:", df["clase"].value_counts().to_dict(), "(0=genuino, 1=falso)")

X, y = df[cols[:-1]].values, df["clase"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pipe = Pipeline([("esc", StandardScaler()),
                 ("mlp", MLPClassifier(max_iter=1500, random_state=42))])
grid = {"mlp__hidden_layer_sizes": [(8,), (16,), (16, 8)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-3, 1e-2]}
busq = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy", n_jobs=-1)
busq.fit(X_tr, y_tr)
print("Mejores parámetros:", busq.best_params_)
print("Exactitud media CV (5-fold): %.4f" % busq.best_score_)

modelo = busq.best_estimator_
y_pred = modelo.predict(X_te)
print("Exactitud en prueba: %.4f" % accuracy_score(y_te, y_pred))
print(classification_report(y_te, y_pred, target_names=["Genuino", "Falso"], digits=4))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=["Genuino", "Falso"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Matriz de confusión (prueba)")
mlp = modelo.named_steps["mlp"]
ax[1].plot(mlp.loss_curve_, label="Pérdida entrenamiento")
ax[1].set_ylabel("Log-loss")
ax[1].set_xlabel("Época"); ax[1].set_title("Curva de aprendizaje"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("resultado_clasificacion.png", dpi=130)

# Ejemplo de uso: clasificar un billete nuevo
nuevo = [[-1.39, 3.34, -1.99, -1.43]]
print("Billete nuevo", nuevo[0], "->", ["GENUINO", "FALSO"][modelo.predict(nuevo)[0]])
