# Práctica 4 - Ejemplo 2: Red neuronal de AJUSTE (fitting / regresión)
# Problema real (sector inmobiliario): estimar el valor medio de una vivienda (MEDV, miles de USD)
# Dataset: Boston Housing (UCI / StatLib), 506 zonas, 13 variables. Se excluye la variable "B" por sesgo ético.
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
def leer(fuente):
    d = pd.read_csv(fuente, header=None)
    if d.shape[1] == 1:                       # algunas copias usan espacios en lugar de comas
        d = pd.read_csv(fuente, header=None, sep=r"\s+")
    d.columns = cols
    return d
try:
    df = leer(URL)
except Exception:
    df = leer("housing.csv")
df = df.drop(columns=["B"])
print("Dataset:", df.shape, "| MEDV min=%.1f max=%.1f media=%.2f" % (df.MEDV.min(), df.MEDV.max(), df.MEDV.mean()))

X, y = df.drop(columns="MEDV").values, df["MEDV"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

red = Pipeline([("esc", StandardScaler()),
                ("mlp", MLPRegressor(max_iter=3000, early_stopping=True, n_iter_no_change=50, random_state=1))])
modelo = TransformedTargetRegressor(regressor=red, transformer=StandardScaler())
grid = {"regressor__mlp__hidden_layer_sizes": [(32,), (64, 32), (32, 16)],
        "regressor__mlp__activation": ["relu", "tanh"],
        "regressor__mlp__alpha": [1e-3, 1e-2, 1e-1],
        "regressor__mlp__learning_rate_init": [1e-3, 3e-3]}
busq = GridSearchCV(modelo, grid, cv=KFold(5, shuffle=True, random_state=42), scoring="r2", n_jobs=-1)
busq.fit(X_tr, y_tr)
print("Mejores parámetros:", {k.split("__")[-1]: v for k, v in busq.best_params_.items()})
print("R2 medio CV (5-fold): %.4f" % busq.best_score_)

mejor = busq.best_estimator_
y_pred = mejor.predict(X_te)
print("Resultados en prueba (%d viviendas):" % len(y_te))
print("  R2   = %.4f" % r2_score(y_te, y_pred))
print("  RMSE = %.3f mil USD" % np.sqrt(mean_squared_error(y_te, y_pred)))
print("  MAE  = %.3f mil USD" % mean_absolute_error(y_te, y_pred))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(y_te, y_pred, s=18, alpha=.75)
lim = [0, 52]; ax[0].plot(lim, lim, "r--", label="Ajuste perfecto")
ax[0].set_xlabel("Valor real MEDV"); ax[0].set_ylabel("Predicción"); ax[0].set_title("Real vs. predicho (prueba)")
ax[0].legend(); ax[0].grid(alpha=.3)
mlp = mejor.regressor_.named_steps["mlp"]
ax[1].plot(mlp.loss_curve_, label="Pérdida entrenamiento (MSE/2)")
ax[1].set_xlabel("Época"); ax[1].set_title("Curva de aprendizaje"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("resultado_regresion.png", dpi=130)

# Ejemplo de uso: estimar el valor de una zona nueva
nueva = X_te[:1]
print("Zona de ejemplo -> valor estimado: %.2f mil USD (real: %.2f)" % (mejor.predict(nueva)[0], y_te[0]))
