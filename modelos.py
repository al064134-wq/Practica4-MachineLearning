"""Carga de datos y entrenamiento de los 3 modelos de la Práctica 4.

Lo usan app.py (Streamlit) y train_models.py. Los hiperparámetros son los
mejores encontrados por la búsqueda en rejilla de los scripts 1_, 2_ y 3_.
Además del modelo final se registran, época por época, las curvas de
entrenamiento / validación / prueba para las gráficas estilo MATLAB.
"""
import io, os, zipfile, urllib.request
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             r2_score, mean_squared_error, mean_absolute_error)

RAIZ = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
DIR_DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def _origen(nombre):
    local = os.path.join(DIR_DATOS, nombre)
    return local if os.path.exists(local) else RAIZ + nombre


def _monitor(red, X_tr, y_tr, X_val, y_val, X_te, y_te, epocas, perdida, clases=None):
    """Réplica de la red entrenada época por época (partial_fit) para registrar
    la pérdida en entrenamiento, validación y prueba (equivalente a plotperform)."""
    curvas = {"train": [], "val": [], "test": []}
    for _ in range(epocas):
        if clases is not None:
            red.partial_fit(X_tr, y_tr, classes=clases)
        else:
            red.partial_fit(X_tr, y_tr)
        for k, (X, y) in {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}.items():
            curvas[k].append(perdida(red, X, y))
    return curvas


# =========================================================================== 1. Clasificación
COLS_BILLETES = ["varianza", "asimetria", "curtosis", "entropia", "clase"]
CLASES_BILLETES = ["Genuino", "Falso"]
DESC_BILLETES = {"varianza": "Varianza de la imagen (wavelet)", "asimetria": "Asimetría (skewness)",
                 "curtosis": "Curtosis", "entropia": "Entropía de la imagen"}


def datos_billetes():
    return pd.read_csv(_origen("banknote_authentication.csv"), header=None, names=COLS_BILLETES)


def entrenar_clasificacion():
    df = datos_billetes()
    X, y = df[COLS_BILLETES[:-1]].values, df["clase"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    params = dict(hidden_layer_sizes=(8,), activation="relu", alpha=1e-4, learning_rate_init=1e-2,
                  random_state=42)
    modelo = Pipeline([("esc", StandardScaler()), ("mlp", MLPClassifier(max_iter=1500, **params))])
    modelo.fit(X_tr, y_tr)

    # curvas de desempeño (15 % del entrenamiento como validación)
    esc = modelo.named_steps["esc"]
    a, v, ya, yv = train_test_split(esc.transform(X_tr), y_tr, test_size=0.15, stratify=y_tr, random_state=0)
    curvas = _monitor(MLPClassifier(**params), a, ya, v, yv, esc.transform(X_te), y_te,
                      epocas=max(60, len(modelo.named_steps["mlp"].loss_curve_)),
                      perdida=lambda r, X_, y_: log_loss(y_, r.predict_proba(X_), labels=[0, 1]),
                      clases=np.array([0, 1]))
    return {"modelo": modelo, "df": df,
            "arquitectura": "[Capa de Entrada: 4 Neuronas] ➔ [Capa Oculta: 8 Neuronas (ReLU)] ➔ "
                            "[Capa de Salida: 1 Neurona (Logística)]",
            "metricas": {"exactitud": accuracy_score(y_te, modelo.predict(X_te)),
                         "f1": f1_score(y_te, modelo.predict(X_te), average="macro"),
                         "n_train": len(y_tr), "n_test": len(y_te)},
            "y_tr": y_tr, "p_tr": modelo.predict_proba(X_tr), "y_te": y_te, "p_te": modelo.predict_proba(X_te),
            "curvas": curvas, "loss": list(modelo.named_steps["mlp"].loss_curve_)}


# =========================================================================== 2. Regresión
COLS_VIVIENDAS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
                  "PTRATIO", "B", "LSTAT", "MEDV"]
DESC_VIVIENDAS = {
    "CRIM": "Criminalidad per cápita", "ZN": "% suelo residencial en lotes grandes",
    "INDUS": "% superficie industrial", "CHAS": "Colinda con el río",
    "NOX": "Óxidos de nitrógeno (ppm/10)", "RM": "Habitaciones por vivienda",
    "AGE": "% viviendas anteriores a 1940", "DIS": "Distancia a centros de empleo",
    "RAD": "Acceso a autopistas (índice)", "TAX": "Impuesto predial (por 10 000 USD)",
    "PTRATIO": "Alumnos por maestro", "LSTAT": "% población de bajos ingresos"}


def datos_viviendas():
    fuente = _origen("housing.csv")
    df = pd.read_csv(fuente, header=None)
    if df.shape[1] == 1:
        df = pd.read_csv(fuente, header=None, sep=r"\s+")
    df.columns = COLS_VIVIENDAS
    return df.drop(columns=["B"])


def entrenar_regresion():
    df = datos_viviendas()
    X, y = df.drop(columns="MEDV").values, df["MEDV"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    params = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-2, learning_rate_init=3e-3,
                  random_state=1)
    red = Pipeline([("esc", StandardScaler()),
                    ("mlp", MLPRegressor(max_iter=3000, early_stopping=True, n_iter_no_change=50, **params))])
    modelo = TransformedTargetRegressor(regressor=red, transformer=StandardScaler())
    modelo.fit(X_tr, y_tr)

    # curvas de desempeño en MSE (miles de USD²)
    esc_x = StandardScaler().fit(X_tr)
    esc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    zx = lambda A: esc_x.transform(A)
    zy = lambda b: esc_y.transform(b.reshape(-1, 1)).ravel()
    a, v, ya, yv = train_test_split(zx(X_tr), zy(y_tr), test_size=0.1, random_state=0)
    sd = float(esc_y.scale_[0])
    curvas = _monitor(MLPRegressor(**params), a, ya, v, yv, zx(X_te), zy(y_te),
                      epocas=len(modelo.regressor_.named_steps["mlp"].loss_curve_),
                      perdida=lambda r, X_, y_: mean_squared_error(y_, r.predict(X_)) * sd ** 2)
    p_tr, p_te = modelo.predict(X_tr), modelo.predict(X_te)
    return {"modelo": modelo, "df": df,
            "arquitectura": "[Capa de Entrada: 12 Neuronas] ➔ [Capa Oculta 1: 64 Neuronas (ReLU)] ➔ "
                            "[Capa Oculta 2: 32 Neuronas (ReLU)] ➔ [Capa de Salida: 1 Neurona (Lineal)]",
            "metricas": {"r2": r2_score(y_te, p_te), "rmse": float(np.sqrt(mean_squared_error(y_te, p_te))),
                         "mae": mean_absolute_error(y_te, p_te), "n_train": len(y_tr), "n_test": len(y_te)},
            "y_tr": y_tr, "p_tr": p_tr, "y_te": y_te, "p_te": p_te,
            "curvas": curvas, "loss": list(modelo.regressor_.named_steps["mlp"].loss_curve_)}


# =========================================================================== 3. Series de tiempo
CANALES = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
ACT = ["Caminar", "Subir esc.", "Bajar esc.", "Sentado", "De pie", "Acostado"]


def _zip_har():
    local = os.path.join(DIR_DATOS, "HAR_Smartphones.zip")
    if not os.path.exists(local):
        os.makedirs(DIR_DATOS, exist_ok=True)
        urllib.request.urlretrieve(RAIZ + "HAR_Smartphones.zip", local)
    return zipfile.ZipFile(local)


def datos_har():
    z = _zip_har()
    nombres = z.namelist()

    def leer(ruta):
        coincide = [n for n in nombres if n.endswith(ruta) and "__MACOSX" not in n]
        return np.loadtxt(io.TextIOWrapper(z.open(coincide[0])), dtype=np.float32)

    def cargar(parte):
        X = np.stack([leer(f"{parte}/Inertial Signals/{c}_{parte}.txt") for c in CANALES], axis=-1)
        y = leer(f"{parte}/y_{parte}.txt").astype(int) - 1
        return X, y

    X_tr, y_tr = cargar("train")
    X_te, y_te = cargar("test")
    return X_tr, y_tr, X_te, y_te


def entrada_har(X):
    """Ventana temporal cruda + magnitud espectral (FFT) de cada canal."""
    fft = np.abs(np.fft.rfft(X, axis=1))[:, 1:33, :]
    return np.hstack([X.reshape(len(X), -1), np.log1p(fft).reshape(len(X), -1)]).astype(np.float32)


def entrenar_series():
    X_tr, y_tr, X_te, y_te = datos_har()
    modelo = Pipeline([("esc", StandardScaler()),
                       ("mlp", MLPClassifier(hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-2,
                                             batch_size=128, learning_rate_init=1e-3, max_iter=60,
                                             early_stopping=True, validation_fraction=0.15,
                                             n_iter_no_change=8, random_state=7))])
    A_tr, A_te = entrada_har(X_tr), entrada_har(X_te)
    modelo.fit(A_tr, y_tr)
    p_tr, p_te = modelo.predict_proba(A_tr), modelo.predict_proba(A_te)
    del A_tr, A_te
    mlp = modelo.named_steps["mlp"]
    # ventanas guardadas para el simulador: una por actividad + muestra aleatoria
    rng = np.random.default_rng(0)
    idx = np.unique(np.concatenate([[100], [np.where(y_te == a)[0][5] for a in range(6)],
                                    rng.choice(len(y_te), 200, replace=False)]))
    return {"modelo": modelo,
            "arquitectura": "[Capa de Entrada: 1 440 (ventana 128×9 + FFT 32×9)] ➔ [Capa Oculta 1: 256 Neuronas] ➔ "
                            "[Capa Oculta 2: 64 Neuronas] ➔ [Capa de Salida: 6 Actividades (Softmax)]",
            "metricas": {"exactitud": accuracy_score(y_te, p_te.argmax(1)),
                         "f1": f1_score(y_te, p_te.argmax(1), average="macro"),
                         "epocas": mlp.n_iter_, "n_train": len(y_tr), "n_test": len(y_te)},
            "y_tr": y_tr, "p_tr": p_tr.astype(np.float32), "y_te": y_te, "p_te": p_te.astype(np.float32),
            "curvas": {"train": list(mlp.loss_curve_), "val": list(1 - np.array(mlp.validation_scores_))},
            "muestras": {"idx": idx, "X": X_te[idx], "y": y_te[idx]}}
