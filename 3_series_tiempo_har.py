# Práctica 4 - Ejemplo 3: Red neuronal de SERIES DE TIEMPO (clasificación de series temporales)
# Problema real (industria de wearables / salud ocupacional): reconocer la actividad de una persona
# a partir de las señales del acelerómetro y giroscopio de un smartphone (50 Hz).
# Dataset: UCI Human Activity Recognition Using Smartphones (30 sujetos, 6 actividades,
# ventanas de 2.56 s = 128 pasos de tiempo x 9 canales).
# Modelo: red neuronal de retardo temporal (TDNN / MLP con ventana de tiempo) sobre las series crudas.
import os, zipfile, urllib.request, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/HAR_Smartphones.zip"
BASE = "UCI HAR Dataset"
if not os.path.isdir(BASE):
    if not os.path.exists("HAR_Smartphones.zip"):
        print("Descargando dataset (~60 MB)..."); urllib.request.urlretrieve(URL, "HAR_Smartphones.zip")
    zipfile.ZipFile("HAR_Smartphones.zip").extractall(".")

CANALES = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
ACT = ["Caminar", "Subir esc.", "Bajar esc.", "Sentado", "De pie", "Acostado"]

def cargar(parte):
    señales = [np.loadtxt(f"{BASE}/{parte}/Inertial Signals/{c}_{parte}.txt") for c in CANALES]
    X = np.stack(señales, axis=-1)                      # (muestras, 128 pasos, 9 canales)
    y = np.loadtxt(f"{BASE}/{parte}/y_{parte}.txt").astype(int) - 1
    return X, y

X_tr, y_tr = cargar("train"); X_te, y_te = cargar("test")
print("Series de entrenamiento:", X_tr.shape, "| prueba:", X_te.shape, "(muestras, pasos, canales)")

def entrada(X):
    # Ventana temporal cruda + magnitud espectral (FFT) de cada canal -> vector de entrada de la red
    fft = np.abs(np.fft.rfft(X, axis=1))[:, 1:33, :]       # primeras 32 frecuencias (0.4-12.5 Hz)
    return np.hstack([X.reshape(len(X), -1), np.log1p(fft).reshape(len(X), -1)])

esc = StandardScaler().fit(entrada(X_tr))
A_tr, A_te = esc.transform(entrada(X_tr)), esc.transform(entrada(X_te))

# Parámetros ajustados (seleccionados por validación sobre sujetos de entrenamiento)
red = MLPClassifier(hidden_layer_sizes=(256, 64), activation="relu", alpha=1e-2, batch_size=128,
                    learning_rate_init=1e-3, max_iter=60, early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=8, random_state=7)
red.fit(A_tr, y_tr)
y_pred = red.predict(A_te)
print("Épocas entrenadas:", red.n_iter_)
print("Exactitud en prueba (9 sujetos nunca vistos): %.4f" % accuracy_score(y_te, y_pred))
print(classification_report(y_te, y_pred, target_names=ACT, digits=4))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
t = np.arange(128) / 50
for i, a in enumerate([0, 3, 5]):
    k = np.where(y_tr == a)[0][0]; ax[0].plot(t, X_tr[k, :, 6] + i * 1.5, label=ACT[a])
ax[0].set_title("Serie total_acc_x por actividad"); ax[0].set_xlabel("Tiempo (s)"); ax[0].legend(fontsize=8)
ax[1].plot(red.loss_curve_, label="Pérdida entrenamiento"); ax[1].plot(1 - np.array(red.validation_scores_), label="Error validación")
ax[1].set_title("Curva de aprendizaje"); ax[1].set_xlabel("Época"); ax[1].legend(); ax[1].grid(alpha=.3)
ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred), display_labels=ACT).plot(ax=ax[2], cmap="Greens", colorbar=False, xticks_rotation=45)
ax[2].set_title("Matriz de confusión (prueba)")
plt.tight_layout(); plt.savefig("resultado_series_tiempo.png", dpi=130)

# Ejemplo de uso: clasificar una ventana de 2.56 s nueva
print("Ventana #100 de prueba -> predicción:", ACT[y_pred[100]], "| real:", ACT[y_te[100]])
