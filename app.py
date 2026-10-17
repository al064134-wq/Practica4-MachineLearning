"""Práctica 4 – Modelos de machine learning (Sistemas Inteligentes, UACAM).

Aplicación web (Streamlit) para probar las tres redes neuronales del reporte:
  1. Pattern Recognition (clasificación) – autenticación de billetes
  2. Fitting / Regresión – valor de viviendas
  3. Series de tiempo – reconocimiento de actividad humana con sensores

Ejecutar en local:  streamlit run app.py
"""
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import modelos as M
import graficas as G

ALUMNO = "Kevin Javier Ruiz Tillit"
USUARIO = "al064134-wq"
REPO = f"https://github.com/{USUARIO}/Practica4-MachineLearning"
COLAB = f"https://colab.research.google.com/github/{USUARIO}/Practica4-MachineLearning/blob/main/"
REPORTE = f"{REPO}/blob/main/reporte/PracticaMachineLearning_KevinJavierRuizTillit.pdf"
DIR_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

st.set_page_config(page_title="Redes Neuronales IA", page_icon="🤖", layout="wide")
st.markdown("""<style>
.arq {background:#0e1117; color:#e8eaed; border-left:4px solid #ff4b4b; border-radius:6px;
      padding:12px 16px; margin:6px 0 18px 0; font-family:monospace; font-size:0.9rem}
.arq b {color:#ff4b4b; letter-spacing:.05em}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- carga de modelos
def _cargar_o_entrenar(nombre, funcion):
    """Usa models/<nombre>.pkl si existe (generado con train_models.py); si no, entrena."""
    ruta = os.path.join(DIR_MODELOS, f"{nombre}.pkl")
    if os.path.exists(ruta):
        try:
            res = joblib.load(ruta)
            if "curvas" in res:
                return res
        except Exception:
            pass                       # versión distinta de scikit-learn -> se vuelve a entrenar
    res = funcion()
    try:
        os.makedirs(DIR_MODELOS, exist_ok=True)
        joblib.dump(res, ruta)
    except Exception:
        pass
    return res


@st.cache_resource(show_spinner="Entrenando la red de clasificación…")
def modelo_clasificacion():
    return _cargar_o_entrenar("model_classification", M.entrenar_clasificacion)


@st.cache_resource(show_spinner="Entrenando la red de regresión…")
def modelo_regresion():
    return _cargar_o_entrenar("model_regression", M.entrenar_regresion)


@st.cache_resource(show_spinner="Descargando el dataset HAR (~60 MB) y entrenando la red de series de tiempo… "
                                "(solo la primera vez, ~1-2 min)")
def modelo_series():
    return _cargar_o_entrenar("model_timeseries", M.entrenar_series)


def arquitectura(tipo, texto):
    st.markdown(f'<div class="arq"><b>ARQUITECTURA DE LA RED ({tipo})</b><br>{texto}</div>',
                unsafe_allow_html=True)


def grafica(titulo, fig):
    st.markdown(f"**{titulo}**")
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------- navegación
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Selecciona el modelo:", ["1. Pattern Recognition (Comercial)",
                                                    "2. Fitting / Regresión (Real)",
                                                    "3. Series de Tiempo (Industrial)"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Alumno:** {ALUMNO}  \nSistemas Inteligentes · UACAM")
st.sidebar.markdown(f"[📂 Repositorio en GitHub]({REPO})  \n[📄 Reporte PDF]({REPORTE})  \n"
                    f"[📓 Colab – Clasificación]({COLAB}1_clasificacion_billetes.ipynb)  \n"
                    f"[📓 Colab – Regresión]({COLAB}2_regresion_viviendas.ipynb)  \n"
                    f"[📓 Colab – Series de tiempo]({COLAB}3_series_tiempo_har.ipynb)")

st.title("🤖 Proyecto de Redes Neuronales")
st.write("Ejecución de modelos de Machine Learning para problemas Reales, Comerciales e Industriales.")

# ============================================================================= 1. CLASIFICACIÓN
if pagina.startswith("1"):
    st.header("💵 Autenticación de Billetes Bancarios (UCI Banknote)")
    r = modelo_clasificacion()
    arquitectura("MLPClassifier", r["arquitectura"])
    m = r["metricas"]
    a, b, c = st.columns(3)
    a.metric("Exactitud en prueba", f"{m['exactitud']:.2%}"); b.metric("F1 macro", f"{m['f1']:.3f}")
    c.metric("Billetes (entrenamiento / prueba)", f"{m['n_train']} / {m['n_test']}")

    st.subheader("Simulador de Predicción")
    st.caption("Características estadísticas de la transformada wavelet de la imagen del billete.")
    df = r["df"]; valores = []
    cols = st.columns(4)
    for col, nombre, d in zip(cols, M.COLS_BILLETES[:-1], [-1.39, 3.34, -1.99, -1.43]):
        with col:
            valores.append(st.slider(M.DESC_BILLETES[nombre], float(df[nombre].min()),
                                     float(df[nombre].max()), float(d), step=0.01))
    if st.button("Verificar Billete", type="primary"):
        prob = r["modelo"].predict_proba([valores])[0]
        if prob[1] < 0.5:
            st.success(f"✅ Billete **GENUINO** — probabilidad {prob[0]:.1%}")
        else:
            st.error(f"🚫 Billete **FALSO** — probabilidad {prob[1]:.1%}")
        st.progress(float(prob[1]), text=f"Probabilidad de falsificación: {prob[1]:.1%}")

    st.subheader("📈 Análisis de Entrenamiento (Estilo MATLAB)")
    yp_tr, yp_te = r["p_tr"].argmax(1), r["p_te"].argmax(1)
    grafica("1. Neural Network Training Performance (plotperform)",
            G.plotperform(r["curvas"], "Cross-Entropy (log-loss)"))
    grafica("2. Neural Network Training State (plottrainstate)", G.plottrainstate(r["curvas"]))
    grafica("3. Error Histogram (ploterrhist)",
            G.ploterrhist(r["y_tr"] - r["p_tr"][:, 1], r["y_te"] - r["p_te"][:, 1]))
    grafica("4. Confusion Matrix (plotconfusion)",
            G.plotconfusion([("Training Confusion Matrix", r["y_tr"], yp_tr),
                             ("Test Confusion Matrix", r["y_te"], yp_te),
                             ("All Confusion Matrix", np.r_[r["y_tr"], r["y_te"]], np.r_[yp_tr, yp_te])],
                            M.CLASES_BILLETES))
    grafica("5. Receiver Operating Characteristic (plotroc)",
            G.plotroc([("Training ROC", r["y_tr"], r["p_tr"]), ("Test ROC", r["y_te"], r["p_te"])],
                      M.CLASES_BILLETES))

# ============================================================================= 2. REGRESIÓN
elif pagina.startswith("2"):
    st.header("🏠 Estimación del Valor de Viviendas (Boston Housing)")
    r = modelo_regresion()
    arquitectura("MLPRegressor", r["arquitectura"])
    m = r["metricas"]
    a, b, c = st.columns(3)
    a.metric("R² en prueba", f"{m['r2']:.4f}"); b.metric("RMSE", f"{m['rmse']:.2f} mil USD")
    c.metric("MAE", f"{m['mae']:.2f} mil USD")

    st.subheader("Simulador de Predicción")
    df = r["df"]; valores = []
    rejilla = st.columns(4)
    for i, c in enumerate([c for c in df.columns if c != "MEDV"]):
        with rejilla[i % 4]:
            if c == "CHAS":
                valores.append(float(st.selectbox(M.DESC_VIVIENDAS[c], [0, 1],
                                                  format_func=lambda v: "Sí" if v else "No")))
            else:
                lo, hi, med = float(df[c].min()), float(df[c].max()), float(df[c].median())
                valores.append(st.slider(M.DESC_VIVIENDAS[c], lo, hi, med))
    if st.button("Estimar Valor de la Vivienda", type="primary"):
        pred = float(r["modelo"].predict([valores])[0])
        st.success(f"🏠 Valor medio estimado: **{pred:.2f} mil USD** (≈ {pred * 1000:,.0f} USD)")

    st.subheader("📈 Análisis de Entrenamiento (Estilo MATLAB)")
    grafica("1. Neural Network Training Performance (plotperform)",
            G.plotperform(r["curvas"], "Mean Squared Error (mse)"))
    grafica("2. Neural Network Training State (plottrainstate)", G.plottrainstate(r["curvas"]))
    grafica("3. Error Histogram (ploterrhist)",
            G.ploterrhist(r["y_tr"] - r["p_tr"], r["y_te"] - r["p_te"]))
    grafica("4. Neural Network Training Regression (plotregression)",
            G.plotregression([("Training", r["y_tr"], r["p_tr"]), ("Test", r["y_te"], r["p_te"]),
                              ("All", np.r_[r["y_tr"], r["y_te"]], np.r_[r["p_tr"], r["p_te"]])]))

# ============================================================================= 3. SERIES DE TIEMPO
else:
    st.header("🏃 Reconocimiento de Actividad Humana con Sensores (UCI HAR)")
    r = modelo_series()
    arquitectura("Time-Series / TDNN", r["arquitectura"])
    m = r["metricas"]
    a, b, c = st.columns(3)
    a.metric("Exactitud (9 personas nuevas)", f"{m['exactitud']:.2%}"); b.metric("F1 macro", f"{m['f1']:.3f}")
    c.metric("Épocas entrenadas", m["epocas"])

    st.subheader("Simulador de Lecturas de Sensores")
    mu = r["muestras"]
    escenario = st.radio("Escenario (elige la actividad que realiza la persona):",
                         ["Ventana #100"] + M.ACT + ["Aleatoria"], horizontal=True)
    if escenario == "Ventana #100" and 100 in mu["idx"]:
        pos = int(np.where(mu["idx"] == 100)[0][0])
    elif escenario in M.ACT:
        pos = int(np.where(mu["y"] == M.ACT.index(escenario))[0][0])
    else:
        pos = int(np.random.randint(len(mu["idx"])))
    X = mu["X"][pos:pos + 1]
    t = np.arange(128) / 50
    fig, ax = plt.subplots(1, 3, figsize=(13, 3), sharex=True)
    for g, nombre in enumerate(["Aceleración del cuerpo", "Giroscopio", "Aceleración total"]):
        for c in range(3):
            ax[g].plot(t, X[0, :, 3 * g + c], lw=1, label="xyz"[c])
        ax[g].set_title(nombre); ax[g].set_xlabel("Tiempo (s)"); ax[g].grid(alpha=.3); ax[g].legend(fontsize=7)
    fig.suptitle(f"Ventana #{mu['idx'][pos]} del conjunto de prueba (2.56 s, 9 canales a 50 Hz)", fontsize=10)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if st.button("Ejecutar Diagnóstico Temporal", type="primary"):
        prob = r["modelo"].predict_proba(M.entrada_har(X))[0]
        pred, real = int(prob.argmax()), int(mu["y"][pos])
        (st.success if pred == real else st.error)(
            f"Actividad detectada: **{M.ACT[pred]}** ({prob[pred]:.1%}) · actividad real: **{M.ACT[real]}**")
        st.bar_chart({"Probabilidad": {a: float(p) for a, p in zip(M.ACT, prob)}})

    st.subheader("📈 Análisis de Series de Tiempo (Estilo MATLAB)")
    yp_tr, yp_te = r["p_tr"].argmax(1), r["p_te"].argmax(1)
    grafica("1. Neural Network Training Performance (plotperform)",
            G.plotperform(r["curvas"], "Pérdida (train: log-loss · val: 1 − exactitud)"))
    grafica("2. Neural Network Training State (plottrainstate)", G.plottrainstate(r["curvas"]))
    err_tr = 1 - r["p_tr"][np.arange(len(r["y_tr"])), r["y_tr"]]
    err_te = 1 - r["p_te"][np.arange(len(r["y_te"])), r["y_te"]]
    grafica("3. Error Histogram (ploterrhist)",
            G.ploterrhist(err_tr, err_te, xlabel="Error = 1 − probabilidad de la clase real"))
    grafica("4. Confusion Matrix (plotconfusion)",
            G.plotconfusion([("Training Confusion Matrix", r["y_tr"], yp_tr),
                             ("Test Confusion Matrix", r["y_te"], yp_te)], M.ACT))
    grafica("5. Receiver Operating Characteristic (plotroc)", G.plotroc([("Test ROC", r["y_te"], r["p_te"])], M.ACT))
    n = min(300, len(r["y_te"]))
    grafica("6. Time-Series Response (plotresponse)", G.plotresponse(r["y_te"][:n], yp_te[:n], M.ACT))
    grafica("7. Error Autocorrelation (ploterrcorr)", G.ploterrcorr((r["y_te"] != yp_te).astype(float)))
