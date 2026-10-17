# Práctica 4 · Modelos de machine learning

**Sistemas Inteligentes** · Facultad de Ingeniería, Universidad Autónoma de Campeche
**Profesor:** Justino Ramírez Ortegón · **Alumno:** Kevin Javier Ruiz Tillit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://practica4-ml-kevin-ruiz.streamlit.app)

Tres redes neuronales de tipo perceptrón multicapa (MLP) implementadas en Python con scikit-learn, cada una aplicada a un problema real con un dataset público:

| # | Tipo de red | Problema real | Dataset | Resultado en prueba | Abrir en Colab |
|---|---|---|---|---|---|
| 1 | Reconocimiento de patrones (clasificación) | Autenticación de billetes genuinos / falsos | UCI Banknote Authentication | Exactitud 100 % | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/al064134-wq/Practica4-MachineLearning/blob/main/1_clasificacion_billetes.ipynb) |
| 2 | Ajuste (fitting / regresión) | Valor medio de viviendas por zona | Boston Housing (UCI / StatLib) | R² = 0.859 · MAE ≈ 2.0 mil USD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/al064134-wq/Practica4-MachineLearning/blob/main/2_regresion_viviendas.ipynb) |
| 3 | Series de tiempo (TDNN) | Reconocimiento de actividad humana con sensores del smartphone | UCI HAR Smartphones | Exactitud 93.5 % | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/al064134-wq/Practica4-MachineLearning/blob/main/3_series_tiempo_har.ipynb) |

## Enlaces

- **Aplicación web interactiva (Streamlit):** https://practica4-ml-kevin-ruiz.streamlit.app
- **Reporte de la práctica (PDF):** [reporte/PracticaMachineLearning_KevinJavierRuizTillit.pdf](reporte/PracticaMachineLearning_KevinJavierRuizTillit.pdf)
- **Formato de reporte de programa:** [reporte/FORMATO_REPORTE_DE_PROGRAMA.pdf](reporte/FORMATO_REPORTE_DE_PROGRAMA.pdf)

## Estructura del repositorio

```
Practica4-MachineLearning/
├── 1_clasificacion_billetes.py / .ipynb   # Ejemplo 1 – red de reconocimiento de patrones
├── 2_regresion_viviendas.py / .ipynb      # Ejemplo 2 – red de ajuste (regresión)
├── 3_series_tiempo_har.py / .ipynb        # Ejemplo 3 – red de series de tiempo
├── app.py                                 # Aplicación web (Streamlit) con los 3 modelos
├── modelos.py                             # Carga de datos y entrenamiento usado por la app
├── graficas.py                            # Gráficas estilo MATLAB (plotperform, plotconfusion, plotroc…)
├── train_models.py                        # (Opcional) pre-entrena y guarda models/*.pkl
├── models/                                # Modelos entrenados (.pkl)
├── reporte/                               # Reporte PDF y formato de reporte
└── requirements.txt
```

## Cómo ejecutarlo

**En Google Colab (sin instalar nada):** abre el botón *Open in Colab* del ejemplo, elige *Entorno de ejecución → Ejecutar todas* (Ctrl + F9) y, si aparece un aviso, pulsa *Ejecutar de todos modos*. El cuaderno descarga el dataset, ajusta los hiperparámetros, entrena la red y muestra las métricas y gráficas.

**En local:**

```bash
pip install -r requirements.txt
python 1_clasificacion_billetes.py
python 2_regresion_viviendas.py
python 3_series_tiempo_har.py
streamlit run app.py          # aplicación web en http://localhost:8501
```

Cada script imprime los resultados en consola y guarda la gráfica como `resultado_*.png`.

## Datasets

Se descargan automáticamente del repositorio público [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) (espejo del UCI Machine Learning Repository).

## Créditos

Código generado con apoyo de IA generativa (Claude Opus 5.5 de Anthropic), revisado, ejecutado y validado por el alumno.
