"""Entrena los 3 modelos y los guarda en models/*.pkl (opcional).

La app de Streamlit entrena automáticamente si no encuentra los .pkl,
así que este script solo sirve para pre-entrenar en local:
    python train_models.py
"""
import os, joblib
import modelos as M

os.makedirs("models", exist_ok=True)
for nombre, funcion in [("model_classification", M.entrenar_clasificacion),
                        ("model_regression", M.entrenar_regresion),
                        ("model_timeseries", M.entrenar_series)]:
    print(f"Entrenando {nombre}...")
    res = funcion()
    joblib.dump(res, f"models/{nombre}.pkl")
    print("  métricas:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["metricas"].items()})
print("Listo. Modelos guardados en la carpeta models/")
