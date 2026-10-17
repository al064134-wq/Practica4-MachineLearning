"""Gráficas de entrenamiento al estilo del Neural Network Toolbox de MATLAB
(plotperform, plottrainstate, ploterrhist, plotconfusion, plotroc,
plotregression, plotresponse, ploterrcorr) hechas con matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

AZUL, VERDE, ROJO = "#0072BD", "#2CA02C", "#D62728"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def _ajustar(fig):
    """tight_layout puede fallar con algunas versiones de matplotlib; no es crítico."""
    try:
        _ajustar(fig)
    except Exception:
        try:
            fig.subplots_adjust(left=.1, right=.97, top=.9, bottom=.14, wspace=.35, hspace=.45)
        except Exception:
            pass


def _limpio(v):
    v = np.nan_to_num(np.asarray(v, dtype=float), nan=1e-6, posinf=1e6, neginf=1e-6)
    return np.maximum(v, 1e-6)


def _mejor(val):
    val = np.asarray(val)
    return int(np.argmin(val)), float(np.min(val))


def plotperform(curvas, ylabel="Error"):
    """Pérdida por época en entrenamiento / validación / prueba (escala log)."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    nombres = {"train": ("Train", AZUL), "val": ("Validation", VERDE), "test": ("Test", ROJO)}
    for k, (nom, col) in nombres.items():
        if k in curvas and len(curvas[k]):
            ax.plot(np.arange(len(curvas[k])), _limpio(curvas[k]), color=col, lw=1.6, label=nom)
    ref = curvas.get("val") or curvas["train"]
    e, v = _mejor(_limpio(ref))
    ax.axvline(e, color=VERDE, ls=":", lw=1.2)
    ax.plot(e, max(v, 1e-6), "o", ms=9, mfc="none", mec=VERDE, label="Best")
    ax.set_yscale("log"); ax.grid(alpha=.35, which="both")
    ax.set_xlabel(f"{len(ref)} Epochs"); ax.set_ylabel(ylabel)
    ax.set_title(f"Best Validation Performance is {v:.4g} at epoch {e}")
    ax.legend(loc="upper right")
    _ajustar(fig)
    return fig


def plottrainstate(curvas):
    """Gradiente (|Δ pérdida|) y validation checks (épocas sin mejorar) por época."""
    tr = _limpio(curvas["train"]); val = _limpio(curvas.get("val") or tr)
    grad = np.abs(np.diff(tr, prepend=tr[0] * 1.05)) + 1e-12
    checks, mejor, c = [], np.inf, 0
    for x in val:
        c = 0 if x < mejor - 1e-12 else c + 1
        mejor = min(mejor, x); checks.append(c)
    fig, ax = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    ax[0].plot(grad, color=AZUL); ax[0].set_yscale("log"); ax[0].grid(alpha=.35, which="both")
    ax[0].set_ylabel("gradient"); ax[0].set_title(f"Gradient = {grad[-1]:.3g}, at epoch {len(tr) - 1}")
    ax[1].plot(checks, color=AZUL, marker=".", ms=4); ax[1].grid(alpha=.35)
    ax[1].set_ylabel("val fail"); ax[1].set_xlabel(f"{len(tr)} Epochs")
    ax[1].set_title(f"Validation Checks = {checks[-1]}, at epoch {len(tr) - 1}")
    _ajustar(fig)
    return fig


def ploterrhist(err_tr, err_te, bins=20, xlabel="Errors = Targets - Outputs"):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    todos = np.concatenate([err_tr, err_te])
    bordes = np.linspace(todos.min(), todos.max(), bins + 1)
    ax.hist([err_tr, err_te], bins=bordes, stacked=True, color=[AZUL, ROJO], label=["Training", "Test"],
            edgecolor="black", linewidth=.4)
    ax.axvline(0, color="orange", lw=2, label="Zero Error")
    ax.set_title(f"Error Histogram with {bins} Bins"); ax.set_xlabel(xlabel); ax.set_ylabel("Instances")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    _ajustar(fig)
    return fig


def _confusion(ax, cm, etiquetas, titulo):
    """Matriz con el formato de plotconfusion: filas = Output, columnas = Target."""
    cm = cm.T                                    # MATLAB: filas = clase predicha
    k, n = len(etiquetas), cm.sum()
    for i in range(k + 1):
        for j in range(k + 1):
            if i < k and j < k:
                v = cm[i, j]; color = "#7DD87D" if i == j else "#F28B82"
                txt = f"{v}\n{v / n:.1%}"
            else:
                if i < k:   tot, ok = cm[i].sum(), cm[i, i]
                elif j < k: tot, ok = cm[:, j].sum(), cm[j, j]
                else:       tot, ok = n, np.trace(cm)
                p = ok / tot if tot else 0
                color = "#B0C4DE" if (i == k and j == k) else "#E0E0E0"
                txt = f"{p:.1%}\n{1 - p:.1%}"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", lw=1.5))
            ax.text(j + .5, i + .5, txt, ha="center", va="center", fontsize=7 if k > 3 else 9,
                    fontweight="bold" if (i == k or j == k) else "normal")
    ax.set_xlim(0, k + 1); ax.set_ylim(k + 1, 0)
    ax.set_xticks(np.arange(k) + .5); ax.set_xticklabels(etiquetas, rotation=45 if k > 3 else 0, fontsize=7)
    ax.set_yticks(np.arange(k) + .5); ax.set_yticklabels(etiquetas, fontsize=7)
    ax.set_xlabel("Target Class"); ax.set_ylabel("Output Class"); ax.set_title(titulo)
    ax.set_aspect("equal")


def plotconfusion(pares, etiquetas):
    """pares = [(titulo, y_real, y_pred), ...]"""
    from sklearn.metrics import confusion_matrix
    fig, ax = plt.subplots(1, len(pares), figsize=(4.2 * len(pares), 4.4))
    for a, (t, y, p) in zip(np.atleast_1d(ax), pares):
        _confusion(a, confusion_matrix(y, p, labels=range(len(etiquetas))), etiquetas, t)
    _ajustar(fig)
    return fig


def plotroc(pares, etiquetas):
    """pares = [(titulo, y_real, probabilidades), ...] – una curva por clase (uno contra todos)."""
    fig, ax = plt.subplots(1, len(pares), figsize=(4.4 * len(pares), 4))
    colores = plt.cm.tab10.colors
    for a, (t, y, P) in zip(np.atleast_1d(ax), pares):
        clases = range(P.shape[1]) if P.shape[1] > 2 else [1]
        for c in clases:
            fpr, tpr, _ = roc_curve(y == c, P[:, c])
            nombre = etiquetas[c] if P.shape[1] > 2 else f"{etiquetas[1]} vs {etiquetas[0]}"
            a.plot(fpr, tpr, color=colores[c % 10], lw=1.6, label=f"{nombre} (AUC={auc(fpr, tpr):.3f})")
        a.plot([0, 1], [0, 1], color="gray", ls="--", lw=.8)
        a.set_xlabel("False Positive Rate"); a.set_ylabel("True Positive Rate"); a.set_title(t)
        a.legend(fontsize=6.5, loc="lower right"); a.grid(alpha=.3)
    _ajustar(fig)
    return fig


def plotregression(pares):
    """pares = [(titulo, targets, outputs), ...] con línea de ajuste y coeficiente R."""
    fig, ax = plt.subplots(1, len(pares), figsize=(4.2 * len(pares), 4))
    colores = [AZUL, ROJO, "black"]
    for a, (t, T, Y), col in zip(np.atleast_1d(ax), pares, colores):
        m, b = np.polyfit(T, Y, 1); r = np.corrcoef(T, Y)[0, 1]
        lim = [min(T.min(), Y.min()), max(T.max(), Y.max())]
        a.plot(T, Y, "o", mfc="none", mec="black", ms=4, lw=0, label="Data")
        a.plot(lim, [m * lim[0] + b, m * lim[1] + b], color=col, lw=1.8, label="Fit")
        a.plot(lim, lim, ":", color="gray", label="Y = T")
        a.set_title(f"{t}: R={r:.5f}"); a.set_xlabel("Target")
        a.set_ylabel(f"Output ~= {m:.2f}*Target + {b:.2f}"); a.legend(fontsize=7, loc="upper left")
        a.grid(alpha=.3); a.set_aspect("equal", adjustable="datalim")
    _ajustar(fig)
    return fig


def plotresponse(y_real, y_pred, etiquetas, paso_s=1.28):
    """Respuesta temporal: actividad real vs. predicha en ventanas consecutivas."""
    t = np.arange(len(y_real)) * paso_s
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].step(t, y_real, where="post", color=AZUL, lw=2, label="Targets")
    ax[0].step(t, y_pred, where="post", color="orange", lw=1.2, ls="--", label="Outputs")
    err = np.where(y_real != y_pred)[0]
    ax[0].plot(t[err], y_pred[err], "x", color=ROJO, ms=6, label="Errors")
    ax[0].set_yticks(range(len(etiquetas))); ax[0].set_yticklabels(etiquetas)
    ax[0].set_title("Response of Output Element 1 for Time-Series"); ax[0].legend(fontsize=7, ncol=3); ax[0].grid(alpha=.3)
    ax[1].bar(t, (y_real != y_pred).astype(int), width=paso_s, color=ROJO)
    ax[1].set_ylabel("Error"); ax[1].set_xlabel("Tiempo (s)"); ax[1].set_yticks([0, 1])
    _ajustar(fig)
    return fig


def ploterrcorr(err, max_lag=20):
    """Autocorrelación del error con límites de confianza del 95 %."""
    e = np.asarray(err, float) - np.mean(err)
    den = np.dot(e, e) or 1.0
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.array([np.dot(e[max(0, -l):len(e) - max(0, l)], e[max(0, l):len(e) - max(0, -l)]) / den for l in lags])
    lim = 1.96 / np.sqrt(len(e))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(lags, r, color=AZUL, width=.5, label="Correlations")
    ax.axhline(0, color="black", lw=.6)
    ax.fill_between([-max_lag - 1, max_lag + 1], -lim, lim, color="red", alpha=.12, label="Confidence Limit")
    ax.plot([-max_lag - 1, max_lag + 1], [lim, lim], "r--", lw=.8); ax.plot([-max_lag - 1, max_lag + 1], [-lim, -lim], "r--", lw=.8)
    ax.set_xlim(-max_lag - 1, max_lag + 1); ax.set_xlabel("Lag"); ax.set_ylabel("Correlation")
    ax.set_title("Autocorrelation of Error 1"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    _ajustar(fig)
    return fig
