"""
Métricas de clasificación reportables — métrica 1 del MVP (Sección 11).

Funciones **puras** (numpy, sin sklearn ni disco salvo donde se dice) que
convierten un par `(y_true, y_pred)` en lo que pide el informe: accuracy global,
accuracy por seña, matriz de confusión 10x10 y el veredicto contra el umbral
`config.ACCURACY_OBJETIVO`. Viven aquí y no en el script para poder testearlas
sin modelo, sin dataset y sin cámara — el mismo criterio que siguen
`_resumen_latencias` en `scripts/demo_vivo.py` y `MonitorRecursos`.

Convención de la matriz de confusión, igual que en `ModelTrainer`:
**filas = clase real, columnas = clase predicha**. La diagonal son los aciertos.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config


def matriz_confusion(y_true, y_pred, n_clases: int) -> np.ndarray:
    """Matriz (n_clases, n_clases) de conteos enteros: filas real, cols predicha.

    Se calcula a mano en vez de con sklearn para que el módulo se pueda importar
    y testear sin la dependencia, y porque el conteo es trivial: un histograma 2D.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true e y_pred deben tener el mismo largo; "
            f"se recibió {yt.shape} y {yp.shape}.")
    if len(yt) and (yt.min() < 0 or yt.max() >= n_clases
                    or yp.min() < 0 or yp.max() >= n_clases):
        raise ValueError(
            f"Hay índices de clase fuera del rango [0, {n_clases - 1}].")

    cm = np.zeros((n_clases, n_clases), dtype=np.int64)
    # bincount sobre el índice aplanado (real * n + predicha) — un solo barrido.
    if len(yt):
        plano = np.bincount(yt * n_clases + yp, minlength=n_clases * n_clases)
        cm = plano.reshape(n_clases, n_clases).astype(np.int64)
    return cm


def accuracy_global(y_true, y_pred) -> float:
    """Fracción de aciertos sobre todas las muestras. Sin muestras -> 0.0."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def metricas_por_clase(cm: np.ndarray, classes: list[str]) -> list[dict]:
    """Desglose por seña a partir de la matriz de confusión.

    `accuracy` por clase es el **recall**: de todas las veces que se ejecutó esa
    seña, cuántas se reconocieron. Es la lectura que pide el informe ("accuracy
    por seña"). Se acompaña de la precisión —de todas las veces que el sistema
    dijo esa seña, cuántas acertó— porque las dos fallan de formas distintas y
    una sola de las dos puede esconder el problema: una clase a la que el modelo
    nunca llama tiene precisión 1.0 y recall 0.0.

    Clases sin muestras (soporte 0) devuelven 0.0 en vez de NaN, para que el CSV
    y el JSON del reporte sean numéricos en todas sus filas.
    """
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (len(classes), len(classes)):
        raise ValueError(
            f"La matriz {cm.shape} no cuadra con {len(classes)} clases.")

    filas = []
    for i, nombre in enumerate(classes):
        aciertos = int(cm[i, i])
        soporte = int(cm[i, :].sum())        # veces que la seña REAL fue esta
        predichas = int(cm[:, i].sum())      # veces que el sistema DIJO esta
        recall = aciertos / soporte if soporte else 0.0
        precision = aciertos / predichas if predichas else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        filas.append({
            "clase": nombre,
            "soporte": soporte,
            "aciertos": aciertos,
            "accuracy": round(recall, 4),     # = recall
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        })
    return filas


def veredicto(accuracy: float,
              objetivo: float = config.ACCURACY_OBJETIVO) -> dict:
    """¿La accuracy alcanza el umbral del MVP? Devuelve también la brecha.

    `brecha` es cuánto falta (positiva) o cuánto sobra (negativa) respecto al
    objetivo — el número que el informe necesita cuando NO se cumple.
    """
    return {
        "accuracy": round(float(accuracy), 4),
        "objetivo": float(objetivo),
        "cumple": bool(accuracy >= objetivo),
        "brecha": round(float(objetivo - accuracy), 4),
    }


def cobertura_con_umbral(y_true, y_pred, confianzas,
                         umbral: float = config.CONF_THRESHOLD) -> dict:
    """Qué pasa cuando se aplica `CONF_THRESHOLD` (lo que hace el sistema real).

    La accuracy "cruda" (argmax, sin umbral) es la métrica 1 del informe, pero
    no describe lo que ve el funcionario en ventanilla: bajo el umbral la seña
    se muestra como "<desconocida>" (FA-01) en vez de como una glosa. Este
    desglose separa las dos cosas:

      * `cobertura`  — fracción de señas que superan el umbral y SÍ se muestran.
      * `precision_mostradas` — de las mostradas, cuántas eran correctas.
      * `n_erroneas_mostradas` — glosas equivocadas que llegaron a pantalla.
        Es el número que importa: mostrar una glosa mala engaña, "no reconocida"
        solo pide repetir. Es el mismo criterio con que se calibró el umbral a
        0.90 el 2026-07-26 (ver `config.CONF_THRESHOLD`).
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    conf = np.asarray(confianzas, dtype=np.float64).ravel()
    if not (len(yt) == len(yp) == len(conf)):
        raise ValueError("y_true, y_pred y confianzas deben tener el mismo largo.")

    base = {"umbral": float(umbral), "n": int(len(yt))}
    if len(yt) == 0:
        return base

    mostradas = conf >= umbral
    n_mostradas = int(mostradas.sum())
    aciertos_mostradas = int(np.sum(yt[mostradas] == yp[mostradas]))
    base.update({
        "n_mostradas": n_mostradas,
        "cobertura": round(n_mostradas / len(yt), 4),
        "precision_mostradas": (round(aciertos_mostradas / n_mostradas, 4)
                                if n_mostradas else 0.0),
        "n_erroneas_mostradas": n_mostradas - aciertos_mostradas,
        "n_ocultas_por_umbral": int(len(yt) - n_mostradas),
    })
    return base


# --------------------------------------------------------------------------- #
# Exportación (las únicas funciones de este módulo que tocan disco)
# --------------------------------------------------------------------------- #
def guardar_matriz_csv(cm: np.ndarray, classes: list[str], ruta: Path) -> Path:
    """Matriz de confusión como CSV, con encabezados de fila y columna.

    Primera columna = clase real, primera fila = clase predicha, para que se
    pueda pegar en el informe sin reinterpretar los ejes.
    """
    cm = np.asarray(cm, dtype=np.int64)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real \\ predicha"] + list(classes))
        for i, nombre in enumerate(classes):
            w.writerow([nombre] + [int(v) for v in cm[i]])
    return ruta


def guardar_por_clase_csv(filas: list[dict], ruta: Path) -> Path:
    """Tabla de accuracy por seña como CSV."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = ["clase", "soporte", "aciertos", "accuracy", "precision", "f1"]
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    return ruta


def graficar_matriz(cm: np.ndarray, classes: list[str], ruta_png: Path,
                    titulo: str = "Matriz de confusión",
                    subtitulo: str = "",
                    cumple: bool | None = None,
                    normalizar: bool = False) -> Path:
    """Matriz de confusión como PNG.

    Si `cumple` es False el título va en **rojo** y con el aviso de que la
    métrica no alcanza el umbral: la imagen que termina pegada en el informe
    tiene que delatar el incumplimiento por sí sola, sin depender de que alguien
    lea la consola donde se generó.

    `normalizar=True` dibuja porcentajes por fila en vez de conteos. Útil cuando
    las clases tienen soportes muy distintos; con el corpus balanceado del MVP
    (50 muestras por glosa) los conteos ya se leen bien.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.int64)
    n = len(classes)

    datos = cm.astype(np.float64)
    if normalizar:
        soportes = datos.sum(axis=1, keepdims=True)
        datos = np.divide(datos, soportes, out=np.zeros_like(datos),
                          where=soportes > 0) * 100.0

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 2.4))
    im = ax.imshow(datos, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")

    color_titulo = "black" if cumple is None else ("darkgreen" if cumple else "red")
    ax.set_title(titulo, color=color_titulo,
                 fontweight="normal" if cumple is None else "bold")
    if subtitulo:
        # `figtext` en vez de un segundo título: queda bajo el eje x sin pelear
        # con las etiquetas rotadas de las clases.
        fig.text(0.5, 0.015, subtitulo, ha="center", fontsize=9,
                 color=color_titulo)

    umbral_color = datos.max() / 2.0 if datos.max() else 0.5
    for i in range(n):
        for j in range(n):
            texto = (f"{datos[i, j]:.0f}%" if normalizar else f"{int(cm[i, j])}")
            ax.text(j, i, texto, ha="center", va="center", fontsize=9,
                    color="white" if datos[i, j] > umbral_color else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0.04, 1, 1) if subtitulo else None)
    ruta_png = Path(ruta_png)
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=120)
    plt.close(fig)
    return ruta_png
