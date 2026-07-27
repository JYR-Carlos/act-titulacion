"""
realce_luz — realce adaptativo de frames con poca luz (Capa 1, Captura).

Motivación medida (`diagnostico_captura.py`, sesión "noche" del 2026-07-26):
con luminancia media 10/255 y el 96% de los píxeles por debajo de 40, MediaPipe
encuentra la palma en solo el 51% de los frames. No es que los keypoints salgan
mal — cuando detecta, el score de handedness es 0.97: simplemente no encuentra
la mano. El detector de palma no tiene contraste con el que trabajar.

Este módulo sube el contraste ANTES de MediaPipe, en dos pasos:

  1. Gamma global, con el exponente calculado para llevar la luminancia media
     del frame a `LUMINANCIA_OBJETIVO`. Autoajustable: una escena a 10/255
     necesita mucho más lift que una a 45/255, y fijar una gamma constante
     serviría solo para una de las dos.
  2. CLAHE sobre el canal L de LAB, que recupera contraste LOCAL (bordes de los
     dedos) sin quemar el resto del cuadro. El color se conserva porque solo se
     toca la luminancia.

ADAPTATIVO A PROPÓSITO: por encima de `UMBRAL_LUMINANCIA` el frame se devuelve
intacto. Así activarlo no degrada una sesión bien iluminada, y el costo en luz
normal se reduce a medir la media (submuestreada: microsegundos).

NO ES GRATIS Y NO ESTÁ ACTIVO POR DEFECTO
-----------------------------------------
El corpus LSA64 del que salió `modelo.onnx` se extrajo SIN realce (los vídeos
están bien iluminados, `luminancia_media` los deja pasar sin tocar). Aplicarlo
en servicio cambia la imagen que ve MediaPipe, así que es una decisión con el
mismo perfil de riesgo que `running_mode` (INTEGRACION_UNITY.md sección 7) y va
detrás de un flag (`--realce`), no encendida de fábrica.

El argumento a favor: en una escena oscura el realce ACERCA la entrada a la
distribución de entrenamiento (imágenes con contraste), no la aleja — el modo
de fallo que corrige es "no hay detección", no "detección desplazada". El
argumento en contra: sube también el ruido del sensor, que con ganancia alta no
es despreciable. Por eso la forma correcta de decidirlo es medirlo:

    python diagnostico_captura.py --etiqueta noche
    python diagnostico_captura.py --etiqueta noche-realce --realce

y comparar `tasa_con_mano` y `racha_max_sin_mano` de los dos JSON. Más luz
física sigue siendo mejor que cualquier realce: esto es la red de seguridad
para cuando no la hay.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Luminancia media (gris 0..255) por encima de la cual NO se toca el frame: hay
# luz suficiente y realzar solo agregaría ruido. Mismo corte que usa el
# veredicto de `diagnostico_captura.py`, para que los números sean comparables.
UMBRAL_LUMINANCIA = 60.0

# Luminancia media a la que se intenta llevar un frame oscuro. Por encima de
# ~100 el ruido del sensor domina la imagen y la detección deja de mejorar.
LUMINANCIA_OBJETIVO = 90.0

# Gamma mínima: por debajo el ruido sube más rápido que la señal de la mano.
GAMMA_MIN = 0.35

# Parámetros de CLAHE. `clipLimit` moderado a propósito: con ganancia alta, un
# límite agresivo convierte el grano del sensor en textura falsa.
CLAHE_CLIP = 2.0
CLAHE_GRID = 8

_clahe = None
_luts: dict[float, np.ndarray] = {}


def luminancia_media(frame_bgr: np.ndarray, submuestreo: int = 4) -> float:
    """Media del gris 0..255 del frame.

    `submuestreo` toma 1 de cada N píxeles por eje: a 640x480 con N=4 mide sobre
    160x120 y cuesta microsegundos, con un error irrelevante para decidir si la
    escena está oscura. Es la misma métrica que reporta `diagnostico_captura.py`
    (ahí sin submuestrear, porque no corre en el bucle de la demo).
    """
    if submuestreo > 1:
        frame_bgr = frame_bgr[::submuestreo, ::submuestreo]
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())


def _lut_gamma(gamma: float) -> np.ndarray:
    """LUT de 256 entradas para `cv2.LUT`, memoizada por gamma redondeada.

    La gamma se recalcula en cada frame y varía poco entre frames vecinos;
    redondear a 2 decimales hace que en la práctica se reutilice siempre la
    misma tabla en vez de construir 30 por segundo.
    """
    clave = round(gamma, 2)
    lut = _luts.get(clave)
    if lut is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip((x ** clave) * 255.0, 0, 255).astype(np.uint8)
        _luts[clave] = lut
    return lut


def gamma_para(luminancia: float, objetivo: float = LUMINANCIA_OBJETIVO) -> float:
    """Exponente que lleva `luminancia` a `objetivo` bajo una corrección gamma.

    De despejar `objetivo/255 = (luminancia/255) ** gamma`. Acotado a
    [`GAMMA_MIN`, 1.0]: por arriba no se oscurece nunca un frame (esa no es la
    tarea de este módulo) y por abajo se corta antes de que el realce sea
    mayormente ruido amplificado.
    """
    if luminancia <= 0.5:      # frame prácticamente negro: log(~0) no informa
        return GAMMA_MIN
    if luminancia >= objetivo:
        return 1.0
    gamma = math.log(objetivo / 255.0) / math.log(luminancia / 255.0)
    return max(GAMMA_MIN, min(1.0, gamma))


def realzar_poca_luz(frame_bgr: np.ndarray,
                     luminancia: float | None = None,
                     umbral: float = UMBRAL_LUMINANCIA,
                     objetivo: float = LUMINANCIA_OBJETIVO) -> np.ndarray:
    """Devuelve el frame realzado, o el mismo frame si ya hay luz suficiente.

    `luminancia`: si el llamador ya la midió (la demo la necesita igual para el
    overlay), se pasa aquí para no medirla dos veces por frame.
    """
    if luminancia is None:
        luminancia = luminancia_media(frame_bgr)
    if luminancia >= umbral:
        return frame_bgr

    realzado = cv2.LUT(frame_bgr, _lut_gamma(gamma_para(luminancia, objetivo)))
    lab = cv2.cvtColor(realzado, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe_compartido().apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _clahe_compartido():
    """CLAHE reutilizado entre frames: crearlo por frame es puro descarte."""
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                                 tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    return _clahe
