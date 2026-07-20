"""
Descarga el modelo `hand_landmarker.task` de MediaPipe Tasks (HandLandmarker).

Este archivo NO viene con `pip install mediapipe` y es necesario para la Capa 1
(HandTrackingProvider). Se guarda en `models/hand_landmarker.task`.

Uso:
    python descargar_modelo.py
"""
from __future__ import annotations

import sys
import urllib.request

from lsch_mr import config
from lsch_mr.consola import configurar_utf8

configurar_utf8()

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")


def main() -> int:
    destino = config.HAND_LANDMARKER_TASK
    if destino.exists():
        print(f"Ya existe: {destino} ({destino.stat().st_size/1e6:.1f} MB)")
        return 0
    print(f"Descargando HandLandmarker desde:\n  {URL}")
    try:
        urllib.request.urlretrieve(URL, destino)
    except Exception as e:
        print(f"ERROR al descargar: {e}\n"
              "Descárgalo manualmente desde el catálogo de MediaPipe y colócalo "
              f"en: {destino}")
        return 1
    print(f"OK -> {destino} ({destino.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
