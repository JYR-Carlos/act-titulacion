"""
FuenteVideo — abstracción de la entrada de imágenes para validación en PC.

Soporta tres orígenes con la misma interfaz:
  * Webcam local          -> índice entero: 0, 1, ...
  * Cámara de teléfono     -> URL de stream (MJPEG/RTSP), p.ej. IP Webcam:
                              http://192.168.1.42:8080/video
  * Archivo de video       -> ruta a un .mp4/.avi/...

Uso de la cámara del teléfono (recomendado, sin drivers extra):
  1. Instalar en el teléfono la app "IP Webcam" (Android) o similar.
  2. Iniciar el servidor; anota la URL que muestra (p.ej. http://IP:8080).
  3. Pasar la URL de video como fuente:  --fuente http://IP:8080/video
  (iOS: apps que exponen MJPEG/RTSP; también sirven DroidCam/Iriun como webcam
   virtual, en cuyo caso se usa el índice de webcam correspondiente.)

En Capa 1 del sistema final la fuente real será el Meta XR SDK; aquí solo
cambia la fuente de keypoints (Sección 8.2 del diseño).
"""
from __future__ import annotations

import time
from typing import Iterator, Optional, Union

import cv2
import numpy as np

FuenteSpec = Union[int, str]


def parse_fuente(valor: str) -> FuenteSpec:
    """Convierte el argumento CLI en índice de webcam (int) o URL/ruta (str)."""
    return int(valor) if str(valor).isdigit() else str(valor)


class FuenteVideo:
    """Envuelve `cv2.VideoCapture` y entrega frames BGR con timestamp en ms."""

    def __init__(self, fuente: FuenteSpec = 0, espejo: bool = False) -> None:
        self.fuente = fuente
        self.espejo = espejo
        self._cap: Optional[cv2.VideoCapture] = None
        self._t0: Optional[float] = None

    def abrir(self) -> "FuenteVideo":
        """Abre la fuente. Lanza `RuntimeError` con la causa probable si falla.

        Es el fallo más común al montar el repo en otra máquina —cámara
        ocupada, índice equivocado, teléfono que no está transmitiendo— así que
        el mensaje dice qué comprobar en vez de dejar un `None` que reviente
        más adelante.

        Idempotente: reabrir una fuente ya abierta la devuelve tal cual. Así se
        puede validar la apertura antes de un `with` sin que `__enter__` cree
        una segunda `VideoCapture` y deje la primera colgando.
        """
        if self._cap is not None and self._cap.isOpened():
            return self
        # CAP_FFMPEG es más robusto para URLs de red (streams del teléfono).
        if isinstance(self.fuente, str) and "://" in self.fuente:
            self._cap = cv2.VideoCapture(self.fuente, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(self.fuente)
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la fuente de video: {self.fuente!r}. "
                "Si es un teléfono, verifica que la app esté transmitiendo, "
                "que PC y teléfono estén en la misma red Wi-Fi y que la URL "
                "termine en /video (IP Webcam)."
            )
        self._t0 = time.monotonic()
        return self

    def leer(self) -> Optional[tuple[np.ndarray, int]]:
        """Devuelve (frame_bgr, timestamp_ms) o None si terminó el stream."""
        if self._cap is None:
            raise RuntimeError("FuenteVideo no abierta; llama a abrir() primero.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self.espejo:
            frame = cv2.flip(frame, 1)
        ts_ms = int((time.monotonic() - (self._t0 or 0.0)) * 1000)
        return frame, ts_ms

    def frames(self) -> Iterator[tuple[np.ndarray, int]]:
        """Itera frames hasta que la fuente se agote."""
        while True:
            r = self.leer()
            if r is None:
                break
            yield r

    def cerrar(self) -> None:
        """Libera la cámara. Idempotente; preferir el uso como context manager.

        Sin esto la webcam queda tomada por el proceso y el siguiente script
        falla al abrirla.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FuenteVideo":
        return self.abrir()

    def __exit__(self, *exc) -> None:
        self.cerrar()
