"""
HandTrackingProvider — Capa 1 (Captura).

Encapsula el hand tracking con MediaPipe Tasks API (`HandLandmarker`).
IMPORTANTE: se usa la Tasks API (NO la API legacy `mp.solutions.hands`,
descontinuada por Google en 2023) — ver CONTEXTO_PROYECTO.md Sección 4.

En el sistema final este provider se reemplaza por el Meta XR SDK; el contrato
`getFrame()` se mantiene idéntico (Sección 8.2).

Requiere el modelo `hand_landmarker.task` en `models/` (no viene con
`pip install mediapipe`). Usa `descargar_modelo.py` para obtenerlo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .tipos import HandFrame, MultiHandFrame


class HandTrackingProvider:
    """Detecta hasta `num_hands` manos por frame y entrega sus 21 keypoints."""

    def __init__(self,
                 model_path: Path = config.HAND_LANDMARKER_TASK,
                 num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 running_mode: str = "video") -> None:
        # Import diferido: mediapipe es pesado y solo hace falta al capturar.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró {model_path}. Descárgalo con:\n"
                "    python descargar_modelo.py\n"
                "o manualmente desde el catálogo de MediaPipe HandLandmarker."
            )

        # "video": stream en vivo con tracking temporal (webcam/teléfono).
        # "image": cada frame independiente; ideal para extracción por lotes de
        #          vídeos (sin exigir timestamps, sin arrastrar tracking entre
        #          archivos).
        self._modo = running_mode.lower()
        rm = (vision.RunningMode.IMAGE if self._modo == "image"
              else vision.RunningMode.VIDEO)

        self._mp_vision = vision
        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=rm,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._ultimo: MultiHandFrame = MultiHandFrame(hands=[], timestamp_ms=0)
        self._last_ts = -1  # guard de timestamps estrictamente crecientes (VIDEO)

    # -- Contrato del diseño (Sección 10.2) --------------------------------- #
    def getFrame(self, image_rgb: np.ndarray, timestamp_ms: int) -> MultiHandFrame:
        """Procesa una imagen RGB y devuelve las manos detectadas.

        `image_rgb`: ndarray (H, W, 3) uint8 en formato RGB.
        El resultado agrupa hasta 2 manos; cada una con landmarks (21,3),
        etiqueta de handedness ('Left'/'Right') y su score.
        """
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(image_rgb))
        if self._modo == "image":
            res = self._landmarker.detect(mp_image)
        else:
            # MediaPipe VIDEO exige timestamps estrictamente crecientes; se
            # fuerza aunque el reloj repita o retroceda.
            ts = int(timestamp_ms)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            res = self._landmarker.detect_for_video(mp_image, ts)

        hands: list[HandFrame] = []
        if res.hand_landmarks:
            for i, lm_list in enumerate(res.hand_landmarks):
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list],
                               dtype=np.float32)
                lado, score = "Right", 1.0
                if res.handedness and i < len(res.handedness):
                    cat = res.handedness[i][0]
                    lado, score = cat.category_name, float(cat.score)
                hands.append(HandFrame(landmarks=pts, hand=lado, score=score,
                                       timestamp_ms=int(timestamp_ms)))

        self._ultimo = MultiHandFrame(hands=hands, timestamp_ms=int(timestamp_ms))
        return self._ultimo

    @property
    def ultimo(self) -> MultiHandFrame:
        return self._ultimo

    def cerrar(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self) -> "HandTrackingProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.cerrar()


def seleccionar_frame(multi: MultiHandFrame, modo: str = config.MODO_MANOS
                      ) -> Optional[np.ndarray]:
    """Reduce una detección multi-mano al tensor de keypoints del modo elegido.

    * "dominante": (21, 3) de la mano con mayor score (o None si no hay manos).
    * "ambas":     (2, 21, 3) [Left, Right]; la mano ausente se rellena con NaN
                   para que las capas superiores decidan cómo tratar la ausencia.
    """
    if modo == "ambas":
        if not multi.visible:
            return None
        vacia = np.full((config.NUM_LANDMARKS, config.NUM_EJES), np.nan, np.float32)
        izq = multi.por_lado("Left")
        der = multi.por_lado("Right")
        return np.stack([
            izq.landmarks if izq else vacia,
            der.landmarks if der else vacia,
        ], axis=0)

    dom = multi.dominante()
    return dom.landmarks if dom else None
