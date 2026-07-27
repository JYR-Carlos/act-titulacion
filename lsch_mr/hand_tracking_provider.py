"""
HandTrackingProvider — Capa 1 (Captura).

Encapsula el hand tracking con MediaPipe Tasks API (`HandLandmarker`).
IMPORTANTE: se usa la Tasks API (NO la API legacy `mp.solutions.hands`,
descontinuada por Google en 2023) — ver docs/CONTEXTO_PROYECTO.md Sección 4.

En el sistema final este provider se reemplaza por el Meta XR SDK; el contrato
`getFrame()` se mantiene idéntico (Sección 8.2).

Requiere el modelo `hand_landmarker.task` en `models/` (no viene con
`pip install mediapipe`). Usa `scripts/descargar_modelo.py` para obtenerlo.

======================================================================
running_mode OFICIAL DEL PROYECTO: "image"   (decisión del 2026-07-26)
======================================================================
Vale para Python y para el port a Unity (plugin de homuler). NO es un
detalle de implementación: es parte del contrato con el modelo.

Por qué: el corpus LSA64 del que salió `modelo.onnx` se extrajo con
`scripts/extraer_lote.py`, que pasa `running_mode="image"`. El modo de servicio
tiene que ser el modo de extracción — si no, el modelo recibe keypoints
de otra distribución y falla en silencio (train/serve skew).

Medido sobre el mismo vídeo del corpus, IMAGE vs VIDEO divergen hasta
1.79 en el NormVector, cinco órdenes de magnitud por encima de la
tolerancia con la que se valida el port (1e-5), y eligen distinta mano
dominante en 16 de 18 frames. No son intercambiables.

Ojo con el default de esta clase: sigue siendo "video", que NO es el modo
oficial. Se mantiene solo para no romper llamadas existentes que exploran
tracking en vivo sin producir datos ni clasificar. Los cuatro usos reales
del proyecto pasan running_mode="image" explícitamente:

    scripts/extraer_lote.py        corpus por lotes (es el que extrajo LSA64)
    scripts/extraer_keypoints.py   corpus desde webcam o archivo
    DatasetRecorder        corpus grabado en sesión
    scripts/demo_vivo.py           demo end-to-end, alimenta modelo.onnx

Si añades un uso nuevo que grabe corpus o alimente al modelo, pásalo
también. No confíes en el default.

Especificación completa, evidencia y qué reabriría la decisión:
docs/INTEGRACION_UNITY.md sección 7. Vectores de referencia frame a frame:
`scripts/generar_secuencia_dorada.py` -> integracion/secuencia_dorada.json.
LIVE_STREAM no se usa nunca (asíncrono: descarta frames).
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
                "    python scripts/descargar_modelo.py\n"
                "o manualmente desde el catálogo de MediaPipe HandLandmarker."
            )

        # "image": cada frame independiente. MODO OFICIAL DEL PROYECTO (ver el
        #          docstring): es el que extrajo el corpus, así que es el único
        #          coherente con `modelo.onnx`. Ignora el timestamp.
        # "video": arrastra el ROI del frame anterior en vez de redetectar la
        #          palma. Detecta más manos y con menos jitter, pero produce
        #          keypoints de OTRA distribución que la del entrenamiento.
        #          Solo para capturar en vivo sin alimentar al modelo.
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
                # Se guarda el timestamp ORIGINAL, no el `ts` corregido por el
                # guard de arriba: el HandFrame documenta cuándo se capturó el
                # frame, no qué reloj vio MediaPipe. Los dos divergen en cuanto
                # el guard se dispara (solo en modo "video").
                hands.append(HandFrame(landmarks=pts, hand=lado, score=score,
                                       timestamp_ms=int(timestamp_ms)))

        self._ultimo = MultiHandFrame(hands=hands, timestamp_ms=int(timestamp_ms))
        return self._ultimo

    @property
    def ultimo(self) -> MultiHandFrame:
        """Última detección, sin volver a inferir. Para el overlay de la demo."""
        return self._ultimo

    def cerrar(self) -> None:
        """Libera el HandLandmarker. Idempotente y no lanza."""
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
