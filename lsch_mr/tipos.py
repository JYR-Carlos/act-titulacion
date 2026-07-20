"""
Tipos de datos del pipeline (modelo de datos, CONTEXTO_PROYECTO.md Sección 6).

Se representan las coordenadas con `numpy.ndarray` por eficiencia; las clases
`dataclass` envuelven los conceptos del diseño (Frame, SignEvent, ClassResult)
manteniendo la trazabilidad con el diagrama de clases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Alias semánticos (ver Sección 6):
#   Frame       -> ndarray (21, 3)      : 21 Keypoint (x, y, z) de una mano
#   SignSequence-> ndarray (T, 21, 3)   : 1..60 Frame
#   NormVector  -> ndarray (63,)        : salida de KeypointNormalizer
Frame = np.ndarray
SignSequence = np.ndarray
NormVector = np.ndarray


@dataclass
class HandFrame:
    """Un frame de una mano detectada, con metadatos de MediaPipe.

    `landmarks` tiene forma (21, 3). `hand` es 'Left'/'Right' según la
    handedness reportada por HandLandmarker; `score` su confianza.
    """
    landmarks: np.ndarray                 # (21, 3) float32
    hand: str = "Right"                   # 'Left' | 'Right'
    score: float = 1.0                    # handedness_score
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32).reshape(21, 3)


@dataclass
class MultiHandFrame:
    """Detección de un frame completo: 0, 1 o 2 manos."""
    hands: list[HandFrame] = field(default_factory=list)
    timestamp_ms: int = 0

    @property
    def visible(self) -> bool:
        return len(self.hands) > 0

    def dominante(self) -> Optional[HandFrame]:
        """Mano de mayor handedness_score (heurística de mano dominante)."""
        if not self.hands:
            return None
        return max(self.hands, key=lambda h: h.score)

    def por_lado(self, lado: str) -> Optional[HandFrame]:
        cand = [h for h in self.hands if h.hand == lado]
        return max(cand, key=lambda h: h.score) if cand else None


class SignEventType(Enum):
    """Eventos emitidos por RestStateDetector (Sección 8)."""
    IDLE = "idle"            # en reposo, sin actividad
    START = "start"          # las manos salieron de reposo -> inicia captura
    CAPTURING = "capturing"  # acumulando frames de la seña
    END = "end"              # retorno sostenido a reposo -> secuencia lista
    DISCARDED = "discarded"  # pérdida de tracking / seña demasiado corta


@dataclass
class SignEvent:
    """Resultado de `RestStateDetector.update(frame)`.

    En `END`, `sequence` contiene la SignSequence completa (T, 21, 3) lista
    para normalizar y clasificar.
    """
    type: SignEventType
    sequence: Optional[SignSequence] = None


@dataclass
class ClassResult:
    """Resultado de `SignClassifier.classify(seq)`."""
    label: str                 # glosa predicha o "<desconocida>"
    index: int                 # índice de clase (-1 si fuera de vocabulario)
    confidence: float          # probabilidad de la clase ganadora
    in_vocab: bool             # confidence >= confThreshold
    scores: np.ndarray         # distribución softmax sobre todas las clases
