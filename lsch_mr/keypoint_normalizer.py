"""
KeypointNormalizer — Capa 2 (Preprocesamiento).

Implementa la especificación de la Sección 10.2 / Sección 6 del diseño:
centrado en la muñeca (L0) + escalado por la distancia L0–L9 (muñeca–base del
dedo medio). El resultado (NormVector, 63 dim) es invariante a traslación y a
escala (tamaño de mano / distancia a la cámara).

Decisión: implementación propia, NO se reutiliza código de terceros
(ver docs/DECISION_PREPROCESAMIENTO.md).
"""
from __future__ import annotations

import numpy as np

from . import config
from .tipos import Frame, NormVector

_EPS = 1e-6


class KeypointNormalizer:
    """Normaliza un Frame (21 keypoints x,y,z) a un NormVector de 63 dimensiones."""

    def __init__(self,
                 wrist_idx: int = config.WRIST_IDX,
                 middle_mcp_idx: int = config.MIDDLE_MCP_IDX) -> None:
        self.wrist_idx = wrist_idx
        self.middle_mcp_idx = middle_mcp_idx

    # -- Contrato del diseño (Sección 10.2) --------------------------------- #
    def normalize(self, frame: Frame) -> NormVector:
        """Centra en L0 y escala por ||L9 - L0||. Devuelve un vector (63,).

        Acepta entradas de forma (21, 3) o (63,). Lanza `ValueError` si la
        forma no corresponde a 21 keypoints (x, y, z). En casos degenerados
        (mano colapsada, distancia L0–L9 ≈ 0) devuelve un vector de ceros,
        evitando la división por cero y valores NaN/Inf.
        """
        pts = self._validar_forma(frame)

        # 1) Centrado en la muñeca (invariancia a traslación).
        origen = pts[self.wrist_idx]
        centrado = pts - origen

        # 2) Escala = distancia L0–L9 (invariancia a escala).
        escala = float(np.linalg.norm(centrado[self.middle_mcp_idx]))
        if escala < _EPS:
            # Caso degenerado: no se puede escalar de forma estable.
            return np.zeros(config.NORMVECTOR_DIM, dtype=np.float32)

        normalizado = centrado / escala

        # 3) Aplanado fila-mayor: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20].
        vec = normalizado.reshape(-1).astype(np.float32)

        if not np.all(np.isfinite(vec)):
            return np.zeros(config.NORMVECTOR_DIM, dtype=np.float32)
        return vec

    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normaliza una SignSequence (T, 21, 3) -> (T, 63)."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[1:] != (config.NUM_LANDMARKS, config.NUM_EJES):
            raise ValueError(
                f"Se esperaba SignSequence (T, {config.NUM_LANDMARKS}, "
                f"{config.NUM_EJES}); se recibió {seq.shape}."
            )
        return np.stack([self.normalize(f) for f in seq], axis=0)

    # -- Utilidades internas ------------------------------------------------ #
    @staticmethod
    def _validar_forma(frame: Frame) -> np.ndarray:
        pts = np.asarray(frame, dtype=np.float32)
        if pts.shape == (config.NORMVECTOR_DIM,):
            pts = pts.reshape(config.NUM_LANDMARKS, config.NUM_EJES)
        if pts.shape != (config.NUM_LANDMARKS, config.NUM_EJES):
            raise ValueError(
                f"Frame inválido: se esperaba forma "
                f"({config.NUM_LANDMARKS}, {config.NUM_EJES}) o "
                f"({config.NORMVECTOR_DIM},); se recibió {pts.shape}."
            )
        return pts
