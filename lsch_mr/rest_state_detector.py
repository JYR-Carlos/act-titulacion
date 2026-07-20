"""
RestStateDetector — Capa 1 (Captura).

Máquina de estados que segmenta el inicio/fin de una seña por reposo
(CONTEXTO_PROYECTO.md Sección 8):

    Reposo --(manos salen de reposo)--> Capturando
    Capturando --(retorno sostenido a reposo, FIN)--> [secuencia despachada] --> Reposo
    Capturando --(pérdida de tracking)--> [descarta parcial] --> Reposo

La métrica de movimiento es el desplazamiento medio por landmark entre frames,
escalado por el tamaño de la mano (distancia L0–L9). Así el umbral es invariante
a la escala / distancia a la cámara, igual que el KeypointNormalizer.

El detector es agnóstico al "payload": segmenta usando la geometría de la mano
dominante (frame 21x3), pero acumula el `payload` que se le entregue (una o dos
manos). Esto permite grabar en modo "dominante" o "ambas" sin cambiar la
máquina de estados (punto abierto de la Sección 6).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import config
from .tipos import SignEvent, SignEventType

_EPS = 1e-6


class RestStateDetector:
    def __init__(self,
                 umbral_movimiento: float = config.REST_UMBRAL_MOVIMIENTO,
                 frames_inicio: int = config.REST_FRAMES_INICIO,
                 frames_fin: int = config.REST_FRAMES_FIN,
                 min_frames: int = config.REST_MIN_FRAMES_SENA) -> None:
        self.umbral = umbral_movimiento
        self.frames_inicio = frames_inicio
        self.frames_fin = frames_fin
        self.min_frames = min_frames
        self.reset()

    def reset(self) -> None:
        self.state = "reposo"
        self._buffer: list[Any] = []
        self._prev: Optional[np.ndarray] = None
        self._mov_count = 0
        self._rest_count = 0

    # -- Contrato del diseño (Sección 10.2) --------------------------------- #
    def update(self, frame: Optional[np.ndarray],
               payload: Any = None) -> SignEvent:
        """Procesa un frame y avanza la máquina de estados.

        `frame`: keypoints (21,3) de la mano dominante, o None si se perdió el
                 tracking. `payload`: lo que se acumula en la secuencia (por
                 defecto, el propio `frame`). Devuelve un `SignEvent`.
        """
        if payload is None:
            payload = frame

        # --- Pérdida de tracking --------------------------------------- #
        if frame is None:
            self._prev = None
            if self.state == "capturando":
                self.reset()
                return SignEvent(SignEventType.DISCARDED)
            self._mov_count = 0
            self._buffer.clear()
            return SignEvent(SignEventType.IDLE)

        frame = np.asarray(frame, dtype=np.float32).reshape(config.NUM_LANDMARKS,
                                                            config.NUM_EJES)
        movimiento = self._movimiento(self._prev, frame)
        self._prev = frame
        en_movimiento = movimiento > self.umbral

        if self.state == "reposo":
            return self._update_reposo(en_movimiento, payload)
        return self._update_capturando(en_movimiento, payload)

    # -- Estados ------------------------------------------------------------ #
    def _update_reposo(self, en_movimiento: bool, payload: Any) -> SignEvent:
        if en_movimiento:
            self._buffer.append(payload)     # incluye los frames de arranque
            self._mov_count += 1
            if self._mov_count >= self.frames_inicio:
                self.state = "capturando"
                self._rest_count = 0
                return SignEvent(SignEventType.START)
            return SignEvent(SignEventType.IDLE)
        # movimiento no sostenido -> se descarta el candidato
        self._mov_count = 0
        self._buffer.clear()
        return SignEvent(SignEventType.IDLE)

    def _update_capturando(self, en_movimiento: bool, payload: Any) -> SignEvent:
        self._buffer.append(payload)
        if en_movimiento:
            self._rest_count = 0
            return SignEvent(SignEventType.CAPTURING)

        self._rest_count += 1
        if self._rest_count < self.frames_fin:
            return SignEvent(SignEventType.CAPTURING)

        # FIN: se recortan los frames finales de reposo sostenido.
        util = self._buffer[: max(0, len(self._buffer) - self.frames_fin)]
        self.reset()
        if len(util) < self.min_frames:
            return SignEvent(SignEventType.DISCARDED)
        # Si el payload son arrays (modo geométrico) se apilan en (T,...);
        # si son objetos (p.ej. MultiHandFrame para grabar) se devuelve la lista.
        if util and isinstance(util[0], np.ndarray):
            seq = np.stack(util, axis=0).astype(np.float32)
        else:
            seq = util
        return SignEvent(SignEventType.END, sequence=seq)

    # -- Métrica de movimiento ---------------------------------------------- #
    @staticmethod
    def _movimiento(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
        if prev is None:
            return 0.0
        escala = float(np.linalg.norm(cur[config.MIDDLE_MCP_IDX] -
                                      cur[config.WRIST_IDX]))
        if escala < _EPS:
            return 0.0
        desplazamiento = float(np.mean(np.linalg.norm(cur - prev, axis=1)))
        return desplazamiento / escala

    # -- Introspección para overlay ---------------------------------------- #
    @property
    def capturando(self) -> bool:
        return self.state == "capturando"

    @property
    def n_frames_buffer(self) -> int:
        return len(self._buffer)
