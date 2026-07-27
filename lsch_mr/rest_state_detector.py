"""
RestStateDetector — Capa 1 (Captura).

Máquina de estados que segmenta el inicio/fin de una seña por reposo
(CONTEXTO_PROYECTO.md Sección 8):

    Reposo --(manos salen de reposo)--> Capturando
    Capturando --(retorno sostenido a reposo, FIN)--> [secuencia despachada] --> Reposo
    Capturando --(sin mano > frames_perdida_max frames seguidos)--> [descarta parcial] --> Reposo

    Un parpadeo de detección de hasta `frames_perdida_max` frames seguidos NO
    descarta la captura: running_mode="image" redetecta la palma en cada frame
    (sin arrastrar el ROI del anterior), así que perder la mano 1-2 frames por
    motion blur o un ángulo raro es esperable, no evidencia de que la seña
    terminó o de que la mano se fue de verdad.

    Esa misma tolerancia protege al CANDIDATO de inicio mientras se está en
    reposo (corregido el 2026-07-26). Antes solo existía en "capturando", y en
    reposo cualquier frame sin mano borraba el contador de movimiento: había que
    encadenar `frames_inicio + 1` detecciones seguidas para arrancar (una extra
    porque tras un hueco `_prev` quedaba en None y el primer frame de vuelta
    medía movimiento 0). Con la detección al 51% que produce una escena oscura
    —medido en `outputs/reports/diagnostico_captura_noche_*.json`— eso deja la
    probabilidad de arrancar en ~7% por intento, y la demo pasa sesiones enteras
    en "ESCUCHANDO" sin segmentar una sola seña ni reportar un solo descarte.
    Con luz buena el síntoma no aparece, que es por qué sobrevivió hasta ahora.

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
                 min_frames: int = config.REST_MIN_FRAMES_SENA,
                 frames_perdida_max: int = config.REST_FRAMES_PERDIDA_MAX) -> None:
        self.umbral = umbral_movimiento
        self.frames_inicio = frames_inicio
        self.frames_fin = frames_fin
        self.min_frames = min_frames
        self.frames_perdida_max = frames_perdida_max
        self.reset()

    def reset(self) -> None:
        self.state = "reposo"
        self._buffer: list[Any] = []
        self._prev: Optional[np.ndarray] = None
        self._mov_count = 0
        self._rest_count = 0
        self._perdida_count = 0

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
            self._perdida_count += 1
            tolerado = self._perdida_count <= self.frames_perdida_max

            if self.state == "capturando":
                if not tolerado:
                    self.reset()
                    return SignEvent(SignEventType.DISCARDED)
                # Parpadeo tolerado: no se apila nada este frame (no hay
                # payload), pero la captura sigue viva.
                return SignEvent(SignEventType.CAPTURING)

            # En reposo la tolerancia protege al candidato de inicio, que solo
            # existe si ya se acumuló movimiento. Sin candidato en curso no hay
            # nada que sostener y se sale por el camino de abajo.
            if tolerado and self._mov_count > 0:
                return SignEvent(SignEventType.IDLE)

            # Hueco largo (o reposo sin candidato): se olvida todo. `_prev` se
            # suelta SOLO aquí. Mientras el parpadeo esté dentro de la
            # tolerancia se conserva, para que el primer frame en que vuelve la
            # mano mida su movimiento contra la última mano vista en vez de
            # devolver 0.0 y contar como reposo — que en capturando adelantaba
            # el cierre de la seña y en reposo tiraba el candidato de inicio.
            self._prev = None
            self._perdida_count = 0
            self._mov_count = 0
            self._buffer.clear()
            return SignEvent(SignEventType.IDLE)

        self._perdida_count = 0  # hubo mano: se cancela cualquier parpadeo en curso
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
