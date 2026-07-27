"""
Esquema CSV del corpus (docs/CONTEXTO_PROYECTO.md Sección 6).

Base documentada:  frame_idx, x0..x20, y0..y20, z0..z20, label
Extensiones (punto abierto de una vs. dos manos): se escribe UNA FILA POR MANO
detectada por frame, agregando las columnas `hand` (Left/Right) y
`handedness_score`. Además se añade `sample_id` para agrupar los frames de una
misma seña (necesario porque el corpus son secuencias, no frames sueltos).

Orden de coordenadas: agrupado por eje (todas las x, luego y, luego z), tal
como está documentado. La reconstrucción a (21,3) en `build_dataset` respeta
este orden.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config
from .tipos import MultiHandFrame

_X = [f"x{i}" for i in range(config.NUM_LANDMARKS)]
_Y = [f"y{i}" for i in range(config.NUM_LANDMARKS)]
_Z = [f"z{i}" for i in range(config.NUM_LANDMARKS)]

COLUMNAS = ["sample_id", "frame_idx", "hand", "handedness_score"] + _X + _Y + _Z + ["label"]


def fila_desde_mano(sample_id, frame_idx: int, landmarks: np.ndarray,
                    hand: str, score: float, label: str) -> dict:
    """Una fila del CSV crudo: los 21 keypoints de UNA mano en UN frame.

    Los keypoints se guardan **sin normalizar**, tal como salen del
    `HandLandmarker`. La normalización es un paso posterior (`build_dataset`),
    para poder rehacerla sin volver a extraer el corpus, que es lo caro.

    Un frame con dos manos produce dos filas con el mismo `sample_id` y
    `frame_idx`, distinguidas por la columna `hand`.
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(config.NUM_LANDMARKS,
                                                          config.NUM_EJES)
    fila = {"sample_id": sample_id, "frame_idx": frame_idx,
            "hand": hand, "handedness_score": round(float(score), 4),
            "label": label}
    for i in range(config.NUM_LANDMARKS):
        fila[f"x{i}"] = round(float(pts[i, 0]), 6)
        fila[f"y{i}"] = round(float(pts[i, 1]), 6)
        fila[f"z{i}"] = round(float(pts[i, 2]), 6)
    return fila


class EscritorCSV:
    """Escritor incremental del CSV crudo del corpus."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNAS)
        self._w.writeheader()

    def escribir_multiframe(self, sample_id, frame_idx: int,
                            multi: MultiHandFrame, label: str) -> int:
        """Escribe una fila por mano detectada. Devuelve cuántas escribió.

        Cero manos escribe cero filas: los huecos del corpus son huecos en
        `frame_idx`, no filas de ceros. Por eso el contador de retorno importa
        —es lo que permite detectar un vídeo sin ninguna detección.
        """
        n = 0
        for h in multi.hands:
            self._w.writerow(fila_desde_mano(sample_id, frame_idx,
                                             h.landmarks, h.hand, h.score, label))
            n += 1
        return n

    def escribir_filas(self, filas: Iterable[dict]) -> None:
        """Escribe filas ya construidas (deben tener las claves de `COLUMNAS`)."""
        for fila in filas:
            self._w.writerow(fila)

    def cerrar(self) -> None:
        """Cierra el archivo. Preferir el uso como context manager."""
        self._f.close()

    def __enter__(self) -> "EscritorCSV":
        return self

    def __exit__(self, *exc) -> None:
        self.cerrar()
