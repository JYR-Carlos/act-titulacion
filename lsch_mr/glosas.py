"""Carga del catálogo de glosas (data/catalogos/Glosas_LSCh_Mappeadas.csv)."""
from __future__ import annotations

import csv
from pathlib import Path

from . import config


def cargar_glosas(path: Path = config.GLOSAS_CSV) -> list[str]:
    """Devuelve la lista de glosas ordenada por `id`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el catálogo de glosas: {path}")
    filas = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filas.append((int(row["id"]), row["glosa"].strip()))
    filas.sort(key=lambda t: t[0])
    return [g for _, g in filas]
