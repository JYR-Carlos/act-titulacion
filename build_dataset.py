"""
build_dataset.py — CSV(s) crudo(s) -> dataset normalizado listo para entrenar.

Resuelve el ítem de backlog "integrar extracción + normalización en un solo
flujo" (CONTEXTO_PROYECTO.md Sección 10). Reconstruye cada SignSequence a partir
de las filas del CSV (una fila por mano por frame), aplica el KeypointNormalizer
y el remuestreo temporal, y guarda un `.npz` con:
    X       -> (N, SEQ_LEN, n_features)
    y       -> (N,)  índices de clase
    classes -> lista de glosas (orden estable)

Modo de manos (punto abierto, Sección 6):
    --modo dominante  (63 dim)  | --modo ambas (126 dim)

Uso:
    python build_dataset.py                       # lee data/raw/*.csv
    python build_dataset.py --entrada data/raw/corpus_x.csv --modo ambas
"""
from __future__ import annotations

import argparse
import csv as _csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from lsch_mr import config
from lsch_mr.caracteristicas import preparar_entrada
from lsch_mr.consola import configurar_utf8
from lsch_mr.glosas import cargar_glosas

configurar_utf8()

_XCOLS = [f"x{i}" for i in range(config.NUM_LANDMARKS)]
_YCOLS = [f"y{i}" for i in range(config.NUM_LANDMARKS)]
_ZCOLS = [f"z{i}" for i in range(config.NUM_LANDMARKS)]


def _leer_csv(path: Path):
    """Devuelve dict[(sample_key, label)] -> dict[frame_idx] -> list[fila]."""
    muestras = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            label = row["label"].strip()
            if not label:
                continue  # sin etiqueta -> no sirve para entrenar
            key = (path.stem, label, row["sample_id"])
            muestras[(key, label)][int(row["frame_idx"])].append(row)
    return muestras


def _landmarks(row) -> np.ndarray:
    pts = np.empty((config.NUM_LANDMARKS, 3), dtype=np.float32)
    for i in range(config.NUM_LANDMARKS):
        pts[i, 0] = float(row[_XCOLS[i]])
        pts[i, 1] = float(row[_YCOLS[i]])
        pts[i, 2] = float(row[_ZCOLS[i]])
    return pts


def _secuencia_dominante(frames: dict) -> np.ndarray:
    seq = []
    for fi in sorted(frames):
        filas = frames[fi]
        mejor = max(filas, key=lambda r: float(r["handedness_score"]))
        seq.append(_landmarks(mejor))
    return np.stack(seq, axis=0)


def _secuencia_ambas(frames: dict) -> np.ndarray:
    vacia = np.full((config.NUM_LANDMARKS, 3), np.nan, np.float32)
    seq = []
    for fi in sorted(frames):
        filas = frames[fi]
        def mano(lado):
            cand = [r for r in filas if r["hand"] == lado]
            if not cand:
                return vacia
            return _landmarks(max(cand, key=lambda r: float(r["handedness_score"])))
        seq.append(np.stack([mano("Left"), mano("Right")], axis=0))
    return np.stack(seq, axis=0)


def _orden_clases(labels_presentes: set[str]) -> list[str]:
    try:
        catalogo = cargar_glosas()
    except FileNotFoundError:
        catalogo = []
    ordenadas = [g for g in catalogo if g in labels_presentes]
    extra = sorted(labels_presentes - set(ordenadas))
    return ordenadas + extra


def main() -> int:
    ap = argparse.ArgumentParser(description="Construcción del dataset normalizado")
    ap.add_argument("--entrada", nargs="*", default=None,
                    help="CSV(s) crudo(s); por defecto data/raw/*.csv")
    ap.add_argument("--salida", default=str(config.DATA_PROCESSED_DIR / "dataset.npz"))
    ap.add_argument("--modo", choices=["dominante", "ambas"],
                    default=config.MODO_MANOS)
    args = ap.parse_args()

    if args.entrada:
        csvs = [Path(p) for p in args.entrada]
    else:
        csvs = sorted(config.DATA_RAW_DIR.glob("*.csv"))
    if not csvs:
        print("No hay CSV de entrada. Graba corpus o extrae keypoints primero.")
        return 1

    # 1) Reconstruir todas las muestras.
    todas = {}
    for c in csvs:
        for (key, label), frames in _leer_csv(c).items():
            todas[key] = (label, frames)
    if not todas:
        print("No se encontraron muestras etiquetadas en los CSV.")
        return 1

    labels_presentes = {lab for lab, _ in todas.values()}
    classes = _orden_clases(labels_presentes)
    idx_de = {c: i for i, c in enumerate(classes)}

    # 2) Construir X, y.
    X, y = [], []
    descartadas = 0
    for (label, frames) in todas.values():
        try:
            if args.modo == "ambas":
                seq = _secuencia_ambas(frames)
            else:
                seq = _secuencia_dominante(frames)
            if len(seq) < 1:
                descartadas += 1
                continue
            X.append(preparar_entrada(seq, modo=args.modo, seq_len=config.SEQ_LEN))
            y.append(idx_de[label])
        except Exception as e:
            descartadas += 1
            print(f"  aviso: muestra '{label}' descartada ({e})")

    if not X:
        print("No se pudo construir ninguna muestra válida.")
        return 1

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida, X=X, y=y, classes=np.array(classes, dtype=object),
                        modo_manos=args.modo)

    print(f"[build_dataset] modo={args.modo}  X={X.shape}  clases={len(classes)}")
    for c in classes:
        print(f"    {c:>12}: {int(np.sum(y == idx_de[c]))} muestras")
    if descartadas:
        print(f"  ({descartadas} muestras descartadas)")
    print(f"[build_dataset] guardado -> {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
