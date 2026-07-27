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
    ap.add_argument("--min-muestras", type=int,
                    default=config.CORPUS_MIN_MUESTRAS_POR_CLASE,
                    help="muestras mínimas por glosa (Sección 12 del diseño; "
                         f"default {config.CORPUS_MIN_MUESTRAS_POR_CLASE})")
    ap.add_argument("--estricto", action="store_true",
                    help="no guardar el dataset si alguna glosa queda bajo el "
                         "mínimo, en vez de solo avisar")
    args = ap.parse_args()

    if args.entrada:
        csvs = [Path(p) for p in args.entrada]
    else:
        csvs = sorted(config.DATA_RAW_DIR.glob("*.csv"))
    if not csvs:
        print(f"No hay CSV de entrada en {config.DATA_RAW_DIR}.\n"
              "  Extrae keypoints de un corpus de vídeos:\n"
              "    python extraer_lote.py --videos-dir <dir> "
              "--anotaciones lsa64_10_annotations.csv --salida data/raw/lsa64_10.csv\n"
              "  o graba corpus propio con:  python grabar_corpus.py --senante s01")
        return 1

    # Se comprueban todos antes de leer ninguno: mejor listar los que faltan de
    # una vez que morir en el primero tras haber procesado los anteriores.
    faltan = [str(c) for c in csvs if not c.exists()]
    if faltan:
        print("No existe(n) el/los CSV de entrada:\n" +
              "".join(f"    {f}\n" for f in faltan) +
              "  Revisa la ruta, o genéralo con extraer_lote.py / grabar_corpus.py.")
        return 1

    # 1) Reconstruir todas las muestras.
    todas = {}
    for c in csvs:
        try:
            for (key, label), frames in _leer_csv(c).items():
                todas[key] = (label, frames)   # key = (archivo, label, sample_id)
        except (OSError, UnicodeDecodeError, _csv.Error) as e:
            print(f"No se pudo leer {c}: {e}\n"
                  "  ¿Es un CSV del esquema de este repo (lsch_mr/csv_esquema.py)?")
            return 1
    if not todas:
        print("Los CSV no tienen ninguna muestra etiquetada.\n"
              f"  Se leyeron {len(csvs)} archivo(s) pero ninguna fila con "
              "sample_id y label utilizables.\n"
              "  Comprueba que las columnas son las de lsch_mr/csv_esquema.py.")
        return 1

    labels_presentes = {lab for lab, _ in todas.values()}
    classes = _orden_clases(labels_presentes)
    idx_de = {c: i for i, c in enumerate(classes)}

    # 2) Construir X, y. Se conserva el sample_id de cada muestra: permite
    #    trazar una fila del dataset hasta su grabación original y evaluar
    #    dejando señantes fuera (ver entrenar.py --cv-grupos).
    X, y, sample_ids = [], [], []
    descartadas = 0
    for (_, _, sample_id), (label, frames) in todas.items():
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
            sample_ids.append(sample_id)
        except Exception as e:
            descartadas += 1
            print(f"  aviso: muestra '{label}' descartada ({e})")

    if not X:
        print("No se pudo construir ninguna muestra válida.")
        return 1

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    # 3) Gate de cobertura del corpus (Sección 12: >= 50 muestras por glosa).
    #    Es más barato descubrir aquí que una glosa quedó corta que después de
    #    entrenar, cuando ya no hay sesión de grabación disponible.
    conteos = {c: int(np.sum(y == idx_de[c])) for c in classes}
    bajo_minimo = {c: n for c, n in conteos.items() if n < args.min_muestras}

    print(f"[build_dataset] modo={args.modo}  X={X.shape}  clases={len(classes)}")
    for c in classes:
        falta = f"  <-- faltan {args.min_muestras - conteos[c]}" if c in bajo_minimo else ""
        print(f"    {c:>12}: {conteos[c]} muestras{falta}")
    if descartadas:
        print(f"  ({descartadas} muestras descartadas)")

    if bajo_minimo:
        print(f"\n[build_dataset] AVISO: {len(bajo_minimo)} de {len(classes)} glosas "
              f"quedan bajo las {args.min_muestras} muestras que fija el diseño. "
              f"La clase más escasa tiene {min(bajo_minimo.values())}.")
        print("  Con clases escasas el split 80/20 deja muy pocas muestras de "
              "validación: reporta la métrica con k-fold (entrenar.py --cv).")
        if args.estricto:
            print("[build_dataset] --estricto: no se guardó el dataset.")
            return 2

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida, X=X, y=y, classes=np.array(classes, dtype=object),
                        modo_manos=args.modo,
                        sample_ids=np.array(sample_ids, dtype=object))
    print(f"[build_dataset] guardado -> {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
