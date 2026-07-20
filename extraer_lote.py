"""
extraer_lote.py — extracción de keypoints por lotes sobre una carpeta de vídeos.

Corre NUESTRO `HandTrackingProvider` (MediaPipe Tasks, 21 keypoints/mano) sobre
cada vídeo y vuelca un único CSV crudo en el esquema propio del proyecto
(`csv_esquema.py`). Cada vídeo se trata como una muestra (sample_id = nombre del
archivo) con su etiqueta.

Sirve para validar el pipeline con datasets de referencia (SWL-LSE VIDEOS_REF,
LSA64, etc.) SIN reutilizar código ni keypoints de terceros: solo se toman los
vídeos y se re-extrae con nuestra Capa 1 (coherente con DECISION_PREPROCESAMIENTO.md).

Etiquetas: se pueden dar con un CSV de anotaciones (columnas FILENAME + LABEL) o
inferir del nombre de archivo.

Uso (SWL-LSE reference set):
    python extraer_lote.py \
        --videos-dir data/external/swl_lse/VIDEOS_REF \
        --anotaciones data/external/swl_lse/videos_ref_annotations.csv \
        --salida data/raw/swl_ref.csv --limite 10
"""
from __future__ import annotations

import argparse
import csv as _csv
from pathlib import Path

from lsch_mr.consola import configurar_utf8
from lsch_mr.csv_esquema import EscritorCSV
from lsch_mr.fuente_video import FuenteVideo

configurar_utf8()

_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv")


def _mapa_etiquetas(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapa = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            cols = {k.strip().upper(): v for k, v in row.items()}
            fname = cols.get("FILENAME") or cols.get("FILE") or cols.get("VIDEO")
            label = cols.get("LABEL") or cols.get("GLOSA") or cols.get("CLASS")
            if fname and label:
                mapa[Path(fname).stem] = label.strip()
    return mapa


def main() -> int:
    ap = argparse.ArgumentParser(description="Extracción de keypoints por lotes")
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--anotaciones", default=None,
                    help="CSV con columnas FILENAME + LABEL")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--limite", type=int, default=0,
                    help="procesar solo los primeros N vídeos (0 = todos)")
    args = ap.parse_args()

    from lsch_mr.hand_tracking_provider import HandTrackingProvider
    import cv2

    vdir = Path(args.videos_dir)
    videos = sorted([p for p in vdir.rglob("*") if p.suffix.lower() in _EXTS])
    if args.limite:
        videos = videos[: args.limite]
    if not videos:
        print(f"No se encontraron vídeos en {vdir}")
        return 1

    etiquetas = _mapa_etiquetas(Path(args.anotaciones) if args.anotaciones else None)
    print(f"[lote] {len(videos)} vídeos | etiquetas mapeadas: {len(etiquetas)}")

    # modo IMAGE: cada frame independiente (sin timestamps ni tracking entre vídeos)
    provider = HandTrackingProvider(num_hands=args.num_hands, running_mode="image")
    total_filas = 0
    procesados = 0
    con_manos = 0
    try:
        with EscritorCSV(Path(args.salida)) as escritor:
            for vid in videos:
                label = etiquetas.get(vid.stem, vid.stem)
                fidx = 0
                filas_video = 0
                with FuenteVideo(str(vid)).abrir() as fv:
                    for frame_bgr, ts_ms in fv.frames():
                        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        multi = provider.getFrame(rgb, ts_ms)
                        filas_video += escritor.escribir_multiframe(
                            vid.stem, fidx, multi, label)
                        fidx += 1
                total_filas += filas_video
                procesados += 1
                if filas_video:
                    con_manos += 1
                print(f"  [{procesados}/{len(videos)}] {vid.name} -> "
                      f"label='{label}' frames={fidx} filas={filas_video}")
    finally:
        provider.cerrar()

    print(f"\n[lote] listo: {procesados} vídeos, {con_manos} con manos detectadas, "
          f"{total_filas} filas -> {args.salida}")
    print("Siguiente:  python build_dataset.py --entrada", args.salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
