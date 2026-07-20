"""
grabar_corpus.py — CLI del DatasetRecorder (CU-02).

Graba el corpus en tomas continuas con segmentación automática por reposo.
Fuente: webcam local o cámara de teléfono.

Uso:
    python grabar_corpus.py --fuente 0
    python grabar_corpus.py --fuente http://192.168.1.42:8080/video
"""
from __future__ import annotations

import argparse

from lsch_mr.consola import configurar_utf8
from lsch_mr.dataset_recorder import DatasetRecorder
from lsch_mr.fuente_video import parse_fuente
from lsch_mr.glosas import cargar_glosas

configurar_utf8()


def main() -> int:
    ap = argparse.ArgumentParser(description="Grabación del corpus (DatasetRecorder)")
    ap.add_argument("--fuente", default="0",
                    help="índice de webcam o URL del teléfono (IP Webcam)")
    ap.add_argument("--sin-espejo", action="store_true",
                    help="no voltear la imagen horizontalmente")
    args = ap.parse_args()

    glosas = cargar_glosas()
    print(f"Glosas cargadas: {glosas}")
    recorder = DatasetRecorder(
        glosas=glosas, fuente=parse_fuente(args.fuente),
        espejo=not args.sin_espejo)
    csv_path = recorder.recordSession()
    print(f"\nCorpus guardado en: {csv_path}")
    print("Siguiente paso:  python build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
