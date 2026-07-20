"""
extraer_keypoints.py — extracción de keypoints con MediaPipe Tasks (HandLandmarker).

Modos:
  --modo webcam   Preview en vivo (webcam local o cámara de teléfono). Muestra
                  los landmarks; opcionalmente vuelca el CSV crudo con --salida.
  --modo video    Batch sobre un archivo de video -> genera el CSV crudo.

Fuente (--fuente):
  * Índice de webcam:            --fuente 0
  * Cámara de teléfono (URL):    --fuente http://192.168.1.42:8080/video
  * Archivo de video:            --fuente ruta/al/video.mp4   (con --modo video)

Ejemplos:
  python extraer_keypoints.py --modo webcam --fuente 0
  python extraer_keypoints.py --modo webcam --fuente http://192.168.1.42:8080/video
  python extraer_keypoints.py --modo video --fuente sena.mp4 --label HOLA --salida data/raw/hola.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lsch_mr import config
from lsch_mr.consola import configurar_utf8
from lsch_mr.csv_esquema import EscritorCSV
from lsch_mr.fuente_video import FuenteVideo, parse_fuente
from lsch_mr.hand_tracking_provider import HandTrackingProvider

configurar_utf8()

_CONEXIONES = [  # esqueleto de la mano para dibujar (índices MediaPipe)
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def _dibujar(cv2, frame, multi):
    h, w = frame.shape[:2]
    for hand in multi.hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
        for a, b in _CONEXIONES:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)
        cv2.putText(frame, f"{hand.hand} {hand.score:.2f}", pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extracción de keypoints (HandLandmarker)")
    ap.add_argument("--modo", choices=["webcam", "video"], required=True)
    ap.add_argument("--fuente", default="0",
                    help="índice de webcam, URL del teléfono o ruta de video")
    ap.add_argument("--label", default="", help="glosa a etiquetar (modo video)")
    ap.add_argument("--salida", default="", help="ruta del CSV crudo de salida")
    ap.add_argument("--espejo", action="store_true", help="voltear horizontal")
    ap.add_argument("--num-hands", type=int, default=2)
    args = ap.parse_args()

    import cv2

    fuente = parse_fuente(args.fuente)
    escritor = None
    if args.salida:
        escritor = EscritorCSV(Path(args.salida))
    elif args.modo == "video":
        salida = config.DATA_RAW_DIR / (Path(str(fuente)).stem + ".csv")
        escritor = EscritorCSV(salida)
        print(f"[extraer] CSV -> {salida}")

    # webcam/teléfono en vivo -> modo VIDEO (tracking temporal);
    # archivo de vídeo en batch -> modo IMAGE (frames independientes).
    provider = HandTrackingProvider(
        num_hands=args.num_hands,
        running_mode="video" if args.modo == "webcam" else "image")
    frame_idx = 0
    n_filas = 0
    try:
        with FuenteVideo(fuente, espejo=args.espejo).abrir() as fv:
            for frame_bgr, ts_ms in fv.frames():
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                multi = provider.getFrame(rgb, ts_ms)
                if escritor is not None:
                    n_filas += escritor.escribir_multiframe(
                        0, frame_idx, multi, args.label)
                frame_idx += 1

                if args.modo == "webcam":
                    _dibujar(cv2, frame_bgr, multi)
                    cv2.putText(frame_bgr,
                                f"manos: {len(multi.hands)}  frame: {frame_idx}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2)
                    cv2.imshow("extraer_keypoints — LSCh-MR", frame_bgr)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
    finally:
        provider.cerrar()
        if escritor is not None:
            escritor.cerrar()
        cv2.destroyAllWindows()

    print(f"[extraer] frames procesados: {frame_idx} | filas escritas: {n_filas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
