"""
demo_vivo.py — validación end-to-end del pipeline en PC (CU-01 + CU-03).

Orquesta en vivo todo el flujo runtime (equivalente en PC a
`TranslationSessionController`), como paso previo a portar a Quest 3
(Sección 8.2): solo cambian la fuente de keypoints (aquí webcam/teléfono en vez
de Meta XR SDK) y el canal de salida (aquí overlay OpenCV en vez de subtítulo
espacial). Preprocesamiento e inferencia son idénticos.

Flujo por seña:
    captura -> RestStateDetector (segmenta por reposo) -> KeypointNormalizer
    -> SignClassifier (ONNX, confThreshold) -> MessageComposer (concatena)

Muestra la glosa reconocida, su confianza, la latencia de inferencia y el
mensaje compuesto.

Uso:
    python demo_vivo.py --fuente 0
    python demo_vivo.py --fuente http://192.168.1.42:8080/video
"""
from __future__ import annotations

import argparse
import time

from lsch_mr import config
from lsch_mr.consola import configurar_utf8
from lsch_mr.fuente_video import FuenteVideo, parse_fuente
from lsch_mr.hand_tracking_provider import HandTrackingProvider, seleccionar_frame
from lsch_mr.message_composer import MessageComposer
from lsch_mr.rest_state_detector import RestStateDetector
from lsch_mr.sign_classifier import SignClassifier
from lsch_mr.tipos import SignEventType


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo end-to-end en PC (CU-01/CU-03)")
    ap.add_argument("--fuente", default="0",
                    help="índice de webcam o URL del teléfono (IP Webcam)")
    ap.add_argument("--sin-espejo", action="store_true")
    ap.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    args = ap.parse_args()

    configurar_utf8()
    import cv2

    clf = SignClassifier(conf_threshold=args.conf)
    print(f"[demo] modo_manos={clf.modo_manos}  clases={clf.classes}")
    detector = RestStateDetector()
    composer = MessageComposer()
    ultimo = ("", 0.0, 0.0)  # (label, conf, latencia_ms)

    with FuenteVideo(parse_fuente(args.fuente), espejo=not args.sin_espejo).abrir() as fv, \
            HandTrackingProvider(num_hands=2) as provider:
        for frame_bgr, ts_ms in fv.frames():
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            multi = provider.getFrame(rgb, ts_ms)
            dom = seleccionar_frame(multi, modo="dominante")
            payload = seleccionar_frame(multi, modo=clf.modo_manos)
            evento = detector.update(dom, payload=payload)

            if evento.type == SignEventType.END and evento.sequence is not None:
                t0 = time.perf_counter()
                res = clf.classify(evento.sequence)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                ultimo = (res.label, res.confidence, lat_ms)
                if res.in_vocab:
                    composer.appendWord(res.label)
                print(f"  -> {res.label:>14}  conf={res.confidence:.2f}  "
                      f"inferencia={lat_ms:.1f} ms")

            _overlay(cv2, frame_bgr, detector, composer, ultimo)
            cv2.imshow("demo_vivo — LSCh-MR", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
        cv2.destroyAllWindows()
    return 0


def _overlay(cv2, frame, detector, composer, ultimo):
    h, w = frame.shape[:2]
    label, conf, lat = ultimo
    estado = "CAPTURANDO" if detector.capturando else "reposo"
    color = (0, 0, 255) if detector.capturando else (0, 180, 0)
    if label:
        txt = f"{label}  ({conf:.2f})" if label != "<desconocida>" else "fuera de vocabulario"
        cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2)
        cv2.putText(frame, f"inferencia: {lat:.0f} ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"{estado} ({detector.n_frames_buffer})", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Subtítulo compuesto (MessageComposer)
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Mensaje: " + composer.texto(), (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())
