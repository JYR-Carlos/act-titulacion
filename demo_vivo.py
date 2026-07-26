"""
demo_vivo.py — validación end-to-end del pipeline en PC (CU-01 + CU-03).

Orquesta en vivo todo el flujo runtime (equivalente en PC a
`TranslationSessionController`), como paso previo a portar a Unity/Sentis
(Sección 8.2): solo cambian la fuente de keypoints (aquí webcam/teléfono en vez
de Meta XR SDK) y el canal de salida (aquí overlay OpenCV en vez de subtítulo
espacial). Preprocesamiento e inferencia son idénticos.

Flujo por seña:
    captura -> RestStateDetector (segmenta por reposo) -> KeypointNormalizer
    -> SignClassifier (ONNX, confThreshold) -> MessageComposer (concatena)

Muestra la glosa reconocida, su confianza, el mensaje compuesto y la latencia
end-to-end contra el umbral de la Sección 11 (`config.LATENCIA_MAX_MS`).

Uso:
    python demo_vivo.py --fuente 0
    python demo_vivo.py --fuente http://192.168.1.42:8080/video

Teclas:
    r        reinicia el mensaje compuesto (FA-01 de CU-03)
    q / ESC  salir
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

from lsch_mr import config
from lsch_mr.consola import configurar_utf8
from lsch_mr.fuente_video import FuenteVideo, parse_fuente
from lsch_mr.hand_tracking_provider import HandTrackingProvider, seleccionar_frame
from lsch_mr.message_composer import MessageComposer
from lsch_mr.rest_state_detector import RestStateDetector
from lsch_mr.sign_classifier import SignClassifier
from lsch_mr.tipos import SignEventType

# Traducción SOLO para lo que se dibuja en pantalla. El clasificador sigue
# devolviendo las glosas de LSA64 en inglés (el corpus del MVP: no hay
# vocabulario LSCh, ver CONTEXTO_UNITY.md sección 2) y eso no cambia aquí; esto
# es la "decisión de presentación" que esa misma sección deja abierta, no un
# renombrado de las clases del modelo. Mantener sincronizado a mano con la
# tabla de GLOSARIO_SENAS_MODELO.md.
ETIQUETAS_ES = {
    "Accept": "Aceptar",
    "Appear": "Aparecer",
    "Call": "Llamar",
    "Give": "Dar",
    "Help": "Ayuda",
    "Last_name": "Apellido",
    "Name": "Nombre",
    "None": "Ninguno",
    "Patience": "Paciencia",
    "Thanks": "Gracias",
}


def _es(label: str) -> str:
    """Traduce una glosa del modelo a español para mostrarla en pantalla.

    Si aparece una clase que no está en el diccionario (p. ej. tras
    reentrenar con otro vocabulario), muestra la glosa cruda en vez de
    fallar — mejor un rótulo en inglés que una excepción en plena demo.
    """
    return ETIQUETAS_ES.get(label, label)


@dataclass
class Medicion:
    """Última seña reconocida y sus latencias (métricas de la Sección 11)."""
    label: str = ""
    conf: float = 0.0
    latencia_inferencia_ms: float = 0.0
    retardo_segmentacion_ms: float = 0.0
    latencia_e2e_ms: float = 0.0

    @property
    def excede_umbral(self) -> bool:
        return self.latencia_e2e_ms > config.LATENCIA_MAX_MS


class _PeriodoFrame:
    """Período medio entre frames (ms), sobre los timestamps de la fuente.

    Se usa para convertir el retardo de segmentación —que el diseño define en
    frames (`REST_FRAMES_FIN`)— a milisegundos comparables con el umbral.
    """

    def __init__(self, ventana: int = 30) -> None:
        self._ts: deque[float] = deque(maxlen=ventana)

    def update(self, ts_ms: float) -> None:
        self._ts.append(float(ts_ms))

    def ms(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        return (self._ts[-1] - self._ts[0]) / (len(self._ts) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo end-to-end en PC (CU-01/CU-03)")
    ap.add_argument("--fuente", default="0",
                    help="índice de webcam o URL del teléfono (IP Webcam)")
    ap.add_argument("--sin-espejo", action="store_true",
                    help="no voltear la vista en pantalla (solo afecta al "
                         "renderizado; el modelo nunca ve el frame volteado)")
    ap.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    args = ap.parse_args()

    configurar_utf8()
    import cv2

    clf = SignClassifier(conf_threshold=args.conf)
    print(f"[demo] modo_manos={clf.modo_manos}  clases={clf.classes}")
    print(f"[demo] umbral de latencia end-to-end: {config.LATENCIA_MAX_MS:.0f} ms")
    print("[demo] teclas:  r = reiniciar mensaje (FA-01 de CU-03)  |  q/ESC = salir")
    detector = RestStateDetector()
    composer = MessageComposer()
    ultimo = Medicion()
    periodo = _PeriodoFrame()

    # running_mode="image" explícito: esta demo alimenta a `modelo.onnx`, que se
    # entrenó con keypoints extraídos en IMAGE. Con el default ("video") el
    # modelo recibiría otra distribución y acertaría menos sin dar ningún error
    # — el mismo train/serve skew que el runtime de Unity tiene que evitar.
    # Decisión cerrada: INTEGRACION_UNITY.md sección 7.
    #
    # espejo=False SIEMPRE en la fuente: el corpus se extrajo sin voltear
    # (extraer_lote.py), así que el frame que alimenta al modelo tiene que ir
    # igual. El espejo de UX se aplica más abajo, solo sobre la copia que se
    # dibuja en pantalla — nunca sobre la que entra a `provider.getFrame`.
    # Ver INTEGRACION_UNITY.md sección 7.6, trampa 7.
    mostrar_espejo = not args.sin_espejo
    with FuenteVideo(parse_fuente(args.fuente), espejo=False).abrir() as fv, \
            HandTrackingProvider(num_hands=2, running_mode="image") as provider:
        for frame_bgr, ts_ms in fv.frames():
            t_captura = time.perf_counter()
            periodo.update(ts_ms)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            multi = provider.getFrame(rgb, ts_ms)
            dom = seleccionar_frame(multi, modo="dominante")
            payload = seleccionar_frame(multi, modo=clf.modo_manos)
            evento = detector.update(dom, payload=payload)

            recien_clasificada = False
            if evento.type == SignEventType.END and evento.sequence is not None:
                t0 = time.perf_counter()
                res = clf.classify(evento.sequence)
                ultimo.latencia_inferencia_ms = (time.perf_counter() - t0) * 1000.0
                ultimo.label, ultimo.conf = res.label, res.confidence
                if res.in_vocab:
                    composer.appendWord(_es(res.label))
                recien_clasificada = True

            composer.tick()   # cierre autónomo del mensaje al expirar el timeout
            # Volteo SOLO de la copia que se muestra (UX); `frame_bgr` sin
            # voltear ya se usó arriba para la inferencia y no se toca aquí.
            frame_mostrado = cv2.flip(frame_bgr, 1) if mostrar_espejo else frame_bgr
            _overlay(cv2, frame_mostrado, detector, composer, ultimo)
            cv2.imshow("demo_vivo — LSCh-MR", frame_mostrado)

            if recien_clasificada:
                # El subtítulo ya está dibujado: aquí termina la cadena que mide
                # la Sección 11. Al tiempo de proceso hay que sumarle el retardo
                # de segmentación — los REST_FRAMES_FIN frames de reposo que el
                # detector necesita para confirmar que la seña terminó, y que el
                # usuario percibe como parte de la espera.
                proceso_ms = (time.perf_counter() - t_captura) * 1000.0
                ultimo.retardo_segmentacion_ms = detector.frames_fin * periodo.ms()
                ultimo.latencia_e2e_ms = ultimo.retardo_segmentacion_ms + proceso_ms
                aviso = "  [!] EXCEDE EL UMBRAL" if ultimo.excede_umbral else ""
                print(f"  -> {ultimo.label:>14} ({_es(ultimo.label)})  "
                      f"conf={ultimo.conf:.2f}  "
                      f"inferencia={ultimo.latencia_inferencia_ms:.1f} ms  "
                      f"e2e={ultimo.latencia_e2e_ms:.0f} ms "
                      f"(<= {config.LATENCIA_MAX_MS:.0f} ms){aviso}")

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord("r"):
                composer.reiniciar()
                print("  [r] mensaje reiniciado (FA-01 de CU-03)")
        cv2.destroyAllWindows()
    return 0


def _overlay(cv2, frame, detector, composer, ultimo: Medicion):
    # Nota: OpenCV no renderiza tildes con las fuentes Hershey — texto sin acentos.
    h, w = frame.shape[:2]
    estado = "CAPTURANDO" if detector.capturando else "reposo"
    color = (0, 0, 255) if detector.capturando else (0, 180, 0)
    if ultimo.label:
        txt = (f"{_es(ultimo.label)}  ({ultimo.conf:.2f})"
               if ultimo.label != "<desconocida>" else "fuera de vocabulario")
        cv2.putText(frame, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2)
        col_lat = (0, 0, 255) if ultimo.excede_umbral else (200, 200, 200)
        cv2.putText(frame,
                    f"e2e: {ultimo.latencia_e2e_ms:.0f} ms  "
                    f"(inferencia {ultimo.latencia_inferencia_ms:.0f} + "
                    f"segmentacion {ultimo.retardo_segmentacion_ms:.0f})",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_lat, 1)
    cv2.putText(frame, f"{estado} ({detector.n_frames_buffer})", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "[r] reiniciar  [q] salir", (w - 240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    # Subtítulo compuesto (MessageComposer)
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Mensaje: " + composer.texto(), (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())
