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
    p        muestra/oculta los puntos (keypoints) de las manos detectadas
             (también hay un botón clicable en la esquina superior derecha)
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


# -- Paleta del overlay (BGR, como espera OpenCV) --------------------------- #
_COLOR_FONDO = (32, 28, 24)
_COLOR_REPOSO = (170, 130, 40)
_COLOR_CAPTURANDO = (0, 170, 255)
_COLOR_EXITO = (70, 190, 90)
_COLOR_OOV = (150, 150, 150)
_COLOR_ALERTA = (60, 60, 230)
_COLOR_TEXTO = (255, 255, 255)
_COLOR_TEXTO_TENUE = (185, 185, 185)
_COLOR_PUNTOS = (0, 255, 255)  # keypoints de las manos (botón "Puntos")


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
    vocab_es = [_es(c) for c in clf.classes]  # mismo orden que clf.classes
    print(f"[demo] modo_manos={clf.modo_manos}  clases={clf.classes}")
    print(f"[demo] umbral de latencia end-to-end: {config.LATENCIA_MAX_MS:.0f} ms")
    print("[demo] teclas:  p = puntos de las manos  |  r = reiniciar mensaje "
          "(FA-01 de CU-03)  |  q/ESC = salir")
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
    aviso_txt, aviso_hasta = "", 0.0  # feedback visual breve para DISCARDED

    # Estado del botón "Puntos" (mostrar/ocultar keypoints de las manos).
    # Diccionario mutable en vez de variable local: lo comparte con el
    # callback del mouse, que corre fuera del cuerpo del bucle.
    NOMBRE_VENTANA = "demo_vivo — LSCh-MR"
    estado_ui = {"mostrar_puntos": False, "boton_rect": (0, 0, 0, 0)}

    def _click_boton(event, x, y, flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = estado_ui["boton_rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            estado_ui["mostrar_puntos"] = not estado_ui["mostrar_puntos"]

    cv2.namedWindow(NOMBRE_VENTANA)
    cv2.setMouseCallback(NOMBRE_VENTANA, _click_boton)

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
            elif evento.type == SignEventType.DISCARDED:
                # Antes de la tolerancia a parpadeos (config.REST_FRAMES_PERDIDA_MAX)
                # esto pasaba en silencio y parecía que "el sistema dejó de andar".
                # Ahora solo llega aquí una pérdida de tracking sostenida real.
                aviso_txt = "Sena descartada: se perdio el seguimiento de la mano"
                aviso_hasta = time.monotonic() + 2.0
                print("  [!] secuencia descartada: perdida de tracking sostenida "
                      f"(> {detector.frames_perdida_max} frames sin mano)")

            composer.tick()   # cierre autónomo del mensaje al expirar el timeout
            # Volteo SOLO de la copia que se muestra (UX); `frame_bgr` sin
            # voltear ya se usó arriba para la inferencia y no se toca aquí.
            frame_mostrado = cv2.flip(frame_bgr, 1) if mostrar_espejo else frame_bgr
            if estado_ui["mostrar_puntos"]:
                _dibujar_puntos(cv2, frame_mostrado, multi, mostrar_espejo)
            aviso_activo = aviso_txt if time.monotonic() < aviso_hasta else ""
            _overlay(cv2, frame_mostrado, detector, composer, ultimo, periodo,
                     vocab_es, args.conf, estado_ui, aviso_activo)
            cv2.imshow(NOMBRE_VENTANA, frame_mostrado)

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
            if tecla == ord("p"):
                estado_ui["mostrar_puntos"] = not estado_ui["mostrar_puntos"]
                print(f"  [p] puntos {'activados' if estado_ui['mostrar_puntos'] else 'ocultos'}")
        cv2.destroyAllWindows()
    return 0


def _panel(cv2, frame, x1, y1, x2, y2, color, alpha=0.55):
    """Rectángulo translúcido — fondo de las barras y tarjetas del overlay."""
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    capa = frame.copy()
    cv2.rectangle(capa, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(capa, alpha, frame, 1 - alpha, 0, dst=frame)


def _envolver(cv2, texto, font, escala, grosor, ancho_max):
    """Parte `texto` en líneas que no superen `ancho_max` píxeles."""
    lineas: list[str] = []
    actual = ""
    for palabra in texto.split():
        candidato = f"{actual} {palabra}".strip()
        (ancho, _), _ = cv2.getTextSize(candidato, font, escala, grosor)
        if ancho > ancho_max and actual:
            lineas.append(actual)
            actual = palabra
        else:
            actual = candidato
    if actual:
        lineas.append(actual)
    return lineas or [""]


def _dibujar_puntos(cv2, frame, multi, mostrar_espejo: bool) -> None:
    """Dibuja los 21 keypoints de cada mano detectada (depuración / demo).

    Los landmarks llegan en coordenadas normalizadas [0,1] del frame SIN
    voltear (el mismo que ve el modelo). Si `frame` está en espejo para la
    vista del usuario, se invierte aquí el eje x para que los puntos calcen
    con la mano tal como se ve en pantalla.
    """
    h, w = frame.shape[:2]
    radio = max(2, int(0.006 * w))
    for mano in multi.hands:
        for x, y, _z in mano.landmarks:
            px = (1.0 - x) * w if mostrar_espejo else x * w
            py = y * h
            cv2.circle(frame, (int(px), int(py)), radio, _COLOR_PUNTOS, -1,
                      cv2.LINE_AA)


def _overlay(cv2, frame, detector, composer, ultimo: Medicion, periodo,
            vocab_es: list[str], umbral_conf: float, estado_ui: dict,
            aviso: str = ""):
    # Nota: OpenCV no renderiza tildes con las fuentes Hershey — por eso las
    # traducciones de ETIQUETAS_ES y los rótulos de aquí van sin acentos.
    F = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    s = max(0.6, min(2.2, h / 480.0))  # escala tipográfica relativa a 480p

    # -- Cabecera: una sola franja con estado + FPS y, debajo, vocabulario -- #
    barra_h = int(34 * s)
    y_vocab = barra_h + int(20 * s)
    header_h = y_vocab + int(10 * s)
    _panel(cv2, frame, 0, 0, w, header_h, _COLOR_FONDO, alpha=0.65)

    capturando = detector.capturando
    color_estado = _COLOR_CAPTURANDO if capturando else _COLOR_REPOSO
    texto_estado = "CAPTURANDO SENA..." if capturando else "ESCUCHANDO"
    cy = barra_h // 2 + int(6 * s)
    cv2.circle(frame, (int(16 * s), barra_h // 2), max(4, int(7 * s)),
              color_estado, -1, cv2.LINE_AA)
    cv2.putText(frame, texto_estado, (int(30 * s), cy), F, 0.55 * s,
                color_estado, max(1, int(1.6 * s)), cv2.LINE_AA)
    if capturando:
        (ancho_estado, _), _ = cv2.getTextSize(texto_estado, F, 0.55 * s,
                                               max(1, int(1.6 * s)))
        cv2.putText(frame, f"({detector.n_frames_buffer} frames)",
                    (int(40 * s) + ancho_estado, cy), F, 0.5 * s,
                    _COLOR_TEXTO_TENUE, 1, cv2.LINE_AA)
    fps_txt = f"{1000.0 / periodo.ms():.0f} FPS" if periodo.ms() > 0 else "-- FPS"
    (fps_w, _), _ = cv2.getTextSize(fps_txt, F, 0.5 * s, 1)
    cv2.putText(frame, fps_txt, (w - fps_w - int(14 * s), cy), F, 0.5 * s,
                _COLOR_TEXTO_TENUE, 1, cv2.LINE_AA)

    # Botón "Puntos": clic (o tecla [p]) para mostrar/ocultar los keypoints de
    # las manos. El rect se recalcula cada frame y se guarda en estado_ui para
    # que el callback del mouse (registrado una sola vez) sepa dónde está.
    mostrar_puntos = estado_ui["mostrar_puntos"]
    boton_txt = "Puntos: ON" if mostrar_puntos else "Puntos: OFF"
    boton_grosor = 1
    (boton_txt_w, boton_txt_h), _ = cv2.getTextSize(boton_txt, F, 0.4 * s, boton_grosor)
    boton_w = boton_txt_w + int(20 * s)
    boton_h = int(22 * s)
    boton_x2 = w - fps_w - int(28 * s)
    boton_x1 = boton_x2 - boton_w
    boton_y1 = max(0, barra_h // 2 - boton_h // 2)
    boton_y2 = boton_y1 + boton_h
    estado_ui["boton_rect"] = (boton_x1, boton_y1, boton_x2, boton_y2)
    color_boton = _COLOR_PUNTOS if mostrar_puntos else _COLOR_TEXTO_TENUE
    cv2.rectangle(frame, (boton_x1, boton_y1), (boton_x2, boton_y2), color_boton,
                  max(1, int(1.4 * s)), cv2.LINE_AA)
    cv2.putText(frame,
                boton_txt,
                (boton_x1 + (boton_w - boton_txt_w) // 2,
                 boton_y1 + (boton_h + boton_txt_h) // 2),
                F, 0.4 * s, color_boton, boton_grosor, cv2.LINE_AA)

    # Vocabulario: qué señas reconoce este MVP (misma franja, fila de abajo).
    cv2.putText(frame, "Vocabulario: " + "  |  ".join(vocab_es),
                (int(12 * s), y_vocab), F, 0.42 * s, _COLOR_TEXTO_TENUE, 1,
                cv2.LINE_AA)

    # -- Tarjeta: última seña reconocida + barra de confianza ---------------- #
    y0 = header_h + int(14 * s)
    alto_tarjeta = int(70 * s)
    _panel(cv2, frame, 0, y0, w, y0 + alto_tarjeta, _COLOR_FONDO, alpha=0.55)
    if aviso:
        cv2.putText(frame, aviso, (int(16 * s), y0 + int(32 * s)), F, 0.65 * s,
                    _COLOR_ALERTA, max(1, int(1.8 * s)), cv2.LINE_AA)
    elif not ultimo.label:
        cv2.putText(frame, "Esperando una sena...", (int(16 * s), y0 + int(32 * s)),
                    F, 0.7 * s, _COLOR_TEXTO_TENUE, max(1, int(1.6 * s)), cv2.LINE_AA)
    else:
        es_oov = ultimo.label == "<desconocida>"
        color_card = _COLOR_OOV if es_oov else _COLOR_EXITO
        texto_label = "Fuera de vocabulario" if es_oov else _es(ultimo.label)
        cv2.putText(frame, texto_label, (int(16 * s), y0 + int(32 * s)), F,
                    0.95 * s, color_card, max(2, int(2 * s)), cv2.LINE_AA)

        # Barra de confianza 0..1, con marca en el umbral de CONF_THRESHOLD.
        bx0, bx1 = int(16 * s), int(16 * s) + int(220 * s)
        by0, by1 = y0 + int(44 * s), y0 + int(58 * s)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (90, 90, 90), 1, cv2.LINE_AA)
        relleno = int((bx1 - bx0) * max(0.0, min(1.0, ultimo.conf)))
        if relleno > 0:
            cv2.rectangle(frame, (bx0, by0), (bx0 + relleno, by1), color_card, -1)
        x_umbral = bx0 + int((bx1 - bx0) * max(0.0, min(1.0, umbral_conf)))
        cv2.line(frame, (x_umbral, by0 - 3), (x_umbral, by1 + 3),
                (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{ultimo.conf * 100:.0f}%  (umbral {umbral_conf * 100:.0f}%)",
                    (bx1 + int(10 * s), by1), F, 0.42 * s, _COLOR_TEXTO_TENUE, 1,
                    cv2.LINE_AA)

        # Latencia end-to-end (Sección 11), discreta en la esquina de la tarjeta.
        color_lat = _COLOR_ALERTA if ultimo.excede_umbral else _COLOR_TEXTO_TENUE
        lat_txt = (f"e2e {ultimo.latencia_e2e_ms:.0f}ms (inf "
                  f"{ultimo.latencia_inferencia_ms:.0f} + seg "
                  f"{ultimo.retardo_segmentacion_ms:.0f})")
        (lat_w, _), _ = cv2.getTextSize(lat_txt, F, 0.42 * s, 1)
        cv2.putText(frame, lat_txt, (w - lat_w - int(14 * s), y0 + alto_tarjeta - int(10 * s)),
                    F, 0.42 * s, color_lat, 1, cv2.LINE_AA)

    # -- Barra inferior: subtítulo compuesto (MessageComposer) --------------- #
    subt_h = int(56 * s)
    y_sub0 = h - subt_h
    _panel(cv2, frame, 0, y_sub0, w, h, (0, 0, 0), alpha=0.7)
    texto = composer.texto()
    grosor_sub = max(1, int(2 * s))
    if texto:
        lineas = _envolver(cv2, texto, F, 0.8 * s, grosor_sub, w - int(24 * s))[-2:]
        paso = int(26 * s)
        y_base = h - int(14 * s) - paso * (len(lineas) - 1)
        for i, linea in enumerate(lineas):
            cv2.putText(frame, linea, (int(12 * s), y_base + i * paso), F,
                        0.8 * s, _COLOR_TEXTO, grosor_sub, cv2.LINE_AA)
    else:
        cv2.putText(frame, "...", (int(12 * s), h - int(16 * s)), F, 0.8 * s,
                    _COLOR_TEXTO_TENUE, grosor_sub, cv2.LINE_AA)

    ayuda = "[p] puntos    [r] reiniciar mensaje    [q] salir"
    (ayuda_w, _), _ = cv2.getTextSize(ayuda, F, 0.45 * s, 1)
    cv2.putText(frame, ayuda, (w - ayuda_w - int(12 * s), y_sub0 - int(10 * s)),
                F, 0.45 * s, _COLOR_TEXTO_TENUE, 1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
