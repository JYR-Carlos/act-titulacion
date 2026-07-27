"""
diagnostico_captura.py — ¿por qué la demo tarda en "tomar" las manos?

Mide la captura con EXACTAMENTE la misma configuración que `demo_vivo.py`
(`num_hands=2`, `running_mode="image"`, sin espejo) y separa las dos causas que
producen el mismo síntoma:

  (A) La escena / la cámara      -> poca luz, exposición larga, motion blur:
                                    MediaPipe no encuentra la palma.
  (B) El throughput del pipeline -> el bucle corre a pocos FPS. Como las
                                    constantes de segmentación están en FRAMES
                                    (REST_FRAMES_*), a menos FPS todo tarda más
                                    en milisegundos aunque la detección sea
                                    perfecta.

El discriminador es el desglose del tiempo por frame:

    t_lectura alto  -> la cámara es el cuello de botella (exposición/luz: (A))
    t_mediapipe alto -> la CPU es el cuello de botella ((B))

Además reporta la tasa de detección y la luminancia, que es la medición directa
de la hipótesis "es la luz".

Uso:
    python diagnostico_captura.py --etiqueta noche
    python diagnostico_captura.py --etiqueta con-lampara --segundos 30
    python diagnostico_captura.py --etiqueta noche-realce --realce

El tercer caso es el A/B del realce por software (`lsch_mr/realce_luz.py`):
mismo entorno, misma duración, y se comparan `tasa_con_mano` y
`racha_max_sin_mano` de los dos JSON. La luminancia se mide SIEMPRE sobre el
frame original, para que siga describiendo la escena y no el post-proceso.

Haz 3 o 4 señas frente a la cámara mientras mide (necesita frames con mano y
con movimiento, no solo la mano quieta). Guarda un JSON en
`outputs/reports/diagnostico_captura_<etiqueta>_<fecha>.json` para poder
comparar dos condiciones de luz después.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time

import numpy as np

from lsch_mr import config
from lsch_mr.consola import configurar_utf8
from lsch_mr.fuente_video import FuenteVideo, parse_fuente
from lsch_mr.hand_tracking_provider import HandTrackingProvider, seleccionar_frame
from lsch_mr.realce_luz import realzar_poca_luz

# Umbrales de lectura del diagnóstico. No son hiperparámetros del pipeline: son
# solo los cortes con los que este script redacta su veredicto.
LUMINANCIA_BAJA = 60.0      # media de gris 0..255 por debajo de esto = escena oscura
NITIDEZ_BAJA = 100.0        # varianza del Laplaciano: por debajo, imagen blanda/borrosa
TASA_DETECCION_OK = 0.70    # fracción de frames con mano que se considera sana
FPS_OK = 20.0               # por debajo, el retardo en frames se nota en ms


def _stats(valores: list[float]) -> dict:
    """Media/min/max/p95 de una serie, o dict vacío si no hay muestras."""
    if not valores:
        return {}
    a = np.asarray(valores, dtype=np.float64)
    return {
        "media": round(float(a.mean()), 2),
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
    }


def _propiedades_camara(cv2, cap) -> dict:
    """Lo que la cámara admite reportar. Muchos drivers devuelven 0 o -1: se
    incluye igual porque un valor de exposición ausente ya es información."""
    props = {
        "ancho": cv2.CAP_PROP_FRAME_WIDTH,
        "alto": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps_declarado": cv2.CAP_PROP_FPS,
        "exposicion": cv2.CAP_PROP_EXPOSURE,
        "ganancia": cv2.CAP_PROP_GAIN,
        "brillo": cv2.CAP_PROP_BRIGHTNESS,
        "auto_exposicion": cv2.CAP_PROP_AUTO_EXPOSURE,
    }
    out = {}
    for nombre, prop in props.items():
        try:
            out[nombre] = round(float(cap.get(prop)), 3)
        except Exception:
            out[nombre] = None
    try:
        out["backend"] = cap.getBackendName()
    except Exception:
        out["backend"] = None
    return out


def _nitidez(cv2, gris: np.ndarray) -> float:
    """Varianza del Laplaciano — proxy estándar de enfoque/motion blur."""
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def _roi_mano(landmarks: np.ndarray, w: int, h: int, margen: float = 0.15):
    """Caja del bbox de la mano en píxeles, con margen. Los landmarks vienen
    normalizados [0,1] y pueden salirse del cuadro: se recorta al frame."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    dx, dy = (xs.max() - xs.min()) * margen, (ys.max() - ys.min()) * margen
    x1 = int(max(0, (xs.min() - dx) * w))
    x2 = int(min(w, (xs.max() + dx) * w))
    y1 = int(max(0, (ys.min() - dy) * h))
    y2 = int(min(h, (ys.max() + dy) * h))
    return x1, y1, x2, y2


def _veredicto(res: dict) -> list[str]:
    """Traduce los números a las causas (A)/(B) del docstring."""
    lineas: list[str] = []
    lum = res["luminancia"].get("media", 0.0)
    tasa = res["deteccion"]["tasa_con_mano"]
    fps_mano = res["throughput"]["fps_efectivo_con_mano"]
    t_read = res["throughput"]["t_lectura_ms"].get("media", 0.0)
    t_mp = res["throughput"]["t_mediapipe_ms"].get("media", 0.0)
    nit_mov = res["nitidez_roi_mano"].get("media")

    if lum < LUMINANCIA_BAJA:
        lineas.append(f"(A) Escena OSCURA: luminancia media {lum:.0f}/255 "
                      f"(< {LUMINANCIA_BAJA:.0f}). Agregar luz frontal.")
    else:
        lineas.append(f"(A) Luminancia {lum:.0f}/255: suficiente. "
                      "La oscuridad no es el factor dominante.")

    if nit_mov is not None and nit_mov < NITIDEZ_BAJA:
        lineas.append(f"(A) Mano BORROSA (nitidez {nit_mov:.0f} < {NITIDEZ_BAJA:.0f}): "
                      "exposición larga por poca luz => motion blur al mover la mano. "
                      "Es el modo en que la luz rompe la detección.")

    if tasa < TASA_DETECCION_OK:
        # Arrancar una captura exige `frames_inicio` detecciones seguidas; con
        # una tasa p esa probabilidad cae como p^frames_inicio. Es la cuenta que
        # explica por qué una demo puede no segmentar NADA en minutos aunque la
        # mano se vea en la mitad de los frames.
        p_inicio = tasa ** config.REST_FRAMES_INICIO
        lineas.append(f"(A) Deteccion INESTABLE: mano en {tasa * 100:.0f}% de los "
                      f"frames (< {TASA_DETECCION_OK * 100:.0f}%). "
                      f"Rachas sin mano de hasta {res['deteccion']['racha_max_sin_mano']} "
                      f"frames; se tolera un parpadeo de "
                      f"{config.REST_FRAMES_PERDIDA_MAX} y a partir de ahi se "
                      f"pierde el candidato de inicio o se descarta la sena en "
                      f"curso. Encadenar los {config.REST_FRAMES_INICIO} frames "
                      f"que hacen falta para arrancar tiene prob. ~"
                      f"{p_inicio * 100:.0f}% por intento.")
        if not res["config_medida"].get("realce_poca_luz"):
            lineas.append("(A) Siguiente medicion: repetir con --realce (o con "
                          "luz frontal) y comparar tasa_con_mano contra este JSON.")
    else:
        lineas.append(f"(A) Deteccion sana: mano en {tasa * 100:.0f}% de los frames.")

    # Que la lectura domine solo es un problema si además el FPS es bajo: una
    # cámara de 30 FPS bloquea ~33 ms en cada `read()` y eso es lo correcto.
    fps_ok = fps_mano is not None and fps_mano >= FPS_OK
    if t_read > t_mp and not fps_ok:
        lineas.append(f"(A) La CAMARA es el cuello de botella: lectura "
                      f"{t_read:.0f} ms vs MediaPipe {t_mp:.0f} ms, y solo "
                      f"{fps_mano} FPS. Sintoma tipico de exposicion larga por "
                      "poca luz: la camara alarga el obturador y entrega menos "
                      "frames por segundo.")
    elif t_read > t_mp:
        lineas.append(f"(A) La lectura domina ({t_read:.0f} ms vs MediaPipe "
                      f"{t_mp:.0f} ms) pero el FPS es sano ({fps_mano}): es la "
                      "cadencia normal de la camara, no un problema.")
    else:
        lineas.append(f"(B) La CPU es el cuello de botella: MediaPipe {t_mp:.0f} ms "
                      f"vs lectura {t_read:.0f} ms. No lo arregla mas luz.")

    if fps_mano and fps_mano < FPS_OK:
        seg = res["consecuencia_en_segmentacion"]
        lineas.append(f"(B) Con mano visible el bucle cae a {fps_mano:.0f} FPS. "
                      f"Como REST_FRAMES_* esta en frames, eso son "
                      f"{seg['retardo_inicio_ms']:.0f} ms para arrancar la captura y "
                      f"{seg['retardo_fin_ms']:.0f} ms de retardo de segmentacion "
                      f"({seg['pct_del_presupuesto']:.0f}% del presupuesto de "
                      f"{config.LATENCIA_MAX_MS:.0f} ms).")
    return lineas


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnostica la captura: luz/camara vs throughput del pipeline")
    ap.add_argument("--fuente", default="0",
                    help="índice de webcam o URL del teléfono (igual que demo_vivo)")
    ap.add_argument("--segundos", type=float, default=20.0,
                    help="duración de la medición")
    ap.add_argument("--etiqueta", default="sin-etiqueta",
                    help="nombre de la condición medida, p. ej. 'noche' o "
                         "'con-lampara' — va en el nombre del JSON")
    ap.add_argument("--sin-ventana", action="store_true",
                    help="no mostrar la vista previa (mide sin el costo de dibujar)")
    ap.add_argument("--realce", action="store_true",
                    help="aplica el realce de poca luz antes de MediaPipe, "
                         "igual que `demo_vivo.py --realce`")
    args = ap.parse_args()

    configurar_utf8()
    import cv2

    print(f"[diag] midiendo {args.segundos:.0f} s con la config de demo_vivo "
          f"(num_hands=2, running_mode=image, "
          f"realce={'ON' if args.realce else 'OFF'})")
    print("[diag] haz 3 o 4 senas frente a la camara mientras mide...")

    luminancia: list[float] = []
    pct_oscuro: list[float] = []
    nitidez_frame: list[float] = []
    nitidez_roi: list[float] = []
    scores: list[float] = []
    t_lectura: list[float] = []
    t_mediapipe: list[float] = []
    t_loop: list[float] = []
    # Períodos separados según hubiera mano o no: es la comparación que muestra
    # por qué la demo se siente lenta justo cuando importa.
    periodos_con_mano: list[float] = []
    periodos_sin_mano: list[float] = []

    n_frames = 0
    n_con_mano = 0
    n_dos_manos = 0
    racha_sin_mano = 0
    racha_max_sin_mano = 0
    props: dict = {}
    t_frame_anterior = None

    with FuenteVideo(parse_fuente(args.fuente), espejo=False).abrir() as fv, \
            HandTrackingProvider(num_hands=2, running_mode="image") as provider:
        props = _propiedades_camara(cv2, fv._cap)
        t_fin = time.monotonic() + args.segundos
        try:
            while time.monotonic() < t_fin:
                t_loop0 = time.perf_counter()

                t0 = time.perf_counter()
                leido = fv.leer()
                t_lectura.append((time.perf_counter() - t0) * 1000.0)
                if leido is None:
                    print("[diag] la fuente se agoto antes de terminar")
                    break
                frame_bgr, ts_ms = leido
                n_frames += 1

                # Todas las métricas de imagen se miden sobre el frame ORIGINAL:
                # describen la escena y la cámara, que es lo que se compara
                # entre condiciones. El realce entra después, solo en la ruta
                # que alimenta a MediaPipe.
                gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                luminancia.append(float(gris.mean()))
                pct_oscuro.append(float((gris < 40).mean() * 100.0))
                nitidez_frame.append(_nitidez(cv2, gris))

                if args.realce:
                    frame_bgr = realzar_poca_luz(frame_bgr,
                                                 luminancia=luminancia[-1])
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                multi = provider.getFrame(rgb, ts_ms)
                t_mediapipe.append((time.perf_counter() - t0) * 1000.0)

                dom = seleccionar_frame(multi, modo="dominante")
                hay_mano = dom is not None
                if hay_mano:
                    n_con_mano += 1
                    racha_sin_mano = 0
                    if len(multi.hands) >= 2:
                        n_dos_manos += 1
                    mano = multi.dominante()
                    if mano is not None:
                        scores.append(float(mano.score))
                    h, w = gris.shape[:2]
                    x1, y1, x2, y2 = _roi_mano(dom, w, h)
                    if x2 > x1 and y2 > y1:
                        nitidez_roi.append(_nitidez(cv2, gris[y1:y2, x1:x2]))
                else:
                    racha_sin_mano += 1
                    racha_max_sin_mano = max(racha_max_sin_mano, racha_sin_mano)

                # Período real entre frames, separado por presencia de mano.
                ahora = time.perf_counter()
                if t_frame_anterior is not None:
                    dt = (ahora - t_frame_anterior) * 1000.0
                    (periodos_con_mano if hay_mano else periodos_sin_mano).append(dt)
                t_frame_anterior = ahora

                if not args.sin_ventana:
                    vista = cv2.flip(frame_bgr, 1)
                    color = (70, 190, 90) if hay_mano else (60, 60, 230)
                    cv2.putText(vista, f"{'MANO' if hay_mano else 'SIN MANO'}  "
                                f"lum={luminancia[-1]:.0f}  mp={t_mediapipe[-1]:.0f}ms",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                                cv2.LINE_AA)
                    cv2.imshow("diagnostico_captura (q = cortar)", vista)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break

                t_loop.append((time.perf_counter() - t_loop0) * 1000.0)
        except KeyboardInterrupt:
            print("\n[diag] interrumpido con Ctrl+C")
        finally:
            cv2.destroyAllWindows()

    if n_frames == 0:
        print("[diag] no se leyo ningun frame; revisa la fuente")
        return 1

    def _fps(periodos: list[float]) -> float | None:
        if not periodos:
            return None
        media = float(np.mean(periodos))
        return round(1000.0 / media, 1) if media > 0 else None

    fps_con_mano = _fps(periodos_con_mano)
    # El retardo de segmentación se paga en el estado "capturando", o sea con la
    # mano visible: es ese FPS el que hay que usar, no el promedio de la sesión.
    periodo_ref = (float(np.mean(periodos_con_mano)) if periodos_con_mano
                   else float(np.mean(periodos_sin_mano or [0.0])))
    retardo_fin = config.REST_FRAMES_FIN * periodo_ref
    resultado = {
        "etiqueta": args.etiqueta,
        "inicio": datetime.datetime.now().isoformat(timespec="seconds"),
        "fuente": args.fuente,
        "config_medida": {
            "num_hands": 2,
            "running_mode": "image",
            "min_hand_detection_confidence": 0.5,
            "realce_poca_luz": args.realce,
            "rest_frames_inicio": config.REST_FRAMES_INICIO,
            "rest_frames_fin": config.REST_FRAMES_FIN,
            "rest_frames_perdida_max": config.REST_FRAMES_PERDIDA_MAX,
        },
        "camara": props,
        "n_frames": n_frames,
        "luminancia": _stats(luminancia),
        "pct_pixeles_oscuros": _stats(pct_oscuro),
        "nitidez_frame": _stats(nitidez_frame),
        "nitidez_roi_mano": _stats(nitidez_roi),
        "score_handedness": _stats(scores),
        "deteccion": {
            "n_con_mano": n_con_mano,
            "tasa_con_mano": round(n_con_mano / n_frames, 3),
            "n_dos_manos": n_dos_manos,
            "racha_max_sin_mano": racha_max_sin_mano,
        },
        "throughput": {
            "t_lectura_ms": _stats(t_lectura),
            "t_mediapipe_ms": _stats(t_mediapipe),
            "t_loop_ms": _stats(t_loop),
            "fps_efectivo_con_mano": fps_con_mano,
            "fps_efectivo_sin_mano": _fps(periodos_sin_mano),
        },
        "consecuencia_en_segmentacion": {
            "periodo_frame_ms": round(periodo_ref, 1),
            "retardo_inicio_ms": round(config.REST_FRAMES_INICIO * periodo_ref, 1),
            "retardo_fin_ms": round(retardo_fin, 1),
            "tolerancia_perdida_ms": round(
                config.REST_FRAMES_PERDIDA_MAX * periodo_ref, 1),
            "presupuesto_ms": config.LATENCIA_MAX_MS,
            "pct_del_presupuesto": round(100.0 * retardo_fin / config.LATENCIA_MAX_MS, 1),
        },
    }
    resultado["veredicto"] = _veredicto(resultado)

    print(f"\n=== Diagnostico de captura — '{args.etiqueta}' "
          f"({n_frames} frames) ===")
    print(f"  luminancia media      {resultado['luminancia']['media']:>7.1f} /255")
    print(f"  nitidez ROI mano      "
          f"{resultado['nitidez_roi_mano'].get('media', float('nan')):>7.1f}")
    print(f"  mano detectada        "
          f"{resultado['deteccion']['tasa_con_mano'] * 100:>7.1f} %"
          f"   (racha max sin mano: {racha_max_sin_mano} frames)")
    print(f"  score handedness      "
          f"{resultado['score_handedness'].get('media', float('nan')):>7.2f}")
    print(f"  t_lectura camara      {resultado['throughput']['t_lectura_ms']['media']:>7.1f} ms")
    print(f"  t_mediapipe           {resultado['throughput']['t_mediapipe_ms']['media']:>7.1f} ms")
    print(f"  FPS con mano          {str(fps_con_mano):>7} "
          f"  (sin mano: {resultado['throughput']['fps_efectivo_sin_mano']})")
    seg = resultado["consecuencia_en_segmentacion"]
    print(f"  -> retardo segmentacion {seg['retardo_fin_ms']:.0f} ms "
          f"= {seg['pct_del_presupuesto']:.0f}% del presupuesto de "
          f"{config.LATENCIA_MAX_MS:.0f} ms")
    print("\n  Veredicto:")
    for linea in resultado["veredicto"]:
        print(f"    - {linea}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = (config.OUTPUTS_REPORTS_DIR /
            f"diagnostico_captura_{args.etiqueta}_{ts}.json")
    try:
        ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\n[diag] guardado: {ruta}")
    except OSError as exc:
        print(f"[diag] no se pudo guardar el diagnostico: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
