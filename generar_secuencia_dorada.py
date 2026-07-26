"""
generar_secuencia_dorada.py — vectores dorados de una SECUENCIA real en movimiento.

Complementa a `generar_vectores_dorados.py`. Aquel congela la Capa 2
(preprocesamiento) sobre una mano SINTÉTICA; este congela la **Capa 1**
(HandLandmarker) sobre una seña REAL del corpus, frame a frame.

Riesgo que mitiga: la seña es un gesto dinámico (1..60 frames). Si Unity corre el
HandLandmarker con un `running_mode` distinto al que produjo el corpus, la
diferencia no es un error puntual de un frame: es un sesgo sistemático presente
en TODA seña dinámica, y se manifiesta como "el modelo acierta poco" sin ninguna
excepción ni error visible.

Qué guarda, por cada frame de la ventana elegida:
  * los 21x3 keypoints crudos de cada mano detectada (tal como salen de
    MediaPipe, ANTES de normalizar),
  * el `handedness` y su score,
  * el `timestamp_ms` exacto que se le pasó al HandLandmarker,
  * el NormVector (63) esperado de la Capa 2 para ese frame.

Y al final la entrada completa del modelo (seq_len x n_features), que cierra la
cadena Capa 1 -> Capa 2 sobre datos reales.

Salidas (las dos con el mismo contenido; el JSON es el autoritativo):
    integracion/secuencia_dorada.json   -> para el test de Unity
    integracion/secuencia_dorada.csv    -> plano, para inspeccionar a ojo

Uso:
    python generar_secuencia_dorada.py                  # genera JSON + CSV
    python generar_secuencia_dorada.py --verificar      # comprueba que sigue vigente
    python generar_secuencia_dorada.py --video data/external/lsa64/videos/039_010_002.mp4

Ver INTEGRACION_UNITY.md, sección 7 ("running_mode oficial").
"""
from __future__ import annotations

import argparse
import csv as _csv
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from lsch_mr import config
from lsch_mr.caracteristicas import preparar_entrada, secuencia_a_features
from lsch_mr.consola import configurar_utf8
from lsch_mr.keypoint_normalizer import KeypointNormalizer

configurar_utf8()

SALIDA_JSON = config.RAIZ / "integracion" / "secuencia_dorada.json"
SALIDA_CSV = config.RAIZ / "integracion" / "secuencia_dorada.csv"

# --------------------------------------------------------------------------- #
# running_mode OFICIAL del proyecto — ver INTEGRACION_UNITY.md sección 7.
#
# IMAGE, porque es el modo con el que `extraer_lote.py` extrajo el corpus LSA64
# del que salió `modelo.onnx`. El modo de servicio DEBE ser el modo de
# extracción: el modelo solo conoce la distribución de keypoints que vio al
# entrenar. Cambiar esto obliga a re-extraer el corpus y reentrenar.
# --------------------------------------------------------------------------- #
RUNNING_MODE_OFICIAL = "image"

# LSA64 051_001_001 = seña 'Thanks', señante 001, repetición 001. Se eligió
# porque MediaPipe detecta mano en sus 122 frames sin un solo hueco, así que
# cualquier ventana consecutiva es utilizable y la comparación entre modos queda
# alineada frame a frame.
VIDEO_POR_DEFECTO = config.RAIZ / "data" / "external" / "lsa64" / "videos" / "051_001_001.mp4"
ANOTACIONES = config.RAIZ / "data" / "external" / "lsa64" / "lsa64_10_annotations.csv"

N_FRAMES = 18          # ventana pedida: 15-20 frames consecutivos en movimiento
TOLERANCIA = 1e-5      # la misma que vectores_dorados.json

# Misma lección que en generar_vectores_dorados.py: normalize divide por
# ||L9-L0||, que aquí vale ~0.06, así que amplifica el redondeo de la ENTRADA
# unas 17x. Las entradas van casi exactas; las salidas, por encima de la
# tolerancia.
_DEC = 6
_DEC_ENTRADA = 12


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def timestamp_ms(frame_idx: int, fps: float) -> int:
    """Tiempo de medios del frame: `round(frame_idx * 1000 / fps)`.

    NO se usa `FuenteVideo`, que deriva el timestamp de `time.monotonic()`
    (reloj de pared). Sobre un archivo eso da un valor distinto en cada corrida
    —depende de lo que tarde el proceso— y un vector dorado tiene que ser
    reproducible. Unity debe usar esta misma fórmula sobre el índice de frame,
    no el tiempo real transcurrido.
    """
    return int(round(frame_idx * 1000.0 / fps))


# --------------------------------------------------------------------------- #
# Extracción
# --------------------------------------------------------------------------- #
def _leer_video(video: Path) -> tuple[list[np.ndarray], dict]:
    """Devuelve (frames RGB, metadatos del vídeo)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video}")

    meta = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        # Sin espejo: `extraer_lote.py` no voltea el frame al extraer el corpus.
        # Voltearlo aquí invertiría el eje x y la handedness (ver trampa 1 de
        # INTEGRACION_UNITY.md).
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"El vídeo no tiene frames legibles: {video}")
    meta["n_frames"] = len(frames)
    return frames, meta


def _detectar(frames: list[np.ndarray], fps: float, modo: str) -> list[dict]:
    """Corre el HandTrackingProvider sobre todos los frames en `modo`.

    Devuelve una lista paralela a `frames`; cada elemento trae las manos
    detectadas y el timestamp que se le pasó al landmarker.
    """
    from lsch_mr.hand_tracking_provider import HandTrackingProvider

    salida = []
    with HandTrackingProvider(num_hands=2, running_mode=modo) as provider:
        for i, rgb in enumerate(frames):
            ts = timestamp_ms(i, fps)
            multi = provider.getFrame(rgb, ts)
            salida.append({
                "frame_idx": i,
                "timestamp_ms": ts,
                "manos": [
                    {"hand": h.hand,
                     "handedness_score": float(h.score),
                     "keypoints": np.asarray(h.landmarks, dtype=np.float32)}
                    for h in multi.hands
                ],
            })
    return salida


def _dominante(det: dict) -> dict | None:
    """Mano de mayor handedness_score, que es la que usa `modo_manos=dominante`."""
    if not det["manos"]:
        return None
    return max(det["manos"], key=lambda m: m["handedness_score"])


# --------------------------------------------------------------------------- #
# Elección de la ventana
# --------------------------------------------------------------------------- #
def elegir_ventana(det_a: list[dict], det_b: list[dict], n: int) -> int:
    """Índice inicial de la ventana de `n` frames consecutivos con MÁS movimiento.

    Se exige mano detectada en los dos modos en todos los frames de la ventana:
    si un modo pierde el tracking donde el otro no, la comparación deja de estar
    alineada y la divergencia medida sería un artefacto, no una diferencia real.

    "Con movimiento" no es un adorno: una ventana estática no distingue un port
    correcto de uno que congela el primer frame.
    """
    norm = KeypointNormalizer()
    T = len(det_a)
    if T < n:
        raise RuntimeError(f"El vídeo solo tiene {T} frames; se piden {n}.")

    valido = np.zeros(T, dtype=bool)
    mov = np.zeros(T, dtype=np.float64)
    prev = None
    for i in range(T):
        da, db = _dominante(det_a[i]), _dominante(det_b[i])
        valido[i] = da is not None and db is not None
        if da is None:
            prev = None
            continue
        # El movimiento se mide sobre el NormVector, no sobre los keypoints
        # crudos: así no cuenta como "movimiento" el simple acercarse a la
        # cámara, que la normalización elimina y el modelo nunca ve.
        actual = norm.normalize(da["keypoints"])
        if prev is not None:
            mov[i] = float(np.mean(np.abs(actual - prev)))
        prev = actual

    mejor_ini, mejor_score = -1, -1.0
    for ini in range(T - n + 1):
        if not valido[ini:ini + n].all():
            continue
        score = float(mov[ini + 1:ini + n].sum())
        if score > mejor_score:
            mejor_ini, mejor_score = ini, score

    if mejor_ini < 0:
        raise RuntimeError(
            f"No hay {n} frames consecutivos con mano detectada en los dos "
            "modos. Prueba con otro vídeo (--video).")
    return mejor_ini


# --------------------------------------------------------------------------- #
# Divergencia entre modos — la evidencia que sostiene la decisión de running_mode
# --------------------------------------------------------------------------- #
def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size),
            "dif_abs_max": round(float(a.max()), 8),
            "dif_abs_media": round(float(a.mean()), 8),
            "dif_abs_mediana": round(float(np.median(a)), 8)}


def medir_divergencia(det_a: list[dict], det_b: list[dict],
                      ini: int, n: int) -> dict:
    """Cuánto se separan los dos running_mode sobre los MISMOS frames.

    Se mide por separado, porque son dos fallos distintos y se confunden si se
    promedian juntos:

    * `emparejado_por_lado` — misma mano física (mismo hand label) en los dos
      modos. Aísla lo que cambia en los landmarks por reusar el ROI del frame
      anterior (VIDEO) en vez de re-detectar la palma (IMAGE).
    * `mano_dominante` — lo que realmente consume el modelo con
      `modo_manos="dominante"`: la mano de mayor handedness_score. Aquí se suma
      el caso peor, que es que cada modo elija una mano FÍSICA DISTINTA; entonces
      no se comparan dos versiones de la misma mano sino dos manos diferentes.
    * `deteccion` — censo sobre todo el vídeo. VIDEO arrastra el tracking, así
      que sostiene manos en frames donde IMAGE, que redetecta desde cero, no ve
      nada. Es la diferencia que más pesa en una seña bimanual.

    Si todo esto diera ~0, el running_mode sería indiferente y no haría falta
    fijarlo en ningún documento.
    """
    norm = KeypointNormalizer()
    por_lado_crudo, por_lado_norm = [], []
    dom_crudo, dom_norm = [], []
    handedness_distinta = 0
    n_manos_distinto = 0

    for i in range(ini, ini + n):
        ma = {m["hand"]: m for m in det_a[i]["manos"]}
        mb = {m["hand"]: m for m in det_b[i]["manos"]}
        for lado in sorted(set(ma) & set(mb)):
            por_lado_crudo.append(np.abs(ma[lado]["keypoints"]
                                         - mb[lado]["keypoints"]).max())
            por_lado_norm.append(np.abs(norm.normalize(ma[lado]["keypoints"])
                                        - norm.normalize(mb[lado]["keypoints"])).max())

        if len(det_a[i]["manos"]) != len(det_b[i]["manos"]):
            n_manos_distinto += 1
        da, db = _dominante(det_a[i]), _dominante(det_b[i])
        if da is None or db is None:
            continue
        if da["hand"] != db["hand"]:
            handedness_distinta += 1
        dom_crudo.append(np.abs(da["keypoints"] - db["keypoints"]).max())
        dom_norm.append(np.abs(norm.normalize(da["keypoints"])
                               - norm.normalize(db["keypoints"])).max())

    def censo(det):
        c = [len(d["manos"]) for d in det]
        return {"frames": len(c), "con_0_manos": c.count(0),
                "con_1_mano": c.count(1), "con_2_manos": c.count(2)}

    return {
        "ventana": {"frame_inicial": ini, "n_frames": n},
        "emparejado_por_lado": {
            "keypoints_crudos": _stats(por_lado_crudo),
            "normvector_63": _stats(por_lado_norm),
        },
        "mano_dominante": {
            "keypoints_crudos": _stats(dom_crudo),
            "normvector_63": _stats(dom_norm),
            "frames_con_mano_dominante_distinta": handedness_distinta,
            "frames_con_distinto_numero_de_manos": n_manos_distinto,
        },
        "deteccion_video_completo": {"modo_a": censo(det_a), "modo_b": censo(det_b)},
        "tolerancia_del_port": TOLERANCIA,
        "lectura": ("Si `normvector_63.dif_abs_max` supera la tolerancia del "
                    "port (1e-5), los dos modos NO son intercambiables: el modo "
                    "de servicio tiene que ser el mismo con el que se extrajo "
                    "el corpus. `frames_con_mano_dominante_distinta` > 0 es más "
                    "grave todavía: el modelo recibiría la otra mano."),
    }


# --------------------------------------------------------------------------- #
# Documento
# --------------------------------------------------------------------------- #
def _r(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), _DEC).tolist()


def _re(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), _DEC_ENTRADA).tolist()


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _etiqueta(video: Path) -> str:
    if not ANOTACIONES.exists():
        return video.stem
    with ANOTACIONES.open(encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            cols = {k.strip().upper(): v for k, v in row.items()}
            if Path(cols.get("FILENAME", "")).stem == video.stem:
                return (cols.get("LABEL") or video.stem).strip()
    return video.stem


def construir_documento(video: Path, n: int, modo: str,
                        comparar_modos: bool = True) -> dict:
    norm = KeypointNormalizer()
    frames, meta = _leer_video(video)
    fps = meta["fps"]

    det = _detectar(frames, fps, modo)
    otro = "video" if modo == "image" else "image"
    det_otro = _detectar(frames, fps, otro) if comparar_modos else det

    ini = elegir_ventana(det, det_otro, n)
    ventana = det[ini:ini + n]

    # Secuencia (T, 21, 3) de la mano dominante: es lo que consume la Capa 2 con
    # modo_manos="dominante", que es el modo del modelo exportado.
    seq_dom = np.stack([_dominante(d)["keypoints"] for d in ventana], axis=0)

    frames_json = []
    for d in ventana:
        dom = _dominante(d)
        frames_json.append({
            "frame_idx": d["frame_idx"],
            "timestamp_ms": d["timestamp_ms"],
            "manos": [
                {"hand": m["hand"],
                 "handedness_score": round(m["handedness_score"], 6),
                 "keypoints_21x3": _re(m["keypoints"])}
                for m in d["manos"]
            ],
            "dominante": {
                "hand": dom["hand"],
                "normvector_63": _r(norm.normalize(dom["keypoints"])),
            },
        })

    doc = {
        "generado": date.today().isoformat(),
        "proposito": ("Vectores dorados de una SECUENCIA real en movimiento, "
                      "para validar la Capa 1 (HandLandmarker) del port a "
                      "Unity frame a frame. Ver INTEGRACION_UNITY.md sec. 7."),
        "tolerancia_absoluta": TOLERANCIA,
        "running_mode": modo,
        "running_mode_oficial": RUNNING_MODE_OFICIAL,
        "advertencia_running_mode": (
            f"Estos vectores se produjeron con running_mode='{modo}'. Unity DEBE "
            "usar el mismo modo o los valores no coinciden: no es un desajuste "
            "de un frame suelto, es un sesgo presente en toda seña dinámica."),
        "timestamps": {
            "formula": "round(frame_idx * 1000 / fps)",
            "fps": round(fps, 6),
            "nota": ("Tiempo de MEDIOS derivado del índice de frame, no reloj de "
                     "pared. En running_mode='image' MediaPipe ignora el "
                     "timestamp por completo (se guarda solo para trazabilidad); "
                     "en 'video' lo exige estrictamente creciente."),
        },
        "video": {
            "archivo": video.name,
            "ruta_relativa": str(video.relative_to(config.RAIZ)).replace("\\", "/"),
            "sha1": _sha1(video),
            "label": _etiqueta(video),
            "fps": round(fps, 6),
            "resolucion": [meta["ancho"], meta["alto"]],
            "n_frames_video": meta["n_frames"],
            "espejo": False,
            "nota_espejo": ("El corpus se extrajo SIN voltear el frame. Unity no "
                            "debe pasar la imagen en espejo al HandLandmarker: "
                            "invertiría el eje x y la handedness."),
        },
        "ventana": {
            "frame_inicial": ini,
            "n_frames": n,
            "criterio": ("los n frames consecutivos con más movimiento "
                         "acumulado del NormVector y mano detectada en todos"),
            "manos_por_frame": [len(d["manos"]) for d in ventana],
            "mano_dominante_por_frame": [_dominante(d)["hand"] for d in ventana],
            "cambia_la_mano_dominante": len(
                {_dominante(d)["hand"] for d in ventana}) > 1,
            "nota_cambio_de_dominante": (
                "En modo 'dominante' la mano se elige por MAYOR handedness_score "
                "en CADA frame por separado, sin memoria. Si aparece la segunda "
                "mano con más score, la secuencia pasa a describir la otra mano "
                "física a mitad de seña. No es un bug del generador: es lo que "
                "hizo build_dataset.py al construir el corpus, así que el modelo "
                "se entrenó con esos saltos y el port en C# tiene que "
                "reproducirlos. No 'arreglar' con memoria del lado de Unity."),
        },
        "config": {
            "SEQ_LEN": config.SEQ_LEN,
            "NUM_LANDMARKS": config.NUM_LANDMARKS,
            "NUM_EJES": config.NUM_EJES,
            "NORMVECTOR_DIM": config.NORMVECTOR_DIM,
            "MODO_MANOS": config.MODO_MANOS,
            "num_hands": 2,
            "min_hand_detection_confidence": 0.5,
            "min_hand_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "modelo_landmarker": config.HAND_LANDMARKER_TASK.name,
            "sha1_landmarker": _sha1(config.HAND_LANDMARKER_TASK),
        },
        "frames": frames_json,
        "esperado": {
            f"features_{n}x{config.NORMVECTOR_DIM}":
                _r(secuencia_a_features(seq_dom, modo="dominante")),
            f"entrada_modelo_{config.SEQ_LEN}x{config.NORMVECTOR_DIM}":
                _r(preparar_entrada(seq_dom, modo="dominante",
                                    seq_len=config.SEQ_LEN)),
        },
    }

    if comparar_modos:
        doc["diagnostico_running_mode"] = {
            "modo_a": modo,
            "modo_b": otro,
            **medir_divergencia(det, det_otro, ini, n),
        }
    return doc


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
_XCOLS = [f"x{i}" for i in range(config.NUM_LANDMARKS)]
_YCOLS = [f"y{i}" for i in range(config.NUM_LANDMARKS)]
_ZCOLS = [f"z{i}" for i in range(config.NUM_LANDMARKS)]
COLUMNAS_CSV = (["frame_idx", "timestamp_ms", "running_mode", "hand",
                 "handedness_score", "es_dominante"] + _XCOLS + _YCOLS + _ZCOLS)


def escribir_csv(doc: dict, destino: Path) -> None:
    """Misma información que el JSON, en el orden de columnas del corpus.

    Nueve decimales (no seis, como el CSV del corpus) porque estos valores son
    la ENTRADA de un test: redondearlos a seis los amplifica ~17x al normalizar
    y el port en C# no podría reproducir el `esperado` aunque fuese correcto.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        w.writeheader()
        for fr in doc["frames"]:
            dom = _dominante(fr)
            for m in fr["manos"]:
                pts = np.asarray(m["keypoints_21x3"], dtype=np.float64)
                fila = {
                    "frame_idx": fr["frame_idx"],
                    "timestamp_ms": fr["timestamp_ms"],
                    "running_mode": doc["running_mode"],
                    "hand": m["hand"],
                    "handedness_score": m["handedness_score"],
                    "es_dominante": int(m is dom),
                }
                for i in range(config.NUM_LANDMARKS):
                    fila[f"x{i}"] = f"{pts[i, 0]:.9f}"
                    fila[f"y{i}"] = f"{pts[i, 1]:.9f}"
                    fila[f"z{i}"] = f"{pts[i, 2]:.9f}"
                w.writerow(fila)


# --------------------------------------------------------------------------- #
# Verificación
# --------------------------------------------------------------------------- #
def verificar(doc: dict) -> int:
    """Re-extrae el mismo vídeo y comprueba que da lo mismo que el JSON.

    Detecta dos cosas distintas: que MediaPipe dejó de ser determinista (cambio
    de versión, de backend o del .task) y que la Capa 2 cambió por debajo.
    """
    video = config.RAIZ / doc["video"]["ruta_relativa"]
    if not video.exists():
        print(f"  [!] no está el vídeo {video}: no se puede verificar")
        return 1

    sha = _sha1(video)
    if sha != doc["video"]["sha1"]:
        print(f"  [!] el vídeo cambió (sha1 {sha} != {doc['video']['sha1']})")
        return 1

    sha_task = _sha1(config.HAND_LANDMARKER_TASK)
    if sha_task != doc["config"].get("sha1_landmarker"):
        print(f"  [!] hand_landmarker.task cambió (sha1 {sha_task}): los "
              "keypoints crudos ya no tienen por qué coincidir")
        return 1

    frames, meta = _leer_video(video)
    det = _detectar(frames, meta["fps"], doc["running_mode"])
    ini = doc["ventana"]["frame_inicial"]
    n = doc["ventana"]["n_frames"]

    fallos = 0
    norm = KeypointNormalizer()
    peor_crudo = peor_norm = 0.0
    for guardado, actual in zip(doc["frames"], det[ini:ini + n]):
        if guardado["frame_idx"] != actual["frame_idx"]:
            print(f"  [!] frame_idx desalineado: {guardado['frame_idx']} vs "
                  f"{actual['frame_idx']}")
            fallos += 1
            continue
        if guardado["timestamp_ms"] != actual["timestamp_ms"]:
            print(f"  [!] frame {actual['frame_idx']}: timestamp "
                  f"{guardado['timestamp_ms']} != {actual['timestamp_ms']}")
            fallos += 1
        if len(guardado["manos"]) != len(actual["manos"]):
            print(f"  [!] frame {actual['frame_idx']}: "
                  f"{len(guardado['manos'])} manos guardadas vs "
                  f"{len(actual['manos'])} ahora")
            fallos += 1
            continue
        for mg, ma in zip(guardado["manos"], actual["manos"]):
            if mg["hand"] != ma["hand"]:
                print(f"  [!] frame {actual['frame_idx']}: handedness "
                      f"{mg['hand']} != {ma['hand']}")
                fallos += 1
            peor_crudo = max(peor_crudo, float(np.abs(
                np.asarray(mg["keypoints_21x3"]) - ma["keypoints"]).max()))
        dom_g = np.asarray(guardado["dominante"]["normvector_63"])
        dom_a = norm.normalize(_dominante(actual)["keypoints"])
        peor_norm = max(peor_norm, float(np.abs(dom_g - dom_a).max()))

    print(f"  keypoints crudos: dif max {peor_crudo:.2e}")
    print(f"  normvector_63:    dif max {peor_norm:.2e} "
          f"(tolerancia {TOLERANCIA:g})")
    if peor_norm > TOLERANCIA:
        print("  [!] la extracción ya no reproduce el JSON guardado")
        fallos += 1

    # La Capa 2 sobre la secuencia guardada: es lo que hace el port en C#.
    seq = np.stack([np.asarray(_dominante(f)["keypoints_21x3"], dtype=np.float32)
                    for f in doc["frames"]], axis=0)
    clave = f"entrada_modelo_{config.SEQ_LEN}x{config.NORMVECTOR_DIM}"
    esperado = np.asarray(doc["esperado"][clave], dtype=np.float64)
    obtenido = preparar_entrada(seq, modo="dominante", seq_len=config.SEQ_LEN)
    dif = float(np.abs(esperado - obtenido).max())
    print(f"  {clave}: dif max {dif:.2e} (desde la entrada guardada)")
    if dif > TOLERANCIA:
        print("  [!] el JSON no es reproducible desde su propia entrada")
        fallos += 1
    return fallos


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Vectores dorados de una secuencia real en movimiento")
    ap.add_argument("--video", default=str(VIDEO_POR_DEFECTO))
    ap.add_argument("--frames", type=int, default=N_FRAMES,
                    help=f"frames consecutivos de la ventana (default {N_FRAMES})")
    ap.add_argument("--modo", choices=["image", "video"],
                    default=RUNNING_MODE_OFICIAL,
                    help=f"running_mode; el oficial es '{RUNNING_MODE_OFICIAL}'")
    ap.add_argument("--salida", default=str(SALIDA_JSON))
    ap.add_argument("--sin-comparar-modos", action="store_true",
                    help="no medir la divergencia IMAGE vs VIDEO (más rápido)")
    ap.add_argument("--verificar", action="store_true",
                    help="no regenera: re-extrae y comprueba que el JSON sigue "
                         "coincidiendo")
    args = ap.parse_args()

    destino = Path(args.salida)

    if args.verificar:
        if not destino.exists():
            print(f"No existe {destino}. Genéralo primero.")
            return 1
        doc = json.loads(destino.read_text(encoding="utf-8"))
        print(f"[secuencia] verificando {destino} "
              f"(running_mode='{doc['running_mode']}') ...")
        fallos = verificar(doc)
        if fallos:
            print(f"\n[secuencia] {fallos} discrepancia(s). Si el cambio es "
                  "intencional, regenera y avisa al equipo de Unity: su test "
                  "de Capa 1 queda invalidado.")
            return 1
        print("\n[secuencia] todo coincide.")
        return 0

    if args.modo != RUNNING_MODE_OFICIAL:
        print(f"[secuencia] AVISO: generando con running_mode='{args.modo}', "
              f"que NO es el oficial ('{RUNNING_MODE_OFICIAL}'). Estos vectores "
              "no sirven para validar el runtime de Unity.")

    video = Path(args.video)
    if not video.exists():
        print(f"No existe el vídeo {video}.")
        return 1

    print(f"[secuencia] {video.name} | running_mode='{args.modo}' | "
          f"{args.frames} frames")
    doc = construir_documento(video, args.frames, args.modo,
                              comparar_modos=not args.sin_comparar_modos)

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    csv_destino = destino.with_suffix(".csv")
    escribir_csv(doc, csv_destino)

    v = doc["ventana"]
    print(f"[secuencia] ventana: frames {v['frame_inicial']}.."
          f"{v['frame_inicial'] + v['n_frames'] - 1} "
          f"(label '{doc['video']['label']}', fps {doc['video']['fps']:.2f})")
    print(f"[secuencia] manos por frame: {v['manos_por_frame']}")
    if v["cambia_la_mano_dominante"]:
        print(f"[secuencia] la mano dominante CAMBIA dentro de la ventana: "
              f"{v['mano_dominante_por_frame']}")
        print("            es fiel al corpus (se elige por score en cada frame, "
              "sin memoria); el port en C# debe reproducirlo.")
    print(f"[secuencia] -> {destino}")
    print(f"[secuencia] -> {csv_destino}")

    diag = doc.get("diagnostico_running_mode")
    if diag:
        print(f"\n[secuencia] divergencia '{diag['modo_a']}' vs '{diag['modo_b']}' "
              f"sobre los mismos frames de la ventana:")
        for etiqueta, bloque in (("misma mano (por lado)", diag["emparejado_por_lado"]),
                                 ("mano dominante", diag["mano_dominante"])):
            kc, nv = bloque["keypoints_crudos"], bloque["normvector_63"]
            print(f"    {etiqueta:22} crudo max {kc['dif_abs_max']:.2e} "
                  f"med {kc['dif_abs_mediana']:.2e} | "
                  f"norm max {nv['dif_abs_max']:.2e} med {nv['dif_abs_mediana']:.2e}")
        md = diag["mano_dominante"]
        print(f"    mano dominante DISTINTA en {md['frames_con_mano_dominante_distinta']}"
              f"/{doc['ventana']['n_frames']} frames")
        print(f"    nº de manos distinto en {md['frames_con_distinto_numero_de_manos']} frames")
        for nombre, c in (("modo_a", diag["deteccion_video_completo"]["modo_a"]),
                          ("modo_b", diag["deteccion_video_completo"]["modo_b"])):
            m = diag[nombre]
            print(f"    deteccion '{m}': {c['frames']} frames -> "
                  f"0 manos:{c['con_0_manos']}  1:{c['con_1_mano']}  2:{c['con_2_manos']}")
        peor = max(diag["emparejado_por_lado"]["normvector_63"]["dif_abs_max"],
                   md["normvector_63"]["dif_abs_max"])
        if peor > TOLERANCIA:
            print(f"    -> {peor:.2e} supera la tolerancia del port ({TOLERANCIA:g}): "
                  "los modos NO son intercambiables.")
        else:
            print(f"    -> por debajo de la tolerancia del port ({TOLERANCIA:g}).")

    print(f"\nUnity debe reproducir estos frames con running_mode="
          f"'{doc['running_mode']}'. Ver INTEGRACION_UNITY.md sección 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
