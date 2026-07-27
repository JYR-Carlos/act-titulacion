"""
evaluar_segmentacion.py — prueba unitaria previa del RestStateDetector.

Compara los límites de seña (inicio / fin) que el detector encuentra
automáticamente contra un **ground truth etiquetado a mano**, y reporta
**precisión y recall** de la detección de límites.

Trabaja sobre el CSV crudo de keypoints (`data/raw/*.csv`), no sobre vídeo: es
determinista, corre en segundos y no necesita cámara ni MediaPipe. El detector
recibe exactamente lo que recibiría en vivo, incluidos los frames **sin mano
detectada** (que en el CSV simplemente no existen y aquí se reponen como
`None`) — así se ejercita de verdad la tolerancia a parpadeos
`REST_FRAMES_PERDIDA_MAX`, que es donde vive el 90% de los fallos reales.

--------------------------------------------------------------------------
  FLUJO DE TRABAJO EN DOS PASOS
--------------------------------------------------------------------------
1) Generar la plantilla de etiquetado y rellenarla a mano viendo los vídeos:

       python scripts/evaluar_segmentacion.py --plantilla etiquetado/limites_gt.csv
       python scripts/evaluar_segmentacion.py --plantilla etiquetado/limites_gt.csv --muestras 40

   Se abre en Excel/LibreOffice y se completan `frame_inicio_gt` y
   `frame_fin_gt` con el primer y el último frame en que la mano se está
   moviendo para hacer la seña. Las filas que se dejen vacías se ignoran, así
   que se puede etiquetar un subconjunto (40-60 muestras ya dan una cifra
   estable) en vez de las 483.

2) Evaluar contra lo etiquetado:

       python scripts/evaluar_segmentacion.py --gt etiquetado/limites_gt.csv

--------------------------------------------------------------------------
  DE DÓNDE SALEN LOS LÍMITES "DETECTADOS"
--------------------------------------------------------------------------
El detector no anuncia los límites en el frame en que ocurren — los confirma
más tarde, por diseño. Para comparar con un humano hay que deshacer ese retardo,
y este script lo hace así:

  * **Inicio**: el evento START se emite cuando se acumulan `REST_FRAMES_INICIO`
    frames de movimiento, pero el primero de ellos ya entró al buffer. El inicio
    reportado es ese primer frame acumulado (se detecta observando cuándo
    `n_frames_buffer` pasa de 0 a 1), no el frame del evento.
  * **Fin**: el evento END se emite tras `REST_FRAMES_FIN` frames de reposo
    sostenido, y el propio detector recorta esos frames de la secuencia. El fin
    reportado es `frame_del_evento - REST_FRAMES_FIN`, que es el último frame
    útil de la seña.

Sin esta corrección, el detector parecería tener un sesgo constante de varios
frames que en realidad es latencia de confirmación, no error de segmentación.
Ese retardo SÍ se mide, pero como latencia (métrica 2), no como error de límite.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _raiz  # noqa: F401  (raíz del repo en sys.path; debe ir antes que lsch_mr)
from lsch_mr import config
from lsch_mr import metricas_segmentacion as ms
from lsch_mr.consola import configurar_utf8, amarillo, negrita, rojo, verde
from lsch_mr.rest_state_detector import RestStateDetector
from lsch_mr.tipos import SignEventType

configurar_utf8()

_XCOLS = [f"x{i}" for i in range(config.NUM_LANDMARKS)]
_YCOLS = [f"y{i}" for i in range(config.NUM_LANDMARKS)]
_ZCOLS = [f"z{i}" for i in range(config.NUM_LANDMARKS)]


# --------------------------------------------------------------------------- #
# Lectura del corpus crudo
# --------------------------------------------------------------------------- #
def _landmarks(row) -> np.ndarray:
    pts = np.empty((config.NUM_LANDMARKS, 3), dtype=np.float32)
    for i in range(config.NUM_LANDMARKS):
        pts[i, 0] = float(row[_XCOLS[i]])
        pts[i, 1] = float(row[_YCOLS[i]])
        pts[i, 2] = float(row[_ZCOLS[i]])
    return pts


def cargar_muestras(csvs: list[Path]) -> dict:
    """dict[sample_id] -> {"label": str, "frames": {frame_idx: (21,3)}}.

    Por frame se queda con la mano de mayor `handedness_score` — la misma regla
    de "mano dominante" que usa `build_dataset.py`, para que la segmentación se
    evalúe sobre la señal que el modelo realmente consume.
    """
    muestras: dict[str, dict] = {}
    for ruta in csvs:
        with ruta.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["sample_id"]
                m = muestras.setdefault(sid, {"label": row.get("label", "").strip(),
                                              "_filas": defaultdict(list)})
                m["_filas"][int(row["frame_idx"])].append(row)

    for sid, m in muestras.items():
        frames = {}
        for fi, filas in m.pop("_filas").items():
            mejor = max(filas, key=lambda r: float(r["handedness_score"]))
            frames[fi] = _landmarks(mejor)
        m["frames"] = frames
    return muestras


def limites_detectados(frames: dict, detector: RestStateDetector) -> dict:
    """Reproduce el detector sobre una muestra y devuelve los límites hallados.

    Recorre el rango COMPLETO de frames (no solo los que tienen mano) pasando
    `None` donde no hubo detección: es lo que ve el detector en vivo y lo que
    activa la tolerancia a parpadeos.
    """
    detector.reset()
    if not frames:
        return {"inicios": [], "fines": [], "descartes": 0,
                "n_frames": 0, "n_con_mano": 0}

    f_min, f_max = min(frames), max(frames)
    inicios: list[int] = []
    fines: list[int] = []
    descartes = 0
    buffer_previo = 0
    inicio_candidato: int | None = None

    for fi in range(f_min, f_max + 1):
        evento = detector.update(frames.get(fi))
        n_buffer = detector.n_frames_buffer

        # 0 -> 1 en el buffer: este es el primer frame que el detector acumula,
        # o sea el inicio real de la seña (ver docstring del módulo).
        if buffer_previo == 0 and n_buffer == 1:
            inicio_candidato = fi

        if evento.type == SignEventType.END:
            inicios.append(inicio_candidato if inicio_candidato is not None else fi)
            fines.append(fi - detector.frames_fin)
            inicio_candidato = None
        elif evento.type == SignEventType.DISCARDED:
            descartes += 1
            inicio_candidato = None

        buffer_previo = detector.n_frames_buffer

    return {
        "inicios": inicios,
        "fines": fines,
        "descartes": descartes,
        "n_frames": f_max - f_min + 1,
        "n_con_mano": len(frames),
    }


# --------------------------------------------------------------------------- #
# Plantilla de etiquetado manual
# --------------------------------------------------------------------------- #
_CAMPOS_GT = ["sample_id", "label", "frame_primero_detectado",
              "frame_ultimo_detectado", "tasa_deteccion",
              "frame_inicio_gt", "frame_fin_gt", "notas"]


def escribir_plantilla(muestras: dict, ruta: Path, limite: int | None) -> Path:
    """Genera el CSV que la persona rellena a mano.

    Se incluyen el primer/último frame CON detección y la tasa de detección como
    ayuda de contexto, pero **no** como sugerencia de respuesta: el ground truth
    debe salir de mirar el vídeo, no de lo que MediaPipe alcanzó a ver. Si se
    rellenara copiando esas columnas, la evaluación mediría el detector contra
    sí mismo y daría una cifra sin ningún valor.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sids = sorted(muestras)
    if limite:
        # Muestreo uniforme a lo largo del corpus en vez de los N primeros: los
        # sample_id están ordenados por clase y señante, así que los primeros N
        # serían todos de la misma seña y del mismo señante.
        paso = max(1, len(sids) // limite)
        sids = sids[::paso][:limite]

    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CAMPOS_GT)
        w.writeheader()
        for sid in sids:
            m = muestras[sid]
            frames = m["frames"]
            if not frames:
                continue
            f_min, f_max = min(frames), max(frames)
            span = f_max - f_min + 1
            w.writerow({
                "sample_id": sid,
                "label": m["label"],
                "frame_primero_detectado": f_min,
                "frame_ultimo_detectado": f_max,
                "tasa_deteccion": round(len(frames) / span, 3) if span else 0.0,
                "frame_inicio_gt": "",
                "frame_fin_gt": "",
                "notas": "",
            })
    return ruta


def leer_gt(ruta: Path) -> dict:
    """Lee el CSV etiquetado. Filas con los dos límites vacíos se ignoran."""
    gt: dict[str, dict] = {}
    incompletas = 0
    with Path(ruta).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ini = (row.get("frame_inicio_gt") or "").strip()
            fin = (row.get("frame_fin_gt") or "").strip()
            if not ini and not fin:
                continue
            if not ini or not fin:
                incompletas += 1
                continue
            try:
                gt[row["sample_id"]] = {"inicio": int(float(ini)),
                                        "fin": int(float(fin)),
                                        "label": row.get("label", "")}
            except ValueError:
                incompletas += 1
    gt["__incompletas__"] = incompletas  # type: ignore[assignment]
    return gt


# --------------------------------------------------------------------------- #
def _imprimir(resultado: dict, contexto: dict) -> None:
    print()
    print(negrita("=" * 74))
    print(negrita("  PRUEBA UNITARIA — RestStateDetector (límites de seña)"))
    print(negrita("=" * 74))
    print(f"  muestras etiquetadas : {resultado['n_muestras']}")
    print(f"  tolerancia           : ±{resultado['tolerancia_frames']} frames")
    print(f"  parámetros detector  : umbral={contexto['umbral']}  "
          f"inicio={contexto['frames_inicio']}  fin={contexto['frames_fin']}  "
          f"perdida_max={contexto['frames_perdida_max']}")
    print(f"  muestras exactas     : {resultado['n_muestras_exactas']}"
          f"/{resultado['n_muestras']}"
          f"  ({resultado.get('pct_muestras_exactas', 0.0)}%)")

    print()
    print(f"    {'límite':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'precisión':>11} "
          f"{'recall':>9} {'F1':>7} {'sesgo':>8}")
    print("    " + "-" * 64)
    for nombre in ("inicio", "fin", "global"):
        r = resultado[nombre]
        err = r.get("error", {})
        sesgo = (f"{err['sesgo_frames']:+.1f}f" if err.get("n") else "--")
        etiqueta = negrita(f"{nombre:<10}") if nombre == "global" else f"{nombre:<10}"
        print(f"    {etiqueta} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
              f"{r['precision']:>11.3f} {r['recall']:>9.3f} {r['f1']:>7.3f} "
              f"{sesgo:>8}")

    err = resultado["global"].get("error", {})
    if err.get("n"):
        print()
        print(f"  error absoluto de los límites acertados: "
              f"medio {err['error_abs_medio_frames']} f  |  "
              f"mediano {err['error_abs_mediano_frames']} f  |  "
              f"máx {err['error_abs_max_frames']} f")
        for nombre in ("inicio", "fin"):
            e = resultado[nombre].get("error", {})
            if e.get("n"):
                signo = ("tarde" if e["sesgo_frames"] > 0 else "temprano")
                print(f"    sesgo del {nombre:<7}: {e['sesgo_frames']:+.2f} frames "
                      f"(el detector marca el {nombre} {signo})")

    # Salud de la captura: sin esto, una recall baja se confunde con un fallo de
    # la máquina de estados cuando la causa puede ser que no había mano que ver.
    print()
    print(negrita("  Contexto de captura (por qué falla lo que falla)"))
    print(f"    tasa de detección de mano (media): {contexto['tasa_deteccion']:.1%}")
    print(f"    secuencias descartadas por pérdida de tracking: "
          f"{contexto['descartes']}")
    print(f"    muestras sin ninguna seña segmentada: {contexto['sin_segmentar']}")
    print(f"    muestras con más de una seña segmentada: {contexto['multiples']}")
    if contexto["tasa_deteccion"] < 0.8:
        print(amarillo(
            "    [!] La mano no se ve en 1 de cada 5 frames o más. Parte del FN "
            "es dropout\n        de MediaPipe, no error de segmentación — "
            "declara las dos cosas por separado."))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Precisión y recall del RestStateDetector contra ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", nargs="*", default=None,
                    help="CSV(s) crudo(s) de keypoints; por defecto data/raw/*.csv")
    ap.add_argument("--plantilla", default=None, metavar="RUTA",
                    help="genera el CSV de etiquetado manual y termina")
    ap.add_argument("--muestras", type=int, default=None,
                    help="con --plantilla: cuántas muestras incluir "
                         "(muestreo uniforme a lo largo del corpus)")
    ap.add_argument("--gt", default=None, metavar="RUTA",
                    help="CSV de ground truth ya etiquetado")
    ap.add_argument("--tolerancia", type=int, default=ms.TOLERANCIA_FRAMES,
                    help=f"frames de tolerancia (default {ms.TOLERANCIA_FRAMES})")
    ap.add_argument("--umbral-movimiento", type=float,
                    default=config.REST_UMBRAL_MOVIMIENTO)
    ap.add_argument("--frames-inicio", type=int, default=config.REST_FRAMES_INICIO)
    ap.add_argument("--frames-fin", type=int, default=config.REST_FRAMES_FIN)
    ap.add_argument("--frames-perdida-max", type=int,
                    default=config.REST_FRAMES_PERDIDA_MAX)
    ap.add_argument("--salida",
                    default=str(config.OUTPUTS_REPORTS_DIR / "segmentacion.json"))
    args = ap.parse_args()

    csvs = ([Path(p) for p in args.entrada] if args.entrada
            else sorted(config.DATA_RAW_DIR.glob("*.csv")))
    csvs = [c for c in csvs if c.exists()]
    if not csvs:
        print(rojo("[segmentacion] No hay CSV de keypoints en data/raw/. "
                   "Ejecuta extraer_lote.py o grabar_corpus.py primero."))
        return 1

    print(f"[segmentacion] leyendo {', '.join(str(c) for c in csvs)} ...")
    muestras = cargar_muestras(csvs)
    print(f"[segmentacion] {len(muestras)} muestras en el corpus.")

    if args.plantilla:
        ruta = escribir_plantilla(muestras, Path(args.plantilla), args.muestras)
        n = sum(1 for _ in ruta.open(encoding="utf-8")) - 1
        print(f"[segmentacion] plantilla de etiquetado -> {ruta}  ({n} filas)")
        print("  Rellena 'frame_inicio_gt' y 'frame_fin_gt' viendo cada vídeo:")
        print("    inicio = primer frame en que la mano empieza a moverse "
              "para hacer la seña")
        print("    fin    = último frame de movimiento, antes de volver a reposo")
        print("  Las filas que dejes vacías se ignoran. Después:")
        print(f"    python scripts/evaluar_segmentacion.py --gt {ruta}")
        return 0

    if not args.gt:
        print(rojo("[segmentacion] Falta --gt (o --plantilla para generarlo). "
                   "Sin ground truth no hay precisión ni recall que medir."))
        return 1

    ruta_gt = Path(args.gt)
    if not ruta_gt.exists():
        print(rojo(f"[segmentacion] No existe {ruta_gt}. Genera la plantilla con "
                   f"--plantilla {ruta_gt}"))
        return 1

    gt = leer_gt(ruta_gt)
    incompletas = gt.pop("__incompletas__", 0)
    if not gt:
        print(rojo(f"[segmentacion] {ruta_gt} no tiene ninguna fila etiquetada "
                   "por completo (hacen falta frame_inicio_gt Y frame_fin_gt)."))
        return 1
    if incompletas:
        print(amarillo(f"[segmentacion] {incompletas} fila(s) con solo uno de los "
                       "dos límites: ignoradas."))

    detector = RestStateDetector(umbral_movimiento=args.umbral_movimiento,
                                 frames_inicio=args.frames_inicio,
                                 frames_fin=args.frames_fin,
                                 frames_perdida_max=args.frames_perdida_max)

    entradas, tasas = [], []
    descartes = sin_segmentar = multiples = 0
    ausentes = []
    for sid, ref in sorted(gt.items()):
        if sid not in muestras:
            ausentes.append(sid)
            continue
        det = limites_detectados(muestras[sid]["frames"], detector)
        descartes += det["descartes"]
        if det["n_frames"]:
            tasas.append(det["n_con_mano"] / det["n_frames"])
        if not det["inicios"]:
            sin_segmentar += 1
        elif len(det["inicios"]) > 1:
            multiples += 1
        entradas.append({
            "sample_id": sid,
            "inicios_detectados": det["inicios"],
            "fines_detectados": det["fines"],
            "inicios_gt": [ref["inicio"]],
            "fines_gt": [ref["fin"]],
        })

    if ausentes:
        print(amarillo(f"[segmentacion] {len(ausentes)} sample_id del ground truth "
                       f"no están en el CSV de keypoints (p.ej. {ausentes[0]}): "
                       "ignorados."))
    if not entradas:
        print(rojo("[segmentacion] Ningún sample_id etiquetado coincide con el "
                   "corpus. ¿La plantilla se generó desde otro CSV?"))
        return 1

    resultado = ms.evaluar_segmentacion(entradas, tolerancia=args.tolerancia)
    contexto = {
        "umbral": args.umbral_movimiento,
        "frames_inicio": args.frames_inicio,
        "frames_fin": args.frames_fin,
        "frames_perdida_max": args.frames_perdida_max,
        "tasa_deteccion": (sum(tasas) / len(tasas)) if tasas else 0.0,
        "descartes": descartes,
        "sin_segmentar": sin_segmentar,
        "multiples": multiples,
        "ground_truth": str(ruta_gt),
        "corpus": [str(c) for c in csvs],
    }
    _imprimir(resultado, contexto)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps({"prueba": "RestStateDetector — límites de seña",
                    "contexto": contexto, "resultado": resultado},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  reporte: {salida}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
