"""
Métricas de segmentación temporal — prueba unitaria previa del RestStateDetector.

Compara los límites de seña (inicio / fin) que el `RestStateDetector` detecta
automáticamente contra un **ground truth etiquetado a mano**, y reporta
precisión y recall de esa detección.

Funciones puras (solo numpy/estándar): el emparejamiento se puede testear sin
detector, sin CSV y sin vídeo. El script que las usa es
`scripts/evaluar_segmentacion.py`.

---------------------------------------------------------------------------
Cómo se cuenta un acierto
---------------------------------------------------------------------------
Un límite no se acierta "exacto": el etiquetador humano decide a ojo en qué
frame empieza el movimiento, y dos personas discrepan en un par de frames. Por
eso un límite detectado cuenta como **verdadero positivo (TP)** si cae a menos
de `tolerancia` frames del límite de referencia correspondiente.

  * TP — límite detectado que empareja con uno de referencia dentro de la tolerancia.
  * FP — límite detectado que no empareja con ninguno: el detector "inventó" un
         inicio o un fin de seña que no existe.
  * FN — límite de referencia que ningún detectado alcanzó: el detector se
         perdió una seña real.

  precisión = TP / (TP + FP)   de los límites que anuncié, cuántos eran de verdad
  recall    = TP / (TP + FN)   de los límites que existían, cuántos encontré

El emparejamiento es **uno a uno y voraz por menor error**: se recorren todos
los pares candidatos ordenados por distancia y se van fijando los más cercanos.
Sin la restricción uno a uno, dos detecciones pegadas podrían reclamar el mismo
límite de referencia y contarse las dos como acierto, inflando la precisión.

La tolerancia por defecto sale del propio diseño: el detector confirma el fin de
una seña tras `REST_FRAMES_FIN` frames de reposo sostenido, así que exigir una
precisión mejor que ese retardo mediría el reloj, no la segmentación.
"""
from __future__ import annotations

from . import config

# Tolerancia por defecto, en frames, para dar por bueno un límite detectado.
TOLERANCIA_FRAMES = max(config.REST_FRAMES_FIN, config.REST_FRAMES_INICIO)


def emparejar_limites(detectados: list[int], referencia: list[int],
                      tolerancia: int = TOLERANCIA_FRAMES) -> dict:
    """Empareja límites detectados con los de referencia, uno a uno.

    Devuelve conteos (`tp`, `fp`, `fn`), los errores con signo de cada pareja
    (`detectado - referencia`, negativo = el detector se adelantó) y las listas
    de límites que quedaron sin pareja, para poder inspeccionar los fallos.
    """
    if tolerancia < 0:
        raise ValueError("La tolerancia no puede ser negativa.")

    det = list(detectados)
    ref = list(referencia)

    # Todos los pares candidatos, del más ajustado al más flojo. Ordenar por
    # error absoluto es lo que hace que el emparejamiento sea estable: el par
    # obvio se fija antes de que un vecino peor pueda robarle el límite.
    pares = [(abs(d - r), i, j)
             for i, d in enumerate(det)
             for j, r in enumerate(ref)
             if abs(d - r) <= tolerancia]
    pares.sort()

    det_usados: set[int] = set()
    ref_usados: set[int] = set()
    errores: list[int] = []
    for _, i, j in pares:
        if i in det_usados or j in ref_usados:
            continue
        det_usados.add(i)
        ref_usados.add(j)
        errores.append(det[i] - ref[j])

    tp = len(errores)
    return {
        "tp": tp,
        "fp": len(det) - tp,
        "fn": len(ref) - tp,
        "errores": errores,
        "detectados_sin_pareja": [d for i, d in enumerate(det) if i not in det_usados],
        "referencia_sin_pareja": [r for j, r in enumerate(ref) if j not in ref_usados],
        "tolerancia": int(tolerancia),
    }


def precision_recall(tp: int, fp: int, fn: int) -> dict:
    """Precisión, recall y F1 a partir de los conteos. Sin datos -> 0.0.

    Se devuelve 0.0 en vez de NaN cuando el denominador es cero para que el
    JSON del reporte sea numérico en todos sus campos.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _estadisticos_error(errores: list[int]) -> dict:
    """Sesgo y dispersión del error de los límites emparejados.

    El **sesgo** (media con signo) es más informativo que el error absoluto: un
    sesgo positivo sistemático en el fin significa que el detector cierra tarde,
    y eso se corrige bajando `REST_FRAMES_FIN` — es justo la palanca que
    `docs/ESTADO_ACTUAL.md` propone si la latencia sale corta.
    """
    if not errores:
        return {"n": 0}
    n = len(errores)
    media = sum(errores) / n
    var = sum((e - media) ** 2 for e in errores) / n
    absolutos = sorted(abs(e) for e in errores)
    return {
        "n": n,
        "sesgo_frames": round(media, 2),
        "desv_frames": round(var ** 0.5, 2),
        "error_abs_medio_frames": round(sum(absolutos) / n, 2),
        "error_abs_mediano_frames": absolutos[n // 2],
        "error_abs_max_frames": absolutos[-1],
    }


def evaluar_segmentacion(muestras: list[dict],
                         tolerancia: int = TOLERANCIA_FRAMES) -> dict:
    """Agrega la evaluación sobre todas las muestras del corpus.

    Cada elemento de `muestras` es un dict con:
        sample_id            identificador de la muestra
        inicios_detectados   list[int]   frames de inicio que dio el detector
        fines_detectados     list[int]   frames de fin que dio el detector
        inicios_gt           list[int]   inicios etiquetados a mano
        fines_gt             list[int]   fines etiquetados a mano

    Inicio y fin se evalúan **por separado** además de en conjunto: fallan de
    formas distintas y por causas distintas (el inicio depende de
    `REST_FRAMES_INICIO` y del umbral de movimiento; el fin, de
    `REST_FRAMES_FIN`), así que un agregado único escondería cuál de los dos
    hay que ajustar.
    """
    acumulado = {
        "inicio": {"tp": 0, "fp": 0, "fn": 0, "errores": []},
        "fin": {"tp": 0, "fp": 0, "fn": 0, "errores": []},
    }
    por_muestra = []

    for m in muestras:
        detalle = {"sample_id": m["sample_id"]}
        for limite, clave_det, clave_gt in (("inicio", "inicios_detectados", "inicios_gt"),
                                            ("fin", "fines_detectados", "fines_gt")):
            r = emparejar_limites(m.get(clave_det, []), m.get(clave_gt, []),
                                  tolerancia)
            for k in ("tp", "fp", "fn"):
                acumulado[limite][k] += r[k]
            acumulado[limite]["errores"].extend(r["errores"])
            detalle[limite] = {
                "detectados": list(m.get(clave_det, [])),
                "referencia": list(m.get(clave_gt, [])),
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                "errores": r["errores"],
            }
        # Una muestra "perfecta" es la que reproduce exactamente los límites
        # etiquetados: sin sobras ni faltas en ninguno de los dos extremos.
        detalle["exacta"] = all(detalle[l]["fp"] == 0 and detalle[l]["fn"] == 0
                                for l in ("inicio", "fin"))
        por_muestra.append(detalle)

    resultado = {
        "tolerancia_frames": int(tolerancia),
        "n_muestras": len(muestras),
        "n_muestras_exactas": sum(1 for d in por_muestra if d["exacta"]),
        "por_muestra": por_muestra,
    }
    for limite in ("inicio", "fin"):
        a = acumulado[limite]
        resultado[limite] = precision_recall(a["tp"], a["fp"], a["fn"])
        resultado[limite]["error"] = _estadisticos_error(a["errores"])

    # Agregado de los dos extremos: es la cifra de titular ("precisión y recall
    # de la detección de límites de seña"), pero se reporta junto al desglose,
    # nunca en lugar de él.
    total = precision_recall(
        acumulado["inicio"]["tp"] + acumulado["fin"]["tp"],
        acumulado["inicio"]["fp"] + acumulado["fin"]["fp"],
        acumulado["inicio"]["fn"] + acumulado["fin"]["fn"])
    total["error"] = _estadisticos_error(
        acumulado["inicio"]["errores"] + acumulado["fin"]["errores"])
    resultado["global"] = total
    if len(muestras):
        resultado["pct_muestras_exactas"] = round(
            100.0 * resultado["n_muestras_exactas"] / len(muestras), 1)
    return resultado
