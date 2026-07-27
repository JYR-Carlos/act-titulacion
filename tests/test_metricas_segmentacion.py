"""Tests del emparejamiento de límites de seña (prueba unitaria del detector)."""
import pytest

from lsch_mr import metricas_segmentacion as ms


def test_emparejamiento_exacto():
    r = ms.emparejar_limites([10, 40], [10, 40], tolerancia=3)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert r["errores"] == [0, 0]


def test_dentro_de_tolerancia_cuenta_como_acierto():
    r = ms.emparejar_limites([12], [10], tolerancia=3)
    assert r["tp"] == 1
    assert r["errores"] == [2]        # con signo: el detector marcó 2 frames tarde


def test_fuera_de_tolerancia_es_fp_y_fn_a_la_vez():
    """Un límite muy desplazado no es 'un acierto flojo': es un límite inventado
    (FP) más un límite real que se perdió (FN)."""
    r = ms.emparejar_limites([30], [10], tolerancia=3)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert r["detectados_sin_pareja"] == [30]
    assert r["referencia_sin_pareja"] == [10]


def test_emparejamiento_es_uno_a_uno():
    """Dos detecciones pegadas no pueden reclamar el mismo límite de referencia.

    Sin la restricción uno a uno las dos contarían como TP y la precisión saldría
    1.0 cuando en realidad el detector partió una seña en dos.
    """
    r = ms.emparejar_limites([10, 11], [10], tolerancia=3)
    assert r["tp"] == 1
    assert r["fp"] == 1               # la segunda detección sobra
    assert r["fn"] == 0


def test_emparejamiento_elige_el_mas_cercano():
    """Con varios candidatos dentro de la tolerancia gana el de menor error."""
    r = ms.emparejar_limites([13, 10], [10], tolerancia=5)
    assert r["tp"] == 1
    assert r["errores"] == [0]        # se emparejó el 10, no el 13


def test_sin_detecciones_todo_es_fn():
    r = ms.emparejar_limites([], [10, 20], tolerancia=3)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 2)


def test_sin_referencia_todo_es_fp():
    r = ms.emparejar_limites([10, 20], [], tolerancia=3)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 2, 0)


def test_tolerancia_negativa_es_error():
    with pytest.raises(ValueError):
        ms.emparejar_limites([1], [1], tolerancia=-1)


def test_tolerancia_cero_exige_coincidencia_exacta():
    assert ms.emparejar_limites([10], [10], tolerancia=0)["tp"] == 1
    assert ms.emparejar_limites([11], [10], tolerancia=0)["tp"] == 0


def test_precision_recall():
    r = ms.precision_recall(tp=8, fp=2, fn=2)
    assert r["precision"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(0.8)
    assert r["f1"] == pytest.approx(0.8)


def test_precision_recall_sin_datos_no_divide_por_cero():
    r = ms.precision_recall(0, 0, 0)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def _muestra(sid, ini_det, fin_det, ini_gt, fin_gt):
    return {"sample_id": sid,
            "inicios_detectados": ini_det, "fines_detectados": fin_det,
            "inicios_gt": ini_gt, "fines_gt": fin_gt}


def test_evaluar_segmentacion_separa_inicio_de_fin():
    """Inicio y fin dependen de parámetros distintos del detector, así que un
    agregado único escondería cuál de los dos hay que ajustar."""
    muestras = [
        # El inicio se acierta; el fin se va muy lejos en las dos muestras.
        _muestra("m1", [10], [90], [10], [50]),
        _muestra("m2", [20], [95], [21], [55]),
    ]
    r = ms.evaluar_segmentacion(muestras, tolerancia=3)

    assert r["inicio"]["tp"] == 2
    assert r["inicio"]["recall"] == 1.0
    assert r["fin"]["tp"] == 0
    assert r["fin"]["recall"] == 0.0
    assert r["global"]["tp"] == 2


def test_evaluar_segmentacion_cuenta_muestras_exactas():
    muestras = [
        _muestra("ok", [10], [50], [10], [50]),
        _muestra("mal", [10, 30], [50, 70], [10], [50]),   # segmentó de más
    ]
    r = ms.evaluar_segmentacion(muestras, tolerancia=3)
    assert r["n_muestras"] == 2
    assert r["n_muestras_exactas"] == 1
    assert r["pct_muestras_exactas"] == 50.0


def test_sesgo_detecta_cierre_tardio_sistematico():
    """El sesgo con signo es lo que dice si hay que bajar REST_FRAMES_FIN."""
    muestras = [_muestra(f"m{i}", [10], [54], [10], [50]) for i in range(5)]
    r = ms.evaluar_segmentacion(muestras, tolerancia=6)
    assert r["fin"]["error"]["sesgo_frames"] == pytest.approx(4.0)
    assert r["inicio"]["error"]["sesgo_frames"] == pytest.approx(0.0)


def test_evaluar_segmentacion_sin_muestras():
    r = ms.evaluar_segmentacion([], tolerancia=3)
    assert r["n_muestras"] == 0
    assert r["global"]["precision"] == 0.0
