"""Tests de las métricas de clasificación (métrica 1 del MVP).

Son las funciones que producen la cifra que va al informe, así que un error aquí
no rompe nada visiblemente: solo reporta un número equivocado. De ahí que se
verifiquen contra casos calculados a mano.
"""
import numpy as np
import pytest

from lsch_mr import metricas_clasificacion as mc

CLASES = ["A", "B", "C"]


def test_matriz_confusion_perfecta():
    y = [0, 1, 2, 0, 1, 2]
    cm = mc.matriz_confusion(y, y, 3)
    assert np.array_equal(cm, np.eye(3, dtype=np.int64) * 2)


def test_matriz_confusion_filas_real_columnas_prediccion():
    """La orientación importa: transponerla intercambia precisión y recall."""
    # Dos muestras reales de A, las dos predichas como B.
    cm = mc.matriz_confusion([0, 0], [1, 1], 3)
    assert cm[0, 1] == 2      # fila A, columna B
    assert cm[1, 0] == 0
    assert cm.sum() == 2


def test_matriz_confusion_rechaza_largos_distintos():
    with pytest.raises(ValueError):
        mc.matriz_confusion([0, 1], [0], 2)


def test_matriz_confusion_rechaza_indices_fuera_de_rango():
    with pytest.raises(ValueError):
        mc.matriz_confusion([0, 3], [0, 0], 3)


def test_accuracy_global():
    assert mc.accuracy_global([0, 1, 2, 0], [0, 1, 2, 1]) == 0.75
    assert mc.accuracy_global([], []) == 0.0


def test_metricas_por_clase_recall_y_precision_difieren():
    """Una clase que el modelo nunca predice tiene precisión 1.0 y recall 0.0.

    Es el caso que justifica reportar las dos: mirando solo la precisión, la
    clase C parecería perfecta cuando en realidad no se reconoce nunca.
    """
    # C (idx 2) aparece 2 veces y siempre se predice como A; nunca se predice C.
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 0, 0]
    cm = mc.matriz_confusion(y_true, y_pred, 3)
    filas = {f["clase"]: f for f in mc.metricas_por_clase(cm, CLASES)}

    assert filas["C"]["soporte"] == 2
    assert filas["C"]["aciertos"] == 0
    assert filas["C"]["accuracy"] == 0.0      # recall
    assert filas["C"]["precision"] == 0.0     # no hay predicciones de C -> 0, no NaN
    assert filas["A"]["accuracy"] == 1.0      # recall: se acertaron las 2 de A
    assert filas["A"]["precision"] == 0.5     # 4 predicciones de A, solo 2 correctas


def test_metricas_por_clase_soporte_cero_no_divide_por_cero():
    cm = mc.matriz_confusion([0, 0], [0, 0], 3)
    filas = mc.metricas_por_clase(cm, CLASES)
    for f in filas:
        assert np.isfinite(f["accuracy"]) and np.isfinite(f["f1"])
    assert filas[1]["soporte"] == 0
    assert filas[1]["accuracy"] == 0.0


def test_metricas_por_clase_rechaza_matriz_incoherente():
    with pytest.raises(ValueError):
        mc.metricas_por_clase(np.zeros((2, 2)), CLASES)


def test_veredicto_marca_incumplimiento_y_brecha():
    v = mc.veredicto(0.80, objetivo=0.85)
    assert v["cumple"] is False
    assert v["brecha"] == pytest.approx(0.05)

    v = mc.veredicto(0.92, objetivo=0.85)
    assert v["cumple"] is True
    assert v["brecha"] < 0


def test_veredicto_en_el_limite_exacto_cumple():
    """0.85 exacto CUMPLE: el diseño dice '>= 85%', no '> 85%'."""
    assert mc.veredicto(0.85, objetivo=0.85)["cumple"] is True


def test_cobertura_con_umbral_separa_mostradas_de_ocultas():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 0, 1]          # 2 aciertos, 2 fallos
    conf = [0.99, 0.95, 0.93, 0.40]  # las 3 primeras superan 0.90
    r = mc.cobertura_con_umbral(y_true, y_pred, conf, umbral=0.90)

    assert r["n_mostradas"] == 3
    assert r["cobertura"] == pytest.approx(0.75)
    assert r["n_erroneas_mostradas"] == 1        # la tercera (real 2, predicha 0)
    assert r["precision_mostradas"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["n_ocultas_por_umbral"] == 1


def test_cobertura_con_umbral_vacia():
    assert mc.cobertura_con_umbral([], [], [])["n"] == 0


def test_guardar_matriz_csv_conserva_los_ejes(tmp_path):
    cm = mc.matriz_confusion([0, 0, 1], [0, 1, 1], 3)
    ruta = mc.guardar_matriz_csv(cm, CLASES, tmp_path / "m.csv")
    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()

    assert lineas[0].split(",") == ["real \\ predicha"] + CLASES
    assert lineas[1].split(",") == ["A", "1", "1", "0"]


def test_graficar_matriz_escribe_png(tmp_path):
    cm = mc.matriz_confusion([0, 1, 2], [0, 1, 2], 3)
    ruta = mc.graficar_matriz(cm, CLASES, tmp_path / "m.png",
                              subtitulo="prueba", cumple=False)
    assert ruta.exists() and ruta.stat().st_size > 0
