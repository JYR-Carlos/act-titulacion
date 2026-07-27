"""Tests de las funciones puras de reporte de scripts/demo_vivo.py: la parte que
calcula las métricas de éxito del MVP (Sección 3) a partir del historial de
señas de una sesión, más la salud de la captura que permite interpretar una
sesión vacía. No tocan cámara ni entran a `main()`."""
from demo_vivo import (EstadoCaptura, _resumen_captura, _resumen_latencias,
                       _resumen_reconocimiento)
from lsch_mr.realce_luz import UMBRAL_LUMINANCIA


def _sena(label="Thanks", label_es="Gracias", conf=0.97, in_vocab=True,
         latencia_e2e_ms=200.0, excede_umbral=False):
    return {
        "label": label,
        "label_es": label_es,
        "conf": conf,
        "in_vocab": in_vocab,
        "latencia_inferencia_ms": 2.0,
        "retardo_segmentacion_ms": 180.0,
        "latencia_e2e_ms": latencia_e2e_ms,
        "excede_umbral": excede_umbral,
    }


def test_resumen_latencias_vacio():
    resumen = _resumen_latencias([])
    assert resumen["n"] == 0
    assert "media_ms" not in resumen


def test_resumen_latencias_calcula_media_min_max():
    historial = [_sena(latencia_e2e_ms=100.0), _sena(latencia_e2e_ms=200.0),
                 _sena(latencia_e2e_ms=300.0)]
    resumen = _resumen_latencias(historial)
    assert resumen["n"] == 3
    assert resumen["media_ms"] == 200.0
    assert resumen["min_ms"] == 100.0
    assert resumen["max_ms"] == 300.0
    assert resumen["n_excede_umbral"] == 0
    assert resumen["pct_excede_umbral"] == 0.0


def test_resumen_latencias_cuenta_las_que_exceden_el_umbral():
    historial = [_sena(latencia_e2e_ms=100.0, excede_umbral=False),
                 _sena(latencia_e2e_ms=600.0, excede_umbral=True),
                 _sena(latencia_e2e_ms=700.0, excede_umbral=True),
                 _sena(latencia_e2e_ms=100.0, excede_umbral=False)]
    resumen = _resumen_latencias(historial)
    assert resumen["n_excede_umbral"] == 2
    assert resumen["pct_excede_umbral"] == 50.0


def test_resumen_reconocimiento_separa_vocabulario_y_cuenta_glosas():
    historial = [
        _sena(label_es="Gracias", in_vocab=True),
        _sena(label_es="Gracias", in_vocab=True),
        _sena(label_es="Nombre", in_vocab=True),
        _sena(label="<desconocida>", label_es="<desconocida>", in_vocab=False),
    ]
    resumen = _resumen_reconocimiento(historial, n_descartadas_tracking=3)
    assert resumen["n_clasificadas"] == 4
    assert resumen["n_en_vocabulario"] == 3
    assert resumen["n_fuera_vocabulario"] == 1
    assert resumen["n_descartadas_tracking"] == 3
    assert resumen["conteo_por_glosa"] == {"Gracias": 2, "Nombre": 1}


def test_resumen_reconocimiento_vacio():
    resumen = _resumen_reconocimiento([], n_descartadas_tracking=0)
    assert resumen["n_clasificadas"] == 0
    assert resumen["conteo_por_glosa"] == {}


def _capturar(estado: EstadoCaptura, n: int, hay_mano: bool, luz: float) -> None:
    for _ in range(n):
        estado.tick(hay_mano=hay_mano, luminancia=luz)


def test_estado_captura_promedia_tasa_y_luz():
    estado = EstadoCaptura()
    _capturar(estado, 30, hay_mano=True, luz=100.0)
    _capturar(estado, 70, hay_mano=False, luz=100.0)
    assert estado.n_frames == 100
    assert estado.tasa_deteccion == 0.3
    assert estado.luminancia_media == 100.0
    assert estado.hay_mano is False     # refleja el último frame, no el promedio


def test_estado_captura_vacio_no_divide_por_cero():
    estado = EstadoCaptura()
    assert estado.tasa_deteccion == 0.0
    assert estado.luminancia_media == 0.0
    assert estado.escena_oscura is False


def test_escena_oscura_solo_bajo_el_umbral():
    oscura, clara = EstadoCaptura(), EstadoCaptura()
    _capturar(oscura, 10, hay_mano=True, luz=UMBRAL_LUMINANCIA - 50.0)
    _capturar(clara, 10, hay_mano=True, luz=UMBRAL_LUMINANCIA + 50.0)
    assert oscura.escena_oscura is True
    assert clara.escena_oscura is False


def test_resumen_captura_reconstruye_la_sesion_de_noche():
    """Los números reales del 2026-07-26: escena a 10/255 y mano vista en la
    mitad de los frames. El reporte tiene que dejar eso explícito — es lo que
    convierte "0 señas" en un diagnóstico en vez de un misterio."""
    estado = EstadoCaptura()
    _capturar(estado, 255, hay_mano=True, luz=9.65)
    _capturar(estado, 242, hay_mano=False, luz=9.65)
    resumen = _resumen_captura(estado, periodo_ms=42.9, realce=False)
    assert resumen["n_frames"] == 497
    assert resumen["n_frames_con_mano"] == 255
    assert resumen["tasa_deteccion_mano"] == 0.513
    assert resumen["luminancia_media"] == 9.7
    assert resumen["escena_oscura"] is True
    assert resumen["realce_poca_luz"] is False
    assert resumen["fps_promedio_camara"] == 23.3


def test_resumen_captura_sin_frames_no_inventa_fps_ni_luz():
    resumen = _resumen_captura(EstadoCaptura(), periodo_ms=0.0, realce=True)
    assert resumen["fps_promedio_camara"] is None
    assert resumen["luminancia_media"] is None
    assert resumen["realce_poca_luz"] is True
