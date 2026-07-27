"""Tests del realce de poca luz: que suba la luminancia cuando la escena está
oscura y que NO toque el frame cuando ya hay luz suficiente (esa garantía es la
que permite dejar el flag encendido sin degradar una sesión bien iluminada)."""
import numpy as np

from lsch_mr.realce_luz import (GAMMA_MIN, LUMINANCIA_OBJETIVO,
                                UMBRAL_LUMINANCIA, gamma_para,
                                luminancia_media, realzar_poca_luz)


def _frame(nivel: int, alto=48, ancho=64) -> np.ndarray:
    """Frame BGR al nivel medio pedido, con textura para que CLAHE y la métrica
    de luminancia trabajen sobre algo parecido a una imagen.

    La textura es ruido pseudoaleatorio (semilla fija) y no un patrón regular: un
    patrón con período 4 se alinearía con el submuestreo `[::4]` y haría que la
    lectura submuestreada midiera siempre los mismos píxeles.
    """
    rng = np.random.default_rng(0)
    ruido = rng.integers(-10, 11, size=(alto, ancho, 1))
    return np.clip(nivel + ruido, 0, 255).astype(np.uint8).repeat(3, axis=2)


def test_luminancia_media_refleja_el_nivel_del_frame():
    assert luminancia_media(_frame(10), submuestreo=1) < 20.0
    assert luminancia_media(_frame(200), submuestreo=1) > 190.0


def test_submuestreo_no_cambia_la_lectura_de_forma_apreciable():
    frame = _frame(30)
    completa = luminancia_media(frame, submuestreo=1)
    submuestreada = luminancia_media(frame, submuestreo=4)
    assert abs(completa - submuestreada) < 5.0


def test_gamma_para_lleva_la_luminancia_al_objetivo():
    # Con gamma = log(obj/255)/log(lum/255), aplicar la gamma a lum/255 devuelve
    # obj/255 por construcción: se verifica que el despeje sea el correcto.
    lum = 20.0
    g = gamma_para(lum, objetivo=LUMINANCIA_OBJETIVO)
    assert GAMMA_MIN <= g <= 1.0
    assert abs(((lum / 255.0) ** g) * 255.0 - LUMINANCIA_OBJETIVO) < 1.0


def test_gamma_para_nunca_oscurece():
    # Por encima del objetivo la gamma es 1.0 (identidad): este módulo aclara,
    # no ajusta exposición en los dos sentidos.
    assert gamma_para(LUMINANCIA_OBJETIVO + 50.0) == 1.0


def test_gamma_para_acotada_en_frame_casi_negro():
    assert gamma_para(0.0) == GAMMA_MIN
    assert gamma_para(1.0) >= GAMMA_MIN


def test_frame_con_luz_suficiente_se_devuelve_intacto():
    frame = _frame(int(UMBRAL_LUMINANCIA) + 60)
    salida = realzar_poca_luz(frame)
    # Mismo objeto: además de no alterar la imagen, no paga la conversión LAB.
    assert salida is frame


def test_frame_oscuro_sube_de_luminancia():
    frame = _frame(10)
    antes = luminancia_media(frame, submuestreo=1)
    salida = realzar_poca_luz(frame)
    despues = luminancia_media(salida, submuestreo=1)
    assert salida is not frame
    assert salida.shape == frame.shape
    assert salida.dtype == np.uint8
    assert despues > antes * 3


def test_luminancia_precalculada_evita_recalcularla():
    """El llamador puede pasar la luminancia que ya midió; la decisión de
    realzar debe tomarse con ESE valor, no con uno propio."""
    frame = _frame(10)
    # Se miente diciendo que el frame es claro: debe salir intacto.
    assert realzar_poca_luz(frame, luminancia=UMBRAL_LUMINANCIA + 1.0) is frame
