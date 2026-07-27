"""
Estabilidad del preprocesador geométrico (Capa 2) — prueba unitaria previa.

`test_keypoint_normalizer.py` ya comprueba las propiedades algebraicas del
`KeypointNormalizer` (centrado, escala unitaria, casos degenerados). Este archivo
comprueba lo que pide el informe: que el `NormVector` de 63 dimensiones sea
**estable ante variaciones de distancia a la cámara y de tamaño de mano**, que es
una afirmación sobre el sistema físico, no sobre el álgebra.

---------------------------------------------------------------------------
Por qué no basta con multiplicar los keypoints por un escalar
---------------------------------------------------------------------------
`test_invariante_escala` multiplica la mano por 3.7 y comprueba que el vector no
cambia. Eso demuestra invariancia a un **escalado uniforme**, que es lo que
ocurre si cambia el tamaño de la mano en un mundo sin perspectiva. Pero alejar
una mano de la cámara **no** es un escalado uniforme: es una proyección
perspectiva, donde cada punto se encoge según SU propia profundidad. Los dedos
adelantados se encogen menos que la palma, así que la forma proyectada cambia
—no solo su tamaño— y el centrado + escalado no puede deshacerlo.

Por eso estos tests usan un **modelo de cámara pinhole** y verifican que la
deriva residual se mantiene acotada, en vez de exigir igualdad exacta que la
física no permite. Las cotas son empíricas: se midieron sobre 40 formas de mano
aleatorias y se dejó margen. Si un cambio en el preprocesamiento las supera, el
test falla y hay que decidir a conciencia si la nueva deriva es aceptable.

Hallazgo que estas cotas documentan: **la deriva por desplazamiento lateral en el
encuadre es unas 3 veces mayor que la deriva por distancia a la cámara.** Signar
en un lado del cuadro no produce el mismo vector que signar en el centro. Es una
limitación real del preprocesamiento (la normalización quita traslación en el
espacio de keypoints, pero no la distorsión de perspectiva fuera de eje), y está
recogida como caso caracterizado más abajo.
"""
import numpy as np
import pytest

from lsch_mr import config
from lsch_mr.caracteristicas import preparar_entrada
from lsch_mr.keypoint_normalizer import KeypointNormalizer

# Cotas de deriva máxima admitida, en unidades del NormVector (donde la
# distancia L0–L9 vale exactamente 1.0). Medidas sobre 40 manos aleatorias:
# distancia 0.058, tamaño 0.032, lateral 0.161 — con margen sobre lo observado.
COTA_DISTANCIA = 0.08     # 40 cm .. 150 cm respecto a la referencia de 60 cm
COTA_TAMANO = 0.05        # mano un 30% más pequeña / más grande
COTA_LATERAL = 0.25       # 10 cm de desplazamiento lateral en el encuadre

DISTANCIA_REF_CM = 60.0


def _mano_3d(seed: int = 0) -> np.ndarray:
    """Mano sintética en 3D, en metros, con la muñeca (L0) en el origen.

    L9 se fija a 9 cm de L0 (distancia muñeca–base del dedo medio de un adulto),
    que es justo la referencia de escala del normalizador.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(scale=0.03, size=(config.NUM_LANDMARKS, 3))
    pts[config.WRIST_IDX] = [0.0, 0.0, 0.0]
    pts[config.MIDDLE_MCP_IDX] = [0.0, 0.09, 0.0]
    return pts


def _proyectar(pts3d: np.ndarray, distancia_cm: float,
               foco: float = 1.0) -> np.ndarray:
    """Proyección pinhole de una mano 3D a coordenadas tipo MediaPipe.

    Devuelve (21, 3) con `x`, `y` normalizadas en el plano imagen y `z` como
    profundidad relativa a la muñeca en una escala comparable a la de `x` — que
    es la convención que documenta MediaPipe y la que asume el resto del
    pipeline (ver la trampa 2 de docs/INTEGRACION_UNITY.md).
    """
    d = distancia_cm / 100.0
    z_camara = pts3d[:, 2] + d                      # profundidad real por punto
    x = foco * pts3d[:, 0] / z_camara + 0.5
    y = foco * pts3d[:, 1] / z_camara + 0.5
    z = foco * (pts3d[:, 2] - pts3d[config.WRIST_IDX, 2]) / d
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _deriva(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.abs(v1 - v2).max())


# --------------------------------------------------------------------------- #
# Forma y validez del vector, pase lo que pase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distancia_cm", [25.0, 40.0, 60.0, 100.0, 200.0])
def test_siempre_63_dimensiones_y_finito(distancia_cm):
    n = KeypointNormalizer()
    v = n.normalize(_proyectar(_mano_3d(1), distancia_cm))
    assert v.shape == (config.NORMVECTOR_DIM,) == (63,)
    assert v.dtype == np.float32
    assert np.all(np.isfinite(v))


def test_escala_unitaria_se_mantiene_a_cualquier_distancia():
    """Tras normalizar, ‖L9 − L0‖ debe valer 1 esté la mano donde esté.

    Es la propiedad que hace comparables dos frames tomados a distancias
    distintas; si se rompiera, la magnitud del vector arrastraría información de
    distancia a la cámara hasta el clasificador.
    """
    n = KeypointNormalizer()
    for distancia in (25.0, 60.0, 150.0):
        v = n.normalize(_proyectar(_mano_3d(2), distancia)).reshape(21, 3)
        d = np.linalg.norm(v[config.MIDDLE_MCP_IDX] - v[config.WRIST_IDX])
        assert d == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Estabilidad ante distancia a la cámara
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("distancia_cm", [40.0, 50.0, 80.0, 100.0, 150.0])
def test_estable_ante_distancia_a_camara(distancia_cm):
    """El mismo gesto a distinta distancia debe dar casi el mismo vector.

    No exactamente el mismo: la perspectiva deforma la mano proyectada. Lo que
    se verifica es que el residuo queda muy por debajo de la magnitud de la
    propia señal (la distancia L0–L9 vale 1.0), o sea que el clasificador ve el
    mismo gesto y no uno distinto.
    """
    n = KeypointNormalizer()
    mano = _mano_3d(3)
    ref = n.normalize(_proyectar(mano, DISTANCIA_REF_CM))
    v = n.normalize(_proyectar(mano, distancia_cm))
    assert _deriva(ref, v) < COTA_DISTANCIA


def test_estable_ante_distancia_sobre_muchas_manos():
    """La cota no puede depender de una forma de mano afortunada."""
    n = KeypointNormalizer()
    peor = 0.0
    for seed in range(40):
        mano = _mano_3d(seed)
        ref = n.normalize(_proyectar(mano, DISTANCIA_REF_CM))
        for distancia in (40.0, 50.0, 80.0, 100.0, 150.0):
            peor = max(peor, _deriva(ref, n.normalize(_proyectar(mano, distancia))))
    assert peor < COTA_DISTANCIA


# --------------------------------------------------------------------------- #
# Estabilidad ante tamaño de mano
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factor", [0.70, 0.85, 1.15, 1.30])
def test_estable_ante_tamano_de_mano(factor):
    """Una mano un 30% más pequeña (niño / adulto) da el mismo vector.

    Aquí sí hay invariancia casi perfecta, porque escalar la mano en 3D a
    distancia fija es muy parecido a un escalado uniforme en el plano imagen.
    """
    n = KeypointNormalizer()
    mano = _mano_3d(4)
    ref = n.normalize(_proyectar(mano, DISTANCIA_REF_CM))
    v = n.normalize(_proyectar(mano * factor, DISTANCIA_REF_CM))
    assert _deriva(ref, v) < COTA_TAMANO


def test_estable_ante_tamano_sobre_muchas_manos():
    n = KeypointNormalizer()
    peor = 0.0
    for seed in range(40):
        mano = _mano_3d(seed)
        ref = n.normalize(_proyectar(mano, DISTANCIA_REF_CM))
        for factor in (0.70, 0.85, 1.15, 1.30):
            peor = max(peor, _deriva(ref, n.normalize(
                _proyectar(mano * factor, DISTANCIA_REF_CM))))
    assert peor < COTA_TAMANO


def test_escalado_uniforme_puro_es_exactamente_invariante():
    """Sin perspectiva de por medio, la invariancia a escala es exacta.

    Separa las dos cosas: lo que se mide arriba como "deriva" es efecto de la
    proyección, no un defecto del normalizador.
    """
    n = KeypointNormalizer()
    proyectada = _proyectar(_mano_3d(5), DISTANCIA_REF_CM)
    assert _deriva(n.normalize(proyectada), n.normalize(proyectada * 4.2)) < 1e-5


# --------------------------------------------------------------------------- #
# Caracterización: la deriva lateral es el término dominante
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("offset_m", [0.05, 0.10])
def test_deriva_lateral_acotada(offset_m):
    """Signar fuera del centro del encuadre SÍ mueve el vector, y bastante.

    Este test no celebra una invariancia: **documenta una limitación**. Con 10 cm
    de desplazamiento lateral la deriva (~0.16) triplica la de alejarse de 60 a
    150 cm (~0.06). La normalización elimina la traslación en el espacio de
    keypoints, pero no la distorsión de perspectiva fuera de eje.

    Consecuencia práctica: conviene signar razonablemente centrado, y el corpus
    debería cubrir posiciones laterales si en uso real la mano se va a los
    bordes. Si esta cota se supera, el preprocesamiento cambió de forma que
    afecta a la robustez posicional — hay que revisarlo, no relajar el número.
    """
    n = KeypointNormalizer()
    mano = _mano_3d(6)
    ref = n.normalize(_proyectar(mano, DISTANCIA_REF_CM))
    desplazada = mano.copy()
    desplazada[:, 0] += offset_m
    assert _deriva(ref, n.normalize(_proyectar(desplazada, DISTANCIA_REF_CM))) < COTA_LATERAL


def test_lateral_deriva_mas_que_la_distancia():
    """Deja constancia del orden de magnitud relativo entre las dos derivas."""
    n = KeypointNormalizer()
    mano = _mano_3d(7)
    ref = n.normalize(_proyectar(mano, DISTANCIA_REF_CM))

    por_distancia = _deriva(ref, n.normalize(_proyectar(mano, 150.0)))
    desplazada = mano.copy()
    desplazada[:, 0] += 0.10
    por_lateral = _deriva(ref, n.normalize(_proyectar(desplazada, DISTANCIA_REF_CM)))

    assert por_lateral > por_distancia


# --------------------------------------------------------------------------- #
# Estabilidad de la secuencia completa (Capa 2 de punta a punta)
# --------------------------------------------------------------------------- #
def test_entrada_del_modelo_estable_ante_distancia():
    """La cadena completa (normalizar + ensamblar + remuestrear a 60) hereda la
    estabilidad: es lo que realmente entra al TCN."""
    mano = _mano_3d(8)
    # Una "seña" de 24 frames con un desplazamiento suave.
    def secuencia(distancia_cm):
        frames = []
        for t in range(24):
            m = mano.copy()
            m[:, 0] += 0.002 * t
            frames.append(_proyectar(m, distancia_cm))
        return np.stack(frames, axis=0)

    ref = preparar_entrada(secuencia(DISTANCIA_REF_CM), modo="dominante")
    lejos = preparar_entrada(secuencia(100.0), modo="dominante")

    assert ref.shape == (config.SEQ_LEN, 63)
    assert lejos.shape == (config.SEQ_LEN, 63)
    assert np.all(np.isfinite(lejos))
    assert _deriva(ref, lejos) < COTA_DISTANCIA
