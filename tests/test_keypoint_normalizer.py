"""Tests del KeypointNormalizer: invariancia a traslación y escala, degenerados."""
import numpy as np
import pytest

from lsch_mr import config
from lsch_mr.keypoint_normalizer import KeypointNormalizer


def _mano_aleatoria(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(config.NUM_LANDMARKS, 3)).astype(np.float32)
    # Aseguramos distancia L0-L9 no nula.
    pts[config.MIDDLE_MCP_IDX] = pts[config.WRIST_IDX] + np.array([0.3, 0.4, 0.0])
    return pts


def test_forma_salida():
    n = KeypointNormalizer()
    v = n.normalize(_mano_aleatoria())
    assert v.shape == (config.NORMVECTOR_DIM,)
    assert v.dtype == np.float32


def test_muñeca_en_origen():
    n = KeypointNormalizer()
    v = n.normalize(_mano_aleatoria()).reshape(21, 3)
    assert np.allclose(v[config.WRIST_IDX], 0.0, atol=1e-5)


def test_invariante_traslacion():
    n = KeypointNormalizer()
    pts = _mano_aleatoria(1)
    v1 = n.normalize(pts)
    v2 = n.normalize(pts + np.array([10.0, -5.0, 2.0], dtype=np.float32))
    assert np.allclose(v1, v2, atol=1e-5)


def test_invariante_escala():
    n = KeypointNormalizer()
    pts = _mano_aleatoria(2)
    v1 = n.normalize(pts)
    v2 = n.normalize(pts * 3.7)
    assert np.allclose(v1, v2, atol=1e-5)


def test_escala_unitaria_l0_l9():
    # Tras normalizar, la distancia L0-L9 debe ser 1.
    n = KeypointNormalizer()
    v = n.normalize(_mano_aleatoria(3)).reshape(21, 3)
    d = np.linalg.norm(v[config.MIDDLE_MCP_IDX] - v[config.WRIST_IDX])
    assert abs(d - 1.0) < 1e-4


def test_degenerado_devuelve_ceros():
    n = KeypointNormalizer()
    pts = np.ones((config.NUM_LANDMARKS, 3), dtype=np.float32)  # todos iguales
    v = n.normalize(pts)
    assert np.all(v == 0.0)
    assert np.all(np.isfinite(v))


def test_acepta_vector_plano():
    n = KeypointNormalizer()
    pts = _mano_aleatoria(4)
    v1 = n.normalize(pts)
    v2 = n.normalize(pts.reshape(-1))
    assert np.allclose(v1, v2)


def test_forma_invalida():
    n = KeypointNormalizer()
    with pytest.raises(ValueError):
        n.normalize(np.zeros((10, 3), dtype=np.float32))


def test_normalize_sequence():
    n = KeypointNormalizer()
    seq = np.stack([_mano_aleatoria(i) for i in range(5)], axis=0)
    out = n.normalize_sequence(seq)
    assert out.shape == (5, config.NORMVECTOR_DIM)
