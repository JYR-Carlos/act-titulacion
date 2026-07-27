"""Tests de ModelTrainer.split_indices: el reparto 80/20 del modelo final.

Importa porque `scripts/evaluar_modelo.py --fuente modelo` se apoya en que este helper
reproduce EXACTAMENTE el split que usó `train()`. Si los dos divergieran, el
script evaluaría el modelo sobre muestras con las que se entrenó y devolvería una
accuracy inflada sin dar ningún error — el mismo tipo de fallo silencioso que el
train/serve skew.
"""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from lsch_mr import config
from lsch_mr.model_trainer import ModelTrainer


def _y_balanceado(n_clases=10, por_clase=50):
    return np.repeat(np.arange(n_clases), por_clase).astype("int64")


def test_reproduce_el_split_de_arrays():
    """Partir índices y luego indexar == partir los arrays directamente.

    Es la propiedad de la que depende toda la evaluación del modelo exportado.
    """
    y = _y_balanceado()
    X = np.random.default_rng(0).normal(size=(len(y), 4, 3)).astype("float32")

    Xtr_ref, Xva_ref, ytr_ref, yva_ref = train_test_split(
        X, y, test_size=config.ENTRENAMIENTO_VAL_SPLIT,
        random_state=config.SEMILLA, stratify=y)

    idx_tr, idx_va = ModelTrainer.split_indices(y, n_clases=10)

    assert np.array_equal(X[idx_va], Xva_ref)
    assert np.array_equal(y[idx_va], yva_ref)
    assert np.array_equal(X[idx_tr], Xtr_ref)
    assert np.array_equal(y[idx_tr], ytr_ref)


def test_particion_sin_solape_y_completa():
    y = _y_balanceado()
    idx_tr, idx_va = ModelTrainer.split_indices(y, n_clases=10)
    assert set(idx_tr).isdisjoint(set(idx_va))
    assert len(set(idx_tr) | set(idx_va)) == len(y)


def test_proporcion_y_estratificacion():
    y = _y_balanceado(n_clases=10, por_clase=50)
    _, idx_va = ModelTrainer.split_indices(y, n_clases=10)
    assert len(idx_va) == 100                     # 20% de 500
    # Estratificado: cada clase aporta la misma cantidad al conjunto retenido.
    conteos = np.bincount(y[idx_va], minlength=10)
    assert set(conteos.tolist()) == {10}


def test_es_determinista_con_la_misma_semilla():
    y = _y_balanceado()
    a = ModelTrainer.split_indices(y, 10, seed=config.SEMILLA)
    b = ModelTrainer.split_indices(y, 10, seed=config.SEMILLA)
    assert np.array_equal(a[1], b[1])


def test_semilla_distinta_da_split_distinto():
    y = _y_balanceado()
    a = ModelTrainer.split_indices(y, 10, seed=42)
    b = ModelTrainer.split_indices(y, 10, seed=7)
    assert not np.array_equal(a[1], b[1])


def test_clase_con_una_sola_muestra_no_estratifica_y_no_revienta():
    """Con una clase de 1 muestra el estratificado es imposible; el split debe
    seguir funcionando en vez de lanzar."""
    y = np.array([0] * 20 + [1] * 20 + [2], dtype="int64")
    idx_tr, idx_va = ModelTrainer.split_indices(y, n_clases=3)
    assert len(idx_tr) + len(idx_va) == len(y)
    assert set(idx_tr).isdisjoint(set(idx_va))
