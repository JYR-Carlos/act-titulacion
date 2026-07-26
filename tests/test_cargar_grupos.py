"""Tests de ModelTrainer.cargar_grupos: extracción del señante desde sample_id.

Importa porque de esto depende que la validación cruzada por señante agrupe
bien: si los grupos salen mal, el fold deja de ser independiente del señante y
la métrica vuelve a estar inflada sin que nada falle de forma visible.
"""
import numpy as np
import pytest

from lsch_mr.model_trainer import ModelTrainer


def _npz(tmp_path, sample_ids):
    ruta = tmp_path / "ds.npz"
    np.savez_compressed(
        ruta,
        X=np.zeros((len(sample_ids), 2, 3), dtype=np.float32),
        y=np.zeros(len(sample_ids), dtype=np.int64),
        classes=np.array(["A"], dtype=object),
        sample_ids=np.array(sample_ids, dtype=object))
    return ruta


def test_campo_numerico_extrae_el_senante(tmp_path):
    ruta = _npz(tmp_path, ["051_003_002", "017_010_005", "040_001_001"])
    grupos = ModelTrainer.cargar_grupos(ruta, "2")
    assert list(grupos) == ["003", "010", "001"]


def test_regex_con_grupo_de_captura(tmp_path):
    ruta = _npz(tmp_path, ["glosa-HOLA-senante-juan-rep-1"])
    grupos = ModelTrainer.cargar_grupos(ruta, r"senante-([a-z]+)")
    assert list(grupos) == ["juan"]


def test_devuelve_none_si_el_dataset_no_guarda_sample_ids(tmp_path):
    ruta = tmp_path / "viejo.npz"
    np.savez_compressed(ruta, X=np.zeros((1, 2, 3), dtype=np.float32),
                        y=np.zeros(1, dtype=np.int64),
                        classes=np.array(["A"], dtype=object))
    assert ModelTrainer.cargar_grupos(ruta, "2") is None


def test_falla_si_el_campo_no_existe(tmp_path):
    """Mejor un error claro que agrupar mal en silencio."""
    ruta = _npz(tmp_path, ["sin_campos"])
    with pytest.raises(ValueError, match="campos"):
        ModelTrainer.cargar_grupos(ruta, "5")


def test_falla_si_la_regex_no_casa(tmp_path):
    ruta = _npz(tmp_path, ["051_003_002"])
    with pytest.raises(ValueError, match="no casa"):
        ModelTrainer.cargar_grupos(ruta, r"senante-([a-z]+)")
