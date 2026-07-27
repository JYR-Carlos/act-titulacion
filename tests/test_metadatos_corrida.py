"""Tests de los metadatos de trazabilidad que acompañan a cada corrida de CV.

Con ~±0.01 de ruido entre corridas —el entrenamiento de Keras no es
bit-determinista aunque se fije la semilla— una cifra suelta no dice nada. Lo
que la hace defendible en el informe es poder atarla a un commit, unos datos y
unos splits concretos. Estos tests cubren esa parte, que no necesita entrenar.
"""
import numpy as np

from lsch_mr.model_trainer import ModelTrainer


def _X(n=6, semilla=0):
    return np.random.default_rng(semilla).normal(size=(n, 4, 3)).astype("float32")


def _metadatos(X=None, y=None, **kw):
    X = _X() if X is None else X
    y = np.array([0, 1, 0, 1, 0, 1], dtype="int64") if y is None else y
    args = {"etiqueta": "dominante", "esquema": "por señante", "n_splits": 2,
            "n_grupos": 3, "splits": [], **kw}
    return ModelTrainer()._metadatos_corrida(X, y, **args)


def test_incluye_todo_lo_que_hace_rastreable_una_cifra():
    m = _metadatos()
    assert set(m) >= {"timestamp_utc", "git", "versiones", "modo_manos",
                      "esquema", "n_splits", "n_grupos", "semilla",
                      "hiperparametros", "dataset", "splits"}
    assert m["modo_manos"] == "dominante"
    assert m["esquema"] == "por señante"
    assert m["n_grupos"] == 3
    assert m["semilla"] == ModelTrainer().seed


def test_registra_las_versiones_que_pueden_mover_la_cifra():
    v = _metadatos()["versiones"]
    assert v["python"].startswith("3.")
    assert v["numpy"] == np.__version__
    for paquete in ("tensorflow", "keras", "scikit-learn"):
        assert paquete in v          # None es válido si no está instalado


def test_el_sha1_del_dataset_identifica_los_datos():
    """Mismo contenido -> mismo sha1; un valor distinto -> sha1 distinto."""
    a = _metadatos(X=_X(semilla=1))["dataset"]["sha1_X"]
    b = _metadatos(X=_X(semilla=1))["dataset"]["sha1_X"]
    otro = _X(semilla=1)
    otro[0, 0, 0] += 0.5
    c = _metadatos(X=otro)["dataset"]["sha1_X"]
    assert a == b and a != c


def test_la_forma_del_dataset_queda_registrada():
    d = _metadatos(X=_X(n=6))["dataset"]
    assert (d["n_muestras"], d["seq_len"], d["n_features"]) == (6, 4, 3)
    assert d["n_clases"] == 2


def test_guarda_los_grupos_que_quedaron_fuera_en_cada_fold():
    splits = [{"fold": 1, "n_train": 4, "n_val": 2, "grupos_fuera": ["003"],
               "val_indices": [0, 1]}]
    assert _metadatos(splits=splits)["splits"] == splits


def test_git_reporta_commit_y_si_el_arbol_estaba_sucio():
    """Una corrida sobre cambios sin commitear no es reproducible: hay que verlo."""
    g = ModelTrainer._git_estado()
    assert set(g) >= {"commit", "sucio"}
    if g["commit"] is not None:       # None si se corre fuera de un repo git
        assert len(g["commit"]) == 40
        assert isinstance(g["sucio"], bool)


def test_el_timestamp_es_utc_iso():
    ts = _metadatos()["timestamp_utc"]
    from datetime import datetime
    assert datetime.fromisoformat(ts).utcoffset().total_seconds() == 0
