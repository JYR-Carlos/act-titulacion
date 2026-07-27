"""Tests de ModelTrainer.cargar_grupos: extracción del señante desde sample_id.

Importa porque de esto depende que la validación cruzada por señante agrupe
bien: si los grupos salen mal, el fold deja de ser independiente del señante y
la métrica vuelve a estar inflada sin que nada falle de forma visible.

La segunda mitad del archivo cubre `verificar_grupos`, que es la guarda contra
ese fallo silencioso: pasar el número de campo equivocado no daba ningún error.
"""
from pathlib import Path

import numpy as np
import pytest

from lsch_mr.model_trainer import ModelTrainer

LSA64_NPZ = Path(__file__).resolve().parent.parent / "data" / "processed" / "lsa64_dominante.npz"


def _corpus_lsa64(n_clases=10, n_senantes=10, n_reps=5):
    """`sample_id`s con la forma real de LSA64: clase_senante_repeticion.

    Un cruce completo, que es justo el caso difícil: el campo del señante y el
    de la repetición parten el corpus en grupos que cubren todas las clases, así
    que solo se distinguen por cardinalidad.
    """
    clases = ["017", "027", "038", "039", "040",
              "050", "051", "053", "056", "063"][:n_clases]
    sids, y = [], []
    for ci, c in enumerate(clases):
        for s in range(1, n_senantes + 1):
            for r in range(1, n_reps + 1):
                sids.append(f"{c}_{s:03d}_{r:03d}")
                y.append(ci)
    return sids, np.asarray(y, dtype="int64")


def _grupos_por_campo(sids, campo):
    return np.array([s.split("_")[campo - 1] for s in sids], dtype=object)


def _verificar(sids, y, campo, **kw):
    return ModelTrainer.verificar_grupos(
        sids, _grupos_por_campo(sids, campo), y, str(campo), **kw)


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


# --------------------------------------------------------------------------- #
# verificar_grupos: la guarda contra agrupar por el campo equivocado
# --------------------------------------------------------------------------- #

def test_el_campo_del_senante_da_diez_grupos():
    """El caso bueno: LSA64 tiene 10 señantes y el señante es el campo 2."""
    sids, y = _corpus_lsa64()
    perfil = _verificar(sids, y, 2)
    assert perfil["n_grupos"] == 10
    assert perfil["tam_min"] == perfil["tam_max"] == 50


def test_imprime_siempre_el_diagnostico(capsys):
    """Sin esta huella en la salida, un --cv-grupos mal puesto no se nota."""
    sids, y = _corpus_lsa64()
    _verificar(sids, y, 2)
    salida = capsys.readouterr().out
    assert "campo 2 del sample_id -> 10 grupos" in salida
    assert "'001', '002', '003'" in salida          # tres ejemplos de grupo
    assert "min=50" in salida and "max=50" in salida
    assert "(500 en total)" in salida


def test_agrupar_por_la_clase_aborta():
    """Campo 1 = la glosa: 10 grupos, pero cada uno con una sola clase."""
    sids, y = _corpus_lsa64()
    with pytest.raises(SystemExit, match="GLOSA"):
        _verificar(sids, y, 1)


def test_agrupar_por_la_repeticion_aborta_y_senala_el_campo_bueno():
    """Campo 3 = repetición: 5 grupos plausibles, pero el 2 da 10 (más finos)."""
    sids, y = _corpus_lsa64()
    with pytest.raises(SystemExit, match=r"campo 2 produce 10"):
        _verificar(sids, y, 3)


def test_forzar_desactiva_la_heuristica_del_campo_dominado():
    sids, y = _corpus_lsa64()
    perfil = _verificar(sids, y, 3, forzar=True)
    assert perfil["n_grupos"] == 5


def test_un_solo_grupo_aborta():
    sids, y = _corpus_lsa64(n_senantes=1)
    with pytest.raises(SystemExit, match="un solo grupo"):
        _verificar(sids, y, 2)


def test_un_grupo_por_muestra_aborta():
    """Agrupar por un identificador único equivale a no agrupar."""
    sids = [f"017_{i:04d}" for i in range(60)]
    y = np.zeros(60, dtype="int64")
    with pytest.raises(SystemExit, match="uno por muestra"):
        _verificar(sids, y, 2)


def test_demasiados_grupos_aborta_salvo_con_forzar():
    sids, y = _corpus_lsa64(n_senantes=60, n_reps=2)
    with pytest.raises(SystemExit, match="máximo plausible 50"):
        _verificar(sids, y, 2)
    assert _verificar(sids, y, 2, forzar=True)["n_grupos"] == 60


def test_el_contador_de_grabar_corpus_no_compite_como_candidato():
    """`s01_0001`: el campo 2 es un contador global, no un rival del campo 1.

    Es la convención de `grabar_corpus.py`. Si la heurística del campo dominado
    lo tomara por un candidato mejor, rechazaría el agrupamiento correcto.
    """
    sids, y = [], []
    for s in range(1, 4):                       # 3 señantes
        for i in range(1, 31):                  # 30 tomas cada uno
            sids.append(f"s{s:02d}_{i:04d}")
            y.append(i % 10)
    perfil = _verificar(sids, np.asarray(y, dtype="int64"), 1)
    assert perfil["n_grupos"] == 3


def test_sample_ids_con_distinto_numero_de_campos_no_rompen_el_diagnostico():
    """Corpus mezclado: la tabla de candidatos solo mira los campos comunes."""
    sids = [f"017_{s:03d}_{r:03d}" if s % 2 else f"017_{s:03d}"
            for s in range(1, 11) for r in range(1, 6)]
    y = np.asarray([i % 10 for i in range(len(sids))], dtype="int64")
    assert _verificar(sids, y, 2)["n_grupos"] == 10


@pytest.mark.skipif(not LSA64_NPZ.exists(), reason="falta data/processed/lsa64_dominante.npz")
def test_dataset_real_de_lsa64_da_diez_senantes():
    """Criterio de aceptación sobre el corpus de verdad, no sobre un fixture."""
    sids = ModelTrainer.cargar_sample_ids(LSA64_NPZ)
    _, y, _ = ModelTrainer.cargar_dataset(LSA64_NPZ)
    grupos = ModelTrainer.cargar_grupos(LSA64_NPZ, "2")
    assert ModelTrainer.verificar_grupos(sids, grupos, y, "2")["n_grupos"] == 10
    with pytest.raises(SystemExit):
        ModelTrainer.verificar_grupos(sids, _grupos_por_campo(sids, 1), y, "1")
    with pytest.raises(SystemExit):
        ModelTrainer.verificar_grupos(sids, _grupos_por_campo(sids, 3), y, "3")
