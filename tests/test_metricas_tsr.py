"""Tests del Task Success Rate (métrica 4 del MVP)."""
import pytest

from lsch_mr import metricas_tsr as tsr


def _registros(exitos: int, total: int, mismo_participante: bool = False):
    return [{"participante": "P01" if mismo_participante else f"P{i + 1:02d}",
             "secuencia_id": f"S{(i % 5) + 1}",
             "exito": i < exitos}
            for i in range(total)]


def test_tsr_basico():
    r = tsr.calcular_tsr(_registros(8, 10))
    assert r["n_pruebas"] == 10
    assert r["n_exitos"] == 8
    assert r["tsr"] == pytest.approx(0.8)
    assert r["cumple"] is True
    assert r["alertas"] == []


def test_bajo_el_umbral_no_cumple():
    r = tsr.calcular_tsr(_registros(7, 10))
    assert r["tsr"] == pytest.approx(0.7)
    assert r["cumple"] is False


def test_pocas_pruebas_no_cumple_aunque_el_porcentaje_sea_perfecto():
    """4/4 = 100% pero el diseño exige un mínimo de 10 pruebas."""
    r = tsr.calcular_tsr(_registros(4, 4))
    assert r["tsr"] == 1.0
    assert r["cumple_umbral_sin_validar_protocolo"] is True
    assert r["cumple"] is False
    assert any("mínimo de 10" in a for a in r["alertas"])


def test_diez_pruebas_a_la_misma_persona_no_cumple():
    """El diseño pide 10 participantes DISTINTOS, no 10 repeticiones."""
    r = tsr.calcular_tsr(_registros(10, 10, mismo_participante=True))
    assert r["tsr"] == 1.0
    assert r["n_participantes"] == 1
    assert r["cumple"] is False
    assert any("distinto" in a for a in r["alertas"])
    assert any("más de una prueba" in a for a in r["alertas"])


def test_participante_sin_identificar_genera_alerta():
    registros = _registros(10, 10)
    registros[3]["participante"] = ""
    r = tsr.calcular_tsr(registros)
    assert any("sin identificador" in a for a in r["alertas"])


def test_wilson_es_asimetrico_y_no_se_sale_de_rango():
    bajo, alto = tsr.wilson(10, 10)
    assert 0.0 <= bajo <= alto <= 1.0
    # Con 10/10, Wald daría [1.0, 1.0] (±0). Wilson mantiene incertidumbre.
    assert bajo < 1.0


def test_wilson_sin_datos_es_ignorancia_total():
    assert tsr.wilson(0, 0) == (0.0, 1.0)


def test_ic_de_8_de_10_no_respalda_el_objetivo():
    """8/10 = 80% justo, pero el IC baja muy por debajo: con n=10 el resultado
    es compatible con un sistema que no cumple. Es el matiz que el informe
    tiene que declarar."""
    r = tsr.calcular_tsr(_registros(8, 10))
    assert r["cumple"] is True
    assert r["ic95_respalda_objetivo"] is False
    assert r["ic95"][0] < 0.80


def test_ic_estrecho_con_muestra_grande_si_respalda():
    r = tsr.calcular_tsr(_registros(95, 100))
    assert r["ic95"][0] >= 0.80
    assert r["ic95_respalda_objetivo"] is True


def test_desglose_por_secuencia_localiza_la_que_falla():
    registros = [
        {"participante": "P01", "secuencia_id": "S1", "exito": True},
        {"participante": "P02", "secuencia_id": "S1", "exito": True},
        {"participante": "P03", "secuencia_id": "S2", "exito": False},
        {"participante": "P04", "secuencia_id": "S2", "exito": False},
    ]
    por = {d["secuencia_id"]: d for d in tsr.desglose_por_secuencia(registros)}
    assert por["S1"]["tsr"] == 1.0
    assert por["S2"]["tsr"] == 0.0


def test_comentarios_ignora_los_vacios():
    registros = [
        {"participante": "P01", "exito": True, "comentarios": "  "},
        {"participante": "P02", "exito": False, "comentarios": "texto muy chico"},
        {"participante": "P03", "exito": True},
    ]
    c = tsr.comentarios_cualitativos(registros)
    assert len(c) == 1
    assert c[0]["participante"] == "P02"
    assert c[0]["exito"] is False


def test_sin_registros_no_divide_por_cero():
    r = tsr.calcular_tsr([])
    assert r["n_pruebas"] == 0
    assert r["tsr"] == 0.0
    assert r["cumple"] is False
