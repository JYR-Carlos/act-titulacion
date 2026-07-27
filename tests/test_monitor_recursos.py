"""Tests de MonitorRecursos: muestreo por intervalo y agregados, con lector y
reloj inyectados (no depende de psutil real ni de tiempo real)."""
from lsch_mr.monitor_recursos import MonitorRecursos


class _Reloj:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def avanzar(self, segundos: float) -> None:
        self.t += segundos


def _lector_fijo(valores):
    """Lector inyectable: entrega (cpu_pct, rss_bytes) de una lista, en orden."""
    it = iter(valores)
    return lambda: next(it)


def test_no_muestrea_antes_del_intervalo():
    reloj = _Reloj()
    mon = MonitorRecursos(lector=_lector_fijo([(10.0, 100 * 1024 ** 2)]),
                          n_cpus=1, intervalo_s=1.0, reloj=reloj)
    assert mon.tick() is False
    assert mon.muestras == []


def test_muestrea_al_cumplirse_el_intervalo():
    reloj = _Reloj()
    mon = MonitorRecursos(lector=_lector_fijo([(50.0, 200 * 1024 ** 2)]),
                          n_cpus=2, intervalo_s=1.0, reloj=reloj)
    reloj.avanzar(1.0)
    assert mon.tick() is True
    assert len(mon.muestras) == 1
    m = mon.muestras[0]
    assert m["t_s"] == 1.0
    assert m["cpu_pct"] == 50.0
    assert m["cpu_pct_normalizado"] == 25.0   # 50% de 1 core / 2 cores
    assert m["memoria_rss_mb"] == 200.0


def test_ultima_muestra_es_none_hasta_la_primera():
    reloj = _Reloj()
    mon = MonitorRecursos(lector=_lector_fijo([(10.0, 100 * 1024 ** 2)]),
                          n_cpus=1, intervalo_s=1.0, reloj=reloj)
    assert mon.ultima_muestra is None
    reloj.avanzar(1.0)
    mon.tick()
    assert mon.ultima_muestra is not None


def test_ultima_muestra_sigue_a_la_mas_reciente():
    """Lo que lee el overlay de demo_vivo.py tiene que ser el mismo dict que se
    guarda en el CSV, no una copia que pueda quedar atrás."""
    reloj = _Reloj()
    lecturas = [(10.0, 100 * 1024 ** 2), (30.0, 150 * 1024 ** 2)]
    mon = MonitorRecursos(lector=_lector_fijo(lecturas), n_cpus=2,
                          intervalo_s=1.0, reloj=reloj)
    for _ in lecturas:
        reloj.avanzar(1.0)
        mon.tick()
    assert mon.ultima_muestra == mon.muestras[-1]
    assert mon.ultima_muestra["cpu_pct_normalizado"] == 15.0  # 30% / 2 cores
    assert mon.ultima_muestra["memoria_rss_mb"] == 150.0


def test_resumen_agrega_media_min_max():
    reloj = _Reloj()
    lecturas = [(10.0, 100 * 1024 ** 2), (30.0, 150 * 1024 ** 2), (20.0, 120 * 1024 ** 2)]
    mon = MonitorRecursos(lector=_lector_fijo(lecturas), n_cpus=1,
                          intervalo_s=1.0, reloj=reloj)
    for _ in lecturas:
        reloj.avanzar(1.0)
        mon.tick()
    resumen = mon.resumen()
    assert resumen["n_muestras"] == 3
    assert resumen["cpu_pct_normalizado"] == {"media": 20.0, "min": 10.0, "max": 30.0}
    assert resumen["memoria_rss_mb"] == {"media": 123.3, "min": 100.0, "max": 150.0}


def test_resumen_vacio_sin_muestras():
    reloj = _Reloj()
    mon = MonitorRecursos(lector=_lector_fijo([]), n_cpus=1, intervalo_s=1.0, reloj=reloj)
    resumen = mon.resumen()
    assert resumen["n_muestras"] == 0
    assert "cpu_pct_normalizado" not in resumen


def test_guardar_csv_escribe_una_fila_por_muestra(tmp_path):
    reloj = _Reloj()
    mon = MonitorRecursos(lector=_lector_fijo([(5.0, 50 * 1024 ** 2), (5.0, 50 * 1024 ** 2)]),
                          n_cpus=1, intervalo_s=1.0, reloj=reloj)
    reloj.avanzar(1.0)
    mon.tick()
    reloj.avanzar(1.0)
    mon.tick()
    ruta = tmp_path / "recursos.csv"
    mon.guardar_csv(ruta)
    contenido = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(contenido) == 3  # header + 2 filas


def test_guardar_csv_no_escribe_nada_sin_muestras(tmp_path):
    reloj = _Reloj()
    mon = MonitorRecursos(lector=_lector_fijo([]), n_cpus=1, intervalo_s=1.0, reloj=reloj)
    ruta = tmp_path / "recursos.csv"
    mon.guardar_csv(ruta)
    assert not ruta.exists()
