"""
MonitorRecursos — instrumentación de uso de recursos (fuera del diagrama de
clases del diseño).

El diseño fija cuatro métricas objetivo del MVP (CONTEXTO_PROYECTO.md Sección
3: accuracy, latencia end-to-end, FPS de renderizado, task success rate).
Ninguna es "uso de CPU/memoria": no hay un umbral de aprobación definido para
recursos. Este módulo no lo inventa — solo muestrea y deja constancia
estructurada (JSON + CSV) para que el dato exista cuando haya que presentarlo
(p. ej. para argumentar viabilidad de ejecución on-device).

`lector` y `reloj` son inyectables (mismo patrón que `MessageComposer`) para
poder testear sin depender de psutil ni de tiempo real; `para_proceso_actual`
es la fábrica que sí usa psutil, pensada para demo_vivo.py.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Optional


class MonitorRecursos:
    def __init__(self,
                 lector: Callable[[], tuple[float, float]],
                 n_cpus: int = 1,
                 intervalo_s: float = 1.0,
                 reloj: Callable[[], float] = time.monotonic) -> None:
        self._lector = lector          # () -> (cpu_pct de 1 core, rss_bytes)
        self._n_cpus = max(1, n_cpus)
        self._intervalo = intervalo_s
        self._reloj = reloj
        self._inicio = reloj()
        self._ultimo_tick = self._inicio
        self._muestras: list[dict] = []

    @classmethod
    def para_proceso_actual(cls, intervalo_s: float = 1.0) -> "MonitorRecursos":
        """Fábrica real: envuelve `psutil.Process()` del proceso en ejecución."""
        import psutil

        proceso = psutil.Process()
        # La primera lectura de psutil.cpu_percent() es siempre 0.0 (no hay
        # intervalo previo contra el que medir) — se descarta aquí para que la
        # primera muestra real ya sea significativa.
        proceso.cpu_percent(interval=None)
        n_cpus = psutil.cpu_count(logical=True) or 1

        def _lector() -> tuple[float, float]:
            return proceso.cpu_percent(interval=None), float(proceso.memory_info().rss)

        return cls(lector=_lector, n_cpus=n_cpus, intervalo_s=intervalo_s)

    def tick(self) -> bool:
        """Toma una muestra si ya pasó `intervalo_s` desde la última.

        Pensada para llamarse una vez por frame: si todavía no toca muestrear
        el costo es solo comparar dos floats. Devuelve True si se registró
        una muestra nueva.
        """
        ahora = self._reloj()
        if ahora - self._ultimo_tick < self._intervalo:
            return False
        self._ultimo_tick = ahora
        cpu_pct, rss_bytes = self._lector()
        self._muestras.append({
            "t_s": round(ahora - self._inicio, 2),
            "cpu_pct": round(cpu_pct, 1),
            "cpu_pct_normalizado": round(cpu_pct / self._n_cpus, 1),
            "memoria_rss_mb": round(rss_bytes / (1024 ** 2), 1),
        })
        return True

    @property
    def muestras(self) -> list[dict]:
        return list(self._muestras)

    def resumen(self) -> dict:
        """Estadísticos agregados. Sin muestras -> solo metadatos de sesión."""
        base = {
            "n_muestras": len(self._muestras),
            "duracion_s": round(self._reloj() - self._inicio, 2),
            "n_cpus_logicos": self._n_cpus,
        }
        if not self._muestras:
            return base
        cpu = [m["cpu_pct_normalizado"] for m in self._muestras]
        mem = [m["memoria_rss_mb"] for m in self._muestras]
        base["cpu_pct_normalizado"] = _stats(cpu)
        base["memoria_rss_mb"] = _stats(mem)
        return base

    def guardar_csv(self, ruta: Path) -> None:
        """Serie temporal completa (una fila por muestra), para graficar."""
        if not self._muestras:
            return
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self._muestras[0].keys()))
            w.writeheader()
            w.writerows(self._muestras)


def _stats(valores: list[float]) -> dict:
    return {
        "media": round(sum(valores) / len(valores), 1),
        "min": round(min(valores), 1),
        "max": round(max(valores), 1),
    }
