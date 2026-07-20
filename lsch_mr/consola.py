"""Configuración de la consola para salida UTF-8 en Windows.

La consola de Windows suele usar cp1252, que no puede codificar algunos
caracteres (p. ej. flechas o símbolos matemáticos) y hace fallar `print(...)`.
Esta utilidad fuerza UTF-8 en stdout/stderr de forma segura.
"""
from __future__ import annotations

import sys


def configurar_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
