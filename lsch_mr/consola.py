"""Configuración de la consola para salida UTF-8 en Windows.

La consola de Windows suele usar cp1252, que no puede codificar algunos
caracteres (p. ej. flechas o símbolos matemáticos) y hace fallar `print(...)`.
Esta utilidad fuerza UTF-8 en stdout/stderr de forma segura.

También expone coloreado ANSI mínimo, que usan los scripts de evaluación para
**destacar en rojo una métrica que no alcanza su umbral** (Sección 11 del
diseño). El color se apaga solo cuando la salida está redirigida a un archivo o
cuando el entorno pide no colorear: un informe no debe quedar sembrado de
secuencias de escape.
"""
from __future__ import annotations

import os
import sys


def configurar_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    _habilitar_ansi()


def _habilitar_ansi() -> None:
    """Activa el procesamiento de secuencias ANSI en la consola de Windows.

    Windows Terminal ya las interpreta, pero `cmd.exe` y algunos hosts antiguos
    necesitan ENABLE_VIRTUAL_TERMINAL_PROCESSING. Si algo falla se ignora: sin
    color el texto sigue siendo legible, con basura de escape no.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32          # type: ignore[attr-defined]
        for handle in (-11, -12):                  # STD_OUTPUT, STD_ERROR
            modo = ctypes.c_uint32()
            h = kernel32.GetStdHandle(handle)
            if kernel32.GetConsoleMode(h, ctypes.byref(modo)):
                kernel32.SetConsoleMode(h, modo.value | 0x0004)
    except Exception:
        pass


def color_activo() -> bool:
    """¿Conviene emitir secuencias ANSI?

    No, si la salida está redirigida (un `> reporte.txt` quedaría ilegible) o si
    el entorno lo desactiva explícitamente: `NO_COLOR` es la convención estándar
    y `LSCH_MR_COLOR=0` permite apagarlo solo para este proyecto.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("LSCH_MR_COLOR") == "0":
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _envolver(texto: str, codigo: str) -> str:
    return f"\033[{codigo}m{texto}\033[0m" if color_activo() else texto


def rojo(texto: str) -> str:
    return _envolver(texto, "1;31")


def verde(texto: str) -> str:
    return _envolver(texto, "1;32")


def amarillo(texto: str) -> str:
    return _envolver(texto, "1;33")


def negrita(texto: str) -> str:
    return _envolver(texto, "1")
