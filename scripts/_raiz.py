"""
Deja la raíz del repositorio en `sys.path`.

Los scripts de esta carpeta se ejecutan como `python scripts/demo_vivo.py`, y en
ese caso `sys.path[0]` es `scripts/`, no la raíz: sin esto `import lsch_mr` falla
con ModuleNotFoundError. Importar este módulo antes que cualquier import del
proyecto es lo que lo arregla.

No hace falta bajo pytest —`pytest.ini` ya declara `pythonpath = . scripts`—,
pero importarlo dos veces no cuesta nada: la comprobación evita duplicar la ruta.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
