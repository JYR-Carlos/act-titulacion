#!/usr/bin/env python3
"""
assembler_repo.py — LSCh-MR
============================
Recolecta todos los archivos de código fuente del repositorio y los empaqueta
en un único archivo de texto listo para subir a claude.ai.

Uso:
    python tools/assembler_repo.py --repo .

Opciones:
    --repo      Directorio raíz del repositorio (obligatorio)
    --output    Nombre del archivo de salida (default: codigo_repo.txt)

No requiere dependencias externas ni API key.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ─── Configuración ────────────────────────────────────────────────────────────

SOURCE_EXTENSIONS = {
    ".py", ".cs", ".js", ".ts",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
}

IGNORE_DIRS = {
    ".git", ".github", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".idea", ".vscode",
    # Unity
    "Library", "Logs", "Temp", "UserSettings", "obj", "bin",
    ".vs", "Packages",
}

MAX_FILE_SIZE = 80_000  # 80 KB por archivo — evita binarios grandes


# ─── Lógica principal ─────────────────────────────────────────────────────────

def collect_files(repo: Path) -> list[dict]:
    found = []
    for root, dirs, filenames in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for name in sorted(filenames):
            path = Path(root) / name
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if path.stat().st_size > MAX_FILE_SIZE:
                print(f"  [SKIP] Archivo muy grande: {path.relative_to(repo)}", file=sys.stderr)
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                found.append({"path": str(path.relative_to(repo)), "content": content})
            except Exception as e:
                print(f"  [WARN] No se pudo leer {path}: {e}", file=sys.stderr)
    return found


def assemble(repo: Path, files: list[dict]) -> str:
    lines = []

    # Encabezado
    lines.append("=" * 70)
    lines.append("CÓDIGO FUENTE DEL REPOSITORIO LSCh-MR")
    lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Repo:     {repo}")
    lines.append(f"Archivos: {len(files)}")
    lines.append("=" * 70)
    lines.append("")

    # Índice de archivos
    lines.append("ÍNDICE DE ARCHIVOS")
    lines.append("-" * 40)
    for i, f in enumerate(files, 1):
        lines.append(f"  {i:>3}. {f['path']}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    # Contenido archivo por archivo
    for f in files:
        ext = Path(f["path"]).suffix.lstrip(".")
        lines.append(f"### ARCHIVO: {f['path']}")
        lines.append(f"```{ext}")
        lines.append(f["content"])
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Empaqueta el código del repo en un único archivo para evaluar en claude.ai",
    )
    parser.add_argument("--repo", required=True, help="Directorio raíz del repositorio")
    parser.add_argument("--output", default="codigo_repo.txt", help="Archivo de salida")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        print(f"[ERROR] No existe el directorio: {repo}", file=sys.stderr)
        sys.exit(1)

    print(f"\n🔍  Escaneando: {repo}")
    files = collect_files(repo)

    if not files:
        print("[WARN] No se encontraron archivos de código fuente.", file=sys.stderr)
        sys.exit(1)

    print(f"    → {len(files)} archivos encontrados:")
    for f in files:
        size = len(f["content"])
        print(f"       {f['path']}  ({size:,} chars)")

    output = Path(args.output)
    output.write_text(assemble(repo, files), encoding="utf-8")
    total_chars = output.stat().st_size

    print(f"\n✅  Archivo generado: {output.resolve()}")
    print(f"    Tamaño total: {total_chars:,} chars (~{total_chars // 4:,} tokens estimados)")
    print()
    print("─" * 60)
    print("PRÓXIMOS PASOS:")
    print("  1. Abre tu proyecto LSCh-MR en claude.ai")
    print(f"  2. Sube el archivo: {output.name}")
    print("  3. Pega el contenido de: tools/prompt_evaluacion_arquitectura.md")
    print("     (el bloque entre ---PROMPT--- y ---PROMPT---)")
    print("  4. Envía — la arquitectura ya está en el Project Knowledge")
    print("─" * 60)
    print()


if __name__ == "__main__":
    main()
