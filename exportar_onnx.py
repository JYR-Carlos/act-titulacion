"""
exportar_onnx.py — CLI del ModelExporter (CU-02).

Exporta el modelo Keras a ONNX (runtime consolidado: Unity Sentis) y valida la
compatibilidad de operadores.

Uso:
    python exportar_onnx.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lsch_mr import config
from lsch_mr.consola import configurar_utf8
from lsch_mr.model_exporter import ModelExporter

configurar_utf8()


def main() -> int:
    ap = argparse.ArgumentParser(description="Exportación a ONNX (ModelExporter)")
    ap.add_argument("--keras", default=str(config.OUTPUTS_MODELS_DIR / "tcn_lsch.keras"))
    ap.add_argument("--onnx", default=str(config.OUTPUTS_MODELS_DIR / "modelo.onnx"))
    ap.add_argument("--opset", type=int, default=config.ONNX_OPSET)
    args = ap.parse_args()

    exporter = ModelExporter(opset=args.opset)
    try:
        res = exporter.export(Path(args.keras), Path(args.onnx))
    except FileNotFoundError as e:
        # Falta el .keras: es lo normal en un clone limpio, no un bug. El repo
        # trae el .onnx ya exportado, así que la mayoría no necesita este paso.
        raise SystemExit(
            f"\n[exportar_onnx] {e}\n"
            "  El repo ya incluye outputs/models/modelo.onnx: solo hace falta "
            "reexportar si reentrenaste.")
    except Exception as e:
        # tf2onnx + Keras 3 es la combinación frágil del entorno; si falla, lo
        # que hay que mirar son las versiones, no el código.
        raise SystemExit(
            f"\n[exportar_onnx] Falló la exportación: {type(e).__name__}: {e}\n"
            "  Comprueba que tensorflow y tf2onnx son las versiones pineadas "
            "en requirements.txt: la exportación es lo primero que rompe una "
            "versión más nueva de TensorFlow.")

    print("\n===== EXPORTACIÓN ONNX =====")
    print(f"  archivo        : {res.path}")
    print(f"  opset          : {res.opset}")
    print(f"  operadores     : {', '.join(res.operadores)}")
    print(f"  sentisCompatible: {res.sentisCompatible}")
    if not res.sentisCompatible:
        print(f"  [!] operadores a revisar para Sentis: {res.no_soportados}")
    else:
        print("  [OK] Todos los operadores estan en la lista soportada por Sentis.")
    print("\nEste modelo.onnx es el único punto de acoplamiento con el runtime "
          "de Unity (deploy a Tomás).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
