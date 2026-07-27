"""
verificar_artefactos.py — comprueba que lo que viaja a Unity es coherente.

`modelo.onnx`, `labels.json` y `integracion/vectores_dorados.json` están
versionados y **viajan juntos**. Si un commit los deja desalineados, el fallo no
aparece aquí: aparece en el repo de Unity, donde el ONNX carga bien, Sentis
infiere sin excepciones y las predicciones son basura. Es el train/serve skew
que describe docs/INTEGRACION_UNITY.md sección 2, y es silencioso por naturaleza.

Este script lo convierte en un fallo ruidoso. Corre en CI y no necesita corpus,
cámara ni TensorFlow.

Uso:
    python scripts/verificar_artefactos.py
    python scripts/verificar_artefactos.py --sin-onnx    # solo los JSON (sin onnxruntime)
"""
from __future__ import annotations

import argparse
import hashlib
import json

import _raiz  # noqa: F401  (raíz del repo en sys.path; debe ir antes que lsch_mr)
from lsch_mr import config
from lsch_mr.consola import configurar_utf8

configurar_utf8()

CV_METRICS = ("cv_metrics_dominante.json", "cv_metrics_dominante_senante.json",
              "cv_metrics_ambas.json", "cv_metrics_ambas_senante.json")


class Verificador:
    """Acumula fallos en vez de abortar en el primero.

    Interesa el parte completo: si el modelo y las etiquetas están
    desalineados, saber además si los vectores dorados siguen vigentes cambia
    lo que hay que regenerar.
    """

    def __init__(self) -> None:
        self.fallos: list[str] = []

    def comprobar(self, condicion: bool, descripcion: str, detalle: str = "") -> bool:
        if condicion:
            print(f"  OK    {descripcion}")
        else:
            print(f"  FALLO {descripcion}" + (f"\n          {detalle}" if detalle else ""))
            self.fallos.append(descripcion)
        return bool(condicion)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verifica la coherencia de los artefactos versionados.")
    ap.add_argument("--sin-onnx", action="store_true",
                    help="no carga el modelo (no requiere onnxruntime)")
    args = ap.parse_args()

    v = Verificador()
    onnx_path = config.OUTPUTS_MODELS_DIR / "modelo.onnx"
    labels_path = config.OUTPUTS_MODELS_DIR / "labels.json"

    # -- 1. Existencia ------------------------------------------------------ #
    print("\n[1/4] Artefactos presentes")
    hay_onnx = v.comprobar(onnx_path.exists(), f"{onnx_path.name} existe",
                           f"falta {onnx_path}; expórtalo con exportar_onnx.py")
    hay_labels = v.comprobar(labels_path.exists(), f"{labels_path.name} existe",
                             f"falta {labels_path}")
    if not (hay_onnx and hay_labels):
        print("\nFALLO: sin modelo y etiquetas no hay nada que verificar.")
        return 1

    # -- 2. labels.json ----------------------------------------------------- #
    print("\n[2/4] labels.json es coherente consigo mismo")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    campos = ("classes", "modo_manos", "seq_len", "n_features")
    if not v.comprobar(all(c in labels for c in campos),
                       "declara classes, modo_manos, seq_len y n_features",
                       f"tiene {sorted(labels)}"):
        return 1

    clases = labels["classes"]
    modo = labels["modo_manos"]
    esperado = config.NORMVECTOR_DIM * (2 if modo == "ambas" else 1)
    v.comprobar(modo in ("dominante", "ambas"),
                f"modo_manos = {modo!r} es un valor válido")
    v.comprobar(labels["n_features"] == esperado,
                f"n_features ({labels['n_features']}) corresponde a "
                f"modo_manos={modo!r}",
                f"para {modo!r} deberían ser {esperado}")
    v.comprobar(len(clases) == len(set(clases)) and len(clases) > 1,
                f"las {len(clases)} clases son únicas")

    # -- 3. El ONNX concuerda con labels.json ------------------------------- #
    print("\n[3/4] El grafo ONNX concuerda con labels.json")
    if args.sin_onnx:
        print("  (saltado por --sin-onnx)")
    else:
        import numpy as np
        import onnxruntime as ort

        sesion = ort.InferenceSession(str(onnx_path),
                                      providers=["CPUExecutionProvider"])
        entrada = sesion.get_inputs()[0]
        salida = sesion.get_outputs()[0]
        forma_in, forma_out = list(entrada.shape), list(salida.shape)
        print(f"        entrada {entrada.name}{forma_in} -> "
              f"salida {salida.name}{forma_out}")

        v.comprobar(len(forma_in) == 3 and forma_in[1] == labels["seq_len"]
                    and forma_in[2] == labels["n_features"],
                    f"la entrada es [batch, {labels['seq_len']}, "
                    f"{labels['n_features']}]", f"el grafo declara {forma_in}")
        v.comprobar(len(forma_out) == 2 and forma_out[1] == len(clases),
                    f"la salida tiene {len(clases)} clases",
                    f"el grafo declara {forma_out}")

        # El softmax va DENTRO del grafo: si alguien lo saca, el C# tendría que
        # aplicarlo y nadie se enteraría hasta ver confianzas absurdas.
        x = np.zeros((1, labels["seq_len"], labels["n_features"]), dtype="float32")
        probs = sesion.run(None, {entrada.name: x})[0]
        v.comprobar(abs(float(probs.sum()) - 1.0) < 1e-4,
                    "la salida ya viene pasada por Softmax (suma 1.0)",
                    f"suma {float(probs.sum()):.4f}; docs/INTEGRACION_UNITY.md §6.5 "
                    "dice que no hay que aplicar softmax dos veces en C#")

    # -- 4. Los vectores dorados siguen vigentes ---------------------------- #
    print("\n[4/4] Los casos dorados corresponden a este modelo")
    # Anclado a la raíz del repo, no al directorio de trabajo: el script se
    # lanza desde `scripts/` y también desde CI, y la ruta tiene que ser la misma.
    dorados = config.RAIZ / "integracion" / "vectores_dorados.json"
    if v.comprobar(dorados.exists(), "integracion/vectores_dorados.json existe"):
        sha1_real = hashlib.sha1(onnx_path.read_bytes()).hexdigest()
        texto = dorados.read_text(encoding="utf-8")
        v.comprobar(sha1_real in texto,
                    f"el sha1 del modelo ({sha1_real[:12]}...) aparece en los dorados",
                    "el caso 'inferencia_onnx_entrada_conocida' quedó invalidado "
                    "al reentrenar: regenera con `python scripts/generar_vectores_dorados.py` "
                    "y avisa al equipo de Unity (docs/INTEGRACION_UNITY.md §5)")

    faltan = [n for n in CV_METRICS
              if not (config.OUTPUTS_REPORTS_DIR / n).exists()]
    v.comprobar(not faltan,
                "están los cuatro cv_metrics_*.json que respaldan el informe",
                "faltan: " + ", ".join(faltan))

    # -- Veredicto ---------------------------------------------------------- #
    print()
    if v.fallos:
        print(f"FALLO: {len(v.fallos)} comprobación(es) no pasaron:")
        for f in v.fallos:
            print(f"  - {f}")
        return 1
    print("OK: los artefactos versionados son coherentes entre sí.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
