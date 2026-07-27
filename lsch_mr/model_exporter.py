"""
ModelExporter — Pipeline offline (CONTEXTO_PROYECTO.md Sección 10.2 / CU-02).

Exporta el modelo Keras entrenado a ONNX (runtime consolidado: ONNX vía Unity
Sentis — Sección 3) y valida que sus operadores sean compatibles con Sentis.

`export()`          -> escribe modelo.onnx y devuelve su ruta.
`validateOperators()` -> revisa el grafo ONNX contra una lista de operadores
                         soportados por Sentis y reporta `sentisCompatible`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# Subconjunto de operadores ONNX soportados por Unity Sentis y usados por el
# TCN. Si el grafo introduce un operador fuera de esta lista, se marca como
# potencialmente incompatible para revisión manual.
OPS_SENTIS_SOPORTADOS = {
    "Conv", "Relu", "Add", "Mul", "Sub", "Div", "Pad", "Cast",
    "MatMul", "Gemm", "Softmax", "Sigmoid", "Tanh",
    "ReduceMean", "GlobalAveragePool", "AveragePool", "MaxPool",
    "Reshape", "Transpose", "Concat", "Slice", "Squeeze", "Unsqueeze",
    "Flatten", "Identity", "Shape", "Gather", "Constant", "ConstantOfShape",
    "BatchNormalization", "Clip", "Erf",
}


@dataclass
class OnnxModel:
    """Resultado de exportar: la ruta y el veredicto de compatibilidad.

    `sentisCompatible` es False si `no_soportados` trae algún operador que
    Unity Sentis no implementa. Vale la pena saberlo aquí y no en el otro repo:
    allí el síntoma sería que el modelo no carga, sin decir por qué.
    """
    path: str
    opset: int
    sentisCompatible: bool
    operadores: list[str] = field(default_factory=list)
    no_soportados: list[str] = field(default_factory=list)


class ModelExporter:
    """Convierte el modelo Keras a ONNX y valida que Sentis pueda cargarlo.

    Usa un opset conservador (13) porque Sentis va por detrás de onnxruntime en
    cobertura de operadores. La combinación tf2onnx + Keras 3 es frágil, así
    que `export()` intenta varias rutas antes de rendirse: ver los fallbacks
    dentro del método.
    """

    def __init__(self, opset: int = config.ONNX_OPSET) -> None:
        self.opset = opset

    # -- Contrato del diseño ------------------------------------------------ #
    def export(self, keras_path: Path = config.OUTPUTS_MODELS_DIR / "tcn_lsch.keras",
               onnx_path: Path = config.OUTPUTS_MODELS_DIR / "modelo.onnx"
               ) -> OnnxModel:
        """Exporta el `.keras` a ONNX y devuelve el veredicto de operadores.

        Lanza `FileNotFoundError` si no hay modelo entrenado todavía.

        Al reexportar, el `.onnx` cambia de sha1 y eso **invalida el octavo
        caso de `integracion/vectores_dorados.json`**, que compara Sentis
        contra onnxruntime. Hay que regenerar los dorados y avisar al equipo de
        Unity (INTEGRACION_UNITY.md §5).
        """
        import tensorflow as tf

        keras_path = Path(keras_path)
        onnx_path = Path(onnx_path)
        if not keras_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo Keras {keras_path}. Entrena primero "
                "(python entrenar.py).")

        model = tf.keras.models.load_model(keras_path)

        # Se intentan varias rutas de exportación en orden. Distintas versiones
        # de TF/Keras (Keras 2 vs Keras 3) soportan unas u otras.
        errores = []
        for estrategia in (self._via_tf2onnx_keras,
                           self._via_saved_model,
                           self._via_keras_nativo):
            try:
                estrategia(model, onnx_path, tf)
                print(f"[ModelExporter] Exportado con: {estrategia.__name__}")
                return self.validateOperators(onnx_path)
            except Exception as e:  # noqa: BLE001
                errores.append(f"{estrategia.__name__}: {e}")

        raise RuntimeError(
            "No se pudo exportar a ONNX con ninguna estrategia:\n  - "
            + "\n  - ".join(errores))

    # -- Estrategias de exportación ---------------------------------------- #
    def _via_tf2onnx_keras(self, model, onnx_path: Path, tf) -> None:
        """Ruta directa con tf2onnx (estable en Keras 2)."""
        import tf2onnx
        entrada = model.inputs[0]
        spec = (tf.TensorSpec([None] + list(entrada.shape[1:]),
                              tf.float32, name="input"),)
        tf2onnx.convert.from_keras(
            model, input_signature=spec, opset=self.opset,
            output_path=str(onnx_path))

    def _via_saved_model(self, model, onnx_path: Path, tf) -> None:
        """Exporta a SavedModel y luego convierte (robusto en Keras 3)."""
        import tempfile

        import tf2onnx
        with tempfile.TemporaryDirectory() as tmp:
            if hasattr(model, "export"):          # Keras 3
                model.export(tmp)
            else:                                  # Keras 2
                tf.saved_model.save(model, tmp)
            tf2onnx.convert.from_saved_model(
                tmp, opset=self.opset, output_path=str(onnx_path))

    def _via_keras_nativo(self, model, onnx_path: Path, tf) -> None:
        """Exportación ONNX nativa de Keras 3 (versiones recientes)."""
        model.export(str(onnx_path), format="onnx")

    def validateOperators(self, onnx_path: Path = config.OUTPUTS_MODELS_DIR / "modelo.onnx"
                          ) -> OnnxModel:
        """Inspecciona el grafo ONNX y verifica compatibilidad con Sentis."""
        import onnx

        onnx_path = Path(onnx_path)
        model = onnx.load(str(onnx_path))
        try:
            onnx.checker.check_model(model)
        except Exception as e:  # pragma: no cover
            print(f"[ModelExporter] Aviso: onnx.checker reportó: {e}")

        ops = sorted({n.op_type for n in model.graph.node})
        no_soportados = [o for o in ops if o not in OPS_SENTIS_SOPORTADOS]
        opset = model.opset_import[0].version if model.opset_import else self.opset
        return OnnxModel(
            path=str(onnx_path), opset=int(opset),
            sentisCompatible=(len(no_soportados) == 0),
            operadores=ops, no_soportados=no_soportados)
