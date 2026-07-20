"""
SignClassifier — Capa 3 (Inferencia).

Clasifica una SignSequence con el modelo ONNX (Sección 10.2). Aplica el umbral
de confianza (`confThreshold`): por debajo del umbral la seña se considera
"fuera de vocabulario" (flujo alternativo de CU-01).

En PC se usa onnxruntime para validar el pipeline end-to-end; en Quest 3 el
mismo modelo.onnx corre con Unity Sentis. El preprocesamiento e inferencia son
idénticos — solo cambia el runtime (Sección 8.2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .caracteristicas import preparar_entrada
from .tipos import ClassResult

_DESCONOCIDA = "<desconocida>"


class SignClassifier:
    def __init__(self,
                 onnx_path: Path = config.OUTPUTS_MODELS_DIR / "modelo.onnx",
                 labels_path: Path = config.OUTPUTS_MODELS_DIR / "labels.json",
                 conf_threshold: float = config.CONF_THRESHOLD,
                 modo_manos: Optional[str] = None) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No existe {onnx_path}. Exporta el modelo primero "
                "(python exportar_onnx.py).")

        self.conf_threshold = conf_threshold
        meta = self._cargar_labels(Path(labels_path))
        self.classes: list[str] = meta.get("classes", [])
        self.modo_manos = modo_manos or meta.get("modo_manos", config.MODO_MANOS)
        self.seq_len = int(meta.get("seq_len", config.SEQ_LEN))

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _cargar_labels(labels_path: Path) -> dict:
        if labels_path.exists():
            return json.loads(labels_path.read_text(encoding="utf-8"))
        return {}

    # -- Contrato del diseño (Sección 10.2) --------------------------------- #
    def classify(self, seq: np.ndarray) -> ClassResult:
        """Clasifica una SignSequence cruda y devuelve un ClassResult.

        `seq` puede venir como:
          * SignSequence cruda (T,21,3) / (T,2,21,3) -> se preprocesa aquí, o
          * entrada ya preparada (seq_len, n_features).
        """
        x = np.asarray(seq, dtype=np.float32)
        # Si no viene ya como (seq_len, n_features), se preprocesa.
        if not (x.ndim == 2 and x.shape[0] == self.seq_len):
            x = preparar_entrada(x, modo=self.modo_manos, seq_len=self.seq_len)

        logits = self.session.run(None, {self.input_name: x[None, ...]})[0][0]
        scores = self._softmax_si_hace_falta(logits)

        idx = int(np.argmax(scores))
        conf = float(scores[idx])
        in_vocab = conf >= self.conf_threshold
        label = self.classes[idx] if (self.classes and in_vocab) else _DESCONOCIDA
        return ClassResult(
            label=label,
            index=idx if in_vocab else -1,
            confidence=conf, in_vocab=in_vocab, scores=scores)

    @staticmethod
    def _softmax_si_hace_falta(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).ravel()
        # El modelo ya termina en softmax; se re-normaliza por robustez.
        s = float(v.sum())
        if v.min() >= 0.0 and abs(s - 1.0) < 1e-3:
            return v
        e = np.exp(v - v.max())
        return (e / e.sum()).astype(np.float32)
