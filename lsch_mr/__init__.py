"""
Paquete LSCh-MR — pipeline de IA / datos (parte de Juan Yampara).

Reúne los componentes runtime y offline definidos en el diseño
(CONTEXTO_PROYECTO.md, Sección 10.2). Los nombres de clases y métodos clave
respetan el diagrama de clases para mantener la trazabilidad diseño↔código.
"""
from __future__ import annotations

from . import config
from .caracteristicas import preparar_entrada, secuencia_a_features
from .keypoint_normalizer import KeypointNormalizer
from .message_composer import MessageComposer
from .rest_state_detector import RestStateDetector
from .tipos import (ClassResult, Frame, HandFrame, MultiHandFrame, NormVector,
                    SignEvent, SignEventType, SignSequence)

__all__ = [
    "config",
    "KeypointNormalizer",
    "RestStateDetector",
    "MessageComposer",
    "preparar_entrada",
    "secuencia_a_features",
    "ClassResult",
    "Frame",
    "NormVector",
    "SignSequence",
    "HandFrame",
    "MultiHandFrame",
    "SignEvent",
    "SignEventType",
    # Los siguientes hacen import diferido de dependencias pesadas
    # (mediapipe/tensorflow/onnx); se importan bajo demanda desde los scripts:
    #   HandTrackingProvider  -> lsch_mr.hand_tracking_provider
    #   DatasetRecorder       -> lsch_mr.dataset_recorder
    #   ModelTrainer          -> lsch_mr.model_trainer
    #   ModelExporter         -> lsch_mr.model_exporter
    #   SignClassifier        -> lsch_mr.sign_classifier
]

__version__ = "0.1.0"
