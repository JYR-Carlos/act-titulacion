"""
Configuración central del pipeline LSCh-MR (parte de IA / datos).

Concentra rutas, constantes del modelo de datos e hiperparámetros compartidos
para que todos los componentes y scripts usen exactamente los mismos valores.
Ver `CONTEXTO_PROYECTO.md` (Secciones 6 y 3) para la trazabilidad diseño↔código.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parent.parent

MODELS_DIR = RAIZ / "models"                 # aquí va hand_landmarker.task
DATA_RAW_DIR = RAIZ / "data" / "raw"         # CSV crudos (keypoints sin normalizar)
DATA_PROCESSED_DIR = RAIZ / "data" / "processed"  # dataset normalizado (.npz)
OUTPUTS_MODELS_DIR = RAIZ / "outputs" / "models"   # .keras + .onnx
OUTPUTS_REPORTS_DIR = RAIZ / "outputs" / "reports"  # matriz de confusión, métricas

# Modelo de landmarks de MediaPipe Tasks (HandLandmarker). NO viene con pip.
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"

# Catálogo de glosas (las 10 señas del corpus cerrado)
GLOSAS_CSV = RAIZ / "Glosas_LSCh_Mappeadas.csv"

for _d in (MODELS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR,
           OUTPUTS_MODELS_DIR, OUTPUTS_REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Modelo de datos (CONTEXTO_PROYECTO.md, Sección 6)
# --------------------------------------------------------------------------- #
NUM_LANDMARKS = 21          # keypoints por mano (estándar MediaPipe / Meta XR SDK)
NUM_EJES = 3                # (x, y, z)
NORMVECTOR_DIM = NUM_LANDMARKS * NUM_EJES  # 63 → salida de KeypointNormalizer

WRIST_IDX = 0               # L0 — muñeca (centro de normalización)
MIDDLE_MCP_IDX = 9          # L9 — base del dedo medio (referencia de escala)

MAX_SEQ_FRAMES = 60         # ventana temporal máxima de una seña (SignSequence)

# --------------------------------------------------------------------------- #
# Preprocesamiento de secuencia para el clasificador
# --------------------------------------------------------------------------- #
# Toda SignSequence (1..60 frames) se remuestrea temporalmente a esta longitud
# fija antes de entrar al TCN. Da una forma de entrada fija (Sentis prefiere
# formas estáticas) y normaliza la duración de la seña.
SEQ_LEN = 60

# Modo de manos: "dominante" -> 63 dim (una mano) | "ambas" -> 126 dim (dos manos).
# PUNTO ABIERTO del equipo (Sección 6). Configurable, sin forzar la decisión.
MODO_MANOS = "dominante"

def n_features(modo: str | None = None) -> int:
    """Dimensión del vector de características por frame según el modo de manos."""
    modo = modo or MODO_MANOS
    return NORMVECTOR_DIM * (2 if modo == "ambas" else 1)

# --------------------------------------------------------------------------- #
# Detección de reposo (RestStateDetector, Sección 8)
# --------------------------------------------------------------------------- #
# Movimiento medio por landmark entre frames, escalado por la distancia L0–L9
# (invariante a escala). Por debajo del umbral se considera "reposo".
REST_UMBRAL_MOVIMIENTO = 0.08   # unidades relativas al tamaño de la mano
REST_FRAMES_INICIO = 3          # frames de movimiento sostenido para iniciar captura
REST_FRAMES_FIN = 6             # frames de reposo sostenido para cerrar la seña
REST_MIN_FRAMES_SENA = 5        # secuencias más cortas se descartan como ruido

# --------------------------------------------------------------------------- #
# Clasificación (SignClassifier)
# --------------------------------------------------------------------------- #
CONF_THRESHOLD = 0.60           # bajo este umbral -> "fuera de vocabulario"

# --------------------------------------------------------------------------- #
# Entrenamiento (ModelTrainer) / TCN
# --------------------------------------------------------------------------- #
TCN_FILTROS = 64
TCN_KERNEL = 3
TCN_DILATACIONES = (1, 2, 4, 8)   # RF ≈ 1 + 2*(k-1)*sum = 61 ≥ 60 frames
TCN_DROPOUT = 0.2

ENTRENAMIENTO_EPOCHS = 120
ENTRENAMIENTO_BATCH = 16
ENTRENAMIENTO_LR = 1e-3
ENTRENAMIENTO_VAL_SPLIT = 0.2
SEMILLA = 42

# --------------------------------------------------------------------------- #
# Exportación ONNX (ModelExporter)
# --------------------------------------------------------------------------- #
ONNX_OPSET = 13   # opset conservador, bien soportado por Unity Sentis
