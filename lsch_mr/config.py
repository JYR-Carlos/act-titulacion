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

# Frames SIN mano detectada tolerados antes de dar la mano por perdida.
# running_mode="image" (el oficial, ver INTEGRACION_UNITY.md sección 7) redetecta
# la palma en cada frame sin arrastrar ROI del frame anterior, así que un
# parpadeo de 1-2 frames (motion blur, ángulo, borde del cuadro) es variación
# normal, no que la mano se haya ido de verdad.
#
# Rige en los DOS estados del RestStateDetector:
#   * capturando -> superarlo descarta la secuencia en curso (evento DISCARDED).
#     Con tolerancia 0 (previo a 2026-07-27) el parpadeo tiraba la seña entera.
#   * reposo     -> superarlo borra el candidato de inicio. Que aquí la
#     tolerancia fuera 0 (previo a 2026-07-26) es lo que hacía que con detección
#     inestable —escena oscura— la demo no arrancara NUNCA una captura.
#
# Mismo orden de magnitud que REST_FRAMES_INICIO: alcanza para absorber el
# parpadeo sin confundirlo con una pérdida sostenida real.
REST_FRAMES_PERDIDA_MAX = 3

# --------------------------------------------------------------------------- #
# Clasificación (SignClassifier)
# --------------------------------------------------------------------------- #
# Bajo este umbral la seña se reporta como "fuera de vocabulario" (FA-01).
#
# Calibrado sobre las predicciones out-of-fold de la validación por señante
# (483 muestras, LSA64). Cifras de la corrida canónica del 2026-07-26,
# `outputs/reports/cv_metrics_dominante_senante.json` (0.911 ± 0.051):
#
#   umbral   cobertura   precisión   glosas erróneas mostradas
#     0.60      94.0%       92.1%       36
#     0.90      79.1%       96.3%       14     <- elegido
#     0.95      73.3%       96.0%       14
#     0.99      61.7%       96.0%       12
#
# Se elige 0.90 porque en ventanilla mostrar una glosa equivocada engaña al
# funcionario, mientras que "no reconocida" solo pide repetir la seña. Y porque
# es la rodilla de la curva: de 0.90 en adelante la precisión deja de subir
# (96.3% -> 96.0%) mientras la cobertura sigue cayendo.
#
# OJO: el modelo está MAL CALIBRADO, y peor de lo que parecía. La confianza
# mediana de sus predicciones ERRÓNEAS es 0.78 y su p95 llega a 1.00: hay fallos
# con confianza máxima. Por eso el umbral es un instrumento romo — pasar de 0.90
# a 0.99 solo quita 2 de las 14 glosas erróneas y cuesta 17 puntos de cobertura.
# No se puede comprar precisión subiendo el umbral. Corregirlo de raíz
# (temperature scaling u otra calibración) queda como trabajo futuro, y cambiaría
# el contrato con Unity porque hoy el softmax va dentro del grafo ONNX.
#
# Recalcular tras reentrenar:  python evaluar_modelo.py --fuente cv
CONF_THRESHOLD = 0.90

# --------------------------------------------------------------------------- #
# Métricas de éxito del MVP (Secciones 11 y 12 del diseño)
# --------------------------------------------------------------------------- #
# Latencia end-to-end: desde que la seña termina realmente hasta que el subtítulo
# queda dibujado. Incluye el retardo de segmentación (REST_FRAMES_FIN frames de
# reposo sostenido), que es inherente al diseño: no se puede descontar de una
# medición honesta porque el sistema no sabe que la seña terminó hasta confirmarlo.
LATENCIA_MAX_MS = 500.0

# Muestras mínimas por glosa en el corpus (Sección 12).
CORPUS_MIN_MUESTRAS_POR_CLASE = 50

# Identificador del señante en una sesión de grabación. Va como primer campo del
# `sample_id`, que es lo que permite evaluar dejando señantes fuera
# (`entrenar.py --cv-grupos 1`). No se puede reconstruir después de grabar.
SENANTE_POR_DEFECTO = "s01"

# Accuracy mínima de clasificación para dar el MVP por cumplido (Sección 11).
ACCURACY_OBJETIVO = 0.85

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

# Validación cruzada estratificada (k-fold) para estimar el desempeño del
# clasificador. El split 80/20 de arriba sigue produciendo el modelo exportado;
# el k-fold entrega la métrica reportable (media ± desviación estándar).
ENTRENAMIENTO_CV_FOLDS = 5

# Nombre de la arquitectura del clasificador, para trazabilidad en las métricas.
ARQUITECTURA = "TCN"

# --------------------------------------------------------------------------- #
# Exportación ONNX (ModelExporter)
# --------------------------------------------------------------------------- #
ONNX_OPSET = 13   # opset conservador, bien soportado por Unity Sentis
