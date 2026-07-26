# Estado actual del código — LSCh-MR (pipeline IA/datos)

> Generado el 2026-07-22 para dar contexto rápido a Claude en una nueva
> conversación. Es una **fotografía del repo en este momento**, no un documento
> de diseño — para diseño ver `CONTEXTO_PROYECTO.md` y
> `DECISION_PREPROCESAMIENTO.md` (no reabrir lo que ahí se consolida).

## Qué es este repo

Implementación en Python de la parte de **Juan Yampara** en el proyecto de
titulación (UTA): pipeline de reconocimiento de Lengua de Señas Chilena (LSCh)
— captura de keypoints con MediaPipe, normalización, dataset, entrenamiento de
un clasificador TCN y exportación a ONNX. La capa espacial (Unity/Quest, de
Tomás) vive en otro repo; el único punto de acoplamiento es `modelo.onnx`.

## Decisiones consolidadas (no reabrir)

- Clasificador **TCN** (no LSTM ni Transformer).
- Runtime **ONNX** (opset 13, `sentisCompatible=True`) consumido con **Unity
  Sentis**.
- Captura con **MediaPipe Tasks `HandLandmarker`** (API Tasks, no la legacy).
- **21 keypoints por mano**, `NormVector` de **63 dim** por mano (centrado en
  muñeca L0 + escala por distancia L0–L9).
- Preprocesamiento **implementado desde cero**, sin reutilizar
  `mvazquezgts/SWL-LSE` (`generate_features.py`): esquema de keypoints
  distinto (61 puntos con pose corporal vs. 21 solo mano), formato de salida
  distinto (GCN vs. vector plano) y licencia no verificada. Ver
  `DECISION_PREPROCESAMIENTO.md`.
- **Demo final: solo cámara** (PC + webcam / cámara de teléfono). El Meta
  Quest 3 **no** es necesario para demostrar el MVP — decisión del 2026-07-21.
  La demo corre en **Unity + Sentis sobre PC con webcam**, no en Python puro;
  `demo_vivo.py` es el banco de pruebas del pipeline en Python, no el
  entregable final. Quedan fuera del camino crítico: Meta XR SDK, medición en
  Snapdragon XR2, subtítulo espacial anclado.

## Estado del pipeline: implementado y verificado end-to-end

Todo el paquete `lsch_mr/` + los CLIs de nivel superior existen y la cadena
completa corre sin errores:

```
captura (HandLandmarker) → KeypointNormalizer → RestStateDetector →
build_dataset → ModelTrainer (TCN, Keras 3) → ModelExporter (ONNX opset 13) →
SignClassifier (onnxruntime, ~3.5 ms/inferencia)
```

- **13/13 tests pytest pasan** (`tests/test_keypoint_normalizer.py`,
  `tests/test_rest_state_detector.py`).
- El smoke test end-to-end se hizo con dataset **sintético**
  (`entrenar.py --sintetico`) y también re-extrayendo keypoints reales del
  dataset de referencia **SWL-LSE VIDEOS_REF** (Zenodo 13691887,
  `data/external/swl_lse/`) con nuestro propio `HandLandmarker`
  (`extraer_lote.py`) — validando el camino vídeo real → keypoints → dataset,
  **no** la accuracy (VIDEOS_REF es 1 muestra/clase, no es entrenable).
- **`modelo.onnx` / `outputs/models/modelo.onnx` actuales NO son el modelo
  final de LSCh.** Son el artefacto de esa validación de pipeline: 300 clases
  de vocabulario médico en español de SWL-LSE (`ABORTO`, `DIABETES`,
  `HOSPITAL`, ...), modo `ambas` manos (126 dim). Sirven para comprobar que la
  cadena mecánica funciona, no para reconocer LSCh todavía.

## Lo que falta (crítico)

- `data/raw/` y `data/processed/` están **vacíos** — el corpus real de las
  **10 glosas LSCh** (`Glosas_LSCh_Mappeadas.csv`: HOLA, GRACIAS, POR-FAVOR,
  AYUDA, SI, NO, DOCUMENTO, ESPERAR, NOMBRE, ATENCION) **no se ha grabado**.
  Sin MR de por medio, la calidad del reconocimiento real ES la demo, así que
  esto es lo más urgente del backlog.
- Punto abierto de diseño (no bloqueante, ya soportado por código): **una
  mano vs. dos manos** — configurable vía `config.MODO_MANOS` /
  `--modo dominante|ambas` en `build_dataset.py` (63 vs. 126 dim). El
  `KeypointNormalizer` no cambia; es solo ensamblado del dataset.
- `outputs/reports/` está vacío — no hay matriz de confusión ni métricas de
  un entrenamiento con corpus real todavía.

## Riesgos abiertos (lado Unity, fuera de este repo pero relevantes)

1. **Hand tracking sin Meta XR SDK**: `HandTrackingProvider` debe
   reimplementarse en Unity sobre webcam (MediaPipe en Unity). El contrato
   `getFrame(): Frame` se mantiene. Un puente Python↔Unity por red
   reabriría la Sección 3 del diseño (prohíbe MQTT/gRPC/REST) — no hacerlo.
2. **Train/serve skew**: si el `KeypointNormalizer` en C# no replica
   exactamente el de Python (centrado L0, escala L0–L9, orden `[Left|Right]`,
   mano ausente → ceros, remuestreo lineal a 60 frames), el modelo devuelve
   basura. Mitigación planeada: vectores de prueba dorados generados desde
   Python.

## Entorno / gotchas de ejecución

- Python **3.12.0** (pyenv-win), **numpy 2.3.5**. Instalado y verificado:
  mediapipe 0.10.35, opencv 4.13, tensorflow 2.21, keras 3.15, tf2onnx 1.17,
  onnxruntime 1.27, pytest. `requirements.txt` ya refleja estos pines
  (numpy>=1.26, tensorflow>=2.16 — el pin original numpy<2/TF 2.15 no
  instalaba en Python 3.12).
- **Keras 3 es el default**; tf2onnx exporta el TCN vía `from_keras`
  directamente, con fallbacks en `ModelExporter` (SavedModel, export nativo)
  por si acaso.
- **Consola Windows en cp1252**: imprimir caracteres fuera de cp1252 (≥, →,
  ✔, ⚠) lanza `UnicodeEncodeError`. Ya solucionado con
  `lsch_mr/consola.py::configurar_utf8()`, llamado en cada CLI. Igual aplica
  al **leer** JSON con caracteres especiales — usar `encoding="utf-8"`
  explícito al abrir archivos, no confiar en el default de Windows.
- En Git Bash, cuidado con `pip install "pkg>=1.2"` (el `>=` puede crear
  archivos espurios por redirección) — preferir PowerShell o comillas
  robustas.

## Working tree (sin commitear)

```
?? .vscode/           # settings.json con el intérprete pyenv del equipo
?? modelo.onnx         # copia en la raíz, idéntica a outputs/models/modelo.onnx
                        # (artefacto de validación SWL-LSE, no el modelo final)
```
Un solo commit en el repo: `b233a95 Versión inicial del proyecto LSCh-MR
(pipeline IA/datos)`.

## Mapa de archivos

```
lsch_mr/                     paquete con los componentes del diseño (Sección 10.2)
  config.py                  rutas, modelo de datos, hiperparámetros
  tipos.py                   Frame, SignSequence, NormVector, SignEvent, ClassResult
  fuente_video.py             FuenteVideo (webcam / cámara de teléfono / archivo)
  hand_tracking_provider.py   HandTrackingProvider (HandLandmarker; modos image/video)
  keypoint_normalizer.py       KeypointNormalizer
  rest_state_detector.py       RestStateDetector
  caracteristicas.py          normalización + remuestreo a SEQ_LEN=60
  csv_esquema.py               esquema CSV del corpus
  dataset_recorder.py          DatasetRecorder
  tcn.py                       arquitectura TCN (causal, dilatada, Sentis-compatible)
  model_trainer.py             ModelTrainer
  model_exporter.py            ModelExporter (ONNX + validación de operadores)
  sign_classifier.py           SignClassifier (onnxruntime)
  message_composer.py          MessageComposer
descargar_modelo.py           descarga hand_landmarker.task
extraer_keypoints.py          CLI extracción (webcam/video)
extraer_lote.py               CLI extracción por lotes (datasets de referencia)
grabar_corpus.py              CLI grabación del corpus (segmentación por reposo)
build_dataset.py              CLI crudo -> dataset normalizado (.npz)
entrenar.py                   CLI entrenamiento (soporta --sintetico)
exportar_onnx.py               CLI exportación a ONNX
demo_vivo.py                   CLI validación end-to-end en PC (banco de pruebas)
tests/                          13 tests unitarios (normalizer + rest state detector)
data/external/swl_lse/          dataset de referencia SWL-LSE (VIDEOS_REF, no entrenable)
data/raw/, data/processed/      VACÍOS — corpus real LSCh pendiente
outputs/models/                 modelo.onnx + labels actuales = artefacto SWL-LSE (300 clases), no el final
```

## Próximo paso recomendado

Grabar el corpus real de las 10 glosas LSCh (`grabar_corpus.py`, protocolo de
~2h con profesor de señas, ≥50 muestras/glosa según el objetivo del MVP),
correr `build_dataset.py` → `entrenar.py` → `exportar_onnx.py` sobre datos
reales, y validar con `demo_vivo.py` antes de portar la normalización a C#
para Unity.
