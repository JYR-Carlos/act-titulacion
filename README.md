# LSCh-MR — Pipeline de IA / datos

Implementación en Python del **pipeline de reconocimiento de Lengua de Señas
Chilena (LSCh)** del proyecto de titulación (UTA). Cubre la parte a cargo de
**Juan Yampara**: captura de keypoints, preprocesamiento, dataset, entrenamiento
del clasificador y exportación a ONNX para desplegar en Unity Sentis.

Este repositorio implementa las capas 1–3 y el pipeline offline del diseño
(ver `CONTEXTO_PROYECTO.md`). La capa 4 espacial en Unity/Quest es de Tomás; el
único punto de acoplamiento es `modelo.onnx`.

> **Decisiones consolidadas (no reabrir):** clasificador **TCN**, runtime
> **ONNX vía Unity Sentis**, captura con **MediaPipe Tasks (`HandLandmarker`)**,
> **21 keypoints** por mano, **NormVector de 63 dim** (centrado L0 + escala
> L0–L9). Ver Sección 3 de `CONTEXTO_PROYECTO.md` y `DECISION_PREPROCESAMIENTO.md`.

---

## Mapa componente del diseño → código

| Componente (Sección 10.2) | Archivo | Método clave |
|---|---|---|
| `HandTrackingProvider` | `lsch_mr/hand_tracking_provider.py` | `getFrame()` |
| `RestStateDetector` | `lsch_mr/rest_state_detector.py` | `update()` |
| `KeypointNormalizer` | `lsch_mr/keypoint_normalizer.py` | `normalize()` |
| `SignClassifier` | `lsch_mr/sign_classifier.py` | `classify()` |
| `MessageComposer` | `lsch_mr/message_composer.py` | `appendWord()` |
| `DatasetRecorder` | `lsch_mr/dataset_recorder.py` | `recordSession()` |
| `ModelTrainer` | `lsch_mr/model_trainer.py` | `train()` |
| `ModelExporter` | `lsch_mr/model_exporter.py` | `export()`, `validateOperators()` |
| TCN (arquitectura) | `lsch_mr/tcn.py` | `build_tcn()` |

Los nombres de clases/métodos respetan el diagrama de clases para mantener la
trazabilidad diseño↔código.

---

## Instalación

Entorno verificado: **Python 3.12 + numpy 2.x** (con `mediapipe` 0.10.x).

```powershell
# (recomendado) entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Descargar el modelo HandLandmarker (no viene con pip install mediapipe)
python descargar_modelo.py
```

> **TensorFlow** solo se necesita para **entrenar** (`entrenar.py`) y **exportar**
> (`exportar_onnx.py`). La captura, la construcción del dataset y la demo de
> inferencia (onnxruntime) funcionan sin TensorFlow.

---

## Usar la cámara del teléfono 📱 (recomendado)

Toda la CLI acepta `--fuente`, que puede ser un índice de webcam **o una URL**:

1. Instala en el teléfono **IP Webcam** (Android) o una app equivalente que
   exponga un stream MJPEG/RTSP. (iOS: apps con MJPEG/RTSP, o `DroidCam`/`Iriun`
   como webcam virtual.)
2. Conecta el teléfono y el PC a la **misma red Wi-Fi**, inicia el servidor de la
   app y anota la URL (p. ej. `http://192.168.1.42:8080`).
3. Usa la URL de video como fuente:

```powershell
python demo_vivo.py --fuente http://192.168.1.42:8080/video
python grabar_corpus.py --fuente http://192.168.1.42:8080/video
python extraer_keypoints.py --modo webcam --fuente http://192.168.1.42:8080/video
```

Con webcam local se usa el índice: `--fuente 0` (o `1`, `2`, …). Con `DroidCam`/
`Iriun` (webcam virtual) también se usa el índice correspondiente.

---

## Flujo de trabajo (CU-02 → CU-01)

```
grabar_corpus.py ──▶ data/raw/*.csv ──▶ build_dataset.py ──▶ data/processed/dataset.npz
        │                                                              │
   (segmentación por reposo,                                     entrenar.py
    RestStateDetector)                                                 │
                                                          outputs/models/tcn_lsch.keras
                                                                       │
                                                              exportar_onnx.py
                                                                       │
                                                          outputs/models/modelo.onnx ──▶ (deploy a Unity/Tomás)
                                                                       │
                                                                 demo_vivo.py  (validación end-to-end en PC)
```

### 1) Grabar el corpus (tomas continuas, segmentación automática)

```powershell
python grabar_corpus.py --fuente 0
```
Teclas `0..9` eligen la glosa activa (según `Glosas_LSCh_Mappeadas.csv`); haz la
seña y vuelve a reposo: la secuencia se guarda sola. `q`/`ESC` termina.

### 2) Construir el dataset normalizado

```powershell
python build_dataset.py                 # lee data/raw/*.csv, modo dominante (63 dim)
python build_dataset.py --modo ambas    # dos manos (126 dim)
python build_dataset.py --estricto      # falla si una glosa queda bajo 50 muestras
```
Avisa cuando alguna glosa no llega al mínimo de muestras del diseño
(`config.CORPUS_MIN_MUESTRAS_POR_CLASE`); con `--estricto` no guarda el dataset.
Es más barato descubrirlo aquí que después de entrenar.

### 3) Entrenar el TCN

```powershell
python entrenar.py                      # usa data/processed/dataset.npz
python entrenar.py --cv                 # + validación cruzada k-fold
python entrenar.py --cv --solo-cv       # solo evaluar, sin producir modelo
python entrenar.py --cv-grupos '^[^_]+_([^_]+)_'   # dejando señantes fuera
python entrenar.py --sintetico          # smoke test sin corpus real
```

Tres cifras con propósitos distintos:

- `train()` → split estratificado 80/20 con early stopping. Produce el **modelo
  exportable** y su `val_accuracy`.
- `evaluar_cv()` → **k-fold estratificado** (k=5): accuracy media ± desviación
  estándar y matriz de confusión out-of-fold. Con ~50 muestras/glosa el 80/20
  deja ~10 muestras de validación por clase, demasiado ruidoso.
- `evaluar_cv(groups=...)` → **k-fold dejando señantes fuera**, con `--cv-grupos`.
  **Es la cifra honesta cuando el corpus tiene más de un señante.**

> **Por qué importa el tercer protocolo.** Si un señante graba varias
> repeticiones de la misma glosa, un k-fold al azar reparte esas repeticiones
> entre entrenamiento y validación: el modelo puede reconocer *a la persona* en
> vez de *la seña*, y la cifra sale inflada. Dejando señantes fuera se mide lo
> que de verdad interesa —generalizar a alguien que el modelo nunca vio—, que es
> exactamente el escenario de ventanilla. El corpus LSCh planificado (≥2
> señantes) tiene el mismo problema, así que conviene evaluarlo igual.
>
> `--cv-grupos` recibe una regex con un grupo de captura que extrae el señante
> del `sample_id`. Para nombres tipo `<clase>_<señante>_<repetición>` el patrón
> es `^[^_]+_([^_]+)_`. Requiere un dataset construido con una versión de
> `build_dataset.py` que guarde `sample_ids`.

Resultados en `outputs/reports/` (`metrics.json`, `cv_metrics.json`, matrices de
confusión en PNG).

### 4) Exportar a ONNX (para Unity Sentis)

```powershell
python exportar_onnx.py
```
Valida que los operadores del grafo estén soportados por Sentis
(`sentisCompatible`). Produce `outputs/models/modelo.onnx`.

### 5) Validar end-to-end en PC (CU-01 + CU-03)

```powershell
python demo_vivo.py --fuente 0
```
Muestra la glosa reconocida, su confianza, el mensaje compuesto
(`MessageComposer`) y la **latencia end-to-end** contrastada con el umbral de
500 ms del diseño. Teclas: `r` reinicia el mensaje (FA-01 de CU-03), `q`/`ESC`
salir.

La latencia reportada incluye el retardo de segmentación (los `REST_FRAMES_FIN`
frames de reposo que hacen falta para confirmar el fin de la seña), no solo la
inferencia: medir únicamente el ONNX daría ~3 ms, una cifra irrelevante frente al
presupuesto de 500 ms.

### 6) Preparar la integración con Unity

```powershell
python generar_vectores_dorados.py             # genera integracion/vectores_dorados.json
python generar_vectores_dorados.py --verificar # comprueba que sigue vigente
```
Congela la salida de referencia del preprocesamiento para que el port a C# pueda
verificarse caso por caso. Ver **`INTEGRACION_UNITY.md`** para la especificación
completa y el código C# de referencia.

> Si alguien cambia el preprocesamiento en Python, `--verificar` falla: el port en
> C# queda invalidado y hay que regenerar el JSON y avisar al equipo de Unity.

---

## Validación con datasets de referencia (Sección 11)

No existe corpus público de LSCh dinámico, así que el pipeline puede validarse
primero con datasets de referencia **re-extrayendo los keypoints con nuestra
propia Capa 1** (`extraer_lote.py` → HandLandmarker, 21/mano). Esto **no reabre**
`DECISION_PREPROCESAMIENTO.md`: se usan los *vídeos*, no el código ni los
keypoints de terceros.

Ejemplo con **SWL-LSE** (Zenodo 13691887), conjunto de referencia `VIDEOS_REF`
(24 MB, 300 vídeos, uno por seña — sirve para probar el camino de vídeo real,
**no para entrenar**, porque es 1 muestra/clase):

```powershell
# 1) Descargar VIDEOS_REF.zip + anotaciones a data/external/swl_lse/ y descomprimir.
#    (VIDEOS_REF.zip = 24 MB; el corpus entrenable real es LSA64 — ver Sección 11.)

# 2) Extraer keypoints de todos los vídeos con NUESTRO HandLandmarker (modo IMAGE):
python extraer_lote.py `
    --videos-dir data/external/swl_lse/VIDEOS_REF `
    --anotaciones data/external/swl_lse/videos_ref_annotations.csv `
    --salida data/raw/swl_ref.csv          # --limite N para un subconjunto

# 3) Construir el dataset normalizado (LSE es bimanual -> modo ambas):
python build_dataset.py --entrada data/raw/swl_ref.csv --modo ambas
```

`extraer_lote.py` es genérico (mapea etiquetas desde un CSV `FILENAME,LABEL` o
del nombre de archivo), así que sirve igual para **LSA64**.

### El corpus del MVP: LSA64

> **Decisión del 2026-07-26: el corpus del MVP es LSA64, lengua de señas
> argentina.** El equipo no tuvo acceso a señantes de LSCh, así que el corpus
> propio no se grabó. SWL-LSE (lengua de señas española) se descartó porque **no
> publica vídeo entrenable**: de sus 8.000 secuencias solo comparte keypoints ya
> extraídos, en esquema Holistic, y usarlos reabriría
> `DECISION_PREPROCESAMIENTO.md`. LSA64 sí trae vídeo, así que los keypoints se
> extraen con nuestra propia Capa 1.

**LSA64** (CC BY-NC-SA 4.0) trae 64 señas × 10 señantes × 5 repeticiones =
**50 muestras por seña**. Tomando 10 señas se obtiene un corpus con
**exactamente la forma que el diseño pedía** (10 clases × 50 muestras).

```powershell
python extraer_lote.py `
    --videos-dir data/external/lsa64/videos `
    --anotaciones data/external/lsa64/lsa64_10_annotations.csv `
    --salida data/raw/lsa64_10.csv

python build_dataset.py --entrada data/raw/lsa64_10.csv --modo ambas
python entrenar.py --cv
```

> ⚠️ **Las etiquetas son las señas originales de LSA64, no glosas LSCh.** Las 10
> clases se eligieron por paralelo semántico con el vocabulario objetivo
> (`Thanks`↔GRACIAS, `Help`↔AYUDA, `Name`↔NOMBRE, …), pero son señas argentinas.
> El sistema está validado sobre **lengua de señas argentina**; la generalización
> a LSCh no está probada y es la limitación principal a declarar en el informe.
> `Glosas_LSCh_Mappeadas.csv` sigue siendo el vocabulario *objetivo de diseño*,
> no el demostrado. La licencia CC BY-NC-SA 4.0 exige atribución, prohíbe uso
> comercial y obliga a licenciar los derivados —el modelo incluido— igual.

**Resultados obtenidos (2026-07-26)** sobre 483 de 500 vídeos con detección:

| Modo | k-fold estratificado | **k-fold por señante** |
|---|---|---|
| `dominante` (63 dim) | 0.985 ± 0.012 | **0.903 ± 0.054** |
| `ambas` (126 dim) | 0.969 ± 0.022 | 0.911 ± 0.056 |

Ambos modos superan el objetivo MVP (≥ 0.85) en la evaluación por señante, que es
la exigente. Los dos modos son indistinguibles entre sí (las diferencias caen
dentro de las desviaciones y del ruido de corrida a corrida, ~±0.01, porque el
entrenamiento de Keras no es bit-determinista aunque se fije la semilla), así que
se elige `dominante` por coste: la mitad de entrada.

**La cifra de LSA64 es una cota inferior.** Los señantes de LSA64 graban con
**guantes de colores** (rosa/verde, que el dataset usa para identificar manos) y
MediaPipe `HandLandmarker` está entrenado sobre manos desnudas. Aun así detecta
manos en 483 de los 500 vídeos; los 17 fallos se concentran en `Patience`
(14 vídeos: mano de canto tapando la cara) y `Appear` (3). El corpus LSCh real
—manos desnudas, cámara propia, encuadre controlado— parte de condiciones
mejores, así que el desempeño del pipeline sobre él no debería ser peor por esta
causa.

---

## Punto abierto: una mano vs. dos manos (Sección 6)

LSCh usa señas a dos manos, pero el `NormVector` documentado es de una mano
(63 dim). El sistema **no fuerza** la decisión: es configurable vía
`config.MODO_MANOS` y el flag `--modo` de `build_dataset.py`:

- `dominante` → 63 dim (mano de mayor `handedness_score`).
- `ambas` → 126 dim (`[Left | Right]` concatenadas; mano ausente → ceros).

El `KeypointNormalizer` **no cambia**: siempre normaliza una mano a 63 dim
(coherente con `DECISION_PREPROCESAMIENTO.md`). El modo de manos es una decisión
de ensamblado del dataset, no de la capa de normalización. La decisión final es
del equipo; el pipeline soporta ambos caminos sin re-trabajo.

---

## Arquitectura del TCN (`lsch_mr/tcn.py`)

Bloques residuales de convolución 1D **causal y dilatada** (kernel=3,
dilataciones `(1,2,4,8)`, 2 convs/bloque) → `GlobalAveragePooling1D` → `Dense` →
`softmax`. Receptive field ≈ `1 + 2·(k-1)·Σdil = 61 ≥ 60` frames, cubriendo la
ventana temporal completa de una seña (`MAX_SEQ_FRAMES=60`). Usa solo operadores
soportados por Unity Sentis (Conv, Relu, Add, Pad, ReduceMean, Gemm, Softmax).

Toda `SignSequence` (1..60 frames) se **remuestrea temporalmente** a `SEQ_LEN=60`
por interpolación lineal, dando una forma de entrada fija (Sentis prefiere formas
estáticas) e independiente de la duración de la seña.

---

## Pruebas

```powershell
pip install pytest
pytest -q
```
`tests/` cubre el `KeypointNormalizer` (invariancia a traslación/escala, casos
degenerados), el `RestStateDetector` (segmentación por reposo, descarte por
pérdida de tracking) y el `MessageComposer` (acumulación, `maxWords`, cierre por
timeout con reloj inyectable, reinicio manual de FA-01).

Además, `python generar_vectores_dorados.py --verificar` comprueba que el
preprocesamiento no cambió respecto a los vectores de referencia del port a C#.

---

## Estructura

```
lsch_mr/                 paquete con los componentes del diseño
  config.py              rutas, modelo de datos, hiperparámetros
  tipos.py               Frame, SignSequence, NormVector, SignEvent, ClassResult
  fuente_video.py        FuenteVideo (webcam / cámara de teléfono / archivo)
  hand_tracking_provider.py   HandTrackingProvider (HandLandmarker)
  keypoint_normalizer.py      KeypointNormalizer
  rest_state_detector.py      RestStateDetector
  caracteristicas.py     normalización + remuestreo -> entrada del modelo
  csv_esquema.py         esquema CSV del corpus
  dataset_recorder.py    DatasetRecorder
  tcn.py                 arquitectura TCN
  model_trainer.py       ModelTrainer
  model_exporter.py      ModelExporter
  sign_classifier.py     SignClassifier
  message_composer.py    MessageComposer
descargar_modelo.py      descarga hand_landmarker.task
extraer_keypoints.py     CLI extracción (webcam/video)
extraer_lote.py          CLI extracción por lotes (dataset de referencia)
grabar_corpus.py         CLI grabación del corpus
build_dataset.py         CLI crudo -> dataset normalizado
entrenar.py              CLI entrenamiento
exportar_onnx.py         CLI exportación ONNX
demo_vivo.py             CLI validación end-to-end en PC (orquestador de referencia)
generar_vectores_dorados.py  vectores de prueba para el port del preproceso a C#
integracion/             artefactos para el repo de Unity (vectores dorados)
INTEGRACION_UNITY.md     contrato entre repos + spec del port a C#
tests/                   pruebas unitarias
```
