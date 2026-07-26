# Estado actual del código — LSCh-MR (pipeline IA/datos)

> Generado el 2026-07-26 para dar contexto rápido a Claude en una nueva
> conversación. Es una **fotografía del repo en este momento**, no un documento
> de diseño — para diseño ver `CONTEXTO_PROYECTO.md`,
> `context/arquitectura_LSCh-MR.md` y `DECISION_PREPROCESAMIENTO.md` (no reabrir
> lo que ahí se consolida).

## Qué es este repo

Implementación en Python de la parte de **Juan Yampara** en el proyecto de
titulación (UTA): pipeline de reconocimiento de Lengua de Señas Chilena (LSCh)
— captura de keypoints con MediaPipe, normalización, dataset, entrenamiento de
un clasificador TCN y exportación a ONNX. La capa espacial (Unity, de Tomás)
vive en otro repo; el punto de acoplamiento son `modelo.onnx` + `labels.json`,
especificados en `INTEGRACION_UNITY.md`.

## Decisiones consolidadas (no reabrir)

- Clasificador **TCN** (no LSTM ni Transformer).
- Runtime **ONNX** (opset 13, `sentisCompatible=True`) consumido con **Unity
  Sentis**.
- Captura con **MediaPipe Tasks `HandLandmarker`** (API Tasks, no la legacy).
- **21 keypoints por mano**, `NormVector` de **63 dim** por mano (centrado en
  muñeca L0 + escala por distancia L0–L9).
- Preprocesamiento **implementado desde cero**, sin reutilizar
  `mvazquezgts/SWL-LSE`. Ver `DECISION_PREPROCESAMIENTO.md`.
- **Demo final: solo cámara** (PC + webcam). El Meta Quest 3 **no** es necesario
  para demostrar el MVP — decisión del 2026-07-21. La demo corre en **Unity +
  Sentis sobre PC con webcam**; `demo_vivo.py` es el orquestador de referencia en
  Python, no el entregable final. Fuera del camino crítico: Meta XR SDK,
  Snapdragon XR2, subtítulo espacial anclado.
- **Modo de manos: `dominante` (63 dim)** — el punto abierto de la Sección 6 se
  resolvió con datos el 2026-07-26 (ver más abajo). Sigue siendo configurable.

## Estado del pipeline: implementado y verificado end-to-end

```
captura (HandLandmarker) → KeypointNormalizer → RestStateDetector →
build_dataset → ModelTrainer (TCN, Keras 3) → ModelExporter (ONNX opset 13) →
SignClassifier (onnxruntime, ~3.5 ms/inferencia) → MessageComposer
```

- **25 tests pytest pasan** (`KeypointNormalizer`, `RestStateDetector`,
  `MessageComposer`, `cargar_grupos`).
- `python generar_vectores_dorados.py --verificar` comprueba que el
  preprocesamiento no cambió respecto a los vectores de referencia del port a C#.

## Corpus del MVP: LSA64 — decisión del 2026-07-26

**El corpus del MVP es LSA64 (lengua de señas argentina), no LSCh.** El equipo no
tiene acceso a señantes de Lengua de Señas Chilena, así que grabar el corpus
propio quedó descartado. Se evaluó también SWL-LSE (lengua de señas española),
pero **no publica vídeo entrenable**: de sus 8.000 secuencias solo comparte
keypoints ya extraídos, en esquema Holistic, y usarlos reabriría
`DECISION_PREPROCESAMIENTO.md`. LSA64 sí trae vídeo, así que los keypoints se
extraen con nuestra propia Capa 1 y esa decisión sigue intacta.

Corpus: 10 señas × 10 señantes × 5 repeticiones = **500 vídeos**, exactamente la
forma que el diseño pedía para el corpus LSCh (10 clases × 50 muestras).
Keypoints re-extraídos con `extraer_lote.py`; 483 de los 500 con detección.

> **Consecuencia para el informe, a declarar explícitamente:** el sistema está
> validado sobre **lengua de señas argentina**, no chilena. El vocabulario
> demostrado son las 10 señas de LSA64 (`Thanks`, `Help`, `Name`, …), no las 10
> glosas de `Glosas_LSCh_Mappeadas.csv`, que siguen siendo el vocabulario
> *objetivo de diseño*. La generalización a LSCh **no está probada** y es trabajo
> futuro. LSA64 es CC BY-NC-SA 4.0: exige atribución, prohíbe uso comercial y
> obliga a licenciar los derivados —el modelo incluido— en los mismos términos.

| Modo | k-fold estratificado | **k-fold por señante** |
|---|---|---|
| `dominante` (63 dim) | 0.985 ± 0.012 | **0.903 ± 0.054** |
| `ambas` (126 dim) | 0.969 ± 0.022 | 0.911 ± 0.056 |

**Cómo leer estas cifras:**

1. **La columna que vale es la de por señante.** Cada señante aporta 5
   repeticiones de cada seña; un k-fold al azar las reparte entre entrenamiento y
   validación, así que el modelo puede reconocer a la persona en vez de la seña.
   La caída de 0.985 → 0.90 al separar por señante mide exactamente esa fuga.
2. **Hay ruido de corrida a corrida de ~±0.01.** El entrenamiento de Keras no es
   bit-determinista aunque se fije la semilla: dos corridas de `dominante` por
   señante dieron 0.915 y 0.903 con los mismos splits. La tabla recoge la corrida
   que produjo el modelo exportado. Al reportar, dar la cifra con su desviación,
   no un valor puntual.
3. **Es una cota inferior.** Los señantes de LSA64 graban con **guantes de
   colores** y MediaPipe está entrenado sobre manos desnudas. Los 17 vídeos sin
   detección se concentran en `Patience` (14, mano de canto sobre la cara) y
   `Appear` (3). Con manos desnudas el pipeline debería ir mejor, no peor.

**`CONF_THRESHOLD` calibrado: 0.60 → 0.90.** Sobre las predicciones out-of-fold
de la validación por señante, el compromiso medido fue:

| Umbral | Cobertura | Precisión | Glosas erróneas mostradas |
|---|---|---|---|
| 0.60 | 96.3% | 92.7% | 34 |
| **0.90** | **81.4%** | **95.9%** | **16** |
| 0.95 | 75.4% | 97.3% | 10 |
| 0.99 | 60.0% | 99.3% | 2 |

Se elige 0.90: en ventanilla, mostrar una glosa equivocada engaña, mientras que
"no reconocida" solo pide repetir. **El modelo está mal calibrado** —la confianza
mediana de sus predicciones erróneas es 0.85 y su p95 llega a 0.99—, así que el
umbral es un instrumento romo. Corregirlo de raíz (temperature scaling) es
trabajo futuro; hacerlo cambiaría el contrato con Unity, porque hoy el softmax
va dentro del grafo ONNX.

**Punto abierto — una mano vs. dos manos.** Los dos modos son
**indistinguibles**: 0.903 ± 0.054 vs 0.911 ± 0.056, con las diferencias muy
por dentro del ruido de corrida a corrida, y eso pese a que 5 de las 10 señas del
proxy son bimanuales. No hay evidencia para preferir `ambas`, así que se elige
`dominante` por coste: la mitad de entrada (63 vs 126) y coincide con lo ya
documentado. **No es una decisión cerrada por accuracy** — para el corpus LSCh la
comparación se re-corre con un comando (`entrenar.py --cv-grupos 1` en ambos
modos) y debe revisarse con sus propios datos, que sí pueden tener señas
bimanuales que la mano dominante no distinga.

## Lo que falta

- **Latencia end-to-end: medida, no cumplida.** Sobre 10 vídeos: mediana ~239 ms,
  rango 171–547 ms, **1 de 9 por encima de los 500 ms**. Y esos vídeos son de
  60 FPS: a 30 FPS el retardo de segmentación se duplica (~200 ms) y varios casos
  quedarían al borde. Falta una corrida sostenida con la cámara del demo. Palanca
  si sale corta: bajar `REST_FRAMES_FIN` de 6 a 4, a costa de cerrar señas antes.
- **Capa 4 / Unity** — no existe todavía; el contrato y los vectores dorados sí.
- **Pruebas con usuarios** (TSR ≥80% en ≥10 pruebas).

> **Si algún día se graba corpus propio:** usar `grabar_corpus.py --senante s01`
> (y `s02`, …) con un identificador distinto por persona. El señante queda dentro
> del `sample_id` y es lo único que permite después la validación por señante —
> **no se puede reconstruir a posteriori**.

## Riesgos abiertos (lado Unity, fuera de este repo)

1. **Hand tracking sin Meta XR SDK**: `HandTrackingProvider` debe
   reimplementarse en Unity sobre webcam. El contrato `getFrame()` se mantiene.
   Un puente Python↔Unity por red reabriría la Sección 3 del diseño — no hacerlo.
2. **Train/serve skew** — *mitigado, no cerrado*. `INTEGRACION_UNITY.md` +
   `integracion/vectores_dorados.json` (7 casos, tolerancia 1e-5) permiten
   verificar el port del preprocesamiento a C#. Riesgo residual: convención de
   ejes y semántica de la `z`, que la normalización no corrige.

## Entorno / gotchas de ejecución

- Python **3.12.0** (pyenv-win), **numpy 2.3.5**, mediapipe 0.10.35, opencv 4.13,
  tensorflow 2.21, keras 3.15, tf2onnx 1.17, onnxruntime 1.27, pytest.
- **Keras 3 es el default**; tf2onnx exporta el TCN vía `from_keras`, con
  fallbacks en `ModelExporter`.
- **Consola Windows en cp1252**: imprimir caracteres fuera de cp1252 lanza
  `UnicodeEncodeError`. Resuelto con `lsch_mr/consola.py::configurar_utf8()`.
  Igual aplica al **leer** archivos — usar `encoding="utf-8"` explícito.
- **Comillas en la línea de comandos**: una regex con `^`, `[`, `(` puede llegar
  mutilada al proceso según el shell. Por eso `--cv-grupos` acepta un número de
  campo (`--cv-grupos 2`) además de una regex.
- En Git Bash, cuidado con `pip install "pkg>=1.2"` (el `>=` puede redirigir).

## Mapa de archivos

```
lsch_mr/                     paquete con los componentes del diseño
  config.py                  rutas, modelo de datos, hiperparámetros, umbrales MVP
  tipos.py                   HandFrame, MultiHandFrame, SignEvent, ClassResult
  fuente_video.py            FuenteVideo (webcam / teléfono / archivo)
  hand_tracking_provider.py  HandTrackingProvider (HandLandmarker; image/video)
  keypoint_normalizer.py     KeypointNormalizer
  rest_state_detector.py     RestStateDetector
  caracteristicas.py         normalización + ensamblado + remuestreo a 60
  csv_esquema.py             esquema CSV del corpus
  dataset_recorder.py        DatasetRecorder (sample_id incluye el señante)
  tcn.py                     arquitectura TCN
  model_trainer.py           ModelTrainer: train() + evaluar_cv() [k-fold / por señante]
  model_exporter.py          ModelExporter (ONNX + validación de operadores)
  sign_classifier.py         SignClassifier (onnxruntime)
  message_composer.py        MessageComposer (con tick() para el timeout)
descargar_modelo.py          descarga hand_landmarker.task
extraer_keypoints.py         CLI extracción (webcam/video)
extraer_lote.py              CLI extracción por lotes (datasets de referencia)
grabar_corpus.py             CLI grabación del corpus (--senante)
build_dataset.py             CLI crudo -> dataset normalizado (+ gate de 50 muestras)
entrenar.py                  CLI entrenamiento (--cv, --cv-grupos, --solo-cv)
exportar_onnx.py             CLI exportación a ONNX
demo_vivo.py                 orquestador de referencia CU-01/CU-03 (latencia e2e, tecla r)
generar_vectores_dorados.py  vectores de prueba para el port a C#
INTEGRACION_UNITY.md         contrato entre repos + spec del port
integracion/                 vectores_dorados.json
tests/                       25 tests unitarios
data/external/lsa64/         corpus proxy LSA64 (no versionado)
data/raw/lsa64_10.csv        keypoints del proxy (no versionado)
outputs/                     modelo, labels y reportes (no versionado)
```

## Próximo paso recomendado

Con el corpus ya resuelto, lo que queda del lado de IA/datos es cerrar las dos
métricas pendientes: calibrar `CONF_THRESHOLD` con las confianzas out-of-fold y
medir la latencia en una sesión sostenida con la cámara real del demo. El resto
del avance del MVP depende de la Capa 4 en Unity.
