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

- **124 tests pytest pasan** (`python -m pytest -q`), repartidos así:

  | Área | Archivo | Tests |
  |---|---|---|
  | Estabilidad del preprocesador (cámara pinhole) | `test_preprocesador_estabilidad.py` | 22 |
  | Métricas de segmentación | `test_metricas_segmentacion.py` | 15 |
  | Métricas de clasificación | `test_metricas_clasificacion.py` | 14 |
  | Métricas de TSR | `test_metricas_tsr.py` | 12 |
  | Reporte de sesión de `demo_vivo` | `test_demo_vivo_metricas.py` | 10 |
  | `RestStateDetector` | `test_rest_state_detector.py` | 8 |
  | Realce de poca luz | `test_realce_luz.py` | 8 |
  | `MonitorRecursos` | `test_monitor_recursos.py` | 8 |
  | `KeypointNormalizer` | `test_keypoint_normalizer.py` | 8 |
  | `MessageComposer` | `test_message_composer.py` | 7 |
  | Split 80/20 del modelo final | `test_split_indices.py` | 6 |
  | `cargar_grupos` | `test_cargar_grupos.py` | 5 |

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

Corrida del **2026-07-26**, las cuatro evaluaciones de una sola vez para que la
tabla y los JSON sean consistentes por construcción:

| Modo | k-fold estratificado | **k-fold por señante** | JSON |
|---|---|---|---|
| `dominante` (63 dim) | 0.981 ± 0.017 | **0.911 ± 0.051** | `cv_metrics_dominante[_senante].json` |
| `ambas` (126 dim) | 0.973 ± 0.020 | 0.905 ± 0.063 | `cv_metrics_ambas[_senante].json` |

Regenerable con los cuatro comandos siguientes (regenerable, **no** reproducible
bit a bit: ver el punto 2 de abajo):

```bash
python entrenar.py --dataset data/processed/lsa64_dominante.npz --cv --solo-cv
python entrenar.py --dataset data/processed/lsa64_dominante.npz --cv-grupos 2 --solo-cv
python entrenar.py --dataset data/processed/lsa64_ambas.npz --cv --solo-cv
python entrenar.py --dataset data/processed/lsa64_ambas.npz --cv-grupos 2 --solo-cv
```

**Cómo leer estas cifras:**

1. **La columna que vale es la de por señante.** Cada señante aporta 5
   repeticiones de cada seña; un k-fold al azar las reparte entre entrenamiento y
   validación, así que el modelo puede reconocer a la persona en vez de la seña.
   La caída de 0.981 → 0.911 al separar por señante mide exactamente esa fuga.
2. **Hay ruido de corrida a corrida de ~±0.01.** El entrenamiento de Keras no es
   bit-determinista aunque se fije la semilla. Con los **mismos splits**,
   `dominante` por señante ha dado 0.903, 0.911, 0.915 y 0.920 en cuatro corridas
   distintas. Al reportar, dar siempre la cifra **con su desviación**, nunca un
   valor puntual, y citar la corrida (la tabla de arriba es la del 2026-07-26,
   con su JSON en disco).

   > Antes esto era peor de lo que parecía: `evaluar_cv` nombraba su salida solo
   > por esquema (`cv_metrics_senante.json`), así que evaluar `ambas` **pisaba**
   > el JSON de `dominante` y había que renombrar a mano. El resultado fue que la
   > cifra de titular de este documento (0.903) acabó sin ningún archivo que la
   > respaldara. Desde el 2026-07-26 el nombre incluye el modo, así que cada
   > corrida tiene su archivo y toda cifra es rastreable.
3. **Es una cota inferior.** Los señantes de LSA64 graban con **guantes de
   colores** y MediaPipe está entrenado sobre manos desnudas. Los 17 vídeos sin
   detección se concentran en `Patience` (14, mano de canto sobre la cara) y
   `Appear` (3). Con manos desnudas el pipeline debería ir mejor, no peor.

**`CONF_THRESHOLD` calibrado: 0.60 → 0.90.** Sobre las predicciones out-of-fold
de la validación por señante (corrida canónica del 2026-07-26), el compromiso
medido fue:

| Umbral | Cobertura | Precisión | Glosas erróneas mostradas |
|---|---|---|---|
| 0.60 | 94.0% | 92.1% | 36 |
| **0.90** | **79.1%** | **96.3%** | **14** |
| 0.95 | 73.3% | 96.0% | 14 |
| 0.99 | 61.7% | 96.0% | 12 |

Se elige 0.90: en ventanilla, mostrar una glosa equivocada engaña, mientras que
"no reconocida" solo pide repetir. Y es la rodilla de la curva — de 0.90 en
adelante la precisión deja de subir (96.3% → 96.0%) mientras la cobertura sigue
cayendo.

**El modelo está mal calibrado, y peor de lo que se creía.** La confianza mediana
de sus predicciones erróneas es 0.78 y su **p95 llega a 1.00**: hay fallos con
confianza máxima. Consecuencia práctica: **no se puede comprar precisión subiendo
el umbral** — pasar de 0.90 a 0.99 solo quita 2 de las 14 glosas erróneas y
cuesta 17 puntos de cobertura. Corregirlo de raíz (temperature scaling) es
trabajo futuro; hacerlo cambiaría el contrato con Unity, porque hoy el softmax
va dentro del grafo ONNX.

> La tabla anterior de este documento (0.90 → 81.4% / 95.9% / 16, y 0.99 → 2
> errores) venía de una corrida cuyo JSON se perdió al sobrescribirse. Con la
> corrida canónica el umbral elegido **no cambia**, pero el argumento sí: antes
> parecía que 0.99 casi eliminaba los errores, y no es así.
> Recalcular tras reentrenar: `python evaluar_modelo.py --fuente cv`.

**Punto abierto — una mano vs. dos manos.** Los dos modos son
**indistinguibles**: 0.911 ± 0.051 (`dominante`) vs 0.905 ± 0.063 (`ambas`), con
las diferencias muy por dentro del ruido de corrida a corrida, y eso pese a que 5
de las 10 señas del proxy son bimanuales.

La evidencia de que son indistinguibles se reforzó el 2026-07-26: **el orden
entre los dos modos se invirtió** respecto a la corrida anterior (entonces
`ambas` 0.911 > `dominante` 0.903; ahora `dominante` 0.911 > `ambas` 0.905). Si
la diferencia fuera real no cambiaría de signo entre corridas — es ruido.

No hay evidencia para preferir `ambas`, así que se elige `dominante` por coste:
la mitad de entrada (63 vs 126) y coincide con lo ya documentado. **No es una
decisión cerrada por accuracy** — para el corpus LSCh la comparación se re-corre
con un comando en ambos modos y debe revisarse con sus propios datos, que sí
pueden tener señas bimanuales que la mano dominante no distinga.

> **Ojo con el número de campo de `--cv-grupos`**: depende de cómo se construyó
> el `sample_id`. En LSA64 (`017_001_001` = clase_señante_repetición) el señante
> es el campo **2**. En un corpus grabado aquí con `grabar_corpus.py --senante s01`
> el `sample_id` es `s01_0001`, así que el señante es el campo **1**. Pasar el
> número equivocado no da error: agrupa por otra cosa y la métrica vuelve a estar
> inflada en silencio.

## Instrumentación de las 4 métricas del MVP

Cada métrica de la Sección 11 tiene ya su herramienta de medición y su evidencia
en `outputs/reports/`. El índice operativo —qué comando produce cada cifra y qué
hay que declarar al reportarla— es **`METRICAS_MVP.md`**; aquí solo el estado:

| # | Métrica | Umbral | Herramienta | Estado |
|---|---|---|---|---|
| 1 | Precisión algorítmica | ≥ 85% | `evaluar_modelo.py` | ✅ cumple (ver tabla de arriba) |
| 2 | Latencia end-to-end | ≤ 500 ms | `demo_vivo.py` · `integracion/unity/` | ⚠️ medida, no cerrada |
| 3 | Rendimiento gráfico | ≥ 72 FPS | `integracion/unity/FrameRateMonitor.cs` | ❌ conflicto de alcance |
| 4 | Task success rate | ≥ 80% | `calcular_tsr.py` + `protocolo/` | ❌ falta ejecutar sesiones |
| — | *(previa)* Límites del `RestStateDetector` | — | `evaluar_segmentacion.py` | ❌ falta etiquetar |
| — | *(previa)* Estabilidad del preprocesador | — | `pytest tests/test_preprocesador_estabilidad.py` | ✅ pasa |

Dos hallazgos de la instrumentación que conviene tener a mano:

- **El preprocesador deriva ~3× más por desplazamiento lateral en el encuadre
  (0.161 a 10 cm) que por distancia a la cámara (0.058 entre 40 y 150 cm)**, en
  unidades donde ‖L9−L0‖ = 1. La normalización quita la traslación en el espacio
  de keypoints, pero no la distorsión de perspectiva fuera de eje. Conviene
  signar centrado; si en uso real la mano se va a los bordes, el corpus debería
  cubrir esas posiciones.
- **En LSA64 la mano solo se detecta en ~75% de los frames** y 405 de las 483
  muestras tienen huecos en `frame_idx`. Al medir la segmentación, buena parte de
  los falsos negativos serán dropout de MediaPipe y no error del detector:
  `evaluar_segmentacion.py` imprime la tasa de detección justamente para poder
  declarar las dos cosas por separado.

## Lo que falta

- **Latencia end-to-end: medida, no cumplida.** Sobre 10 vídeos: mediana ~239 ms,
  rango 171–547 ms, **1 de 9 por encima de los 500 ms**. Y esos vídeos son de
  60 FPS: a 30 FPS el retardo de segmentación se duplica (~200 ms) y varios casos
  quedarían al borde. Falta una corrida sostenida con la cámara del demo. Palanca
  si sale corta: bajar `REST_FRAMES_FIN` de 6 a 4, a costa de cerrar señas antes.
- **Capa 4 / Unity** — el runtime (`SpatialSubtitleRenderer`, Spatial UI) sigue
  sin existir; el contrato y los vectores dorados sí. Lo que **ya** hay en este
  repo es la **instrumentación** de las métricas 2 y 3 en C#
  (`integracion/unity/`, 3 archivos): telemetría de latencia por frontera de capa
  y contador de FPS. Está escrita contra la API estándar de Unity y **no se ha
  compilado** — aquí no hay proyecto Unity donde hacerlo.
- **FPS (métrica 3): conflicto de alcance sin resolver.** El umbral de 72 FPS es
  la tasa de refresco del Quest 3, que la decisión del 2026-07-21 sacó del camino
  crítico. En PC con VSync el FPS queda clavado en el refresco del monitor y
  compararlo con 72 no significa nada. Hay que elegir: declarar la métrica como
  no evaluada en el hardware objetivo, o conseguir un Quest 3 una tarde. Ver
  `METRICAS_MVP.md`.
- **Pruebas con usuarios** (TSR ≥80% en ≥10 pruebas). El guion del escenario, la
  planilla y el cálculo ya están (`protocolo/`, `calcular_tsr.py`); falta
  ejecutar las sesiones.
- **Ground truth de los límites de seña.** `evaluar_segmentacion.py` mide
  precisión/recall del `RestStateDetector`, pero necesita que alguien etiquete a
  mano el inicio/fin de una muestra del corpus viendo los vídeos
  (`--plantilla` genera el CSV a rellenar).

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
  glosas.py                  catálogo de glosas (orden estable de clases)
  consola.py                 UTF-8 en consola Windows + coloreado ANSI
  realce_luz.py              realce adaptativo de frames oscuros
  monitor_recursos.py        MonitorRecursos (CPU/memoria del proceso)
  metricas_clasificacion.py  accuracy, matriz de confusión, veredicto (métrica 1)
  metricas_segmentacion.py   precisión/recall de límites de seña (prueba previa)
  metricas_tsr.py            task success rate + IC de Wilson (métrica 4)
descargar_modelo.py          descarga hand_landmarker.task
extraer_keypoints.py         CLI extracción (webcam/video)
extraer_lote.py              CLI extracción por lotes (datasets de referencia)
grabar_corpus.py             CLI grabación del corpus (--senante)
build_dataset.py             CLI crudo -> dataset normalizado (+ gate de 50 muestras)
entrenar.py                  CLI entrenamiento (--cv, --cv-grupos, --solo-cv)
exportar_onnx.py             CLI exportación a ONNX
demo_vivo.py                 orquestador de referencia CU-01/CU-03 (latencia e2e, tecla r)
diagnostico_captura.py       CLI diagnóstico de captura (luz, tasa de detección)
generar_vectores_dorados.py  vectores de prueba (Capa 2) para el port a C#
generar_secuencia_dorada.py  secuencia de prueba (Capa 1+2) para el port a C#
evaluar_modelo.py            CLI métrica 1: accuracy + matriz de confusión
evaluar_segmentacion.py      CLI prueba previa: límites del RestStateDetector
calcular_tsr.py              CLI métrica 4: task success rate
METRICAS_MVP.md              cómo se mide cada una de las 4 métricas del MVP
INTEGRACION_UNITY.md         contrato entre repos + spec del port
DECISION_PREPROCESAMIENTO.md por qué el preprocesamiento es propio
GLOSARIO_SENAS_MODELO.md     las 10 glosas del modelo y su traducción
integracion/                 vectores_dorados.json, secuencia_dorada.json
integracion/unity/           instrumentación C# de latencia y FPS (métricas 2 y 3)
protocolo/                   guion y planilla del TSR (métrica 4)
tests/                       124 tests unitarios
data/external/lsa64/         corpus proxy LSA64 (no versionado)
data/raw/lsa64_10.csv        keypoints del proxy (no versionado)
outputs/                     modelo, labels y reportes (no versionado)
```

## Próximo paso recomendado

Con el corpus resuelto y `CONF_THRESHOLD` ya calibrado, del lado de IA/datos
queda **una sola cosa medible aquí**: la latencia end-to-end en una sesión
sostenida con la cámara real del demo (`python demo_vivo.py --fuente 0`, que deja
el reporte solo).

Lo demás ya no es código, es trabajo de campo con la instrumentación que ya
existe (ver `METRICAS_MVP.md` para el comando de cada una):

1. **Etiquetar el ground truth** de límites de seña (40-60 muestras bastan) y
   correr `evaluar_segmentacion.py`.
2. **Ejecutar las sesiones de TSR** con 10+ participantes según
   `protocolo/PROTOCOLO_TSR.md`.
3. **Decidir qué se declara sobre la métrica 3** (FPS), que depende de si hay
   Quest 3 o no.

El resto del avance del MVP sigue dependiendo de la Capa 4 en Unity.
