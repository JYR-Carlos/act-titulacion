# Arquitectura de Referencia — Sistema LSCh-MR
> **Propósito de este documento:** Describir la arquitectura diseñada en el informe consolidado final del proyecto LSCh-MR, para que pueda ser comparada contra la implementación real del repositorio y detectar discrepancias entre diseño e implementación.

---

## 1. Descripción General

**LSCh-MR** es un sistema de traducción en tiempo real de señas dinámicas de la Lengua de Señas Chilena (LSCh) a subtítulos flotantes en español, destinado a contextos de atención al público municipal (ventanilla GORE).

El sistema adopta una **arquitectura en capas con procesamiento en el borde (Edge Computing)**: todo el procesamiento ocurre localmente en el dispositivo, sin dependencia de red ni servicios cloud. La comunicación entre capas se realiza exclusivamente mediante **llamadas en memoria** (interfaces C# en runtime), sin protocolos de red ni IPC distribuido.

**Alcance del principio "sin red".** Aplica a la comunicación **entre capas** y a la inferencia: ninguna capa consulta un servicio remoto, y no existe puente por MQTT/gRPC/REST entre componentes. No aplica a la **fuente de imagen**: durante el desarrollo se admite usar la cámara de un teléfono como sustituto de webcam mediante un stream MJPEG local, que ocupa el lugar del hardware de captura y no forma parte del camino de producción. Un puente de red entre capas sí violaría este principio.

---

## 2. Plataforma de Despliegue

### MVP actual: PC + Webcam
El MVP se ejecuta sobre **PC con cámara web estándar**, con subtítulos desplegados en monitor. La extracción de keypoints se realiza con **MediaPipe Hands (Tasks API)**, que entrega el mismo esquema de 21 landmarks por mano que el Meta XR SDK.

### Objetivo final: Meta Quest 3
La migración al visor consiste únicamente en reemplazar:
- Fuente de keypoints: MediaPipe Hands → Meta Hand Tracking API (Meta XR SDK v74+)
- Canal de presentación: ventana en monitor → Spatial UI en Unity (Realidad Mixta)

Las capas de preprocesamiento e inferencia no requieren modificación en la migración.

### Alcance de la demostración (decisión 2026-07-21)
La demostración del MVP se hace **solo con cámara**: el Meta Quest 3 deja de estar en el camino crítico. El runtime demostrable es **Unity + Sentis sobre PC con webcam**, conservando el `modelo.onnx` como único punto de acoplamiento. Quedan fuera del alcance demostrable —y por tanto no verificables en este informe— la integración con Meta XR SDK, la medición en Snapdragon XR2 y el subtítulo espacial anclado al rostro.

Consecuencia para la Capa 1: `HandTrackingProvider` debe reimplementarse en Unity sobre webcam; el contrato `getFrame()` se mantiene y solo cambia la implementación, que es exactamente lo que prevé la Sección 8.2.

### Alcance de este documento respecto a los repositorios
El sistema vive en **dos repositorios**: el subsistema de IA/datos en Python (Juan Yampara) y el runtime Unity/C# (Capa 4, Tomás Silva). Las afirmaciones de conformidad entre diseño e implementación de este documento están verificadas **solo contra el repositorio de IA/datos**. La Capa 4 (`SpatialSubtitleRenderer`, Spatial UI, render Unity) debe auditarse por separado antes de afirmar que los diagramas coinciden con la totalidad del software desarrollado.

---

## 3. Arquitectura en Capas

El pipeline se organiza en **4 capas funcionales** que fluyen de forma secuencial (captura → preprocesamiento → inferencia → presentación), más un **pipeline offline** paralelo para construcción del corpus y entrenamiento del modelo.

```
┌──────────────────────────────────────────────────────────────┐
│  HARDWARE (PC + Webcam / Meta Quest 3)                       │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  CAPA 1 — Captura Cinemática                                 │
│  HandTrackingProvider · RestStateDetector                    │
└───────────────────────┬──────────────────────────────────────┘
                        │ Frame (21 Keypoints × 3 ejes = 63 valores)
┌───────────────────────▼──────────────────────────────────────┐
│  CAPA 2 — Preprocesamiento                                   │
│  KeypointNormalizer                                          │
└───────────────────────┬──────────────────────────────────────┘
                        │ NormVector (63-dim, invariante a escala y distancia)
┌───────────────────────▼──────────────────────────────────────┐
│  CAPA 3 — Inferencia (Edge AI)                               │
│  SignClassifier  [modelo TCN exportado a ONNX]               │
└───────────────────────┬──────────────────────────────────────┘
                        │ ClassResult (etiqueta + confianza)
┌───────────────────────▼──────────────────────────────────────┐
│  CAPA 4 — Presentación (Spatial UI)                          │
│  MessageComposer · SpatialSubtitleRenderer                   │
└──────────────────────────────────────────────────────────────┘

PIPELINE OFFLINE (paralelo, no corre en runtime):
  DatasetRecorder → ModelTrainer → ModelExporter → OnnxModel
```

---

## 4. Descripción de Capas

### Capa 1 — Captura Cinemática
Consume el stream de hand tracking (MediaPipe Hands en MVP, Meta XR SDK en producción) y obtiene las coordenadas articulares `(x, y, z)` de los **21 keypoints por mano** (63 valores por frame por mano) en tiempo real.

**Componentes:**
- `HandTrackingProvider`: encapsula el acceso al proveedor de hand tracking y entrega objetos `Frame` con los 21 keypoints.
- `RestStateDetector`: **componente crítico**. Máquina de estados que detecta cuándo las manos abandonan y regresan a la posición de reposo, delimitando temporalmente el inicio y fin de cada seña. Emite `SignEvent` para disparar la inferencia.

### Capa 2 — Preprocesamiento
Normaliza geométricamente los keypoints crudos para producir un vector invariante a la escala de la mano y a la distancia del usuario respecto a la cámara.

**Componente:**
- `KeypointNormalizer`: centra el esqueleto en la **muñeca (landmark L0 / wristId=0)** y escala todas las distancias relativas a la **longitud del segmento muñeca–dedo medio (L9 / refDistId=9)**. Produce un `NormVector` de **63 dimensiones**.

**Fórmula de normalización:**
```
keypoint_norm = (keypoint_raw - wrist_position) / dist(wrist, middle_finger_mcp)
```

En casos degenerados (mano colapsada, `dist ≈ 0`) el normalizador devuelve un vector de ceros en vez de dividir por cero.

**Ensamblado y remuestreo (frontera Capa 2 → Capa 3).** Antes de entrar al clasificador, la secuencia de `NormVector` pasa por dos pasos adicionales:

1. **Ensamblado según el modo de manos** (ver punto abierto en la Sección 6): una mano → 63 dim; dos manos → 126 dim concatenando `[Left | Right]`, con la mano ausente rellenada con ceros.
2. **Remuestreo temporal a `SEQ_LEN = 60` frames** por interpolación lineal. Da una entrada de forma fija —Unity Sentis prefiere formas estáticas— e independiente de la duración real de la seña.

El detalle exacto de ambos pasos, junto con los vectores de prueba para portarlos a C#, está en `INTEGRACION_UNITY.md`.

### Capa 3 — Inferencia (Edge AI)
Recibe la `SignSequence` (secuencia de `NormVector` correspondiente a una seña segmentada) y emite la etiqueta de texto traducida.

**Componente:**
- `SignClassifier`: carga el modelo ONNX y ejecuta la inferencia sobre la secuencia. Retorna un `ClassResult` con etiqueta y score de confianza. Si la confianza no supera `confThreshold`, activa el flujo alternativo FA-01 (seña fuera de vocabulario).

**Decisiones arquitectónicas consolidadas (no abiertas):**
- Arquitectura del modelo: **TCN (Temporal Convolutional Network)** — LSTM y Transformer Encoder descartados.
- Formato de exportación: **ONNX** — TensorFlow Lite descartado.
- Runtime de inferencia: **Unity Sentis ≥ 2.1** — Barracuda descartado.

### Capa 4 — Presentación (Spatial UI)
Recibe los resultados de clasificación y los presenta como subtítulos flotantes, gestionando la composición de mensajes multi-seña.

**Componentes:**
- `MessageComposer`: concatena etiquetas sucesivas en un buffer de mensaje. Respeta `maxWords` y `continuityTimeout`. Al expirar el timer, cierra el mensaje y lo entrega al renderer.
- `SpatialSubtitleRenderer`: calcula el anclaje espacial óptimo (`computeAnchor()`) y renderiza el subtítulo flotante evitando ocluir el rostro del emisor. Garantiza renderizado a ≥72 FPS.

---

## 5. Pipeline Offline (Entrenamiento)

Corre fuera del sistema en tiempo real, en entorno Python. Produce el `OnnxModel` que consume `SignClassifier`.

| Clase | Responsabilidad |
|---|---|
| `DatasetRecorder` | Orquesta grabación + etiquetado. Exporta CSV crudo (esquema en la Sección 6). Metas: ≥50 muestras/seña, 10 señas, ≥2 señantes. El gate de muestras mínimas lo aplica `build_dataset.py` al construir el dataset. |
| `ModelTrainer` | Entrena modelo TCN. Produce objeto `Metrics` (accuracy + matriz de confusión), atributo `architecture = "TCN"`. **Dos modos con propósitos distintos:** `train()` hace un split estratificado 80/20 con early stopping y produce el **modelo exportable**; `evaluar_cv()` hace **validación cruzada estratificada k-fold** (k=5) y produce la **métrica reportable** (accuracy media ± desviación estándar + matriz de confusión out-of-fold). Con corpus pequeño el 80/20 deja pocas muestras de validación por clase, así que la cifra que se reporta es la de k-fold. |
| `ModelExporter` | Exporta modelo a `OnnxModel`. Ejecuta `validateOperators()` para verificar compatibilidad ONNX/Unity Sentis (`sentisCompatible = true`) antes de habilitar el modelo para runtime. |

---

## 6. Clases de Entidad y Tipos de Datos

| Tipo | Descripción | Dimensiones |
|---|---|---|
| `Keypoint` | Coordenada articular cruda `(x, y, z)` | 3 valores float |
| `Frame` | Snapshot de una mano: exactamente 21 `Keypoint` | 63 valores float |
| `HandFrame` | Realización concreta de `Frame`: landmarks + handedness (`Left`/`Right`) + score de handedness + timestamp | 21×3 + metadatos |
| `MultiHandFrame` | Detección de un frame completo: 0, 1 o 2 `HandFrame`. Es lo que devuelve `getFrame()` | 0–2 manos |
| `NormVector` | Keypoints normalizados (salida de `KeypointNormalizer`) | 63 dim |
| `SignSequence` | Secuencia temporal correspondiente a una seña | T frames (variable) |
| `EntradaModelo` | `SignSequence` ensamblada y remuestreada, lista para el clasificador | 60 × 63 (o 60 × 126) |
| `SignEvent` | Evento emitido por `RestStateDetector` | — |
| `ClassResult` | Etiqueta de texto + score de confianza (salida de `SignClassifier`) | — |
| `OnnxModel` | Artefacto compartido entre pipeline offline (lo produce) y runtime (lo consume) | — |
| `DatasetCSV` | CSV crudo: `sample_id, frame_idx, hand, handedness_score, x0..x20, y0..y20, z0..z20, label` | 63 valores crudos/mano/frame |

**Nota sobre `SignSequence` y los 60 frames.** La captura **no** está acotada a 60 frames: el `RestStateDetector` acumula tantos frames como dure la seña. Los 60 son la longitud fija a la que el remuestreo temporal lleva cualquier secuencia antes de la inferencia (Sección 4, Capa 2). Una seña más larga se comprime, no se trunca.

**Nota sobre el `DatasetCSV`.** El CSV crudo guarda coordenadas **sin normalizar** —la normalización se aplica al construir el dataset, no al grabar— y añade tres columnas que el esquema mínimo no contemplaba pero que son funcionalmente necesarias: `sample_id` (agrupa los frames de una misma muestra), `hand` y `handedness_score` (permiten reconstruir la mano dominante y el modo de dos manos a partir del mismo archivo).

### Punto abierto — una mano vs. dos manos

El sistema soporta **ambos modos**, configurables (`MODO_MANOS`), sin que el `KeypointNormalizer` cambie (siempre produce 63 dim por mano); lo que cambia es el ensamblado del vector de entrada:

| Modo | Entrada al clasificador | Cuándo conviene |
|---|---|---|
| `dominante` | 60 × 63 | Vocabulario de señas mayoritariamente de una mano. |
| `ambas` | 60 × 126, orden `[Left \| Right]`, mano ausente → ceros | Vocabulario con señas bimanuales. |

La decisión definitiva depende de cuántas de las 10 glosas del corpus sean bimanuales, y debe respaldarse con la métrica k-fold de ambos modos sobre el mismo corpus.

### Nomenclatura
Los métodos citados como contrato (`getFrame()`, `appendWord()`, `validateOperators()`) conservan su nombre literal en la implementación. Los **atributos** siguen PEP 8 en Python: `confThreshold` es `conf_threshold`, `maxWords` es `max_words`, `continuityTimeout` es `continuity_timeout`. La diferencia es de convención de lenguaje, no de diseño.

---

## 7. Máquinas de Estado

### 7.1 RestStateDetector

```
         manos abandonan posición neutral
[REPOSO] ─────────────────────────────────► [CAPTURANDO]
    ▲                                             │
    │  retorno sostenido a reposo (→ emite FIN)  │ acumula frames
    └─────────────────────────────────────────────┘
    ▲
    │  pérdida de tracking (EX-01) → descarta secuencia parcial
    └─────────── [CAPTURANDO] ──────────────────────────────────
```

Estados: `Reposo`, `Capturando` (exactamente dos).

Eventos de salida: `IDLE`, `START`, `CAPTURING`, `END` (dispara `SignSequence` a `SignClassifier`), `DISCARDED` (EX-01: pérdida de tracking o seña más corta que el mínimo). Los cinco valores existen para que el orquestador pueda dar retroalimentación visual del estado; solo `END` y `DISCARDED` alteran la máquina de estados de dos estados.

Ambas transiciones exigen evidencia **sostenida**, no un único frame: `REST_FRAMES_INICIO = 3` frames de movimiento para abrir la captura y `REST_FRAMES_FIN = 6` frames de reposo para cerrarla. Los frames de reposo finales se recortan de la secuencia antes de clasificar.

### 7.2 MessageComposer (buffer de composición)

El buffer transita por estados de composición activa, espera de continuidad, y cierre de mensaje (timeout o gesto de reset manual). El gesto de reinicio manual activa el flujo alternativo FA-01 de CU-03 y limpia el buffer de forma anticipada.

**Implementación del timeout.** No hay hilo ni temporizador activo: el cierre por expiración se evalúa (a) al llegar la siguiente palabra y (b) en la llamada `tick()` que el orquestador hace una vez por frame. El efecto observable es el que describe el diseño —el mensaje se cierra al expirar el timer, sin necesidad de que el usuario siga señando— pero la comprobación es síncrona con el bucle de render, no asíncrona.

---

## 8. Casos de Uso

| ID | Nombre | Actores principales |
|---|---|---|
| CU-01 | Traducción en tiempo real de seña dinámica a texto | Ciudadano Sordo (primario), Funcionario (secundario), Meta XR SDK / MediaPipe |
| CU-02 | Construcción y entrenamiento del corpus LSCh | Ingeniero de IA (primario), Señante Nativo (secundario), TensorFlow/Keras |
| CU-03 | Composición de mensaje multi-seña | Ciudadano Sordo (primario) — **include obligatorio a CU-01** |

**Flujo principal CU-01 (resumen):**
1. Ciudadano seña → MediaPipe/SDK extrae keypoints
2. `RestStateDetector` detecta inicio de seña
3. `KeypointNormalizer` normaliza cada frame en tiempo real
4. `RestStateDetector` detecta fin de seña → emite `SignSequence`
5. `SignClassifier` infiere etiqueta → `ClassResult`
6. Si confianza ≥ `confThreshold`: `SpatialSubtitleRenderer` renderiza texto
7. Si confianza < `confThreshold`: muestra indicador "seña no reconocida" (FA-01)

**Excepciones CU-01:**
- `EX-01`: Pérdida de tracking → `RestStateDetector` resetea a Reposo, descarta secuencia parcial.
- `EX-02`: FPS < 72 → Unity prioriza renderizado, registra log de degradación.

---

## 9. Stack Tecnológico Consolidado

| Categoría | Tecnología | Versión | Notas |
|---|---|---|---|
| Captura (MVP) | MediaPipe Hands (Tasks API) | ≥0.10 | 21 landmarks/mano, webcam estándar |
| Captura (producción) | Meta Hand Tracking API (Meta XR SDK) | v74+ | Mismo esquema 21 landmarks |
| Hardware final | Meta Quest 3 | — | Snapdragon XR2 Gen 2 + NPU |
| Motor / Renderizado | Unity + C# | 6 LTS | ≥72 FPS requerido |
| Preprocesamiento | Python + NumPy | ≥1.26 | Solo pipeline offline |
| Framework entrenamiento | TensorFlow / Keras | ≥2.16 | Entrena modelo TCN |
| Modelo de IA | TCN | — | LSTM y Transformer Encoder descartados |
| Exportación | ONNX | — | TFLite descartado |
| Runtime inferencia | Unity Sentis | ≥2.1 | Barracuda descartado |
| Interfaz espacial | TextMesh Pro / Unity UI Toolkit | — | Subtítulos flotantes anclados |
| Control de versiones | Git / GitHub | — | Repositorios por integrante |

---

## 10. Responsabilidades por Integrante

| Integrante | Componentes propios |
|---|---|
| **Juan Yampara** (IA y datos) | `HandTrackingProvider`, `RestStateDetector`, `KeypointNormalizer`, `SignClassifier`, `DatasetRecorder`, `ModelTrainer`, `ModelExporter` |
| **Tomás Silva** (Unity/XR) | `MessageComposer`, `SpatialSubtitleRenderer`, rendering Unity, Spatial UI |

**Orquestador de referencia.** El diseño nombra los componentes pero no el elemento que los encadena en runtime. En el repositorio de IA/datos ese papel lo cumple `demo_vivo.py`: instancia las cuatro capas, ejecuta el bucle captura → segmentación → normalización → inferencia → composición y dibuja el subtítulo. Es la **implementación de referencia de CU-01 + CU-03** y el banco de pruebas contra el que debe contrastarse el orquestador equivalente en Unity; no es el entregable final.

El repositorio de IA/datos incluye también una copia de validación de `MessageComposer` en Python, para poder probar CU-03 sin la capa Unity. La versión productiva es la de Tomás.

**Contrato entre repositorios.** El acoplamiento es solo por archivos (`modelo.onnx` + `labels.json`), nunca por red. La especificación para portar el preprocesamiento a C#, junto con vectores de prueba dorados que detectan divergencias, está en `INTEGRACION_UNITY.md`.

---

## 11. Métricas de Éxito del MVP

| Métrica | Umbral | Cómo se mide |
|---|---|---|
| Accuracy de clasificación | ≥ 85% | Media de la validación cruzada k-fold **dejando señantes fuera** (ver nota). |
| Latencia end-to-end (captura → subtítulo) | ≤ 500 ms | Medida en el orquestador y contrastada contra `LATENCIA_MAX_MS`. |
| FPS de renderizado sostenido | ≥ 72 FPS | Capa 4 (Unity); no medible desde el subsistema de IA/datos. |
| Task Success Rate (escenario ventanilla) | ≥ 80% en ≥ 10 pruebas | Pruebas con usuarios, posteriores al corpus. |

**Qué incluye la latencia end-to-end.** El presupuesto de 500 ms se mide desde que la seña termina realmente hasta que el subtítulo queda dibujado, y se compone de:

1. **Retardo de segmentación** — los `REST_FRAMES_FIN = 6` frames de reposo que el detector necesita para confirmar el fin de la seña. A 30 FPS son ~200 ms, y es la parte dominante del presupuesto. Es inherente al diseño: el sistema no puede saber que la seña terminó antes de confirmarlo.
2. **Preprocesamiento** — normalización, ensamblado y remuestreo de la secuencia.
3. **Inferencia** — ejecución del ONNX (~3,5 ms en CPU de PC).
4. **Composición y render** del subtítulo.

Medir solo (3) daría una cifra irrelevante frente al umbral. La medición debe reportar el total.

---

## 12. Corpus LSCh (Dataset propio)

> **Corpus efectivamente usado en el MVP (decisión 2026-07-26).** El equipo no tuvo acceso a señantes de LSCh, de modo que el corpus propio no llegó a grabarse y **el MVP se construye y valida sobre LSA64** (lengua de señas argentina): 10 señas × 10 señantes × 5 repeticiones, la misma forma que especifica esta sección. También se evaluó SWL-LSE (lengua de señas española), descartado porque no publica vídeo entrenable —solo keypoints ya extraídos, en esquema Holistic, cuyo uso reabriría `DECISION_PREPROCESAMIENTO.md`—. El vocabulario demostrado son las 10 señas de LSA64, no las glosas del catálogo LSCh, que quedan como vocabulario objetivo de diseño. **La generalización a LSCh no está probada y es trabajo futuro.** LSA64 es CC BY-NC-SA 4.0: atribución obligatoria, uso no comercial y derivados bajo la misma licencia.

Especificación original del corpus (objetivo de diseño):

- **10 señas dinámicas** del vocabulario transaccional GORE (trámite, documento, firma, identidad, esperar, etc.)
- **≥ 50 muestras por seña**, ≥ 2 señantes nativos de LSCh
- **Formato CSV crudo:** `sample_id, frame_idx, hand, handedness_score, x0..x20, y0..y20, z0..z20, label` (63 valores **sin normalizar** por mano y frame; la normalización se aplica al construir el dataset)
- **Evaluación:** split 80/20 estratificado para producir el modelo exportable, y **validación cruzada estratificada k-fold (k=5)** para la cifra de accuracy que se reporta
- Referencias de validación de pipeline (no son corpus LSCh):
  - **SWL-LSE** (Zenodo, CC BY 4.0): 300 señas sanitarias en LSE. Se usa su conjunto de vídeos de referencia, con **1 muestra por clase**, por lo que sirve para validar la cadena vídeo → keypoints → dataset, **no** para medir accuracy.
  - **LSA64** (CC BY-NC-SA 4.0): 64 señas de lengua de señas argentina, 10 señantes × 5 repeticiones. Al tener 50 muestras por seña permite construir un corpus proxy **de la misma forma que el objetivo** (10 clases × 50 muestras) y obtener una medida de accuracy honesta del pipeline mientras el corpus LSCh no está grabado. Sus señantes graban con **guantes de colores**, condición adversa para un detector entrenado sobre manos desnudas: la cifra obtenida sobre este corpus es una **cota inferior** del desempeño del pipeline, no una estimación centrada.

En ambos casos los keypoints se **re-extraen con la Capa 1 propia**; no se reutilizan los keypoints ni el preprocesamiento de terceros (ver `DECISION_PREPROCESAMIENTO.md`).

---

## 13. Limitaciones Declaradas del Diseño

- **El sistema está validado sobre lengua de señas argentina (LSA64), no chilena.** Es la limitación principal del trabajo: por indisponibilidad de señantes de LSCh no existe corpus chileno, y ninguna cifra de este informe puede presentarse como desempeño sobre LSCh. Lo que se demuestra es que la arquitectura reconoce señas dinámicas aisladas con la accuracy objetivo; aplicarla a LSCh requiere grabar su corpus y repetir la evaluación, sin cambios de código previstos.
- Las cifras son una **cota inferior**: los señantes de LSA64 graban con guantes de colores, condición adversa para un detector entrenado sobre manos desnudas.
- El corpus tiene 10 señantes, pero de una sola lengua y un solo entorno de grabación (fondo, iluminación y encuadre uniformes). No hay evidencia sobre robustez a condiciones de ventanilla real.
- La conformidad diseño↔implementación está verificada solo para el subsistema de IA/datos; la Capa 4 vive en otro repositorio y requiere auditoría propia.
- Vocabulario cerrado de 10 señas; ampliación via Transfer Learning (FA-01 de CU-02).
- El umbral `confThreshold` de `SignClassifier` debe calibrarse empíricamente sobre el corpus real.
- La compatibilidad de operadores ONNX con Unity Sentis debe validarse con un modelo toy antes de la exportación final (`ModelExporter.validateOperators()`).
