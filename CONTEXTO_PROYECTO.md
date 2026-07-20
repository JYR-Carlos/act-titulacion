# LSCh-MR — Contexto del Proyecto

> Proyecto de titulación, Ingeniería en Computación e Informática, Universidad de Tarapacá (UTA).
> "Herramienta de asistencia para traducción de lenguaje de señas a subtítulos mediante realidad mixta"

## 1. Resumen

Sistema de reconocimiento de Lengua de Señas Chilena (LSCh) en tiempo real, que traduce
señas dinámicas discretas a subtítulos espaciales dentro de un visor de Realidad Mixta
(Meta Quest 3). Todo el pipeline corre **100% on-device / edge**, sin dependencia de
red ni servidores Cloud. Caso de uso objetivo: comunicación entre un ciudadano sordo y
un funcionario público (contexto originalmente acotado a atención en el GORE/Municipalidad
de Arica) sin necesidad de un intérprete presente.

Etapa actual: el diseño UML formal (Actividad 3 — Diseño de la Solución) ya está
entregado y consolidado. El proyecto está entrando en fase de implementación del
pipeline de datos e IA.

## 2. Equipo y responsabilidades

| Integrante | Área | Componentes a su cargo |
|---|---|---|
| **Juan Yampara** | Pipeline de IA / datos | `HandTrackingProvider`, `RestStateDetector`, `KeypointNormalizer`, `SignClassifier`, `DatasetRecorder`, `ModelTrainer`, `ModelExporter` |
| **Tomás Silva** | Spatial UI / Unity | `MessageComposer`, `SpatialSubtitleRenderer` |

Trabajo se coordina con Kanban en GitHub Projects (columnas: Backlog, En progreso,
En revisión, Bloqueado, Completado; WIP máx. 2 tareas simultáneas por integrante).

## 3. Decisiones de arquitectura — CONSOLIDADAS (no reabrir sin motivo fuerte)

Estas decisiones ya están documentadas y evaluadas formalmente. **No se deben proponer
alternativas (LSTM, Transformer, TFLite, GCN, microservicios, etc.) sin que el equipo lo
decida explícitamente** — hacerlo genera inconsistencia entre el código y el diseño
entregado, lo cual es un criterio de evaluación penalizado en el proyecto académico.

- **Arquitectura de clasificación:** TCN (Temporal Convolutional Network).
  Descartadas: LSTM, Transformer Encoder.
- **Formato/runtime de exportación:** ONNX vía Unity Sentis.
  Descartado: TensorFlow Lite / Barracuda.
- **Comunicación entre componentes:** interfaces C# en memoria, sin red ni protocolos
  (no MQTT, no gRPC, no REST). Todo corre en el mismo proceso/dispositivo.
- **Despliegue:** inferencia edge-only, monolítica, on-device (Meta Quest 3, Snapdragon
  XR2 Gen 2).
- **Reconocimiento:** señas dinámicas discretas (con pausas de reposo entre señas), no
  continuo. CSLR (Continuous Sign Language Recognition) queda fuera de alcance, definido
  como trabajo futuro.

### Métricas objetivo del MVP

- Accuracy ≥ 85% (corpus cerrado de 10 señas)
- Latencia end-to-end ≤ 500 ms
- Renderizado ≥ 72 FPS
- Task Success Rate ≥ 80% (mínimo 10 pruebas de usuario)

## 4. Stack tecnológico

**Runtime (Tomás / Unity):**
Meta Quest 3 · Meta XR SDK · Unity 6 LTS · C# · Unity Sentis (inferencia ONNX) ·
TextMesh Pro / Unity UI Toolkit (subtítulos)

**Pipeline de datos e IA (Juan / Python):**
Python · NumPy · MediaPipe Hands (usar **Tasks API / `HandLandmarker`**, NO la API
legacy `mp.solutions.hands`, descontinuada por Google en 2023) · TensorFlow/Keras ·
exportación a ONNX

**Validación intermedia (Sección 8.2 del diseño):** el MVP se valida primero en PC +
webcam (MediaPipe Hands) antes de portar a Quest 3 (Meta XR SDK). Solo cambian la fuente
de keypoints y el canal de salida; preprocesamiento e inferencia se mantienen intactos.

**Gestión de versiones:** Git / GitHub, repos separados por integrante, integración por
releases etiquetados.

## 5. Arquitectura del sistema — capas

```
Capa 1 (Captura)         → HandTrackingProvider, RestStateDetector
Capa 2 (Preprocesamiento) → KeypointNormalizer
Capa 3 (Inferencia)       → SignClassifier
Capa 4 (Presentación)     → MessageComposer, SpatialSubtitleRenderer
```

Orquestador: `TranslationSessionController` (compone todos los componentes runtime,
expone `Initialize()`, `OnFrameUpdate()`, `Shutdown()`).

Pipeline offline (paralelo, no runtime): `DatasetRecorder` → `ModelTrainer` →
`ModelExporter` → produce `modelo.onnx`, que se despliega («deploy») al pipeline
runtime. Es el único punto de acoplamiento entre offline y runtime.

## 6. Modelo de datos

- **Keypoint:** `(x, y, z)` — coordenada articular normalizada.
- **Frame:** exactamente 21 `Keypoint` (una mano, MediaPipe/Meta XR SDK estándar).
- **SignSequence:** entre 1 y 60 `Frame` (ventana temporal máxima de una seña).
- **NormVector:** 63 valores (21 keypoints × 3 ejes), salida de `KeypointNormalizer`.
  Centrado en la muñeca (landmark **L0**) y escalado por la distancia L0–L9
  (muñeca–base del dedo medio). Invariante a traslación y a escala (tamaño de mano /
  distancia a la cámara).

### Esquema CSV del corpus (dataset propio)

```
frame_idx, x0..x20, y0..y20, z0..z20, label
```
63 columnas de coordenadas + índice de frame + etiqueta (glosa). **Punto abierto:** el
esquema documentado asume una mano por fila; como LSCh usa señas a dos manos, el
script de extracción actual (`extraer_keypoints.py`) agrega columnas `hand`
(Left/Right) y `handedness_score`, escribiendo una fila por mano detectada por frame.
Falta decidir en el equipo si el `KeypointNormalizer`/`SignClassifier` consume una sola
mano dominante o ambas concatenadas (126 dim).

Meta del corpus: 10 señas dinámicas cerradas (ver `Glosas_LSCh_Mappeadas.csv`),
≥50 muestras por seña, ≥2 señantes nativos.

## 7. Componentes — contratos (de Sección 10.2)

**Pipeline runtime**

| Clase | Método clave | Responsabilidad |
|---|---|---|
| `HandTrackingProvider` | `getFrame(): Frame` | Encapsula el hand tracking del Meta XR SDK |
| `RestStateDetector` | `update(frame): SignEvent` | Máquina de estados; segmenta inicio/fin de seña por reposo |
| `KeypointNormalizer` | `normalize(frame): NormVector` | Centrado + escalado geométrico (ver Sección 6) |
| `SignClassifier` | `classify(seq): ClassResult` | Clasifica `SignSequence` con el modelo ONNX; usa `confThreshold` |
| `MessageComposer` | `appendWord(label): void` | Concatena palabras respetando `maxWords` y `continuityTimeout` |
| `SpatialSubtitleRenderer` | `render(text): void`, `computeAnchor()` | Ancla y renderiza el subtítulo flotante sin ocluir el rostro del emisor |

**Pipeline offline**

| Clase | Método clave | Responsabilidad |
|---|---|---|
| `DatasetRecorder` | `recordSession()` | Graba y etiqueta el corpus a nivel de frame |
| `ModelTrainer` | `train(): Metrics` | Entrena el TCN, produce accuracy + matriz de confusión |
| `ModelExporter` | `export(): OnnxModel`, `validateOperators()` | Exporta a ONNX y valida compatibilidad con Unity Sentis (`sentisCompatible`) |

## 8. Máquinas de estado

**`RestStateDetector`:** `Reposo` → (manos salen de reposo) → `Capturando` →
(retorno sostenido a reposo, evento `FIN`) → secuencia despachada a inferencia →
`Reposo`. Pérdida de tracking durante `Capturando` descarta la secuencia parcial y
vuelve a `Reposo` sin clasificar.

**Buffer de `MessageComposer`:** `Vacío` → (primera palabra) → `Acumulando` →
(se excede `maxWords` o vence `continuityTimeout`) → `Desplegado` → (tiempo de
lectura) → `Vacío`. Gesto de reinicio manual permite volver a `Vacío` en cualquier
momento desde `Acumulando`.

## 9. Casos de uso críticos

- **CU-01 — Traducción en tiempo real de una seña:** flujo completo captura →
  segmentación por reposo → normalización → clasificación → render del subtítulo.
  Incluye flujo alternativo de "seña fuera de vocabulario" (umbral de confianza) y
  excepciones de pérdida de tracking y degradación de FPS.
- **CU-02 — Construcción y entrenamiento del corpus:** grabación, extracción de
  keypoints, etiquetado, entrenamiento del TCN, exportación a ONNX. Incluye ampliación
  de vocabulario vía transfer learning.
- **CU-03 — Composición de subtítulo por concatenación:** agrupa resultados sucesivos
  de CU-01 en un mensaje compuesto, con `include` obligatorio hacia CU-01 (cada palabra
  del mensaje es una ejecución completa de CU-01).

## 10. Estado actual del código

Ya implementado y probado (en este repo / entregado por Claude):

- **`extraer_keypoints.py`** — extracción de keypoints con MediaPipe Tasks
  (`HandLandmarker`). Dos modos: `--modo webcam` (preview en vivo) y `--modo video`
  (batch, genera el CSV crudo). Requiere descargar `hand_landmarker.task` aparte
  (no incluido en `pip install mediapipe`).
- **`keypoint_normalizer.py`** — implementación de `KeypointNormalizer` según
  Sección 10.2 (centrado L0 + escalado L0–L9). Probado: invariante a traslación y
  escala, maneja división por cero en casos degenerados, valida forma de entrada.

Pendiente (backlog inmediato):

- Integrar `extraer_keypoints.py` + `keypoint_normalizer.py` en un solo flujo que
  produzca el CSV/dataset ya normalizado listo para entrenamiento.
- Decidir y resolver el punto abierto de una mano vs. dos manos (Sección 6).
- Implementar `RestStateDetector` (máquina de estados, Sección 8).
- Protocolo de grabación con el profesor de lengua de señas disponible (sesiones
  limitadas a 2 horas totales) — grabación en tomas continuas con segmentación
  automática posterior vía `RestStateDetector`.
- Definir arquitectura concreta del TCN (capas, kernel size, receptive field) y
  arrancar `ModelTrainer`.

## 11. Datasets de referencia

No existe corpus público de LSCh dinámico — el corpus propio es un aporte académico
original. Para validar el pipeline (extracción → normalización → clasificación) antes
de tener el corpus propio completo:

- **LSA64** (Lengua de Señas Argentina): señas dinámicas aisladas — mismo paradigma
  discreto que el proyecto. Mejor opción para validar el pipeline end-to-end.
- **SWL-LSE** ([Zenodo 10.5281/zenodo.13691887](https://zenodo.org/records/13691887)):
  300 señas aisladas en Lengua de Signos Española, dominio salud, keypoints ya
  extraídos con MediaPipe, CC BY 4.0, 3.5 GB. Buena opción secundaria — dominio
  institucional/atención más cercano semánticamente que LSA64.
- Descartado: **LSE-FS-UVigo** (Zenodo 15797079) — es deletreo (fingerspelling)
  continuo, no señas dinámicas aisladas; no calza con el enfoque discreto del MVP.

## 12. Convenciones de trabajo

- Todo el contenido del informe académico y la comunicación del equipo es en
  **español**. Comentarios de código y nombres de variables/funciones en español o
  inglés estándar según convención técnica habitual; los nombres de clases y métodos
  ya fijados en el diagrama de clases (Sección 10.2) deben respetarse tal cual están
  documentados, para mantener trazabilidad diseño↔código.
- Cambios de arquitectura (algoritmos, formatos, protocolos) deben evaluarse contra la
  Sección 3 (no reabrir decisiones consolidadas sin justificación explícita).
- Antes de adoptar código o datasets de terceros, verificar que el paradigma
  (discreto vs. continuo, arquitectura de modelo, esquema de keypoints) sea compatible
  con lo ya diseñado — evitar dependencias que fuercen a re-abrir decisiones cerradas.
