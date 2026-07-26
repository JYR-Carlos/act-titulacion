# Arquitectura de Referencia — Sistema LSCh-MR
> **Propósito de este documento:** Describir la arquitectura diseñada en el informe consolidado final del proyecto LSCh-MR, para que pueda ser comparada contra la implementación real del repositorio y detectar discrepancias entre diseño e implementación.

---

## 1. Descripción General

**LSCh-MR** es un sistema de traducción en tiempo real de señas dinámicas de la Lengua de Señas Chilena (LSCh) a subtítulos flotantes en español, destinado a contextos de atención al público municipal (ventanilla GORE).

El sistema adopta una **arquitectura en capas con procesamiento en el borde (Edge Computing)**: todo el procesamiento ocurre localmente en el dispositivo, sin dependencia de red ni servicios cloud. La comunicación entre capas se realiza exclusivamente mediante **llamadas en memoria** (interfaces C# en runtime), sin protocolos de red ni IPC distribuido.

---

## 2. Plataforma de Despliegue

### MVP actual: PC + Webcam
El MVP se ejecuta sobre **PC con cámara web estándar**, con subtítulos desplegados en monitor. La extracción de keypoints se realiza con **MediaPipe Hands (Tasks API)**, que entrega el mismo esquema de 21 landmarks por mano que el Meta XR SDK.

### Objetivo final: Meta Quest 3
La migración al visor consiste únicamente en reemplazar:
- Fuente de keypoints: MediaPipe Hands → Meta Hand Tracking API (Meta XR SDK v74+)
- Canal de presentación: ventana en monitor → Spatial UI en Unity (Realidad Mixta)

Las capas de preprocesamiento e inferencia no requieren modificación en la migración.

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
| `DatasetRecorder` | Orquesta grabación + etiquetado. Exporta CSV con esquema `frame_idx, x0..x20, y0..y20, z0..z20, label`. Metas: ≥50 muestras/seña, 10 señas, ≥2 señantes. |
| `ModelTrainer` | Entrena modelo TCN sobre el CSV. Produce objeto `Metrics` (accuracy global + matriz de confusión). Atributo `architecture = "TCN"`. Split: 80% train / 20% test con validación cruzada. |
| `ModelExporter` | Exporta modelo a `OnnxModel`. Ejecuta `validateOperators()` para verificar compatibilidad ONNX/Unity Sentis (`sentisCompatible = true`) antes de habilitar el modelo para runtime. |

---

## 6. Clases de Entidad y Tipos de Datos

| Tipo | Descripción | Dimensiones |
|---|---|---|
| `Keypoint` | Coordenada articular cruda `(x, y, z)` | 3 valores float |
| `Frame` | Snapshot de una mano: exactamente 21 `Keypoint` | 63 valores float |
| `NormVector` | Keypoints normalizados (salida de `KeypointNormalizer`) | 63 dim |
| `SignSequence` | Secuencia temporal de `NormVector` correspondiente a una seña | 1 a 60 frames |
| `SignEvent` | Evento emitido por `RestStateDetector` (INICIO o FIN de seña) | — |
| `ClassResult` | Etiqueta de texto + score de confianza (salida de `SignClassifier`) | — |
| `OnnxModel` | Artefacto compartido entre pipeline offline (lo produce) y runtime (lo consume) | — |
| `DatasetCSV` | Archivo CSV con esquema `frame_idx, x0..x20, y0..y20, z0..z20, label` | 63 valores norm/frame |

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

Estados: `Reposo`, `Capturando`
Eventos de salida: `FIN` (dispara `SignSequence` a `SignClassifier`), `EX-01` (descarta y resetea)

### 7.2 MessageComposer (buffer de composición)

El buffer transita por estados de composición activa, espera de continuidad, y cierre de mensaje (timeout o gesto de reset manual). El gesto de reinicio manual activa el flujo alternativo FA-01 de CU-03 y limpia el buffer de forma anticipada.

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

---

## 11. Métricas de Éxito del MVP

| Métrica | Umbral |
|---|---|
| Accuracy de clasificación | ≥ 85% |
| Latencia end-to-end (captura → subtítulo) | ≤ 500 ms |
| FPS de renderizado sostenido | ≥ 72 FPS |
| Task Success Rate (escenario ventanilla) | ≥ 80% en ≥ 10 pruebas |

---

## 12. Corpus LSCh (Dataset propio)

- **10 señas dinámicas** del vocabulario transaccional GORE (trámite, documento, firma, identidad, esperar, etc.)
- **≥ 50 muestras por seña**, ≥ 2 señantes nativos de LSCh
- **Formato CSV:** `frame_idx, x0..x20, y0..y20, z0..z20, label` (63 valores normalizados / frame)
- Split: 80% entrenamiento / 20% test con validación cruzada
- Referencia de validación de pipeline: dataset **SWL-LSE** (Zenodo, CC BY 4.0, 300 señas dinámicas aisladas)

---

## 13. Limitaciones Declaradas del Diseño

- Corpus de señante único en fase MVP → requiere documentación explícita de esta limitación en el informe académico.
- Vocabulario cerrado de 10 señas; ampliación via Transfer Learning (FA-01 de CU-02).
- El umbral `confThreshold` de `SignClassifier` debe calibrarse empíricamente sobre el corpus real.
- La compatibilidad de operadores ONNX con Unity Sentis debe validarse con un modelo toy antes de la exportación final (`ModelExporter.validateOperators()`).
