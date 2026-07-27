# Prompt: Evaluación de Conformidad Arquitectónica — Sistema LSCh-MR

## Instrucción de uso
Copia todo lo que está dentro del bloque `---PROMPT---` y pégalo como primer mensaje en una conversación nueva de Claude. Adjunta el archivo `docs/arquitectura_LSCh-MR.md` y todos los archivos de código fuente del repositorio (`.py`, `.cs`, etc.) antes de enviar.

---PROMPT---

Eres un evaluador de ingeniería de software. Tu tarea es comparar la **arquitectura diseñada** (descrita en el documento adjunto `docs/arquitectura_LSCh-MR.md`) contra la **implementación real** (los archivos de código fuente que se adjuntan), e identificar todas las conformidades y discrepancias entre ambas.

Este análisis está orientado a la entrega final de titulación, donde la rúbrica exige que "los diagramas y la arquitectura **coincidan exactamente** con la estructura real del software desarrollado." Las discrepancias tienen consecuencias directas sobre la nota.

---

## Paso 1 — Inventario del código

Antes de comparar, haz un inventario del código recibido. Lista:
- Todos los archivos `.py` con sus clases y funciones principales
- Todos los archivos `.cs` con sus clases y métodos principales
- Cualquier otro archivo relevante (configuración, modelos, scripts)

Presenta el inventario en formato de tabla: `Archivo | Clases/Funciones principales | Capa del sistema (según diseño)`

---

## Paso 2 — Comparación clase por clase

Para cada clase definida en el diseño, busca su equivalente en el código y evalúa:

### Pipeline Runtime (Python y/o C#)
Evalúa cada una de estas clases del diseño:

| Clase diseñada | Buscar en código | Verificar |
|---|---|---|
| `HandTrackingProvider` | Clase o módulo que lee keypoints desde MediaPipe o Meta XR SDK | ¿Produce objetos Frame con exactamente 21 keypoints? ¿Expone los 63 valores (x,y,z) por mano? |
| `RestStateDetector` | Clase o módulo de segmentación temporal | ¿Opera como máquina de estados? ¿Tiene estados Reposo/Capturando? ¿Emite evento equivalente a SignEvent al detectar fin de seña? ¿Descarta secuencias parciales ante pérdida de tracking? |
| `KeypointNormalizer` | Clase o función de normalización | ¿Centra en muñeca (landmark 0)? ¿Escala por distancia muñeca–dedo medio (landmark 9)? ¿Produce vector de exactamente 63 dimensiones? |
| `SignClassifier` | Clase o módulo de inferencia | ¿Carga modelo ONNX (no TFLite)? ¿Recibe una secuencia de frames (no un frame suelto)? ¿Retorna etiqueta + score de confianza? ¿Tiene umbral de confianza (confThreshold)? |
| `MessageComposer` | Clase o módulo de composición de texto | ¿Mantiene buffer de palabras? ¿Tiene timeout de continuidad? ¿Soporta reset manual? |
| `SpatialSubtitleRenderer` | Clase o módulo de renderizado | ¿Calcula anclaje espacial dinámico? ¿Evita ocluir el rostro del emisor? |

### Pipeline Offline (Python)
Evalúa:

| Clase diseñada | Verificar en código |
|---|---|
| `DatasetRecorder` | ¿Exporta CSV con esquema `frame_idx, x0..x20, y0..y20, z0..z20, label`? ¿Maneja etiquetado a nivel de frame? |
| `ModelTrainer` | ¿Entrena TCN (no LSTM, no Transformer)? ¿Divide dataset 80/20 con validación cruzada? ¿Genera accuracy + matriz de confusión? |
| `ModelExporter` | ¿Exporta a ONNX (no TFLite)? ¿Verifica compatibilidad de operadores con Unity Sentis antes de guardar? |

---

## Paso 3 — Verificación de tipos de datos y contratos

Verifica que los tipos de datos intermedios del diseño estén respetados en el código:

1. **Dimensión del vector de entrada al clasificador:** ¿El código pasa exactamente 63 valores por frame (o 126 si usa dos manos concatenadas)? Nota: el diseño especifica 63-dim para una mano; si el código usa dos manos, esto es una discrepancia que debe documentarse.
2. **Longitud máxima de secuencia:** ¿El código limita la secuencia a un máximo de 60 frames por seña?
3. **Arquitectura del modelo:** ¿El código instancia o carga un modelo TCN? ¿No hay LSTM ni Transformer?
4. **Formato del modelo guardado:** ¿El archivo de modelo es `.onnx`? ¿No hay archivos `.tflite` ni `.h5`?
5. **Runtime de inferencia:** ¿En el lado Unity, se usa Unity Sentis para cargar el ONNX? ¿No hay Barracuda?
6. **Comunicación entre capas:** ¿Toda la comunicación entre componentes es en memoria (llamadas a funciones/métodos)? ¿No hay sockets, HTTP, ni archivos intermedios en el pipeline de tiempo real?

---

## Paso 4 — Verificación de máquinas de estado

Para el `RestStateDetector`:
- ¿Hay exactamente dos estados operativos (Reposo y Capturando)?
- ¿La transición Reposo → Capturando se activa por detección de movimiento?
- ¿La transición Capturando → Reposo (con emisión de seña) requiere un retorno **sostenido** a reposo (no solo un frame)?
- ¿La pérdida de tracking descarta la secuencia sin emitir clasificación?

Para el `MessageComposer`:
- ¿Hay un timer/timeout que cierra el mensaje automáticamente?
- ¿Hay un mecanismo de reset manual equivalente a FA-01 de CU-03?

---

## Paso 5 — Verificación de métricas y umbrales

Busca en el código si están implementados (aunque sea como constantes o configuración):

| Métrica | Valor del diseño | ¿Presente en código? | Valor real en código |
|---|---|---|---|
| Latencia máxima end-to-end | ≤ 500 ms | | |
| FPS mínimo de renderizado | ≥ 72 FPS | | |
| Umbral de confianza del clasificador | `confThreshold` (valor a calibrar) | | |
| Señas en el corpus | 10 señas | | |
| Muestras mínimas por seña | ≥ 50 | | |

---

## Paso 6 — Identificación de elementos extra (en código pero no en diseño)

Lista todo lo que existe en el código pero **no está descrito en el diseño**:
- Clases adicionales no contempladas
- Métodos con responsabilidades que cruzan capas
- Dependencias de librerías no mencionadas en el stack tecnológico
- Cualquier llamada de red o archivo intermedio en el pipeline de tiempo real

---

## Paso 7 — Informe de discrepancias

Presenta el resultado final en este formato exacto:

### ✅ Conformidades
Lista de elementos del diseño que están correctamente implementados en el código.

### ⚠️ Discrepancias menores
Elementos que existen pero con diferencias en nombre, signatura de método, o valores de atributos. Para cada una: descripción, impacto en rúbrica, y acción recomendada (¿cambiar el código o actualizar el documento?).

### ❌ Discrepancias críticas
Elementos del diseño ausentes en el código, o implementaciones que contradicen decisiones arquitectónicas consolidadas (ej. uso de TFLite en lugar de ONNX, LSTM en lugar de TCN). Para cada una: descripción, impacto en rúbrica, y acción recomendada.

### 🆕 Elementos extra (en código, no en diseño)
Lista de lo que existe en el código y no está en el documento de arquitectura. Para cada uno: descripción y si debe agregarse al documento de diseño o si es prescindible.

### 📋 Lista de acciones priorizadas
Ordena todas las acciones recomendadas de mayor a menor impacto sobre la rúbrica del criterio "Rigor del Diseño Arquitectónico" (25% del informe consolidado):

1. [CRÍTICA] Acción X — razón
2. [IMPORTANTE] Acción Y — razón
3. [MENOR] Acción Z — razón

---PROMPT---
