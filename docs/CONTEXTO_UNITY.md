# Contexto para el trabajo en Unity — LSCh-MR

> Documento de traspaso. Está escrito para alguien (o algo) que empieza en frío en
> el **repositorio de Unity**, sin haber visto el repositorio de Python.
> Cópialo allí junto con los tres archivos del contrato (sección 4).
>
> Generado el 2026-07-26 desde el repo de IA/datos, commit `526f4a4` + cambios de
> calibración.

---

## 1. Qué es el sistema

**LSCh-MR** traduce señas dinámicas a subtítulos en pantalla, para atención de
público en ventanilla municipal. Es un proyecto de titulación (UTA) con dos
integrantes y **dos repositorios**:

| Integrante | Responsabilidad | Repositorio |
|---|---|---|
| Juan Yampara | IA y datos: captura, preprocesamiento, entrenamiento, ONNX | Python (el de origen) |
| Tomás Silva | Runtime Unity: hand tracking en Unity, inferencia con Sentis, UI de subtítulos | **este** |

El acoplamiento entre ambos es **solo por archivos**. No hay ni puede haber
comunicación por red entre ellos (ver sección 5).

### Arquitectura en 4 capas

```
Capa 1  Captura          HandTrackingProvider · RestStateDetector
Capa 2  Preprocesamiento KeypointNormalizer (+ ensamblado + remuestreo)
Capa 3  Inferencia       SignClassifier  [modelo TCN en ONNX]
Capa 4  Presentación     MessageComposer · SpatialSubtitleRenderer
```

Las cuatro existen **en Python** y funcionan end-to-end. En Unity hay que
construir la 1, la 2, la 3 (envolviendo Sentis) y la 4.

---

## 2. Estado real hoy — leer antes de planificar

**En Unity no existe nada todavía.** Cero archivos `.cs`. Todo lo que sigue está
hecho y medido del lado Python.

**El modelo ya está entrenado y exportado**, con métrica honesta:

| | |
|---|---|
| Arquitectura | TCN (Conv1D causal dilatada) |
| Clases | 10 |
| Accuracy | **0.92 ± 0.07** en validación cruzada dejando señantes fuera |
| Umbral de confianza | **0.90** (calibrado, ver sección 6) |
| Inferencia en PC | ~0.4 ms (onnxruntime, CPU) |

**El vocabulario NO está en español.** El corpus del MVP es **LSA64 (lengua de
señas argentina)**, porque el equipo no tuvo acceso a señantes de Lengua de Señas
Chilena. Las 10 etiquetas que devuelve el modelo son, en este orden:

```
0 Accept   1 Appear   2 Call    3 Give      4 Help
5 Last_name 6 Name    7 None    8 Patience  9 Thanks
```

Esto importa para la UI: el subtítulo mostrará `Thanks`, no `GRACIAS`. Si se
quiere mostrar en español, la traducción de la etiqueta es una decisión de
presentación (Capa 4) y debe quedar documentada como tal — **no** renombrar las
clases del modelo, que se corresponden con señas argentinas reales.

---

## 3. La demo que hay que construir

**Unity + Sentis sobre PC con webcam.** El Meta Quest 3 salió del camino crítico
(decisión del 2026-07-21): no hay visor, no hay Meta XR SDK, no hay subtítulo
espacial anclado en el mundo. El subtítulo va en pantalla.

Consecuencia importante: **la Capa 1 hay que reimplementarla sobre webcam en
Unity**. El plan original tomaba los 21 keypoints del Meta XR SDK; sin visor, hay
que obtenerlos de otra forma (MediaPipe en Unity o equivalente) manteniendo el
contrato `getFrame()`. Es la mayor incógnita técnica que queda.

> ⚠️ **Un puente Python↔Unity por red no es una opción.** La Sección 3 del diseño
> prohíbe MQTT, gRPC y REST entre componentes, y está consolidada. Si la única
> salida fuese pasar keypoints desde Python por sockets, eso es un cambio de
> arquitectura que hay que subir al equipo, no una decisión de implementación.

---

## 4. El contrato: cinco archivos

Cópialos desde el repo de Python a este:

| Archivo | Origen | Qué es |
|---|---|---|
| `modelo.onnx` | `outputs/models/` | El clasificador. **sha1 `a1238b32…`** |
| `labels.json` | `outputs/models/` | Metadatos del modelo |
| `vectores_dorados.json` | `integracion/` | 8 casos de prueba de la **Capa 2** (mano sintética) |
| `secuencia_dorada.json` | `integracion/` | 18 frames de una seña real: valida la **Capa 1** (extracción) y la cadena completa |
| `secuencia_dorada.csv` | `integracion/` | Lo mismo en plano, para mirar a ojo |

`labels.json` **manda** sobre cualquier constante que escribas en C#:

```json
{ "classes": [...10 etiquetas...], "modo_manos": "dominante",
  "seq_len": 60, "n_features": 63 }
```

Firma del ONNX: entrada `input` de forma `[batch, 60, 63]` float32; salida
`output` de forma `[batch, 10]`, **ya pasada por Softmax** — no aplicar softmax
otra vez. Opset 13. Operadores usados: `Add, Conv, MatMul, Pad, ReduceMean, Relu,
Softmax, Squeeze, Transpose, Unsqueeze`, todos verificados contra la lista que
soporta Sentis.

Los archivos van **siempre juntos**: si el modelo se reentrena, cambian todos. Un
`labels.json` desparejado del `.onnx` ya causó un fallo en este proyecto.

**La especificación completa del preprocesamiento a portar, con código C# de
referencia, está en `INTEGRACION_UNITY.md`** (cópialo también). No la reconstruyas
leyendo el código Python: el documento existe justo para evitar eso.

---

## 5. Decisiones consolidadas — no reabrir

Proponer alternativas a esto genera inconsistencia entre el código y el diseño ya
entregado, y eso es un criterio de evaluación penalizado en el proyecto.

- **Clasificador: TCN.** LSTM y Transformer descartados.
- **Runtime: ONNX vía Unity Sentis ≥ 2.1.** TensorFlow Lite y Barracuda descartados.
- **Comunicación entre capas: en memoria.** Sin red, sin IPC.
- **21 keypoints por mano, `NormVector` de 63 dim**, centrado en muñeca (L0) y
  escalado por la distancia L0–L9.
- **Reconocimiento de señas dinámicas aisladas**, con pausas de reposo entre
  señas. El reconocimiento continuo (CSLR) es trabajo futuro declarado.
- **Preprocesamiento propio**, sin reutilizar código ni keypoints de terceros.
- **`running_mode` del `HandLandmarker`: `IMAGE`.** No `VIDEO`, no `LIVE_STREAM`
  — ni en Python ni en Unity. Es el modo con el que se extrajo el corpus, así que
  es el único coherente con `modelo.onnx`. Medido sobre el mismo vídeo, `VIDEO`
  produce keypoints que divergen hasta **1.79** en el `NormVector` (la tolerancia
  del port es `1e-5`) y elige distinta mano dominante en 16 de 18 frames.
  Detalle, evidencia y qué lo reabriría: `INTEGRACION_UNITY.md` sección 7.
- **La imagen NO va en espejo** al `HandLandmarker`: invertiría el eje `x` y la
  handedness. Si quieres vista espejo para el usuario, voltea solo lo que dibujas.

---

## 6. Qué construir, en qué orden

**Primero esto, antes de tocar nada de UI:**

1. **Portar el `KeypointNormalizer` a C#** siguiendo `INTEGRACION_UNITY.md`.
2. **Hacer pasar los 8 vectores dorados** con tolerancia 1e-5, como test.

La razón de ese orden: si el preprocesamiento en C# diverge aunque sea un poco
del de Python, el modelo recibe una entrada distinta a la que vio en
entrenamiento y **devuelve basura — pero el ONNX carga bien, la inferencia corre
sin excepciones y la demo parece funcionar**. El fallo es silencioso y se
confunde con "el modelo está mal entrenado". Con los vectores dorados se detecta
en minutos; sin ellos se pierde una tarde larga.

Después:

3. **Capa 1 en Unity**: obtener 21 landmarks por mano desde la webcam, con el
   plugin de homuler en `running_mode: IMAGE` (sección 5 — no es negociable) y
   `numHands: 2`, umbrales de confianza `0.5`. `secuencia_dorada.json` trae 18
   frames consecutivos de una seña real para comparar frame a frame: primero la
   handedness y el número de manos, que son discretos, y después los valores.
4. **Capa 3**: cargar el ONNX en Sentis, tensor `(1, 60, 63)` float32.
   El caso `inferencia_onnx_entrada_conocida` de los vectores dorados comprueba
   que Sentis da los mismos scores que onnxruntime.
5. **`RestStateDetector`**: máquina de 2 estados (`reposo` / `capturando`).
   Abre con 3 frames de movimiento sostenido, cierra con 6 de reposo sostenido,
   y recorta esos 6 finales de la secuencia. Pérdida de tracking sostenida en
   captura (> 3 frames seguidos sin mano, `REST_FRAMES_PERDIDA_MAX`) → descarta
   sin clasificar (excepción EX-01).
   > **Corregido el 2026-07-26**: un parpadeo de 1-3 frames sin mano detectada
   > (normal en `running_mode=IMAGE`, que redetecta la palma en cada frame sin
   > arrastrar el ROI del anterior) ya NO descarta la captura — antes lo hacía
   > con un solo frame perdido, y eso tiraba señas completas por un parpadeo de
   > detección, no por una pérdida real. Si el port en C# no replica esta
   > tolerancia, va a descartar señas mucho más seguido que Python con la misma
   > cámara.
6. **Capa 4**: `MessageComposer` (buffer de palabras, `maxWords`,
   `continuityTimeout`, reinicio manual) y el render del subtítulo.

### Umbral de confianza

Si el score máximo es **< 0.90**, la seña se reporta como no reconocida (flujo
alternativo FA-01) en vez de mostrar una glosa. Ese valor está calibrado sobre
datos reales, no es arbitrario:

| Umbral | Cobertura | Precisión | Glosas erróneas mostradas |
|---|---|---|---|
| 0.60 | 94.0% | 92.1% | 36 |
| **0.90** | **79.1%** | **96.3%** | **14** |
| 0.99 | 61.7% | 96.0% | 12 |

Criterio: en ventanilla, mostrar una glosa equivocada engaña al funcionario,
mientras que "no reconocida" solo pide repetir la seña. Espera que
**aproximadamente 1 de cada 5 señas se rechace** — es el comportamiento
diseñado, no un fallo.

**No subas el umbral en C# buscando más precisión.** El modelo está mal
calibrado: la p95 de la confianza de sus predicciones erróneas es 1.00, o sea que
hay fallos con confianza máxima que ningún umbral filtra. Pasar de 0.90 a 0.99
solo quita 2 de las 14 glosas erróneas y cuesta 17 puntos de cobertura.

---

## 7. Métricas del MVP que dependen de Unity

| Métrica | Umbral | Estado |
|---|---|---|
| FPS de renderizado | ≥ 72 | **Sin medir** — responsabilidad de esta capa |
| Latencia end-to-end | ≤ 500 ms | Medida en Python: mediana 239 ms, pero la cola se pasa |
| Task Success Rate | ≥ 80% en ≥ 10 pruebas | **Sin empezar** — necesita la demo funcionando |
| Accuracy | ≥ 85% | ✅ cumplida (0.92) |

Sobre la latencia: el presupuesto de 500 ms se mide desde que la seña **termina
realmente** hasta que el subtítulo está dibujado. Su componente dominante no es
la inferencia (~0.4 ms) sino el **retardo de segmentación**: los 6 frames de
reposo que hacen falta para confirmar que la seña acabó. A 30 FPS son ~200 ms de
los 500. Si el presupuesto se pasa, la palanca es bajar esos 6 frames a 4, a
costa de cerrar señas antes y arriesgar cortes prematuros.

---

## 8. Riesgos conocidos

1. **Train/serve skew** — el principal. Mitigado por los vectores dorados, no
   eliminado. Riesgo residual: la normalización quita traslación y escala, pero
   **no** un cambio de convención de ejes. Si en Unity la `y` crece hacia arriba
   y en la extracción de entrenamiento crecía hacia abajo, los vectores parecen
   válidos y el modelo falla. Lo mismo con la `z`, cuya semántica cambia según la
   fuente de tracking.
2. **Hand tracking en Unity sin Meta XR SDK** — sin resolver. Si la solución que
   se elija entrega landmarks en una convención distinta a MediaPipe, hay que
   convertir antes de normalizar, o reentrenar.
3. **El modelo está mal calibrado** — la confianza mediana de sus predicciones
   *erróneas* es 0.85. Por eso el umbral es un instrumento romo y no conviene
   subirlo más: cuesta mucha cobertura y quita pocos errores.
4. **El corpus es argentino, no chileno.** La generalización a LSCh no está
   probada. Es la limitación principal declarada del trabajo y debe aparecer en
   el informe, no solo en un anexo.

---

## 9. Dónde está cada cosa (repo de Python)

| Quiero… | Ver |
|---|---|
| La spec del port a C# y el código de referencia | `INTEGRACION_UNITY.md` |
| El diseño completo y sus decisiones | `arquitectura_LSCh-MR.md` |
| Estado del pipeline, resultados y qué falta | `ESTADO_ACTUAL.md` |
| Por qué el preprocesamiento es propio | `DECISION_PREPROCESAMIENTO.md` |
| Cómo se corre el pipeline | `README.md` |
| La implementación de referencia del runtime | `scripts/demo_vivo.py` |
