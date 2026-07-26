# Integración con Unity/Sentis — contrato y port del preprocesamiento a C#

> Para el repo de Unity. Este documento y `integracion/vectores_dorados.json` son
> todo lo que hace falta de este lado: no hay que leer el código Python para
> portar la Capa 2 correctamente.

> ⚠️ **Antes de escribir una línea del extractor de manos en Unity, lee la
> [sección 7](#7-running_mode-oficial-image-decisión-cerrada).** El `running_mode`
> del HandLandmarker es **IMAGE**, no VIDEO ni LIVE_STREAM. Elegir mal ese
> parámetro no da un error: da un sesgo silencioso en **toda** seña dinámica.

## 1. Qué se comparte entre los dos repos

El acoplamiento entre el subsistema de IA/datos (Python) y el runtime (Unity) es
**solo por archivos**, nunca por red — la Sección 3 del diseño prohíbe MQTT/gRPC/REST,
y un puente Python↔Unity reabriría esa decisión.

| Archivo | Qué es |
|---|---|
| `outputs/models/modelo.onnx` | El clasificador TCN entrenado, opset 13. |
| `outputs/models/labels.json` | Metadatos del modelo: clases, `modo_manos`, `seq_len`, `n_features`. |
| `integracion/vectores_dorados.json` | Casos de prueba de la **Capa 2** (mano sintética, un frame). |
| `integracion/secuencia_dorada.json` | Casos de prueba de la **Capa 1 + 2** sobre una seña real: 18 frames consecutivos en movimiento. Ver sección 7. |
| `integracion/secuencia_dorada.csv` | Lo mismo en plano, para inspeccionar a ojo. El JSON es el autoritativo. |

`labels.json` **manda** sobre cualquier constante hardcodeada en C#: si el modelo
se reentrena en modo `ambas`, `n_features` pasa de 63 a 126 y el C# debe seguirlo
leyendo del JSON, no de una constante.

```json
{ "classes": ["...", "..."], "modo_manos": "ambas", "seq_len": 60, "n_features": 126 }
```

Firma del ONNX actual: entrada `input` de forma `[batch, 60, n_features]` float32;
salida `output` de forma `[batch, n_classes]` **ya pasada por Softmax** (no aplicar
softmax otra vez en C#).

## 2. El riesgo que este documento existe para evitar

**Train/serve skew.** El modelo aprendió sobre vectores producidos por el
preprocesamiento de Python. Si el C# produce vectores aunque sea ligeramente
distintos, el modelo devuelve predicciones sin sentido — pero el ONNX carga bien,
la inferencia corre sin excepciones y la demo *parece* funcionar. El fallo es
silencioso y se confunde fácilmente con "el modelo está mal entrenado".

Por eso el port no se da por bueno hasta que reproduce los vectores dorados.

## 3. La cadena a replicar, paso por paso

```
landmarks crudos (21 x 3 por mano)
  -> [A] normalización geométrica por mano      -> 63 floats
  -> [B] ensamblado según modo_manos            -> 63 o 126 floats por frame
  -> [C] remuestreo temporal a seq_len=60       -> [60, n_features]
  -> [D] tensor (1, 60, n_features) float32     -> Sentis
  -> [E] argmax + umbral de confianza           -> glosa o "<desconocida>"
```

### [A] Normalización geométrica (por mano, por frame)

1. Origen = landmark **L0** (muñeca). Restarlo a los 21 puntos.
2. Escala = `‖L9 − L0‖` **después de centrar** (L9 = base del dedo medio),
   norma euclidiana en 3D.
3. Si `escala < 1e-6` → devolver **63 ceros** (mano colapsada; no dividir).
4. Dividir los 21 puntos centrados por la escala.
5. Aplanar **fila-mayor**: `[x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]`.
6. Si algún valor resulta NaN o infinito → devolver **63 ceros** (todo el vector,
   no solo el componente afectado).

El resultado es invariante a traslación y a escala uniforme. **No** es invariante
al sistema de coordenadas: ver la sección 6 (trampas).

### [B] Ensamblado según `modo_manos`

- `"dominante"` → 63 floats: la mano con mayor `handedness_score`.
- `"ambas"` → 126 floats: `[Left(63) | Right(63)]`, **en ese orden**.
  Mano ausente → sus 63 posiciones van a **cero** (no se omite, no se duplica la
  otra mano, no se repite el último frame válido).

### [C] Remuestreo temporal a 60 frames

Interpolación **lineal**, característica por característica, de `T` frames a
exactamente 60. Equivale a `numpy.interp` sobre `linspace(0,1,T) → linspace(0,1,60)`:

```
pos  = j * (T - 1) / (60 - 1)        para j = 0..59
i0   = floor(pos);  i1 = min(i0+1, T-1);  frac = pos - i0
out[j][f] = feats[i0][f] * (1 - frac) + feats[i1][f] * frac
```

Casos límite: `T == 60` → copia directa; `T == 1` → repetir ese frame 60 veces.

> Ojo: el diseño describe la SignSequence como "1 a 60 frames", pero el detector
> **no** corta la captura a 60. Una seña larga puede traer más de 60 frames y el
> remuestreo la comprime. La longitud fija la impone este paso, no la captura.

### [D] Tensor de entrada

Forma `(1, 60, n_features)`, `float32`, en el mismo orden fila-mayor. Verificar
contra la forma declarada por el ONNX en vez de asumirla.

### [E] Decisión

`argmax` sobre la salida. Si el máximo `< 0.90` (`CONF_THRESHOLD`), la seña se
reporta como `"<desconocida>"` — es el flujo alternativo FA-01 de CU-01, no un
error.

El umbral **ya no es provisional**: se calibró el 2026-07-26 sobre las
predicciones out-of-fold de la validación por señante (483 muestras, LSA64).
A 0.90 la cobertura es 81.4% con 95.9% de precisión; a 0.60 sería 96.3% y 92.7%.
Se eligió 0.90 porque mostrar una glosa equivocada en ventanilla engaña al
funcionario, mientras que "no reconocida" solo pide repetir la seña. El valor vive
en `lsch_mr/config.py`; si cambia allí, cambia aquí.

## 4. C# de referencia

```csharp
public static class KeypointNormalizer
{
    public const int NumLandmarks = 21;
    public const int NormVectorDim = 63;
    const int WristIdx = 0;
    const int MiddleMcpIdx = 9;
    const float Eps = 1e-6f;

    /// Devuelve 63 floats. Mano ausente o degenerada -> todo ceros.
    public static float[] Normalize(Vector3[] frame)
    {
        var vec = new float[NormVectorDim];          // ceros por defecto
        if (frame == null || frame.Length != NumLandmarks) return vec;

        Vector3 origen = frame[WristIdx];
        Vector3 refe   = frame[MiddleMcpIdx] - origen;
        float escala   = Mathf.Sqrt(refe.x * refe.x + refe.y * refe.y + refe.z * refe.z);
        if (escala < Eps) return vec;                // mano colapsada

        for (int i = 0; i < NumLandmarks; i++)
        {
            Vector3 p = (frame[i] - origen) / escala;
            if (!IsFinite(p.x) || !IsFinite(p.y) || !IsFinite(p.z))
                return new float[NormVectorDim];     // vector completo a ceros
            vec[i * 3 + 0] = p.x;
            vec[i * 3 + 1] = p.y;
            vec[i * 3 + 2] = p.z;
        }
        return vec;
    }

    static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
}

public static class FeatureAssembler
{
    /// modo "ambas": [Left | Right], mano ausente -> 63 ceros.
    public static float[] Frame(Vector3[] left, Vector3[] right, bool ambas)
    {
        if (!ambas) return KeypointNormalizer.Normalize(right ?? left);

        var izq = KeypointNormalizer.Normalize(left);
        var der = KeypointNormalizer.Normalize(right);
        var salida = new float[KeypointNormalizer.NormVectorDim * 2];
        izq.CopyTo(salida, 0);
        der.CopyTo(salida, KeypointNormalizer.NormVectorDim);
        return salida;
    }
}

public static class TemporalResampler
{
    public static float[,] To(IReadOnlyList<float[]> feats, int seqLen)
    {
        int T = feats.Count, F = feats[0].Length;
        var outp = new float[seqLen, F];
        if (T == 0) return outp;

        for (int j = 0; j < seqLen; j++)
        {
            double pos = (T == 1) ? 0.0 : (double)j * (T - 1) / (seqLen - 1);
            int i0 = (int)System.Math.Floor(pos);
            int i1 = System.Math.Min(i0 + 1, T - 1);
            float frac = (float)(pos - i0);
            for (int f = 0; f < F; f++)
                outp[j, f] = feats[i0][f] * (1f - frac) + feats[i1][f] * frac;
        }
        return outp;
    }
}
```

## 5. Cómo validar el port

`integracion/vectores_dorados.json` trae 8 casos con entrada y salida esperada,
tolerancia absoluta **1e-5**:

| Caso | Qué detecta si falla |
|---|---|
| `normalize_mano_tipica` | La fórmula base. |
| `normalize_invariancia_traslacion` | No se centró en L0. |
| `normalize_invariancia_escala` | No se dividió por ‖L9−L0‖. |
| `normalize_mano_degenerada` | División por cero / NaN propagados. |
| `ensamblado_ambas_mano_ausente` | Orden [Left\|Right] o relleno con ceros mal hecho. |
| `remuestreo_temporal_17_a_60` | Interpolación temporal incorrecta. |
| `end_to_end_secuencia_cruda_a_entrada_del_modelo` | La cadena completa, dimensiones reales. |
| `inferencia_onnx_entrada_conocida` | Que **Sentis** dé los mismos scores que onnxruntime. |

Los siete primeros dependen solo del preprocesamiento y son estables. El octavo
depende del modelo: incluye el `sha1` del `.onnx` con el que se generó y **queda
invalidado al reentrenar**, así que el `.onnx`, el `labels.json` y el
`vectores_dorados.json` viajan siempre juntos. Si el `sha1` del modelo que tienes
en Unity no coincide con el del JSON, ese caso no aplica — pero los otros siete sí.

Escribe un test en C# que cargue el JSON y compare. Mientras no pasen, un
resultado malo de la demo no distingue entre "modelo malo" y "preprocesamiento mal
portado".

Del lado Python, `python generar_vectores_dorados.py --verificar` comprueba que el
JSON sigue coincidiendo con el pipeline. **Si alguien cambia el preprocesamiento en
Python, ese comando falla y el port en C# queda invalidado** — hay que regenerar el
JSON y avisar al equipo de Unity.

## 6. Trampas conocidas

1. **Sistema de coordenadas.** La normalización quita traslación y escala, pero
   **no** cambios de convención de ejes. Si en Unity la `y` crece hacia arriba y en
   la captura de entrenamiento crecía hacia abajo, los vectores se ven "válidos" y
   el modelo falla. Hay que alimentar el C# con landmarks en la misma convención
   que usó la extracción de keypoints, o aplicar la conversión antes de normalizar.
2. **La `z` no es comparable entre fuentes.** MediaPipe entrega una profundidad
   relativa en una escala propia; el Meta XR SDK entrega metros. El escalado por
   ‖L9−L0‖ compensa el factor global, pero no una `z` con semántica distinta. Si se
   cambia la fuente de tracking respecto a la del corpus, hay que reentrenar.
3. **Orden de manos.** Es `[Left | Right]` según la handedness que reporta el
   tracker, no "primera detectada / segunda detectada".
4. **`float32` en todo.** Acumular en `double` y truncar al final da diferencias por
   encima de la tolerancia en las secuencias largas.
5. **No aplicar softmax dos veces.** El grafo ONNX ya termina en Softmax.
6. **`labels.json` y `modelo.onnx` van en pareja.** Si se copia uno sin el otro, el
   índice de clase deja de corresponder a la glosa (y `n_features` puede no cuadrar
   con la entrada del modelo, que es un fallo ruidoso pero confuso).

## 7. `running_mode` oficial: **IMAGE** (decisión cerrada)

> **Decisión del 2026-07-26. El `running_mode` del `HandLandmarker` es `IMAGE`,
> en Python y en Unity. No se reabre sin re-extraer el corpus y reentrenar.**

### 7.1 Por qué IMAGE y no VIDEO

Porque es el modo con el que se extrajo el corpus del que salió `modelo.onnx`.
El corpus del MVP es LSA64 y se extrajo con `extraer_lote.py`, que crea el
provider así:

```python
# extraer_lote.py:77
provider = HandTrackingProvider(num_hands=args.num_hands, running_mode="image")
```

El modo de servicio tiene que ser el modo de extracción. El modelo solo conoce la
distribución de keypoints que vio al entrenar; si Unity produce keypoints de otra
distribución, el ONNX carga bien, Sentis infiere sin excepciones y las
predicciones son basura. Es el mismo train/serve skew de la sección 2, pero en la
Capa 1 en vez de la Capa 2 — y los vectores dorados de la Capa 2 **no lo
detectan**, porque parten de landmarks ya dados.

> **Esto no era consistente en Python hasta el 2026-07-26.** `HandTrackingProvider`
> tiene `running_mode="video"` por defecto, y solo `extraer_lote.py` lo
> sobrescribía. Los otros tres usos iban en `VIDEO`: `demo_vivo.py` (que alimenta
> a `modelo.onnx`, o sea que la propia demo de referencia tenía el skew),
> `DatasetRecorder` y `extraer_keypoints.py --modo webcam` (que escriben corpus).
> Los tres pasan ahora `running_mode="image"` explícito. Se menciona porque
> explica por qué el default de la clase sigue siendo `"video"` y por qué no hay
> que fiarse de él.

### 7.2 La diferencia entre los dos modos es enorme, no marginal

Medido con `generar_secuencia_dorada.py` sobre el mismo vídeo del corpus
(`051_001_001.mp4`, seña `Thanks`, 122 frames), pasando los **mismos** frames por
los dos modos. Tolerancia del port: `1e-5`.

| Qué se compara | dif. abs. máx | mediana |
|---|---|---|
| Misma mano física (emparejada por `handedness`), keypoints crudos | `5.25e-02` | `2.81e-02` |
| Misma mano física, `NormVector` (63) | `1.39e+00` | `7.74e-01` |
| Mano **dominante** (lo que consume el modelo), `NormVector` | `1.79e+00` | `1.45e+00` |

Y el censo de detección sobre los 122 frames del vídeo:

| Modo | 0 manos | 1 mano | 2 manos |
|---|---|---|---|
| `IMAGE` | 0 | 70 | 52 |
| `VIDEO` | 0 | 6 | **116** |

Tres cosas que leer aquí:

1. La divergencia del `NormVector` está **cinco órdenes de magnitud por encima**
   de la tolerancia con la que se valida el port (`1e-5`). No es ruido numérico.
2. `VIDEO` sostiene la segunda mano en 116 de 122 frames; `IMAGE`, en 52. `VIDEO`
   arrastra el ROI del frame anterior en vez de redetectar la palma, así que
   mantiene vivo un tracking que `IMAGE` pierde. En una seña bimanual eso cambia
   la seña entera, no un frame.
3. En 16 de los 18 frames de la ventana dorada, los dos modos eligen una **mano
   física distinta** como dominante. El modelo no recibiría una versión ruidosa
   de la misma mano: recibiría la otra.

> El tamaño del efecto es en parte propio de LSA64 — sus señantes graban con
> guantes de colores y `HandLandmarker` está entrenado sobre manos desnudas, así
> que la detección va justa y el arrastre de tracking de `VIDEO` la rescata más
> de lo que lo haría con manos desnudas. La **dirección** de la conclusión no
> depende de eso: los modos no son intercambiables, y el corpus manda.

### 7.3 LIVE_STREAM: no usarlo nunca

No se usa en Python y no debe usarse en Unity. `LIVE_STREAM` es asíncrono: entrega
resultados por callback y **descarta frames** para no acumular latencia. Una seña
es una secuencia de 1..60 frames consecutivos; perder frames de forma no
determinista la acorta y la deforma de un modo que no se puede reproducir ni
testear. `IMAGE` es síncrono y sin estado, que es justo lo que este pipeline
necesita.

### 7.4 Qué implica en el plugin de homuler

Con `MediaPipeUnityPlugin`, el equivalente es construir el `HandLandmarker` en
modo imagen y llamar a la variante **síncrona** de detección, una vez por frame:

```csharp
var options = new HandLandmarkerOptions(
    baseOptions: new BaseOptions(modelAssetPath: "hand_landmarker.bytes"),
    runningMode: Tasks.Vision.Core.RunningMode.IMAGE,   // <- NO VIDEO, NO LIVE_STREAM
    numHands: 2,
    minHandDetectionConfidence: 0.5f,
    minHandPresenceConfidence: 0.5f,
    minTrackingConfidence: 0.5f);

// Una llamada por frame, sin timestamp: en IMAGE el timestamp no existe.
handLandmarker.TryDetect(image, imageProcessingOptions, ref result);
```

Los tres umbrales de confianza son parte del contrato tanto como el
`running_mode`: son los que usó la extracción (`0.5` los tres, que son también
los valores por defecto de `HandTrackingProvider`). `numHands` es 2 aunque el
modelo actual sea `dominante`, porque la mano dominante se elige **entre las dos
detectadas** por `handedness_score`.

### 7.5 Timestamps

**En `IMAGE` no hay timestamp.** MediaPipe no lo recibe ni lo usa; Unity no tiene
nada que replicar. `secuencia_dorada.json` guarda un `timestamp_ms` por frame
solo para trazabilidad, calculado como tiempo de medios:

```
timestamp_ms = round(frame_idx * 1000 / fps)
```

Esto queda documentado por si algún día se cambia a `VIDEO`, y porque explica un
detalle que hay que conocer del lado Python: `FuenteVideo` deriva su timestamp de
`time.monotonic()`, es decir, **reloj de pared**. Sobre un archivo de vídeo eso da
un valor distinto en cada corrida, según lo que tarde el proceso. Es otra razón
por la que `VIDEO` no serviría para generar vectores dorados reproducibles sin
cambiar antes esa fuente de tiempo.

Para el registro, si alguna vez se pasa a `VIDEO`: `HandTrackingProvider.getFrame`
fuerza timestamps estrictamente crecientes (`ts = self._last_ts + 1` cuando el
reloj repite o retrocede), pero el `HandFrame` que devuelve guarda el
`timestamp_ms` **original**, no el corregido. Los dos valores divergen en cuanto
el guard se dispara.

### 7.6 Dos trampas más de la Capa 1

7. **No pasar la imagen en espejo.** El corpus se extrajo sin voltear el frame
   (`extraer_lote.py` usa `FuenteVideo(str(vid))`, con `espejo=False`). Voltear
   invierte el eje `x` **y** la `handedness`. Unity debe alimentar al
   `HandLandmarker` con el frame **sin voltear**; si se quiere mostrar la vista
   en espejo, se voltea solo lo que se dibuja.
   > `demo_vivo.py` volteaba por defecto **y pasaba ese mismo frame volteado al
   > modelo** (bug corregido el 2026-07-26): construye ahora `FuenteVideo(...,
   > espejo=False)` y aplica `cv2.flip` solo sobre la copia que va a
   > `cv2.imshow`, después de que `provider.getFrame` ya consumió el frame sin
   > voltear. Es exactamente el patrón que Unity tiene que replicar.
8. **La mano dominante se elige por frame, sin memoria.** `modo_manos="dominante"`
   toma en cada frame la mano de mayor `handedness_score`, de forma independiente.
   Si la segunda mano aparece con más score a mitad de seña, la secuencia pasa a
   describir la otra mano física. Ocurre dentro de la propia ventana dorada
   (`Right` en los frames 85–90, `Left` en los 91–102). Es una debilidad real del
   pipeline, pero es exactamente lo que hizo `build_dataset.py` al construir el
   corpus, así que el modelo se entrenó con esos saltos: **el port en C# tiene que
   reproducirlos, no arreglarlos.**

### 7.7 Cómo validar la Capa 1 en Unity

`integracion/secuencia_dorada.json` trae 18 frames consecutivos y en movimiento de
una seña real, con:

| Campo | Qué es |
|---|---|
| `frames[i].manos[j].keypoints_21x3` | Landmarks crudos tal como salen del `HandLandmarker`, antes de normalizar. |
| `frames[i].manos[j].hand` / `handedness_score` | La handedness detectada y su confianza. |
| `frames[i].timestamp_ms` | El tiempo de medios del frame (ver 7.5). |
| `frames[i].dominante.normvector_63` | La salida esperada de la Capa 2 para ese frame. |
| `esperado.entrada_modelo_60x63` | La secuencia completa ya remuestreada: la entrada exacta del ONNX. |
| `video.sha1` / `config.sha1_landmarker` | Con qué vídeo y qué `.task` se generó. Si no coinciden, el caso no aplica. |
| `diagnostico_running_mode` | La medición de 7.2, regenerable. |

Dos tests distintos, y conviene no mezclarlos:

* **Capa 2 sola** (no necesita cámara ni MediaPipe): cargar
  `frames[i].manos[].keypoints_21x3`, pasarlos por el `KeypointNormalizer` de C# y
  comparar con `dominante.normvector_63`; después la secuencia entera contra
  `esperado.entrada_modelo_60x63`. Esto ya se puede escribir hoy.
* **Capa 1** (necesita el plugin corriendo): alimentar el mismo vídeo, con el
  mismo `.task` y `running_mode=IMAGE`, y comparar los keypoints crudos frame a
  frame. Aquí la tolerancia `1e-5` es demasiado estricta si el decodificador de
  vídeo de Unity no entrega píxeles idénticos a los de OpenCV; empieza comparando
  la handedness y el número de manos por frame, que son discretos, y trata la
  diferencia numérica como una métrica a observar, no como un assert.

Del lado Python, `python generar_secuencia_dorada.py --verificar` re-extrae el
vídeo y comprueba que todo sigue coincidiendo (verificado: `5e-13` en los
keypoints crudos, `5e-7` en el `NormVector` — MediaPipe es determinista en CPU).
Si falla, o cambió el `.task`, o cambió el preprocesamiento: en los dos casos el
test de Unity queda invalidado y hay que avisar.

### 7.8 Qué reabriría esta decisión

Solo una cosa: **re-extraer el corpus completo con `running_mode="video"` y
reentrenar**. Si se hace, hay que regenerar `vectores_dorados.json`,
`secuencia_dorada.json`, `modelo.onnx` y `labels.json`, y actualizar esta sección.
Mientras el `.onnx` que viaja a Unity venga de un corpus extraído en `IMAGE`, el
runtime va en `IMAGE`.

## 8. Lo que este documento NO cubre

`SpatialSubtitleRenderer`, la Spatial UI y el render de Unity son de la Capa 4 y
viven en el repo de Unity. Aquí solo se especifica hasta la salida del clasificador.
