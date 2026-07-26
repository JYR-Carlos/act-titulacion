# Integración con Unity/Sentis — contrato y port del preprocesamiento a C#

> Para el repo de Unity. Este documento y `integracion/vectores_dorados.json` son
> todo lo que hace falta de este lado: no hay que leer el código Python para
> portar la Capa 2 correctamente.

## 1. Qué se comparte entre los dos repos

El acoplamiento entre el subsistema de IA/datos (Python) y el runtime (Unity) es
**solo por archivos**, nunca por red — la Sección 3 del diseño prohíbe MQTT/gRPC/REST,
y un puente Python↔Unity reabriría esa decisión.

| Archivo | Qué es |
|---|---|
| `outputs/models/modelo.onnx` | El clasificador TCN entrenado, opset 13. |
| `outputs/models/labels.json` | Metadatos del modelo: clases, `modo_manos`, `seq_len`, `n_features`. |
| `integracion/vectores_dorados.json` | Casos de prueba para validar el port a C#. |

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

`argmax` sobre la salida. Si el máximo `< 0.60` (`CONF_THRESHOLD`), la seña se
reporta como `"<desconocida>"` — es el flujo alternativo FA-01 de CU-01, no un
error. El umbral es provisional hasta calibrarlo con el corpus real.

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

## 7. Lo que este documento NO cubre

`SpatialSubtitleRenderer`, la Spatial UI y el render de Unity son de la Capa 4 y
viven en el repo de Unity. Aquí solo se especifica hasta la salida del clasificador.
