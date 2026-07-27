# Instrumentación de métricas para el runtime Unity (métricas 2 y 3)

Tres scripts C# que miden **latencia end-to-end** (métrica 2) y **FPS**
(métrica 3) desde dentro del pipeline de Unity, y dejan la evidencia en CSV/JSON.

| Archivo | Qué mide |
|---|---|
| `PipelineTelemetry.cs` | Timestamps por frontera de capa → latencia e2e por seña. |
| `FrameRateMonitor.cs` | FPS en ventana deslizante, caídas, degradación, stuttering. |
| `MetricsRecorder.cs` | `MonoBehaviour` que cablea los dos y vuelca los archivos. |

---

## ⚠️ Antes de usarlo: dos avisos de alcance

**1. Estos archivos no se compilan en este repositorio.** Aquí solo vive el
subsistema de IA/datos en Python; no hay proyecto Unity ni ningún otro `.cs`
(ver `docs/ESTADO_ACTUAL.md`). El código está escrito contra la API pública de Unity
(`MonoBehaviour`, `Time`, `Application`, `WaitForEndOfFrame`) y no depende de
Sentis ni del Meta XR SDK, pero **no ha sido compilado ni ejecutado**. Al
integrarlo en el repo de Unity espera tener que ajustar los `namespace` y los
nombres de los métodos del controlador que llama a los ganchos.

**2. El umbral de 72 FPS es del Meta Quest 3, que está fuera del alcance actual.**
La decisión del 2026-07-21 (`docs/ESTADO_ACTUAL.md`) sacó el Quest 3 del camino
crítico: la demo corre en **PC con webcam**. En PC, con VSync activo, el FPS
queda clavado en la tasa del monitor (típicamente 60 Hz) y la comparación
contra 72 no significa nada. Opciones, en orden de honestidad decreciente:

- Declarar en el informe que la métrica 3 **no se evaluó en el hardware objetivo**
  y reportar el FPS de PC con su propio umbral (`targetFps = 60`, o el refresco
  real del monitor).
- Si consigues un Quest 3 prestado aunque sea una tarde, esta instrumentación ya
  está lista: `targetFps = 72`, sesión de 5+ minutos, `adb pull` de los CSV.

En ambos casos, di cuál de los dos hiciste. Reportar "72 FPS sostenidos" medidos
en un PC a 60 Hz con VSync sería una cifra inventada.

---

## Instalación

1. Copia los tres `.cs` al proyecto de Unity, p. ej. `Assets/Scripts/Telemetry/`.
2. Crea un GameObject vacío en la escena (`MetricsRecorder`) y añádele el
   componente `MetricsRecorder`.
3. Ajusta en el inspector:
   - `targetFps` → 72 (Quest 3) o el refresco real del monitor (PC).
   - `restFramesEnd` → **debe coincidir con `config.REST_FRAMES_FIN` de Python**
     (hoy 6). Si cambia allí, cambia aquí, o las latencias de los dos lados dejan
     de ser comparables.
   - `expectedSessionSeconds` → 300 para la sesión de 5 minutos.

## Cableado: cuatro ganchos

Desde tu controlador de sesión de traducción (el equivalente en Unity de
`scripts/demo_vivo.py`), llama a los cuatro Mark* en el orden del pipeline:

```csharp
public class TranslationSessionController : MonoBehaviour
{
    [SerializeField] MetricsRecorder metrics;

    void Update()
    {
        // --- Capa 1: captura de keypoints ---
        var frame = handTrackingProvider.GetFrame();
        metrics.OnKeypointsCaptured();              // (1) SIEMPRE, una vez por frame

        // --- Capa 1: segmentación ---
        var evt = restStateDetector.Update(frame);

        if (evt.Type == SignEventType.Discarded)
        {
            metrics.OnSignDiscarded();              // cancela la seña en curso
            return;
        }

        if (evt.Type != SignEventType.End) return;
        metrics.OnSignSegmented();                  // (2) fin de seña confirmado

        // --- Capa 2 + 3: preprocesamiento e inferencia ---
        var input  = FeaturePipeline.Prepare(evt.Sequence);   // ver docs/INTEGRACION_UNITY.md
        var result = signClassifier.Classify(input);
        metrics.OnInferenceComplete();              // (3) el clasificador respondió

        // --- Capa 4: presentación ---
        if (result.InVocabulary) messageComposer.AppendWord(result.Gloss);
        spatialSubtitleRenderer.SetText(messageComposer.Text);
        metrics.OnSubtitleRendered(                 // (4) JUSTO DESPUÉS del SetText
            result.Gloss, result.Confidence, result.InVocabulary);
    }
}
```

`OnKeypointsCaptured()` va **en todos los frames**, no solo cuando hay seña: es
lo que fija el instante del último frame capturado y lo que estima el periodo de
la cámara.

## Por qué el gancho 4 no sella el tiempo inmediatamente

`OnSubtitleRendered` lanza una corrutina que espera a `WaitForEndOfFrame` antes
de cerrar la medición. Asignar el texto en `Update()` **no lo pone en pantalla**:
eso ocurre al final del frame, cuando la UI se compone y se presenta. Sellar el
instante del `SetText` descontaría un frame entero de la métrica (~14 ms a 72
FPS, ~17 ms a 60). El presupuesto es de 500 ms, así que no cambia el veredicto,
pero es la diferencia entre medir "cuándo lo pedí" y "cuándo el usuario puede
leerlo" — y la métrica está definida como lo segundo.

## Qué se escribe, y dónde

Al salir (o al pausar en Android, porque quitarse el visor puede matar la app sin
pasar por `OnApplicationQuit`) se escriben tres archivos en
`Application.persistentDataPath/metricas/`:

**`sesion_<fecha>_senas.csv`** — una fila por seña. Las cinco primeras columnas
son las que pide el informe:

```
timestamp_utc,t_sesion_s,sena,confianza,en_vocabulario,latencia_ms,
fps_promedio,fps_min,excede_umbral,retardo_segmentacion_ms,proceso_ms,
captura_a_segmentacion_ms,segmentacion_a_inferencia_ms,inferencia_a_render_ms
```

**`sesion_<fecha>_fps.csv`** — serie temporal de FPS (una muestra cada 5 s), para
graficar la curva y ver a ojo si hay degradación:

```
t_s,fps_promedio,fps_min,fps_1pct_low,frames_bajo_objetivo
```

**`sesion_<fecha>_resumen.json`** — veredicto de las dos métricas, listo para
citar.

En Quest 3 la ruta es `/storage/emulated/0/Android/data/<package>/files/metricas`:

```
adb shell ls /storage/emulated/0/Android/data/<tu.package>/files/metricas
adb pull /storage/emulated/0/Android/data/<tu.package>/files/metricas ./metricas_quest
```

## Cómo se decide cada veredicto

**Latencia (métrica 2)** — `cumple` es true solo si **ninguna** seña superó los
500 ms. El diseño dice "latencia ≤ 500 ms", no "latencia media ≤ 500 ms": una
media de 300 ms con un tercio de las señas por encima del segundo sería una
experiencia mala que la media taparía. El JSON trae además mediana, p95, máximo y
el porcentaje que excede, para poder matizarlo en el informe.

**FPS (métrica 3)** — `cumple` exige las tres cosas del enunciado a la vez:

| Palabra del enunciado | Cómo se comprueba |
|---|---|
| "≥72 FPS" | `fps_promedio >= objetivo` |
| "sostenidos" | ≤1% de frames bajo el objetivo **y** sin degradación progresiva |
| "sin degradación" | la segunda mitad de la sesión no va >5% más lenta que la primera |
| "sin stuttering" | `n_stutters` = frames que duran más del doble del frame mediano |
| "5+ minutos" | `sesion_de_5_min` — **no** entra en `cumple`, míralo aparte |

`sesion_de_5_min` se reporta por separado a propósito: una sesión de 40 segundos
puede dar `cumple: true` y no demuestra nada sobre sostenibilidad. Comprueba
siempre los dos campos.

## Comparabilidad con las cifras de Python

`scripts/demo_vivo.py` mide la misma métrica con la misma fórmula
(`retardo_segmentacion + tiempo_de_proceso`) y la guarda en
`outputs/reports/demo_sesion_*.json`. Las dos series son comparables **siempre
que `restFramesEnd` coincida con `config.REST_FRAMES_FIN`**. Si tienes las dos,
compáralas: una diferencia grande delata que el port del pipeline a C# no está
haciendo el mismo trabajo que el de Python.
