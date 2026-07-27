# Cómo medir las 4 métricas de éxito del MVP

Índice operativo: qué comando produce cada cifra del informe, dónde queda la
evidencia y qué hay que declarar al reportarla. Los umbrales viven en
`lsch_mr/config.py`; si cambian allí, cambian aquí.

| # | Métrica | Umbral | Cómo se obtiene | Estado |
|---|---------|--------|-----------------|--------|
| 1 | Precisión algorítmica | ≥ 85% | `evaluar_modelo.py` | ✅ **0.911 ± 0.051** (k-fold por señante) |
| 2 | Latencia end-to-end | ≤ 500 ms | `demo_vivo.py` (PC) · `MetricsRecorder.cs` (Unity) | ⚠️ medida, no cerrada |
| 3 | Rendimiento gráfico | ≥ 72 FPS | `FrameRateMonitor.cs` | ❌ requiere runtime Unity |
| 4 | Task Success Rate | ≥ 80% | `calcular_tsr.py` | ❌ pendiente de sesión |
| — | *(previa)* Límites del RestStateDetector | — | `evaluar_segmentacion.py` | ❌ pendiente de etiquetado |
| — | *(previa)* Estabilidad del preprocesador | — | `pytest tests/test_preprocesador_estabilidad.py` | ✅ pasa |

---

## Antes que nada: qué corpus se está midiendo

**El corpus del MVP es LSA64 (lengua de señas argentina), no LSCh.** El
vocabulario demostrado son 10 señas de LSA64 (`Thanks`, `Help`, `Name`, …), no
las 10 glosas de `Glosas_LSCh_Mappeadas.csv`, que siguen siendo el vocabulario
*objetivo de diseño*. **La generalización a LSCh no está probada.** Ver
`ESTADO_ACTUAL.md` para el razonamiento completo y las implicaciones de licencia
(LSA64 es CC BY-NC-SA 4.0). Toda cifra de este documento hereda esa salvedad.

---

## Métrica 1 — Precisión algorítmica (≥ 85%)

```bash
# La cifra reportable: k-fold dejando SEÑANTES fuera (es el default)
python evaluar_modelo.py --fuente cv

# Desempeño del ONNX exportado sobre su 20% retenido
python evaluar_modelo.py --fuente modelo --dataset data/processed/lsa64_dominante.npz
```

Si `cv_metrics_dominante_senante.json` no existe todavía:

```bash
python entrenar.py --dataset data/processed/lsa64_dominante.npz --cv-grupos 2 --solo-cv
```

(`--cv-grupos 2` = el señante es el 2.º campo del `sample_id`: `017_001_001` →
señante `001`. **Ojo**: en un corpus grabado con `grabar_corpus.py --senante s01`
el `sample_id` es `s01_0001` y el señante es el campo **1**. Pasar el número
equivocado no da error, solo agrupa mal e infla la métrica en silencio.)

**Salidas** en `outputs/reports/`: `evaluacion_cv.json`,
`evaluacion_cv_matriz.png` (título en rojo si no cumple),
`evaluacion_cv_matriz.csv`, `evaluacion_cv_por_clase.csv`.

### Qué reportar, y qué NO

El informe pide "20% de test **con validación cruzada**", que son dos protocolos
distintos. Los dos existen y dan cifras muy distintas:

Cifras de la corrida canónica del 2026-07-26 (las cuatro evaluaciones se
lanzaron juntas, así que la tabla y los JSON son consistentes por construcción):

| Protocolo | Accuracy | Qué mide |
|---|---|---|
| Split 80/20 aleatorio | **0.990** | Nada defendible: repeticiones del mismo señante caen a ambos lados. |
| 5-fold estratificado | 0.981 ± 0.017 | Mismo problema, repartido entre folds. |
| **5-fold por señante** | **0.911 ± 0.051** | **Generalización a una persona nueva. Esta es la cifra.** |

La caída de 0.99 a 0.91 **no es ruido**: es exactamente la fuga de información
que el k-fold por señante elimina. Reportar 0.99 sobreestimaría el sistema.

Reporta la media **con su desviación** (±0.051), nunca un valor puntual, y cita
la corrida: el entrenamiento de Keras no es bit-determinista y cuatro corridas
con los **mismos splits** han dado 0.903, 0.911, 0.915 y 0.920. Cada corrida deja
ahora su propio JSON (el nombre incluye el modo de manos), así que toda cifra del
informe debe poder rastrearse hasta un archivo concreto de `outputs/reports/`.

El script imprime además el efecto de `CONF_THRESHOLD = 0.90`: cobertura 79.1%,
precisión 96.3%, 14 glosas erróneas mostradas. Es la calibración documentada en
`config.py`, y es lo que ve realmente el funcionario en ventanilla.

> **El umbral no compra precisión.** La p95 de la confianza de las predicciones
> erróneas es 1.00: hay fallos con confianza máxima. Subir de 0.90 a 0.99 solo
> quita 2 de las 14 glosas erróneas y cuesta 17 puntos de cobertura. Si el
> informe necesita más precisión, la vía es calibrar el modelo (temperature
> scaling), no mover el umbral.

---

## Métrica 2 — Latencia end-to-end (≤ 500 ms)

**En PC** (ya instrumentado, es el entregable actual):

```bash
python demo_vivo.py --fuente 0
```

Al salir deja `outputs/reports/demo_sesion_<fecha>.json` con la sección
`latencia_e2e_ms`: media, min, max, p95, cuántas señas exceden el umbral.

**En Unity**: `integracion/unity/MetricsRecorder.cs` (ver el README de esa
carpeta para el cableado de los cuatro ganchos).

### La definición incluye el retardo de segmentación

```
latencia_e2e = retardo_segmentacion + tiempo_de_proceso
retardo_segmentacion = REST_FRAMES_FIN × periodo_medio_de_frame
```

El detector no sabe que la seña terminó hasta ver `REST_FRAMES_FIN` (=6) frames
de reposo sostenido. Ese retardo es inherente al diseño y el usuario lo percibe
como espera, así que **no se puede descontar** de una medición honesta. A 30 FPS
ya son ~200 ms de los 500 disponibles.

**Estado**: medida sobre 10 vídeos a 60 FPS → mediana ~239 ms, rango 171–547 ms,
**1 de 9 por encima del umbral**. Falta una corrida sostenida con la cámara real
del demo. Palanca si sale corta: bajar `REST_FRAMES_FIN` de 6 a 4, a costa de
cerrar las señas antes.

---

## Métrica 3 — Rendimiento gráfico (≥ 72 FPS sostenidos, 5+ min)

`integracion/unity/FrameRateMonitor.cs`. **No se puede medir desde este repo**:
no hay proyecto Unity aquí (0 archivos `.cs` propios del runtime).

### Conflicto de alcance que hay que resolver antes de reportar

El umbral de 72 FPS es la tasa de refresco del **Meta Quest 3**, pero la decisión
del 2026-07-21 sacó el Quest 3 del camino crítico: la demo corre en **PC con
webcam**. En PC con VSync el FPS queda clavado en el refresco del monitor
(típicamente 60 Hz) y compararlo contra 72 no significa nada.

Dos salidas honestas, elige una y decláralo:

1. **Declarar la métrica como no evaluada en el hardware objetivo** y reportar el
   FPS de PC con su propio umbral (`targetFps` = refresco real del monitor).
2. **Conseguir un Quest 3 aunque sea una tarde.** La instrumentación ya está
   lista: `targetFps = 72`, sesión de 5+ minutos, `adb pull` de los CSV.

Lo que no vale es reportar "72 FPS sostenidos" medidos en un PC a 60 Hz.

El monitor separa las tres exigencias del enunciado, porque un promedio solo no
las cubre: **≥72** (media), **sostenidos** (≤1% de frames bajo objetivo y sin
degradación >5% entre la primera y la segunda mitad de la sesión) y **sin
stuttering** (frames que duran más del doble del frame mediano). La duración de
5 minutos se reporta aparte (`sesion_de_5_min`) y **no** entra en el veredicto:
una sesión de 40 s puede dar `cumple: true` sin demostrar nada.

---

## Métrica 4 — Task Success Rate (≥ 80%, mín. 10 participantes)

```bash
copy protocolo\planilla_tsr_plantilla.csv protocolo\planilla_tsr.csv
# ... ejecutar las pruebas rellenando la planilla ...
python calcular_tsr.py --planilla protocolo\planilla_tsr.csv
```

Guion completo del escenario de ventanilla simulada, criterio de éxito y
checklist de sesión: **`protocolo/PROTOCOLO_TSR.md`**. Léelo entero antes de la
primera sesión — el criterio de éxito debe fijarse *antes* de ver resultados.

El script valida el protocolo, no solo el porcentaje: 10 pruebas a la misma
persona, o 4 pruebas al 100%, **no cumplen** aunque el número salga bonito.

### Reporta el intervalo, no solo el porcentaje

Con n=10 la incertidumbre es enorme: **8/10 = 80% tiene un IC 95% de
[49%, 94%]**. Un 8/10 no demuestra que el sistema supere el 80% real, solo que lo
alcanzó en esa muestra. Si el tiempo lo permite, 15–20 participantes estrechan
mucho el intervalo.

---

## Pruebas unitarias previas

### RestStateDetector — precisión y recall de los límites de seña

Dos pasos: generar la plantilla, etiquetar a mano viendo los vídeos, evaluar.

```bash
python evaluar_segmentacion.py --plantilla etiquetado/limites_gt.csv --muestras 40
# ... rellenar frame_inicio_gt y frame_fin_gt viendo cada vídeo ...
python evaluar_segmentacion.py --gt etiquetado/limites_gt.csv
```

Reporta precisión/recall **por separado para inicio y fin** (dependen de
parámetros distintos del detector) y el sesgo con signo, que es lo que dice si
`REST_FRAMES_FIN` está cerrando tarde.

> **Aviso sobre LSA64**: la mano solo se detecta en ~75% de los frames (los
> señantes usan guantes de colores y MediaPipe está entrenado con manos
> desnudas). Buena parte de los FN serán **dropout de MediaPipe**, no error de
> segmentación. El script imprime la tasa de detección justamente para poder
> separar las dos cosas — decláralas por separado en el informe.

### Preprocesador geométrico — estabilidad del NormVector de 63 dim

```bash
python -m pytest tests/test_preprocesador_estabilidad.py -v
```

Verifica con un modelo de cámara **pinhole** (no un simple escalado) que el
vector se mantiene estable ante distancia a cámara y tamaño de mano. Derivas
máximas medidas sobre 40 formas de mano:

| Variación | Deriva máx. | Cota del test |
|---|---|---|
| Distancia 40–150 cm | 0.058 | 0.08 |
| Tamaño de mano ±30% | 0.032 | 0.05 |
| **Desplazamiento lateral 10 cm** | **0.161** | 0.25 |

(en unidades donde la distancia L0–L9 vale 1.0)

**Hallazgo a declarar**: la deriva por desplazamiento lateral en el encuadre es
~3× la deriva por distancia a la cámara. La normalización elimina la traslación
en el espacio de keypoints, pero **no** la distorsión de perspectiva fuera de
eje. Consecuencia práctica: conviene signar centrado en el cuadro.

---

## Correr todo lo automatizable de una vez

```bash
python -m pytest -q                                    # pruebas unitarias
python evaluar_modelo.py --fuente cv                   # métrica 1
python calcular_tsr.py                                 # métrica 4 (si hay planilla)
```

`evaluar_modelo.py` y `calcular_tsr.py` devuelven **código de salida 2** cuando
la métrica no alcanza su umbral, para poder encadenarlos en un script de
verificación sin parsear la salida.
