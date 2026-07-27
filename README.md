# LSCh-MR — Pipeline de IA / datos

[![tests](https://github.com/JYR-Carlos/act-titulacion/actions/workflows/tests.yml/badge.svg)](https://github.com/JYR-Carlos/act-titulacion/actions/workflows/tests.yml)

Reconocimiento de lengua de señas a partir de vídeo: captura de keypoints con
MediaPipe, normalización geométrica, segmentación por reposo, clasificación con
una TCN y exportación a ONNX para consumir desde Unity Sentis.

Proyecto de titulación de la Universidad de Tarapacá. Este repositorio es la
parte de **IA y datos**; la capa espacial en Unity vive en otro repo.

---

## Qué es y qué no es

**Es** la implementación de las capas 1–3 del diseño y todo el pipeline
offline:

```
webcam/vídeo -> HandTrackingProvider -> KeypointNormalizer -> RestStateDetector
             -> build_dataset -> ModelTrainer (TCN) -> ModelExporter (ONNX)
             -> SignClassifier -> MessageComposer
```

**No es** el runtime de realidad mixta. `SpatialSubtitleRenderer`, la Spatial UI
y el render de Unity son la **Capa 4** y viven en el repositorio del otro
integrante. Aquí no hay ningún proyecto Unity ni se compila C#.

**El acoplamiento entre los dos repos es solo por archivos**, nunca por red —
la Sección 3 del diseño prohíbe MQTT/gRPC/REST:

| Archivo | Qué es |
|---|---|
| `outputs/models/modelo.onnx` | El clasificador TCN entrenado, opset 13, `sentisCompatible`. |
| `outputs/models/labels.json` | Clases, `modo_manos`, `seq_len`, `n_features`. **Manda sobre cualquier constante en C#.** |
| `integracion/vectores_dorados.json` | 8 casos de prueba de la Capa 2 para validar el port a C#. |
| `integracion/secuencia_dorada.json` | 18 frames reales consecutivos: valida la Capa 1 + 2 junta. |

La especificación completa del port —fórmulas, código C# de referencia,
trampas conocidas y el `running_mode` obligatorio— está en
**[`docs/INTEGRACION_UNITY.md`](docs/INTEGRACION_UNITY.md)**. Ese documento y los vectores
dorados son todo lo que hace falta de este lado: no hay que leer el código
Python para portar la Capa 2 correctamente.

> **Decisiones consolidadas (no se reabren):** clasificador **TCN**, runtime
> **ONNX vía Unity Sentis**, captura con **MediaPipe Tasks `HandLandmarker`** en
> `running_mode=IMAGE`, **21 keypoints** por mano, **NormVector de 63 dim**
> (centrado en L0 + escala por ‖L9−L0‖), preprocesamiento propio
> ([`docs/DECISION_PREPROCESAMIENTO.md`](docs/DECISION_PREPROCESAMIENTO.md)), modo de manos
> `dominante`.

---

## Requisitos e instalación

- **Python 3.12.0** — la versión exacta está en `.python-version` (pyenv-win la
  toma sola).
- Una **webcam** para la demo en vivo. No hace falta Quest 3 ni ningún visor.
- ~2 GB de disco para las dependencias (TensorFlow es la mayor parte).

Las versiones de `requirements.txt` están **pineadas con `==`** a propósito: la
combinación `tf2onnx` + Keras 3 es frágil y una versión más nueva de TensorFlow
rompe la exportación a ONNX, con un fallo que solo aparece al final del
pipeline. Son las versiones con las que se produjo el `modelo.onnx` que viaja a
Unity.

### Instalación en un comando

```powershell
.\setup.ps1
```

Crea el entorno virtual, instala `requirements.txt`, descarga el modelo
`hand_landmarker.task` de MediaPipe (no viene con `pip install mediapipe`) y
corre la batería de pruebas. Termina diciendo si el entorno quedó listo.

`.\setup.ps1 -SinPruebas` salta los tests; `-Recrear` rehace el venv desde cero.

### O paso a paso

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/descargar_modelo.py     # baja models/hand_landmarker.task
python -m pytest -q
```

En Linux/macOS el equivalente es `make setup` (ver [`Makefile`](Makefile)).

> **TensorFlow solo hace falta para entrenar y exportar.** La captura, el
> dataset y la inferencia con `onnxruntime` funcionan sin él, así que la demo
> corre en una instalación mucho más liviana si te saltas las dos líneas de
> entrenamiento de `requirements.txt`.

---

## Quickstart: la demo en 3 comandos

El repositorio incluye el modelo entrenado, así que **no hay que reentrenar
nada** para ver el sistema funcionando:

```powershell
git clone https://github.com/JYR-Carlos/act-titulacion.git
cd act-titulacion
.\setup.ps1
python scripts/demo_vivo.py --fuente 0
```

Haz una seña frente a la cámara y vuelve a reposo: la secuencia se cierra sola
y aparece la glosa reconocida con su confianza, el mensaje compuesto y la
latencia end-to-end contra el umbral de 500 ms. `r` reinicia el mensaje, `q` o
`ESC` salen.

Las 10 señas que el modelo conoce, con vídeo de referencia para comprobar si
las estás haciendo bien, están en
**[`docs/GLOSARIO_SENAS_MODELO.md`](docs/GLOSARIO_SENAS_MODELO.md)**. Son señas
**argentinas** (LSA64), no chilenas — ver [Limitaciones](#limitaciones-declaradas).

Al salir, la demo deja `outputs/reports/demo_sesion_<fecha>.json` con la
distribución de latencia, la tasa de detección de mano y el uso de CPU/memoria.

**Si no reconoce nada**, casi siempre es luz:

```powershell
python scripts/diagnostico_captura.py --etiqueta noche   # 20 s de medición
python scripts/demo_vivo.py --realce                     # realce adaptativo de contraste
```

`scripts/diagnostico_captura.py` separa las dos causas que dan el mismo síntoma: la
escena (poca luz, exposición larga, motion blur) y el throughput del pipeline.
Más luz física siempre gana; el realce es la red de seguridad.

También puedes usar el teléfono como cámara: `--fuente` acepta una URL además
de un índice.

```powershell
python scripts/demo_vivo.py --fuente http://192.168.1.42:8080/video
```

---

## Reproducir el entrenamiento completo

Solo hace falta si vas a cambiar el corpus o los hiperparámetros. Los cuatro
pasos, con los comandos exactos:

### 1. Descargar LSA64

El corpus **no está versionado aquí** (pesa ~1.5 GB y su licencia no permite
redistribuirlo). Descárgalo de su fuente original:

<https://facundoq.github.io/datasets/lsa64/>

Descomprime los vídeos en `data/external/lsa64/videos/`. El repo incluye
`data/catalogos/lsa64_10_annotations.csv`, que mapea las 500 muestras de las 10 señas usadas a
su etiqueta — sin él, `scripts/extraer_lote.py` tomaría el nombre de archivo como
etiqueta y saldrían 500 clases de una muestra cada una.

### 2. Extraer keypoints con nuestra propia Capa 1

```powershell
python scripts/extraer_lote.py `
    --videos-dir data/external/lsa64/videos `
    --anotaciones data/catalogos/lsa64_10_annotations.csv `
    --salida data/raw/lsa64_10.csv
```

Corre `HandLandmarker` en `running_mode=IMAGE` sobre cada vídeo. Es el paso
lento: ~500 vídeos. Se usan los *vídeos* de LSA64, nunca keypoints de terceros,
que es lo que mantiene intacta `docs/DECISION_PREPROCESAMIENTO.md`.

### 3. Construir el dataset normalizado

```powershell
python scripts/build_dataset.py --entrada data/raw/lsa64_10.csv --modo dominante `
    --salida data/processed/lsa64_dominante.npz
python scripts/build_dataset.py --entrada data/raw/lsa64_10.csv --modo ambas `
    --salida data/processed/lsa64_ambas.npz
```

Avisa si alguna glosa no llega al mínimo de muestras del diseño
(`--estricto` hace que falle en vez de avisar). Guarda `sample_ids`, que es lo
único que después permite la validación por señante.

### 4. Entrenar y exportar

```powershell
python scripts/entrenar.py --dataset data/processed/lsa64_dominante.npz
python scripts/exportar_onnx.py
```

`scripts/exportar_onnx.py` valida que todos los operadores del grafo estén soportados
por Sentis antes de dar la exportación por buena.

> **Al reentrenar, `integracion/vectores_dorados.json` queda invalidado** en su
> octavo caso, que incluye el `sha1` del `.onnx`. Regenera los dorados y avisa
> al equipo de Unity:
>
> ```powershell
> python scripts/generar_vectores_dorados.py
> python scripts/generar_secuencia_dorada.py
> ```

---

## Reproducir las métricas del informe

```powershell
python scripts/reproducir_metricas.py
```

Ejecuta las cuatro evaluaciones de validación cruzada (2 modos × 2 esquemas)
más `scripts/evaluar_modelo.py`, y escribe las tablas exactas que aparecen en el
informe. Las cuatro se lanzan juntas para que la tabla y los JSON sean
consistentes por construcción.

**Tarda**: entrena 20 modelos. Usa `--dry-run` para ver los comandos sin
ejecutarlos, y `--solo-tablas` para regenerar las tablas desde los JSON que ya
están en `outputs/reports/` sin reentrenar nada.

Los comandos individuales, por si prefieres lanzarlos por separado:

```powershell
python scripts/entrenar.py --dataset data/processed/lsa64_dominante.npz --cv --solo-cv
python scripts/entrenar.py --dataset data/processed/lsa64_dominante.npz --cv-grupos 2 --solo-cv
python scripts/entrenar.py --dataset data/processed/lsa64_ambas.npz --cv --solo-cv
python scripts/entrenar.py --dataset data/processed/lsa64_ambas.npz --cv-grupos 2 --solo-cv
python scripts/evaluar_modelo.py --fuente cv
```

> **`--cv-grupos 2` significa "el señante es el 2.º campo del `sample_id`"**
> (`017_001_001` = clase_señante_repetición). Pasar el número equivocado
> **agrupaba mal en silencio** e inflaba la métrica; ahora el comando imprime
> siempre cuántos grupos detectó y aborta si el agrupamiento es implausible. En
> un corpus grabado con `scripts/grabar_corpus.py --senante s01` el señante es el
> campo **1**.

Qué comando produce cada cifra del informe y qué hay que declarar al
reportarla: **[`docs/METRICAS_MVP.md`](docs/METRICAS_MVP.md)**.

---

## Mapa de arquitectura: diseño ↔ código

Los nombres de clases y métodos respetan el diagrama de clases del diseño para
mantener la trazabilidad. Componentes de la Sección 10.2:

| Componente (diseño UML) | Módulo | Método clave |
|---|---|---|
| `HandTrackingProvider` | `lsch_mr/hand_tracking_provider.py` | `getFrame()` |
| `KeypointNormalizer` | `lsch_mr/keypoint_normalizer.py` | `normalize()`, `normalize_sequence()` |
| `RestStateDetector` | `lsch_mr/rest_state_detector.py` | `update()`, `reset()` |
| `SignClassifier` | `lsch_mr/sign_classifier.py` | `classify()` |
| `MessageComposer` | `lsch_mr/message_composer.py` | `appendWord()`, `tick()` |
| `DatasetRecorder` | `lsch_mr/dataset_recorder.py` | `recordSession()` |
| `ModelTrainer` | `lsch_mr/model_trainer.py` | `train()`, `evaluar_cv()` |
| `ModelExporter` | `lsch_mr/model_exporter.py` | `export()`, `validateOperators()` |
| `SpatialSubtitleRenderer`, Spatial UI | **no está aquí** | Capa 4, repo de Unity |

Módulos de soporte, sin componente propio en el diagrama:

| Módulo | Rol |
|---|---|
| `lsch_mr/tipos.py` | Modelo de datos: `HandFrame`, `MultiHandFrame`, `SignEvent`, `ClassResult`. |
| `lsch_mr/config.py` | Rutas, hiperparámetros y los umbrales de las métricas del MVP. |
| `lsch_mr/fuente_video.py` | `FuenteVideo`: webcam, URL de teléfono o archivo. |
| `lsch_mr/caracteristicas.py` | Ensamblado por modo de manos + remuestreo temporal a 60 frames. |
| `lsch_mr/tcn.py` | Arquitectura del clasificador (`build_tcn()`). |
| `lsch_mr/csv_esquema.py` | Esquema del CSV crudo del corpus. |
| `lsch_mr/glosas.py` | Catálogo de glosas, orden estable de clases. |
| `lsch_mr/realce_luz.py` | Realce adaptativo de frames oscuros (`demo_vivo --realce`). |
| `lsch_mr/monitor_recursos.py` | `MonitorRecursos`: CPU y memoria del proceso. |
| `lsch_mr/consola.py` | UTF-8 en consola Windows + coloreado ANSI. |
| `lsch_mr/metricas_clasificacion.py` | Accuracy, matriz de confusión, veredicto (métrica 1). |
| `lsch_mr/metricas_segmentacion.py` | Precisión/recall de los límites de seña (prueba previa). |
| `lsch_mr/metricas_tsr.py` | Task success rate + intervalo de Wilson (métrica 4). |

### Arquitectura del clasificador

Bloques residuales de convolución 1D **causal y dilatada** (kernel 3,
dilataciones `(1,2,4,8)`, 2 convoluciones por bloque) →
`GlobalAveragePooling1D` → `Dense` → `Softmax`. El campo receptivo es
`1 + 2·(k−1)·Σdil = 61 ≥ 60` frames, así que cubre la ventana temporal completa
de una seña. Usa solo operadores soportados por Sentis: Conv, Relu, Add, Pad,
ReduceMean, Gemm, Softmax.

Toda secuencia (1..60 frames) se remuestrea por interpolación lineal a
`SEQ_LEN = 60`, lo que da una forma de entrada fija — Sentis prefiere formas
estáticas — e independiente de la duración de la seña.

---

## Resultados

Corrida canónica del **2026-07-26**, sobre 483 de los 500 vídeos de LSA64 (los
17 restantes no tuvieron ninguna detección de mano). Las cuatro evaluaciones se
lanzaron juntas, así que la tabla y los JSON son consistentes por construcción.

| Modo | k-fold estratificado | **k-fold por señante** | Evidencia |
|---|---|---|---|
| `dominante` (63 dim) | 0.981 ± 0.017 | **0.911 ± 0.051** | [`cv_metrics_dominante.json`](outputs/reports/cv_metrics_dominante.json) · [`_senante.json`](outputs/reports/cv_metrics_dominante_senante.json) |
| `ambas` (126 dim) | 0.973 ± 0.020 | 0.905 ± 0.063 | [`cv_metrics_ambas.json`](outputs/reports/cv_metrics_ambas.json) · [`_senante.json`](outputs/reports/cv_metrics_ambas_senante.json) |

**La columna que vale es la de por señante** (objetivo del MVP: ≥ 0.85). Cada
señante aporta 5 repeticiones de cada seña; un k-fold al azar las reparte entre
entrenamiento y validación, así que el modelo puede reconocer *a la persona* en
vez de *la seña*. La caída de 0.981 → 0.911 mide exactamente esa fuga. Un split
80/20 aleatorio da 0.990 y no es defendible.

> **Reporta siempre la media con su desviación, y cita la corrida.** El
> entrenamiento de Keras no es bit-determinista aunque se fije la semilla: con
> los **mismos splits**, `dominante` por señante ha dado 0.903, 0.911, 0.915 y
> 0.920 en cuatro corridas. Cada corrida deja su propio JSON, así que toda cifra
> del informe se puede rastrear hasta un archivo concreto.

**Una mano vs. dos manos: indistinguibles.** 0.911 ± 0.051 contra 0.905 ± 0.063,
con las diferencias muy por dentro del ruido de corrida a corrida — y el orden
entre los dos modos **se invirtió** respecto a la corrida anterior. Si la
diferencia fuera real no cambiaría de signo. Se elige `dominante` por coste (la
mitad de entrada), no por accuracy.

**Umbral de confianza calibrado: `CONF_THRESHOLD = 0.90`.** Medido sobre las
predicciones out-of-fold de la validación por señante:

| Umbral | Cobertura | Precisión | Glosas erróneas mostradas |
|---|---|---|---|
| 0.60 | 94.0% | 92.1% | 36 |
| **0.90** | **79.1%** | **96.3%** | **14** |
| 0.95 | 73.3% | 96.0% | 14 |
| 0.99 | 61.7% | 96.0% | 12 |

En ventanilla, mostrar una glosa equivocada engaña al funcionario, mientras que
"no reconocida" solo pide repetir. Y 0.90 es la rodilla de la curva: a partir de
ahí la precisión deja de subir mientras la cobertura sigue cayendo.

**Latencia end-to-end: medida, no cumplida.** Sobre 10 vídeos del corpus a
60 FPS la mediana es ~239 ms con rango 171–547 ms, y **1 de 9 señas supera los
500 ms**. Las sesiones de webcam versionadas en `outputs/reports/` dan media
~291 ms sin excesos, pero son sesiones cortas. A 30 FPS el retardo de
segmentación se duplica (~200 ms de los 500 disponibles) y varios casos
quedarían al borde. Falta una corrida sostenida con la cámara del demo. Palanca
si sale corta: bajar `REST_FRAMES_FIN` de 6 a 4, a costa de cerrar las señas
antes.

Estado de las cuatro métricas del MVP y qué falta para cerrar cada una:
**[`docs/METRICAS_MVP.md`](docs/METRICAS_MVP.md)** y
**[`docs/ESTADO_ACTUAL.md`](docs/ESTADO_ACTUAL.md)**.

---

## Limitaciones declaradas

Ninguna de estas es un defecto oculto: están medidas y hay que declararlas al
reportar cualquier cifra de este repo.

**1. El corpus es argentino, no chileno.** El MVP está validado sobre
**LSA64 (lengua de señas argentina)**. El equipo no tuvo acceso a señantes de
LSCh, así que grabar corpus propio quedó descartado. El vocabulario demostrado
son 10 señas de LSA64 (`Thanks`, `Help`, `Name`, …), no las glosas de
`data/catalogos/Glosas_LSCh_Mappeadas.csv`, que siguen siendo el vocabulario *objetivo de
diseño*. **La generalización a LSCh no está probada** y es trabajo futuro. Son
idiomas distintos: no hay garantía de que la seña argentina de "gracias" se
parezca a la chilena.

**2. El modelo está mal calibrado, y peor de lo que parecía.** La confianza
mediana de sus predicciones **erróneas** es 0.78 y su **p95 llega a 1.00**: hay
fallos con confianza máxima. Consecuencia práctica: **no se puede comprar
precisión subiendo el umbral** — pasar de 0.90 a 0.99 solo quita 2 de las 14
glosas erróneas y cuesta 17 puntos de cobertura. Una confianza alta **no** es
garantía de acierto. Corregirlo de raíz (temperature scaling) es trabajo futuro
y cambiaría el contrato con Unity, porque hoy el softmax va dentro del grafo
ONNX.

**3. Los señantes de LSA64 usan guantes de colores.** `HandLandmarker` está
entrenado sobre manos desnudas, así que la detección va justa: **la mano solo
aparece en ~75% de los frames** y 17 de los 500 vídeos no tienen ninguna
detección (14 de ellos son `Patience`, con la mano de canto tapando la cara).
Por eso la cifra de LSA64 es una **cota inferior**: con manos desnudas el
pipeline debería ir mejor, no peor. También significa que al medir la
segmentación, buena parte de los falsos negativos serán dropout de MediaPipe y
no error del `RestStateDetector` — hay que declararlos por separado.

**4. El preprocesador deriva ~3× más por desplazamiento lateral que por
distancia a la cámara** (0.161 a 10 cm contra 0.058 entre 40 y 150 cm, en
unidades donde ‖L9−L0‖ = 1). La normalización quita la traslación en el espacio
de keypoints, pero no la distorsión de perspectiva fuera de eje. Conviene signar
centrado en el cuadro.

**5. Métricas 3 y 4 sin cerrar.** El rendimiento gráfico (≥72 FPS) es la tasa de
refresco del Quest 3, que quedó fuera del camino crítico; en un PC con VSync el
FPS se clava en el refresco del monitor y compararlo con 72 no significa nada.
El task success rate necesita ejecutar las sesiones con participantes: el guion,
la planilla y el cálculo ya están en `data/tsr/` y `scripts/calcular_tsr.py`.

---

## Atribución y licencia

El **código** de este repositorio es MIT — ver [`LICENSE`](LICENSE).

El **corpus LSA64 y todo lo derivado de él** —incluidos `modelo.onnx`,
`labels.json` y los `.npz` de `data/processed/`— están bajo
**CC BY-NC-SA 4.0**: exige atribución, **prohíbe el uso comercial** y obliga a
licenciar los derivados en los mismos términos.

> Ronchetti, F., Quiroga, F., Estrebou, C., Lanzarini, L., Rosete, A. (2016).
> *LSA64: An Argentinian Sign Language Dataset.* XXII Congreso Argentino de
> Ciencias de la Computación (CACIC).

Los detalles completos, y qué archivo hereda qué licencia, están en
**[`NOTICE.md`](NOTICE.md)**. Léelo antes de reutilizar el modelo.

---

## Gotchas de ejecución en Windows

- **La consola va en cp1252.** Imprimir un carácter fuera de ese juego lanza
  `UnicodeEncodeError`. Todos los scripts llaman a
  `lsch_mr/consola.py::configurar_utf8()` al arrancar; si escribes uno nuevo,
  hazlo también, y usa `encoding="utf-8"` **explícito** al leer y escribir
  archivos, no el default de la plataforma.
- **Comillas y regex en la línea de comandos.** Una regex con `^`, `[` o `(`
  puede llegar mutilada al proceso según el shell. Por eso `--cv-grupos` acepta
  un número de campo (`--cv-grupos 2`) además de una regex.
- **En Git Bash, cuidado con `pip install "pkg>=1.2"`**: el `>=` puede
  interpretarse como redirección. Otra razón para pinear con `==`.
- **PowerShell parte las líneas con `` ` ``**, no con `\`. Los ejemplos de este
  README ya vienen en sintaxis PowerShell.

---

## Pruebas

```powershell
python -m pytest -q
```

135 pruebas en ~2 s, sin necesidad de cámara, corpus ni TensorFlow. Hoy
**ninguna** prueba necesita hardware de captura; el marcador `camera` está
registrado en `pytest.ini` para las que lleguen a necesitarlo, y CI las
excluiría con `-m "not camera"`.

| Área | Archivo | Tests |
|---|---|---|
| Estabilidad del preprocesador (cámara pinhole) | `test_preprocesador_estabilidad.py` | 22 |
| `cargar_grupos` + guardas de `--cv-grupos` | `test_cargar_grupos.py` | 16 |
| Métricas de segmentación | `test_metricas_segmentacion.py` | 15 |
| Métricas de clasificación | `test_metricas_clasificacion.py` | 14 |
| Métricas de TSR | `test_metricas_tsr.py` | 12 |
| Reporte de sesión de `demo_vivo` | `test_demo_vivo_metricas.py` | 10 |
| `RestStateDetector` | `test_rest_state_detector.py` | 8 |
| Realce de poca luz | `test_realce_luz.py` | 8 |
| `MonitorRecursos` | `test_monitor_recursos.py` | 8 |
| `KeypointNormalizer` | `test_keypoint_normalizer.py` | 9 |
| `MessageComposer` | `test_message_composer.py` | 7 |
| Split 80/20 del modelo final | `test_split_indices.py` | 6 |

Además, dos comprobaciones que no son pytest pero valen lo mismo:

```powershell
python scripts/generar_vectores_dorados.py --verificar   # el preprocesamiento no cambió
python scripts/generar_secuencia_dorada.py --verificar   # re-extrae el vídeo y compara
```

Si cualquiera de las dos falla, el port a C# queda invalidado y hay que avisar
al equipo de Unity.

---

## Estructura del repositorio

```
lsch_mr/        el paquete: los componentes del diseño (Capas 1-3)
scripts/        los CLI, uno por tarea. Se ejecutan desde la raíz:
                  python scripts/demo_vivo.py --fuente 0
tests/          pruebas unitarias (pytest)
docs/           diseño, arquitectura, métricas y contrato con Unity
data/
  catalogos/    CSV de referencia versionados (glosas, anotaciones LSA64)
  external/     corpus descargado (LSA64) — no versionado
  raw/          keypoints crudos extraídos del corpus — no versionado
  processed/    datasets normalizados .npz (los dos del informe, versionados)
  tsr/          planilla de las pruebas con usuarios (métrica 4)
models/         hand_landmarker.task de MediaPipe — lo baja descargar_modelo.py
outputs/
  models/       modelo.onnx + labels.json (versionados: es lo que consume Unity)
  reports/      métricas y evidencia de cada corrida
integracion/    lo que viaja al repo de Unity: vectores dorados y código C#
tools/          utilidades de desarrollo ajenas al pipeline
```

Los scripts se ejecutan **desde la raíz del repositorio**. Cada uno importa
`scripts/_raiz.py`, que deja la raíz en `sys.path` para que `import lsch_mr`
funcione sin instalar el paquete.

---

## Documentación del repositorio

| Documento | Para qué |
|---|---|
| [`docs/INTEGRACION_UNITY.md`](docs/INTEGRACION_UNITY.md) | Contrato entre repos y spec del port del preprocesamiento a C#. **Léelo antes de tocar Unity.** |
| [`docs/arquitectura_LSCh-MR.md`](docs/arquitectura_LSCh-MR.md) | El documento de arquitectura del sistema completo (las cuatro capas). |
| [`docs/CONTEXTO_UNITY.md`](docs/CONTEXTO_UNITY.md) | Contexto del subsistema de presentación (Capa 4), que vive en otro repo. |
| [`docs/METRICAS_MVP.md`](docs/METRICAS_MVP.md) | Qué comando produce cada cifra del informe y qué declarar al reportarla. |
| [`docs/ESTADO_ACTUAL.md`](docs/ESTADO_ACTUAL.md) | Fotografía del repo: qué está hecho, qué falta y por qué. |
| [`docs/GLOSARIO_SENAS_MODELO.md`](docs/GLOSARIO_SENAS_MODELO.md) | Las 10 glosas del modelo, su significado y un vídeo de referencia. |
| [`docs/DECISION_PREPROCESAMIENTO.md`](docs/DECISION_PREPROCESAMIENTO.md) | Por qué el preprocesamiento es propio y no reutilizado. |
| [`docs/CONTEXTO_PROYECTO.md`](docs/CONTEXTO_PROYECTO.md) | El diseño: casos de uso, secciones y modelo de datos. |
| [`NOTICE.md`](NOTICE.md) | Atribución del corpus y herencia de licencias. |
| [`docs/PROTOCOLO_TSR.md`](docs/PROTOCOLO_TSR.md) | Guion del escenario de ventanilla para las pruebas con usuarios. |
| [`integracion/unity/README.md`](integracion/unity/README.md) | Cableado de la instrumentación C# de latencia y FPS. |

### Scripts de línea de comandos

| Script | Qué hace |
|---|---|
| `scripts/descargar_modelo.py` | Descarga `hand_landmarker.task`. |
| `scripts/extraer_lote.py` | Extracción de keypoints por lotes sobre una carpeta de vídeos. |
| `scripts/extraer_keypoints.py` | Extracción desde webcam o un vídeo suelto. |
| `scripts/grabar_corpus.py` | Grabación de corpus propio (`--senante`, que va dentro del `sample_id`). |
| `scripts/build_dataset.py` | CSV crudo → dataset normalizado `.npz`. |
| `scripts/entrenar.py` | Entrenamiento y validación cruzada. |
| `scripts/exportar_onnx.py` | Exportación a ONNX + validación de operadores para Sentis. |
| `scripts/demo_vivo.py` | Demo end-to-end en PC (CU-01/CU-03). |
| `scripts/diagnostico_captura.py` | Diagnóstico de captura: luz y cámara contra throughput. |
| `scripts/evaluar_modelo.py` | Métrica 1: accuracy y matriz de confusión. |
| `scripts/evaluar_segmentacion.py` | Prueba previa: límites de seña del `RestStateDetector`. |
| `scripts/calcular_tsr.py` | Métrica 4: task success rate con intervalo de Wilson. |
| `scripts/reproducir_metricas.py` | Las cuatro evaluaciones del informe de una vez. |
| `scripts/verificar_artefactos.py` | Comprueba que `modelo.onnx`, `labels.json` y los vectores dorados siguen alineados entre sí. Corre en CI. |
| `scripts/generar_vectores_dorados.py` | Casos dorados de la Capa 2 para el port a C#. |
| `scripts/generar_secuencia_dorada.py` | Casos dorados de la Capa 1 + 2 sobre una seña real. |

`scripts/evaluar_modelo.py` y `scripts/calcular_tsr.py` devuelven **código de salida 2** cuando
la métrica no alcanza su umbral, para poder encadenarlos en un script de
verificación sin parsear la salida.
