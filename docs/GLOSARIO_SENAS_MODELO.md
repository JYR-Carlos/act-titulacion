# Glosario de señas del modelo (`modelo.onnx`)

> Para verificar a ojo si una prueba de `scripts/demo_vivo.py` reconoció bien la seña:
> busca la glosa que mostró la demo en la tabla de abajo y compara tu gesto
> contra el vídeo de referencia. Es la forma más directa de saber "si acerté",
> sin depender solo del número de confianza.

## Por qué hace falta este documento

El modelo se entrenó con **LSA64** (lengua de señas argentina), no con LSCh,
porque el equipo no tuvo acceso a señantes de LSCh (ver `README.md`, sección
"El corpus del MVP: LSA64", y `CONTEXTO_UNITY.md` sección 2). Eso tiene dos
consecuencias que este glosario existe para resolver:

1. Las 10 glosas del modelo **no son palabras de LSCh**, son señas argentinas.
   No hay garantía de que "Thanks" en LSA64 se parezca al gesto que un señante
   de LSCh haría para lo mismo — son idiomas distintos.
2. El modelo devuelve el nombre de la clase **en inglés** (así está en
   `outputs/models/labels.json`), que es como los autores de LSA64 tradujeron
   sus propias glosas. No son glosas en español oficiales.

## Tabla de glosas

| Índice | Glosa (modelo, inglés) | Significado (traducción literal) | Vídeo de referencia (señante 001) |
|---|---|---|---|
| 0 | `Accept`    | Aceptar   | `data/external/lsa64/videos/050_001_001.mp4` |
| 1 | `Appear`    | Aparecer  | `data/external/lsa64/videos/053_001_001.mp4` |
| 2 | `Call`      | Llamar    | `data/external/lsa64/videos/017_001_001.mp4` |
| 3 | `Give`      | Dar       | `data/external/lsa64/videos/063_001_001.mp4` |
| 4 | `Help`      | Ayuda / Ayudar | `data/external/lsa64/videos/056_001_001.mp4` |
| 5 | `Last_name` | Apellido  | `data/external/lsa64/videos/027_001_001.mp4` |
| 6 | `Name`      | Nombre    | `data/external/lsa64/videos/039_001_001.mp4` |
| 7 | `None`      | Ninguno *(ver nota)* | `data/external/lsa64/videos/038_001_001.mp4` |
| 8 | `Patience`  | Paciencia | `data/external/lsa64/videos/040_001_001.mp4` |
| 9 | `Thanks`    | Gracias   | `data/external/lsa64/videos/051_001_001.mp4` |

El índice es el orden de `classes` en `outputs/models/labels.json` (el que usa
`argmax` en `SignClassifier`). El mapeo glosa -> vídeo sale de
`data/external/lsa64/lsa64_10_annotations.csv`.

> **Nota sobre `None`:** de las 10 es la glosa menos transparente — "Ninguno"
> es la traducción literal del inglés, no una interpretación lingüística
> verificada de la seña original en LSA. Si tu prueba mostró `None` y no
> tiene sentido con lo que señaste, mira el vídeo antes de asumir que el
> modelo se equivocó.

### Otros vídeos del mismo signo

Cada glosa tiene 50 muestras en el corpus (10 señantes × 5 repeticiones). El
patrón de nombre de archivo es `{clase:03d}_{señante:03d}_{repeticion:03d}.mp4`
(p. ej. `051_003_002.mp4` = `Thanks`, señante 003, repetición 2). Se eligió el
señante `001` para esta tabla por consistencia, pero si tu duda es "¿esta seña
varía mucho entre personas?", vale la pena mirar más de un señante antes de
concluir que hubo un error de reconocimiento.

## Cómo leerlo junto con la confianza

`CONF_THRESHOLD = 0.90` (`lsch_mr/config.py`) filtra lo que se muestra como
reconocido. Pero el modelo está **mal calibrado**: la confianza mediana de sus
predicciones *erróneas* es **0.78**, y su percentil 95 llega a **1.00** (corrida
canónica del 2026-07-26, `outputs/reports/cv_metrics_dominante_senante.json`).
Es decir: **una confianza alta no es garantía de acierto** — hay fallos con
confianza máxima. Si la demo mostró una glosa con confianza 0.95+ y aun así no
coincide con lo que señaste, no es contradictorio: es exactamente el
comportamiento que esa calibración advierte. El vídeo de referencia sigue
siendo la forma más confiable de confirmar.

## Traducción en pantalla de `scripts/demo_vivo.py`

`scripts/demo_vivo.py` ahora traduce estas 10 glosas al español **solo para lo que se
dibuja en pantalla** (el diccionario `ETIQUETAS_ES` en el propio script): el
subtítulo compuesto y el rótulo de la última seña reconocida se ven en
español, para no confundir a quien mira la demo. El clasificador y la consola
de diagnóstico siguen usando el nombre de clase en inglés tal como sale de
`modelo.onnx` — no se renombraron las clases del modelo, es una traducción de
presentación (Capa 4), igual que ya preveía `CONTEXTO_UNITY.md` sección 2.

Si cambias una traducción en la tabla de arriba, actualiza también
`ETIQUETAS_ES` en `scripts/demo_vivo.py` (y viceversa) — no hay una fuente única
automática entre los dos todavía; para una demo puntual no se justificó
construir ese enlace.
