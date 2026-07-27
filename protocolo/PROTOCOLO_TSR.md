# Protocolo de pruebas con usuarios — Task Success Rate (métrica 4 del MVP)

> Escenario de **ventanilla simulada**. Objetivo: medir si una persona que **no
> sabe lengua de señas** puede entender lo que le está diciendo un señante,
> leyendo únicamente los subtítulos que produce el sistema.
>
> Umbral del MVP: **TSR ≥ 80%** sobre un **mínimo de 10 pruebas con
> participantes distintos**.

---

## 0. Antes de empezar: dos cosas que hay que declarar en el informe

**1. El vocabulario demostrado es LSA64 (lengua de señas argentina), no LSCh.**
El corpus del MVP son las 10 señas de LSA64 y no las 10 glosas de
`Glosas_LSCh_Mappeadas.csv` (ver `ESTADO_ACTUAL.md`). Las secuencias de este
protocolo están construidas con el vocabulario que el modelo **realmente**
reconoce. No se puede pedir a un señante que ejecute una seña chilena y esperar
que el sistema la traduzca.

**2. Quien ejecuta las señas debe poder reproducir las de LSA64.** Si nadie del
equipo sabe LSCh ni LSA64, la persona que signa aprende las 10 señas mirando los
vídeos del corpus (`data/external/lsa64/videos/`) antes de la sesión. Esto es una
limitación real del montaje y **debe declararse**: un señante no nativo ejecuta
las señas de forma más lenta y regular que un usuario real, lo que favorece al
sistema. El TSR obtenido es, por tanto, una **cota superior** de lo que daría con
señantes nativos.

---

## 1. Montaje

```
   [ SEÑANTE ]  ---- ejecuta 3 señas ---->  [ CÁMARA ]
                                                |
                                          [ SISTEMA ]
                                                |
                                          [ PANTALLA con subtítulo ]
                                                |
                                        [ PARTICIPANTE ] lee y responde
```

- **Participante** = hace de funcionario de ventanilla. **No sabe lengua de
  señas** y **no ha visto antes las señas del corpus**. Si ya vio una sesión
  anterior, queda descartado como participante (aprendió las señas por
  repetición, no por el subtítulo).
- **Señante** = ejecuta las 3 señas. Siempre la misma persona en todas las
  pruebas, para no introducir variabilidad entre participantes.
- **Observador** = rellena la planilla. Puede ser el señante si trabaja solo,
  pero conviene separarlo.

**Colocación crítica:** el participante ve la **pantalla**, no las manos del
señante. Si ve las manos, puede deducir el gesto por mímica (varias señas del
corpus son icónicas: `Give`, `Call`) y la prueba dejaría de medir el subtítulo.
Sitúa al participante de espaldas al señante, o pon una separación visual.

---

## 2. Guiones de secuencia (3 señas encadenadas)

Cinco guiones, para que no todos los participantes vean lo mismo. Rota:
participante 1 → S1, participante 2 → S2, … participante 6 → S1, etc.

| ID | Señas (glosa del modelo) | Subtítulo esperado | Intención que el participante debe captar |
|----|--------------------------|--------------------|-------------------------------------------|
| **S1** | `Name` + `Last_name` + `Help` | "Nombre Apellido Ayuda" | Quiere identificarse y pedir ayuda con un trámite. |
| **S2** | `Help` + `Patience` + `Thanks` | "Ayuda Paciencia Gracias" | Pide ayuda, entiende que debe esperar y agradece. |
| **S3** | `Call` + `Name` + `Accept` | "Llamar Nombre Aceptar" | Pide que lo llamen por su nombre y acepta. |
| **S4** | `Give` + `Last_name` + `Thanks` | "Dar Apellido Gracias" | Entrega/pide un dato con su apellido y agradece. |
| **S5** | `Appear` + `Name` + `Patience` | "Aparecer Nombre Paciencia" | Pregunta si su nombre ya salió y acepta esperar. |

> Las traducciones al español vienen de `ETIQUETAS_ES` en `demo_vivo.py`; si esa
> tabla cambia, esta también.
>
> **Nota honesta sobre el diseño de la tarea:** el subtítulo es un telegrama de
> tres palabras, no una frase. La tarea del participante es inferir la
> **intención** a partir de esas tres palabras — que es exactamente lo que
> pasaría en una ventanilla real con este MVP. Por eso el criterio de éxito de
> abajo se define sobre la intención y no sobre repetir las tres palabras: pedir
> que las repita mediría lectura, no comunicación.

---

## 3. Procedimiento por prueba

1. **Instrucciones al participante** (leer literalmente, para no sesgar):

   > «Vas a atender a una persona sorda en una ventanilla de atención
   > ciudadana. No conoces lengua de señas. En la pantalla van a aparecer
   > subtítulos con lo que esa persona te está diciendo. Cuando terminen, dime
   > con tus palabras qué crees que necesita. No puedes preguntarle nada ni
   > pedirle que repita.»

2. El señante ejecuta las 3 señas del guion asignado, **con pausa de reposo
   entre cada una** (el `RestStateDetector` segmenta por reposo: sin pausa, dos
   señas se funden en una).

3. El participante lee el subtítulo y **dice en voz alta** qué entendió.
   El observador **transcribe literalmente** su respuesta en la columna
   `interpretacion`. Transcribir literal, no resumir: la respuesta cruda es lo
   que permite después discutir un caso dudoso.

4. El observador marca `exito` = **si** / **no** según el criterio de §4.

5. Preguntas cualitativas (siempre, haya o no éxito):
   - «¿Se leía bien el subtítulo? ¿Tamaño, contraste, tiempo en pantalla?»
   - «¿Te habría servido esto para atender a esa persona?»
   - «¿Qué te faltó?»

   Se anotan en `comentarios`.

---

## 4. Criterio de éxito (binario, decidido ANTES de la sesión)

**Éxito** = el participante expresa la **intención correcta** de la secuencia sin
haber recibido ninguna comunicación adicional (ni gestos del señante, ni pistas
del observador, ni una segunda pasada).

- ✅ S1 → «quiere dar su nombre y apellido y que le ayuden con algo» → **éxito**
- ✅ S2 → «pide ayuda y dice que va a esperar» → **éxito**
- ❌ S1 → «creo que dice algo de un nombre… ¿pide un documento?» → **fallo**
  (no captó la petición de ayuda)
- ❌ Cualquier caso en que el participante pida repetir y se le conceda → **fallo**
  (y se anota en `comentarios`).

**Reglas para que el criterio no se relaje sobre la marcha:**

- Se decide **en el momento**, antes de ver el resultado acumulado. No se
  reinterpretan pruebas pasadas cuando el porcentaje va quedando corto.
- Ante duda genuina entre éxito y fallo → se marca **fallo**. El sesgo debe ir en
  contra del sistema, no a favor.
- Si el sistema muestra `<desconocida>` para una de las 3 señas (confianza bajo
  `CONF_THRESHOLD` = 0.90) y aun así el participante capta la intención, **es
  éxito**: se está midiendo la tarea comunicativa, no el reconocimiento. El caso
  se anota en `comentarios` porque es informativo.
- Si el sistema no segmenta una seña y el subtítulo queda con 2 palabras, la
  prueba **cuenta igual** (no se repite). Repetir solo las pruebas que salen mal
  es lo que convierte un 60% en un 90% ficticio.

---

## 5. Registro y cálculo

Copia la plantilla en blanco y rellénala durante la sesión:

```
copy protocolo\planilla_tsr_plantilla.csv protocolo\planilla_tsr.csv
```

Se puede rellenar en Excel y guardar como `.xlsx` (el script lo lee si tienes
`openpyxl`) o mantenerla como CSV.

Al terminar:

```
python calcular_tsr.py --planilla protocolo\planilla_tsr.csv
```

El script calcula el porcentaje, verifica que haya ≥10 pruebas con participantes
distintos, muestra el desglose por secuencia y los comentarios, y escribe
`outputs/reports/tsr.json`.

### Qué reportar en el informe

Reporta **el porcentaje con su intervalo de confianza**, no el porcentaje solo.
Con n=10 la horquilla es muy ancha: 8/10 = 80% tiene un IC 95% de
aproximadamente [49%, 94%]. Es decir, **un 8/10 no demuestra que el sistema
supere el 80% real** — solo que lo alcanzó en esa muestra. Si el presupuesto de
tiempo lo permite, 15-20 participantes estrechan el intervalo bastante y hacen la
afirmación mucho más defendible.

---

## 6. Checklist de sesión

- [ ] El señante domina las 10 señas del corpus (`GLOSARIO_SENAS_MODELO.md`).
- [ ] `python demo_vivo.py --fuente 0` corre y reconoce las 3 señas del guion en
      un ensayo previo, **sin participante delante**.
- [ ] Luz frontal suficiente — comprueba `deteccion` > 70% en la cabecera del
      overlay. Con poca luz la demo no segmenta nada
      (ver `outputs/reports/diagnostico_captura_noche_*.json`).
- [ ] El participante no ve las manos del señante.
- [ ] El participante no ha presenciado una sesión anterior.
- [ ] Planilla abierta y lista.
- [ ] Consentimiento verbal para anotar sus comentarios de forma anonimizada
      (los participantes se identifican como P01, P02… nunca por su nombre).
- [ ] Al terminar la sesión, guardar también el reporte que deja `demo_vivo.py`
      (`outputs/reports/demo_sesion_*.json`): trae la latencia end-to-end real de
      esas mismas señas, que es la métrica 2.
