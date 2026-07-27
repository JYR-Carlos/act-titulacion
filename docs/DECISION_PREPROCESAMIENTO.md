# Decisión — Preprocesamiento de Keypoints: implementación propia, no reutilización de terceros

> Complementa `CONTEXTO_PROYECTO.md`. Este archivo documenta una decisión puntual
> tomada en el equipo para evitar que se reabra o se malinterprete más adelante
> (ya generó confusión una vez dentro del equipo — ver Sección 4).

## Contexto

Al buscar un dataset de referencia para validar el pipeline (ver Sección 11 de
`CONTEXTO_PROYECTO.md`), se encontró el repositorio
[`mvazquezgts/SWL-LSE`](https://github.com/mvazquezgts/SWL-LSE), que incluye un
script `generate_features.py` para el preprocesamiento/normalización de keypoints,
usado para entrenar su propio clasificador.

Se evaluó si convenía reutilizar ese script como base para el `KeypointNormalizer`
del proyecto, dado que el equipo aún no tenía nada implementado en esa capa.

## Decisión

**No se reutiliza el código de preprocesamiento de `SWL-LSE`.** Se implementa el
`KeypointNormalizer` desde cero, siguiendo exactamente la especificación ya
documentada en la Sección 10.2 del diseño (centrado en muñeca L0 + escalado por
distancia L0–L9). Resultado: `keypoint_normalizer.py`, ya implementado y probado
(invariante a traslación y escala).

## Por qué se descartó la alternativa

1. **Esquema de keypoints incompatible.** `generate_features.py` normaliza 61
   puntos (19 de pose corporal + 21 mano izquierda + 21 mano derecha), porque
   alimenta un modelo que usa esqueleto completo. El diseño de LSCh-MR captura
   **solo 21 keypoints de una mano** desde el Meta XR SDK — no hay pose corporal
   en la Capa 1 del sistema. El código no calza con el esquema de datos propio.

2. **Formato de salida incompatible.** Los tipos de feature de ese script
   (`C4_xyzc`, `C3_xyc`) están pensados como entrada para una **Graph
   Convolutional Network (MS-G3D)**, no para el vector plano de 63 dimensiones por
   frame que el `SignClassifier` (TCN) de este proyecto espera recibir.

3. **La técnica en sí es simple y ya está especificada.** Centrar en un punto de
   referencia y escalar por una distancia de referencia es una técnica estándar de
   normalización de keypoints de mano — no requiere código de terceros, y ya
   estaba completamente definida en el diseño propio (Sección 10.2) antes de
   buscar referencias externas.

4. **Licencia no verificada.** El repo de SWL-LSE no declara un archivo LICENSE
   visible; reutilizar código sustancial de ahí sin confirmar términos no es
   apropiado para un trabajo de titulación.

## Reafirmación — no confundir con esto

Esta decisión es **solo sobre preprocesamiento**. La arquitectura de clasificación
**no cambia y sigue siendo TCN**, tal como está consolidado en la Sección 10.2 del
diseño y en la Sección 3 de `CONTEXTO_PROYECTO.md`. LSTM y Transformer siguen
descartados.

*(Nota de coordinación de equipo: esto se aclaró porque un mensaje de Tomás dio a
entender que la elección entre LSTM/TCN/Transformer seguía abierta. No lo está.
Si vuelve a surgir esa confusión, referenciar directamente la Sección 3 de
`CONTEXTO_PROYECTO.md`.)*

## Estado

✅ Implementado — `keypoint_normalizer.py` (clase `KeypointNormalizer`, método
`normalize(frame) -> NormVector`). Probado: invariante a traslación, invariante a
escala, maneja división por cero en casos degenerados, valida forma de entrada
(21 keypoints x,y,z).

## Para Claude Code

- No sugerir ni importar código de `mvazquezgts/SWL-LSE` (ni `generate_features.py`
  ni el resto del pipeline `msg3d/`) al trabajar en preprocesamiento o clasificación
  de este proyecto.
- Si se necesita extender el preprocesamiento (por ejemplo, para el punto abierto de
  una mano vs. dos manos, ver Sección 6 de `CONTEXTO_PROYECTO.md`), hacerlo
  extendiendo `KeypointNormalizer` directamente, no importando una librería externa
  pensada para otro esquema de datos.
