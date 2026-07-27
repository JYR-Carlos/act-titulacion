# Atribución y licencias

Este repositorio contiene material bajo **dos licencias distintas**. La
distinción no es formal: el corpus con el que se entrenó el modelo prohíbe el
uso comercial, y esa restricción **se hereda** en el modelo y en todo lo
derivado de él.

| Qué | Licencia | Dónde |
|---|---|---|
| El código propio | MIT | [`LICENSE`](LICENSE) |
| El modelo entrenado y los datos derivados de LSA64 | **CC BY-NC-SA 4.0** | esta nota |
| `hand_landmarker.task` de MediaPipe | Apache 2.0 | no se redistribuye |

---

## 1. Corpus LSA64 — atribución obligatoria

El MVP se entrenó con **LSA64: A Dataset for Argentinian Sign Language**, del
Instituto de Investigación en Informática LIDI de la Universidad Nacional de
La Plata.

> Ronchetti, F., Quiroga, F., Estrebou, C. A., Lanzarini, L. C., & Rosete, A.
> (2016). **LSA64: An Argentinian Sign Language Dataset.** XXII Congreso
> Argentino de Ciencias de la Computación (CACIC 2016).

Sitio del corpus: <https://facundoq.github.io/datasets/lsa64/>

**LSA64 se distribuye bajo Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)**, que exige tres cosas:

- **BY (atribución)** — hay que citar a los autores. Esta nota lo hace, y el
  informe también debe hacerlo.
- **NC (no comercial)** — **no se puede usar con fines comerciales.** Ni el
  corpus ni nada derivado de él, incluido el modelo de este repositorio.
- **SA (compartir igual)** — cualquier derivado se licencia **en los mismos
  términos**, no bajo una licencia más permisiva.

Texto completo: <https://creativecommons.org/licenses/by-nc-sa/4.0/>

### Qué archivos de este repositorio son derivados de LSA64

Un modelo entrenado sobre un corpus es una obra derivada de ese corpus. Estos
archivos **están bajo CC BY-NC-SA 4.0**, no bajo MIT:

| Archivo | Por qué |
|---|---|
| `outputs/models/modelo.onnx` | Entrenado sobre LSA64. Sus pesos codifican el corpus. |
| `outputs/models/labels.json` | Los nombres de clase son las glosas de LSA64. |
| `data/processed/lsa64_dominante.npz` | Keypoints extraídos de los vídeos de LSA64. |
| `data/processed/lsa64_ambas.npz` | Ídem. |
| `lsa64_10_annotations.csv` | Mapeo de archivo a etiqueta, metadatos del corpus. |
| `outputs/reports/cv_metrics_*.json` | Contienen las predicciones out-of-fold muestra a muestra. |
| `integracion/secuencia_dorada.json` / `.csv` | Keypoints de un vídeo real del corpus (`051_001_001`). |

**Consecuencia práctica: `modelo.onnx` no se puede usar en un producto
comercial.** Si el sistema llegara a desplegarse en una institución con fines
comerciales, hay que reentrenar sobre un corpus con licencia compatible — que
es, de todos modos, lo que exige la validez lingüística: LSA64 es lengua de
señas **argentina** y el objetivo de diseño es la chilena.

### Qué NO está en este repositorio

**Los vídeos de LSA64 no se redistribuyen.** `data/external/` está fuera del
control de versiones. Para reproducir la extracción hay que descargar el corpus
de su fuente original, aceptando sus términos (ver el README, sección
"Reproducir el entrenamiento completo").

Lo que sí está versionado son **keypoints ya extraídos y normalizados**: 21
coordenadas por mano y por frame, sin imagen. No permiten reconstruir el vídeo
ni identificar a los señantes, pero siguen siendo obra derivada y por eso
heredan la licencia.

---

## 2. MediaPipe — HandLandmarker

La Capa 1 usa `HandLandmarker` de **MediaPipe Tasks** (Google), bajo
**Apache License 2.0**.

El archivo de pesos `models/hand_landmarker.task` **no se redistribuye aquí**:
lo descarga `scripts/descargar_modelo.py` desde el catálogo oficial de MediaPipe, y
`.gitignore` lo excluye.

- MediaPipe: <https://github.com/google-ai-edge/mediapipe>
- Apache 2.0: <https://www.apache.org/licenses/LICENSE-2.0>

---

## 3. Dependencias de Python

Las de `requirements.txt` se instalan desde PyPI y conservan sus propias
licencias, todas permisivas (BSD, MIT o Apache 2.0): NumPy, OpenCV,
scikit-learn, Matplotlib, ONNX, ONNX Runtime, TensorFlow, tf2onnx, psutil y
pytest. Ninguna se redistribuye en este repositorio.

---

## 4. Cómo citar este trabajo

Si el pipeline o el modelo se usan en un trabajo académico, cita **el corpus
LSA64** (referencia de la sección 1) junto con este repositorio, y declara
explícitamente que **el sistema está validado sobre lengua de señas argentina,
no chilena**. Las limitaciones que hay que declarar al reportar cualquier cifra
están en el README, sección "Limitaciones declaradas".
