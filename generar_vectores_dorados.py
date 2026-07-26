"""
generar_vectores_dorados.py — vectores de prueba para portar la Capa 2 a C#.

Riesgo que mitiga (train/serve skew): si el preprocesamiento en Unity/C# no
replica EXACTAMENTE el de Python, el modelo recibe una entrada distinta a la que
vio en entrenamiento y devuelve basura — con un ONNX que carga sin errores y una
demo que "funciona", así que el fallo es silencioso.

Este script congela la salida de referencia de cada paso del preprocesamiento en
`integracion/vectores_dorados.json`. La implementación en C# se considera
correcta cuando reproduce todos los casos dentro de la tolerancia declarada.

Uso:
    python generar_vectores_dorados.py              # genera el JSON
    python generar_vectores_dorados.py --verificar  # comprueba que sigue vigente

Ver INTEGRACION_UNITY.md para la especificación y el código C# de referencia.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from lsch_mr import config
from lsch_mr.caracteristicas import (preparar_entrada, remuestrear_tiempo,
                                     secuencia_a_features)
from lsch_mr.consola import configurar_utf8
from lsch_mr.keypoint_normalizer import KeypointNormalizer

configurar_utf8()

SALIDA = config.RAIZ / "integracion" / "vectores_dorados.json"
TOLERANCIA = 1e-5
_DEC = 6          # decimales guardados; por encima de la tolerancia declarada


def _mano_base() -> np.ndarray:
    """Mano sintética pero plausible, en coordenadas de imagen normalizadas.

    Determinista a propósito: los vectores dorados deben poder regenerarse en
    cualquier máquina sin depender de un corpus grabado.
    """
    rng = np.random.default_rng(config.SEMILLA)
    palma = np.array([0.50, 0.60, 0.00], dtype=np.float32)      # L0, muñeca
    pts = [palma]
    # 4 puntos por dedo, 5 dedos: se abren en abanico desde la muñeca.
    for dedo in range(5):
        ang = -1.20 + dedo * 0.55
        for art in range(1, 5):
            radio = 0.045 * art + 0.02
            pts.append(np.array([
                palma[0] + radio * np.sin(ang),
                palma[1] - radio * np.cos(ang),
                0.01 * art * (1 if dedo % 2 == 0 else -1),
            ], dtype=np.float32))
    mano = np.stack(pts, axis=0).astype(np.float32)
    # Ruido determinista: evita que una implementación "casi correcta" acierte
    # por simetría del caso sintético.
    mano += rng.normal(0.0, 0.004, size=mano.shape).astype(np.float32)
    return mano.astype(np.float32)


def _secuencia_dos_manos(n_frames: int) -> np.ndarray:
    """Secuencia cruda (T, 2, 21, 3) [Left, Right] con la izquierda ausente al inicio."""
    base = _mano_base()
    seq = []
    for t in range(n_frames):
        desplazamiento = np.array([0.004 * t, -0.003 * t, 0.0], dtype=np.float32)
        der = base + desplazamiento
        if t < 3:   # la mano izquierda entra en escena en el frame 3
            izq = np.full_like(base, np.nan)
        else:
            izq = base * 0.9 - desplazamiento
        seq.append(np.stack([izq, der], axis=0))
    return np.stack(seq, axis=0).astype(np.float32)


def _r(a: np.ndarray) -> list:
    return np.round(np.asarray(a, dtype=np.float64), _DEC).tolist()


def construir_casos() -> list[dict]:
    norm = KeypointNormalizer()
    mano = _mano_base()
    casos: list[dict] = []

    casos.append({
        "id": "normalize_mano_tipica",
        "descripcion": "KeypointNormalizer.normalize sobre una mano bien formada.",
        "entrada": {"frame_21x3": _r(mano)},
        "esperado": {"normvector_63": _r(norm.normalize(mano))},
    })

    trasladada = mano + np.array([0.17, -0.09, 0.05], dtype=np.float32)
    casos.append({
        "id": "normalize_invariancia_traslacion",
        "descripcion": ("La misma mano desplazada debe dar el MISMO NormVector. "
                        "Si falla, el centrado en L0 no se aplicó."),
        "entrada": {"frame_21x3": _r(trasladada)},
        "esperado": {"normvector_63": _r(norm.normalize(trasladada)),
                     "igual_a_caso": "normalize_mano_tipica"},
    })

    escalada = mano * np.float32(2.5)
    casos.append({
        "id": "normalize_invariancia_escala",
        "descripcion": ("La misma mano escalada x2.5 debe dar el MISMO NormVector. "
                        "Si falla, no se dividió por ||L9-L0||."),
        "entrada": {"frame_21x3": _r(escalada)},
        "esperado": {"normvector_63": _r(norm.normalize(escalada))},
    })

    degenerada = np.repeat(mano[:1], config.NUM_LANDMARKS, axis=0)
    casos.append({
        "id": "normalize_mano_degenerada",
        "descripcion": ("Todos los landmarks colapsados: ||L9-L0|| < 1e-6. "
                        "Debe devolver ceros, NO NaN ni división por cero."),
        "entrada": {"frame_21x3": _r(degenerada)},
        "esperado": {"normvector_63": _r(np.zeros(config.NORMVECTOR_DIM))},
    })

    par = np.stack([np.full_like(mano, np.nan), mano], axis=0)[None, ...]
    casos.append({
        "id": "ensamblado_ambas_mano_ausente",
        "descripcion": ("Modo 'ambas': orden [Left|Right]; la mano ausente se "
                        "rellena con 63 ceros, no se omite ni se duplica la otra."),
        "entrada": {"frame_2x21x3_left_nan": _r(par[0])},
        "esperado": {"features_126": _r(secuencia_a_features(par, modo="ambas")[0])},
    })

    corta = np.arange(17 * 4, dtype=np.float32).reshape(17, 4) / 10.0
    casos.append({
        "id": "remuestreo_temporal_17_a_60",
        "descripcion": ("Interpolación lineal en el tiempo, por característica, "
                        "de T=17 a SEQ_LEN=60. Dimensión reducida (F=4) a "
                        "propósito: aísla el remuestreo del resto."),
        "entrada": {"features_17x4": _r(corta)},
        "esperado": {"features_60x4": _r(remuestrear_tiempo(corta, config.SEQ_LEN))},
    })

    seq = _secuencia_dos_manos(23)
    casos.append({
        "id": "end_to_end_secuencia_cruda_a_entrada_del_modelo",
        "descripcion": ("Cadena completa: SignSequence cruda (T=23, 2 manos) -> "
                        "normalización por mano -> ensamblado [Left|Right] -> "
                        "remuestreo a 60. Es la entrada exacta que recibe el ONNX."),
        "entrada": {"secuencia_23x2x21x3": _r(seq)},
        "esperado": {"entrada_modelo_60x126":
                     _r(preparar_entrada(seq, modo="ambas", seq_len=config.SEQ_LEN))},
    })

    return casos


def caso_modelo() -> dict | None:
    """Caso opcional: entrada conocida -> scores del ONNX actual.

    Permite verificar que Unity Sentis produce la misma salida que onnxruntime
    para la misma entrada, cerrando la cadena completa y no solo el
    preprocesamiento. Se omite si todavía no hay modelo exportado.

    OJO: este caso queda invalidado cada vez que se reentrena el modelo. Hay que
    regenerar el JSON junto con el `.onnx` y enviarlos siempre en pareja.
    """
    onnx_path = config.OUTPUTS_MODELS_DIR / "modelo.onnx"
    labels_path = config.OUTPUTS_MODELS_DIR / "labels.json"
    if not onnx_path.exists() or not labels_path.exists():
        return None

    import onnxruntime as ort

    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    modo = meta.get("modo_manos", config.MODO_MANOS)
    seq_len = int(meta.get("seq_len", config.SEQ_LEN))

    # Entrada determinista, coherente con el modo del modelo entrenado.
    seq = _secuencia_dos_manos(23)
    if modo != "ambas":
        seq = seq[:, 1]                      # se queda con la mano derecha
    entrada = preparar_entrada(seq, modo=modo, seq_len=seq_len)

    sesion = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    nombre = sesion.get_inputs()[0].name
    scores = sesion.run(None, {nombre: entrada[None, ...].astype(np.float32)})[0][0]

    idx = int(np.argmax(scores))
    clases = meta.get("classes", [])
    return {
        "id": "inferencia_onnx_entrada_conocida",
        "descripcion": ("Entrada conocida -> scores del modelo. Verifica que "
                        "Sentis coincide con onnxruntime. SE INVALIDA AL "
                        "REENTRENAR: regenerar junto con el .onnx."),
        "modelo": {"modo_manos": modo, "seq_len": seq_len,
                   "n_clases": len(clases), "sha1_onnx": _sha1(onnx_path)},
        "entrada": {f"entrada_modelo_{seq_len}x{entrada.shape[1]}": _r(entrada)},
        "esperado": {"scores": _r(scores),
                     "indice_ganador": idx,
                     "etiqueta_ganadora": clases[idx] if clases else None},
    }


def _sha1(path: Path) -> str:
    import hashlib

    return hashlib.sha1(path.read_bytes()).hexdigest()


def construir_documento() -> dict:
    casos = construir_casos()
    cm = caso_modelo()
    if cm is not None:
        casos.append(cm)
    return {
        "generado": date.today().isoformat(),
        "proposito": ("Vectores de referencia para verificar el port del "
                      "preprocesamiento (Capa 2) a C#/Unity. Ver INTEGRACION_UNITY.md."),
        "tolerancia_absoluta": TOLERANCIA,
        "config": {
            "SEQ_LEN": config.SEQ_LEN,
            "NUM_LANDMARKS": config.NUM_LANDMARKS,
            "NUM_EJES": config.NUM_EJES,
            "NORMVECTOR_DIM": config.NORMVECTOR_DIM,
            "WRIST_IDX": config.WRIST_IDX,
            "MIDDLE_MCP_IDX": config.MIDDLE_MCP_IDX,
            "EPS_ESCALA": 1e-6,
            "ORDEN_MANOS": ["Left", "Right"],
            "CONF_THRESHOLD": config.CONF_THRESHOLD,
        },
        "casos": casos,
    }


def _comparar(doc: dict) -> int:
    """Recalcula los casos y los compara con el JSON guardado."""
    actuales = {c["id"]: c for c in construir_casos()}
    cm = caso_modelo()
    if cm is not None:
        actuales[cm["id"]] = cm
    fallos = 0
    for caso in doc["casos"]:
        cid = caso["id"]
        if cid not in actuales:
            print(f"  [!] {cid}: ya no existe en el generador")
            fallos += 1
            continue
        for clave, esperado in caso["esperado"].items():
            if not isinstance(esperado, list):
                continue
            obtenido = actuales[cid]["esperado"][clave]
            dif = np.max(np.abs(np.array(esperado) - np.array(obtenido)))
            estado = "ok" if dif <= TOLERANCIA else "DIFIERE"
            if dif > TOLERANCIA:
                fallos += 1
            print(f"  {cid}.{clave}: {estado} (dif max {dif:.2e})")
    return fallos


def main() -> int:
    ap = argparse.ArgumentParser(description="Vectores dorados para el port a C#")
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--verificar", action="store_true",
                    help="no regenera: comprueba que el JSON guardado sigue "
                         "coincidiendo con el pipeline actual")
    args = ap.parse_args()
    destino = Path(args.salida)

    if args.verificar:
        if not destino.exists():
            print(f"No existe {destino}. Genéralo primero.")
            return 1
        doc = json.loads(destino.read_text(encoding="utf-8"))
        print(f"[dorados] verificando {destino} ...")
        fallos = _comparar(doc)
        if fallos:
            print(f"\n[dorados] {fallos} discrepancia(s): el preprocesamiento "
                  "cambió respecto a los vectores dorados.")
            print("  Si el cambio es intencional, regenera el JSON y avisa al "
                  "equipo de Unity: su port en C# queda invalidado.")
            return 1
        print("\n[dorados] todo coincide.")
        return 0

    doc = construir_documento()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[dorados] {len(doc['casos'])} casos -> {destino}")
    for c in doc["casos"]:
        print(f"    {c['id']}")
    print("\nEl port en C# debe reproducir todos los casos con tolerancia "
          f"{TOLERANCIA:g}. Ver INTEGRACION_UNITY.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
