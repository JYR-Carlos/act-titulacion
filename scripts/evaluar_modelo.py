"""
evaluar_modelo.py — MÉTRICA 1 del MVP: precisión algorítmica (Sección 11).

Evalúa el clasificador ya entrenado y produce lo que pide el informe:
accuracy global, accuracy por seña, matriz de confusión (PNG + CSV) y el
veredicto contra el umbral `config.ACCURACY_OBJETIVO` (0.85), **destacado en
rojo si no se alcanza**.

A diferencia de `entrenar.py --cv`, este script **no entrena nada**: carga el
modelo que ya existe y lo mide. Eso lo hace barato de repetir y, sobre todo,
mide el artefacto que realmente se entrega (`modelo.onnx`), no una réplica.

--------------------------------------------------------------------------
  DOS NÚMEROS DISTINTOS, Y NO SON INTERCAMBIABLES — leer antes de reportar
--------------------------------------------------------------------------
El informe pide "conjunto de test separado (20%) **con validación cruzada**".
Son dos protocolos distintos y este script expone los dos por separado, porque
mezclarlos es la forma más fácil de reportar una cifra que no se sostiene:

  --fuente modelo   Evalúa `modelo.onnx` sobre el 20% que `ModelTrainer.train()`
                    dejó fuera. Es el desempeño del artefacto exportado, pero el
                    split es ALEATORIO: repeticiones de un mismo señante caen a
                    los dos lados, así que el modelo puede reconocer a la persona
                    en vez de la seña. Sobre LSA64 esto da ~0.98.

  --fuente cv       Rinde el reporte desde las predicciones out-of-fold que ya
                    guardó `entrenar.py --cv`. Cada muestra fue predicha por un
                    modelo que no la vio. Con `cv_metrics_dominante_senante.json` (k-fold
                    dejando SEÑANTES fuera) da ~0.90 ± 0.05.

**La cifra defendible es la segunda.** La caída de 0.98 a 0.90 no es ruido: es
exactamente la fuga de información que el k-fold por señante elimina. Reportar
0.98 como "accuracy del sistema" sobreestima lo que hará con una persona nueva.
Ver la tabla y su explicación en `docs/ESTADO_ACTUAL.md`.

Uso:
    # Recomendado para el informe (k-fold por señante, ya calculado):
    python scripts/evaluar_modelo.py --fuente cv --cv-json outputs/reports/cv_metrics_dominante_senante.json

    # Desempeño del ONNX exportado sobre su 20% retenido:
    python scripts/evaluar_modelo.py --fuente modelo --dataset data/processed/lsa64_dominante.npz

    # Lo mismo con el .keras en vez del ONNX (para descartar un fallo de export):
    python scripts/evaluar_modelo.py --fuente modelo --backend keras --dataset ...

Salidas (en outputs/reports/, prefijo configurable con --prefijo):
    <prefijo>.json              resumen completo, listo para citar
    <prefijo>_matriz.png        matriz de confusión (título en rojo si no cumple)
    <prefijo>_matriz.csv        la misma matriz, para pegar en el informe
    <prefijo>_por_clase.csv     accuracy / precisión / F1 por seña
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _raiz  # noqa: F401  (raíz del repo en sys.path; debe ir antes que lsch_mr)
from lsch_mr import config
from lsch_mr import metricas_clasificacion as mc
from lsch_mr.consola import configurar_utf8, negrita, rojo, verde

configurar_utf8()


# --------------------------------------------------------------------------- #
# Fuente 1: el modelo exportado, sobre el 20% retenido por train()
# --------------------------------------------------------------------------- #
def _predecir_onnx(onnx_path: Path, X: np.ndarray) -> np.ndarray:
    """Probabilidades (N, n_clases) del ONNX. El grafo ya termina en Softmax."""
    import onnxruntime as ort

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"No existe {onnx_path}. Expórtalo con: python scripts/exportar_onnx.py")
    sesion = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    nombre = sesion.get_inputs()[0].name
    # Por lotes: un corpus de cientos de secuencias de (60, 63) cabe holgado en
    # memoria, pero el lote evita picos innecesarios y va más rápido que 1 a 1.
    salidas = []
    for i in range(0, len(X), 64):
        lote = X[i:i + 64].astype(np.float32)
        salidas.append(sesion.run(None, {nombre: lote})[0])
    return np.concatenate(salidas, axis=0)


def _predecir_keras(keras_path: Path, X: np.ndarray) -> np.ndarray:
    if not keras_path.exists():
        raise FileNotFoundError(
            f"No existe {keras_path}. Entrénalo con: python scripts/entrenar.py")
    import tensorflow as tf

    modelo = tf.keras.models.load_model(keras_path)
    return modelo.predict(X.astype(np.float32), verbose=0)


def _evaluar_modelo(args) -> dict:
    """Evalúa el modelo exportado sobre el split de validación de `train()`."""
    from lsch_mr.model_trainer import ModelTrainer

    ds = Path(args.dataset)
    if not ds.exists():
        raise FileNotFoundError(
            f"No existe el dataset {ds}. Constrúyelo con: python scripts/build_dataset.py")
    X, y, classes = ModelTrainer.cargar_dataset(ds)

    if args.todo_el_dataset:
        idx = np.arange(len(y))
        descripcion = (f"TODO el dataset ({len(idx)} muestras) — incluye datos de "
                       "entrenamiento, NO es una estimación honesta")
    else:
        # Mismo reparto que produjo el modelo: ver ModelTrainer.split_indices.
        _, idx = ModelTrainer.split_indices(y, len(classes),
                                            val_split=args.test_split,
                                            seed=args.semilla)
        descripcion = (f"conjunto de test retenido: {len(idx)} de {len(y)} muestras "
                       f"({args.test_split:.0%}), split estratificado aleatorio "
                       f"semilla={args.semilla}")

    probs = (_predecir_keras(Path(args.keras), X[idx]) if args.backend == "keras"
             else _predecir_onnx(Path(args.onnx), X[idx]))
    y_true = y[idx]
    y_pred = np.argmax(probs, axis=1)
    confianzas = np.max(probs, axis=1)

    return {
        "fuente": "modelo",
        "backend": args.backend,
        "artefacto": str(Path(args.keras if args.backend == "keras" else args.onnx)),
        "dataset": str(ds),
        "protocolo": descripcion,
        "esquema": "split 80/20 estratificado aleatorio",
        "advertencia_esquema": (
            "El split es aleatorio: repeticiones de un mismo señante caen en "
            "entrenamiento y en test a la vez, así que esta cifra sobreestima "
            "la generalización a una persona nueva. Para el informe usa "
            "--fuente cv con cv_metrics_dominante_senante.json."),
        "classes": classes,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "confianzas": [float(c) for c in confianzas],
    }


# --------------------------------------------------------------------------- #
# Fuente 2: las predicciones out-of-fold que ya guardó entrenar.py --cv
# --------------------------------------------------------------------------- #
def _evaluar_desde_cv(args) -> dict:
    ruta = Path(args.cv_json)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Genéralo con:\n"
            f"    python scripts/entrenar.py --dataset {args.dataset} --cv-grupos 2 --solo-cv\n"
            "(--cv-grupos 2 = el señante es el 2º campo del sample_id, "
            "p.ej. 017_001_001 -> señante 001)")

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    oof = datos.get("out_of_fold") or {}
    if not oof.get("y_true"):
        raise ValueError(
            f"{ruta} no guarda predicciones out-of-fold. Fue generado por una "
            "versión anterior de ModelTrainer: vuelve a correr entrenar.py --cv.")

    classes = datos.get("classes") or _classes_del_labels(args)
    n = len(oof["y_true"])
    esquema = datos.get("esquema", "estratificado")
    return {
        "fuente": "cv",
        "artefacto": str(ruta),
        "protocolo": (f"validación cruzada {datos.get('n_splits')}-fold "
                      f"({esquema}), {n} muestras out-of-fold — cada una predicha "
                      "por un modelo que no la vio en entrenamiento"),
        "esquema": f"{datos.get('n_splits')}-fold {esquema}",
        "n_grupos": datos.get("n_grupos", 0),
        "fold_accuracies": datos.get("fold_accuracies", []),
        "mean_accuracy_folds": datos.get("mean_accuracy"),
        "std_accuracy_folds": datos.get("std_accuracy"),
        "advertencia_esquema": (
            "" if esquema == "por señante" else
            "Esquema ESTRATIFICADO: los folds reparten al azar, así que un mismo "
            "señante aparece en entrenamiento y validación. Para la cifra "
            "defendible usa cv_metrics_dominante_senante.json (k-fold por señante)."),
        "classes": classes,
        "y_true": oof["y_true"],
        "y_pred": oof["y_pred"],
        "confianzas": oof.get("confianzas", []),
    }


def _classes_del_labels(args) -> list[str]:
    ruta = Path(args.labels)
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8")).get("classes", [])
    return []


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def _imprimir(resumen: dict) -> None:
    classes = resumen["classes"]
    print()
    print(negrita("=" * 74))
    print(negrita("  MÉTRICA 1 — PRECISIÓN ALGORÍTMICA (accuracy)"))
    print(negrita("=" * 74))
    print(f"  artefacto   : {resumen['artefacto']}")
    print(f"  protocolo   : {resumen['protocolo']}")
    print(f"  n muestras  : {resumen['n_muestras']}   clases: {len(classes)}")

    if resumen.get("fold_accuracies"):
        por_fold = ", ".join(f"{a:.3f}" for a in resumen["fold_accuracies"])
        print(f"  por fold    : {por_fold}")
        print(f"  media±desv  : {resumen['mean_accuracy_folds']:.3f} ± "
              f"{resumen['std_accuracy_folds']:.3f}")

    # -- Accuracy por seña ------------------------------------------------- #
    print()
    print(negrita("  Accuracy por seña"))
    print(f"    {'seña':<14} {'soporte':>8} {'aciertos':>9} {'accuracy':>9} "
          f"{'precision':>10} {'F1':>7}")
    print("    " + "-" * 60)
    for fila in resumen["por_clase"]:
        # Una clase concreta por debajo del objetivo se marca aunque el global
        # cumpla: es donde se esconde la seña que el sistema nunca acierta.
        acc_txt = f"{fila['accuracy']:.3f}"
        acc_txt = (verde(f"{acc_txt:>9}") if fila["accuracy"] >= resumen["veredicto"]["objetivo"]
                   else rojo(f"{acc_txt:>9}"))
        print(f"    {fila['clase']:<14} {fila['soporte']:>8} {fila['aciertos']:>9} "
              f"{acc_txt} {fila['precision']:>10.3f} {fila['f1']:>7.3f}")

    # -- Veredicto global --------------------------------------------------- #
    v = resumen["veredicto"]
    print()
    linea = (f"  ACCURACY GLOBAL: {v['accuracy']:.4f}  "
             f"(objetivo MVP >= {v['objetivo']:.2f})")
    if v["cumple"]:
        print(verde(linea + "   -> CUMPLE"))
    else:
        print(rojo(linea + "   -> NO CUMPLE"))
        print(rojo(f"  Faltan {v['brecha']:.4f} ({v['brecha'] * 100:.2f} puntos "
                   "porcentuales) para alcanzar el umbral."))

    if resumen.get("advertencia_esquema"):
        print()
        print("  [!] " + resumen["advertencia_esquema"])

    # -- Umbral de confianza ------------------------------------------------ #
    cob = resumen.get("cobertura_umbral") or {}
    if cob.get("n"):
        print()
        print(negrita(f"  Con CONF_THRESHOLD = {cob['umbral']:.2f} "
                      "(lo que ve realmente el funcionario)"))
        print(f"    cobertura            : {cob['cobertura']:.1%} "
              f"({cob['n_mostradas']} de {cob['n']} señas se muestran)")
        print(f"    precisión mostradas  : {cob['precision_mostradas']:.1%}")
        print(f"    glosas erróneas en pantalla: {cob['n_erroneas_mostradas']}")
        print(f"    ocultas como '<desconocida>': {cob['n_ocultas_por_umbral']}")

    print()
    for clave, ruta in resumen["archivos"].items():
        print(f"  {clave:<12}: {ruta}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluación del clasificador — métrica 1 del MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", choices=["cv", "modelo"], default="cv",
                    help="'cv' = predicciones out-of-fold ya calculadas "
                         "(recomendado para el informe); 'modelo' = evaluar el "
                         "artefacto exportado sobre su 20%% retenido")
    # Por defecto, el k-fold por señante del modo `dominante`: es el esquema
    # defendible y el modo con el que se exportó `modelo.onnx` (labels.json).
    # El nombre incluye el modo porque evaluar los dos modos con el mismo nombre
    # es justo lo que dejó cifras huérfanas en el informe (ver evaluar_cv).
    ap.add_argument("--cv-json",
                    default=str(config.OUTPUTS_REPORTS_DIR
                                / "cv_metrics_dominante_senante.json"),
                    help="JSON de entrenar.py --cv (solo con --fuente cv)")
    ap.add_argument("--dataset",
                    default=str(config.DATA_PROCESSED_DIR / "dataset.npz"))
    ap.add_argument("--backend", choices=["onnx", "keras"], default="onnx",
                    help="qué artefacto medir (solo con --fuente modelo)")
    ap.add_argument("--onnx", default=str(config.OUTPUTS_MODELS_DIR / "modelo.onnx"))
    ap.add_argument("--keras", default=str(config.OUTPUTS_MODELS_DIR / "tcn_lsch.keras"))
    ap.add_argument("--labels", default=str(config.OUTPUTS_MODELS_DIR / "labels.json"))
    ap.add_argument("--test-split", type=float, default=config.ENTRENAMIENTO_VAL_SPLIT,
                    help=f"fracción retenida (default {config.ENTRENAMIENTO_VAL_SPLIT})")
    ap.add_argument("--semilla", type=int, default=config.SEMILLA,
                    help="debe coincidir con la del entrenamiento para que el "
                         "split retenido sea el mismo")
    ap.add_argument("--todo-el-dataset", action="store_true",
                    help="evaluar sobre TODAS las muestras (incluye las de "
                         "entrenamiento). Solo para depurar; no reportable")
    ap.add_argument("--objetivo", type=float, default=config.ACCURACY_OBJETIVO)
    ap.add_argument("--umbral-conf", type=float, default=config.CONF_THRESHOLD)
    ap.add_argument("--prefijo", default=None,
                    help="prefijo de los archivos de salida "
                         "(default: evaluacion_<fuente>)")
    ap.add_argument("--normalizar-matriz", action="store_true",
                    help="dibujar la matriz en %% por fila en vez de conteos")
    args = ap.parse_args()

    try:
        datos = (_evaluar_desde_cv(args) if args.fuente == "cv"
                 else _evaluar_modelo(args))
    except (FileNotFoundError, ValueError) as exc:
        print(rojo(f"[evaluar_modelo] {exc}"))
        return 1

    classes = datos["classes"]
    if not classes:
        print(rojo("[evaluar_modelo] No se pudo determinar la lista de clases "
                   f"(revisa {args.labels})."))
        return 1

    y_true, y_pred = datos["y_true"], datos["y_pred"]
    cm = mc.matriz_confusion(y_true, y_pred, len(classes))
    acc = mc.accuracy_global(y_true, y_pred)
    v = mc.veredicto(acc, args.objetivo)
    por_clase = mc.metricas_por_clase(cm, classes)
    cobertura = (mc.cobertura_con_umbral(y_true, y_pred, datos["confianzas"],
                                         args.umbral_conf)
                 if datos.get("confianzas") else {})

    prefijo = args.prefijo or f"evaluacion_{datos['fuente']}"
    base = config.OUTPUTS_REPORTS_DIR / prefijo
    subtitulo = (f"accuracy global {acc:.3f}  |  objetivo >= {args.objetivo:.2f}  |  "
                 + ("CUMPLE" if v["cumple"] else "NO CUMPLE"))
    archivos = {
        "matriz PNG": str(mc.graficar_matriz(
            cm, classes, base.with_name(base.name + "_matriz.png"),
            titulo=f"Matriz de confusión — {datos['esquema']}",
            subtitulo=subtitulo, cumple=v["cumple"],
            normalizar=args.normalizar_matriz)),
        "matriz CSV": str(mc.guardar_matriz_csv(
            cm, classes, base.with_name(base.name + "_matriz.csv"))),
        "por clase": str(mc.guardar_por_clase_csv(
            por_clase, base.with_name(base.name + "_por_clase.csv"))),
    }

    resumen = {
        "metrica": "1 - precisión algorítmica (accuracy)",
        "fuente": datos["fuente"],
        "artefacto": datos["artefacto"],
        "protocolo": datos["protocolo"],
        "esquema": datos["esquema"],
        "advertencia_esquema": datos.get("advertencia_esquema", ""),
        "n_muestras": len(y_true),
        "classes": classes,
        "veredicto": v,
        "por_clase": por_clase,
        "confusion_matrix": cm.tolist(),
        "cobertura_umbral": cobertura,
        "fold_accuracies": datos.get("fold_accuracies", []),
        "mean_accuracy_folds": datos.get("mean_accuracy_folds"),
        "std_accuracy_folds": datos.get("std_accuracy_folds"),
    }
    ruta_json = base.with_suffix(".json")
    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    archivos["resumen"] = str(ruta_json)
    resumen["archivos"] = archivos

    _imprimir(resumen)
    # Código de salida != 0 cuando no se cumple: permite encadenarlo en un script
    # de verificación de las 4 métricas sin tener que parsear la salida.
    return 0 if v["cumple"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
