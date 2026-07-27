"""
entrenar.py — CLI del ModelTrainer (CU-02).

Entrena el TCN sobre el dataset normalizado y reporta accuracy + matriz de
confusión.

Uso:
    python entrenar.py                                  # data/processed/dataset.npz
    python entrenar.py --dataset data/processed/dataset.npz
    python entrenar.py --sintetico                      # dataset falso (smoke test)
    python entrenar.py --cv                             # + validación cruzada k-fold
    python entrenar.py --cv --cv-folds 5 --solo-cv      # solo evaluar, sin exportar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lsch_mr import config
from lsch_mr.consola import configurar_utf8
from lsch_mr.model_trainer import ModelTrainer

configurar_utf8()


def _dataset_sintetico(n_por_clase=40, n_clases=10):
    """Genera un dataset falso separable para validar el pipeline end-to-end."""
    rng = np.random.default_rng(config.SEMILLA)
    F = config.n_features()
    X, y = [], []
    base = rng.normal(size=(n_clases, config.SEQ_LEN, F)).astype("float32")
    for c in range(n_clases):
        for _ in range(n_por_clase):
            X.append(base[c] + rng.normal(scale=0.15, size=(config.SEQ_LEN, F)).astype("float32"))
            y.append(c)
    classes = [f"CLASE_{c}" for c in range(n_clases)]
    return np.stack(X), np.asarray(y, dtype="int64"), classes


def _etiqueta_dataset(ds: Path, X: np.ndarray) -> str:
    """Etiqueta del modo de manos, para nombrar los reportes de la CV.

    Se lee del propio `.npz` (`build_dataset.py` guarda ahí `modo_manos`), y si
    ese campo no está —datasets antiguos— se deduce del ancho del vector de
    características. Sin esto, evaluar los dos modos sobrescribe el mismo
    `cv_metrics.json` y las cifras del informe dejan de ser rastreables.
    """
    try:
        data = np.load(ds, allow_pickle=True)
        if "modo_manos" in data:
            return str(data["modo_manos"])
    except Exception:
        pass
    return "ambas" if X.shape[2] == config.NORMVECTOR_DIM * 2 else "dominante"


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento del TCN (ModelTrainer)")
    ap.add_argument("--dataset", default=str(config.DATA_PROCESSED_DIR / "dataset.npz"))
    ap.add_argument("--sintetico", action="store_true",
                    help="usar un dataset sintético para smoke-testing")
    ap.add_argument("--epochs", type=int, default=config.ENTRENAMIENTO_EPOCHS)
    ap.add_argument("--cv", action="store_true",
                    help="ejecutar validación cruzada estratificada k-fold "
                         "antes de entrenar el modelo final")
    ap.add_argument("--cv-folds", type=int, default=config.ENTRENAMIENTO_CV_FOLDS,
                    help=f"número de folds (default {config.ENTRENAMIENTO_CV_FOLDS})")
    ap.add_argument("--solo-cv", action="store_true",
                    help="solo validación cruzada: no entrena ni guarda el "
                         "modelo final (implica --cv)")
    ap.add_argument("--cv-grupos", default=None, metavar="CAMPO|REGEX",
                    help="evalúa dejando señantes fuera. Número de campo del "
                         "sample_id separado por '_' (p.ej. 2 para "
                         "clase_senante_repeticion), o una regex con un grupo "
                         "de captura. Implica --cv")
    args = ap.parse_args()
    if args.solo_cv or args.cv_grupos:
        args.cv = True

    if args.sintetico:
        print("[entrenar] Usando dataset SINTÉTICO (solo para probar el pipeline).")
        X, y, classes = _dataset_sintetico()
        etiqueta = "sintetico"
    else:
        ds = Path(args.dataset)
        if not ds.exists():
            print(f"No existe el dataset {ds}. Ejecuta build_dataset.py o usa "
                  "--sintetico.")
            return 1
        X, y, classes = ModelTrainer.cargar_dataset(ds)
        etiqueta = _etiqueta_dataset(ds, X)

    print(f"[entrenar] X={X.shape}  clases={classes}")
    trainer = ModelTrainer(epochs=args.epochs, cv_folds=args.cv_folds)

    grupos = None
    if args.cv_grupos:
        if args.sintetico:
            print("[entrenar] --cv-grupos no aplica al dataset sintético.")
            return 1
        grupos = ModelTrainer.cargar_grupos(Path(args.dataset), args.cv_grupos)
        if grupos is None:
            print("[entrenar] El dataset no guarda 'sample_ids'; regenéralo con "
                  "build_dataset.py para poder evaluar por señante.")
            return 1

    cv = None
    if args.cv:
        print(f"\n[entrenar] Validación cruzada estratificada "
              f"({args.cv_folds}-fold) — entrena {args.cv_folds} modelos.")
        cv = trainer.evaluar_cv(X, y, classes, groups=grupos, etiqueta=etiqueta)
        print("\n===== VALIDACIÓN CRUZADA =====")
        print(f"  accuracy       : {cv.resumen()}")
        print("  por fold       : " +
              ", ".join(f"{a:.3f}" for a in cv.fold_accuracies))
        print(f"  objetivo MVP   : >= {config.ACCURACY_OBJETIVO:.2f}  -> "
              f"{'CUMPLE' if cv.mean_accuracy >= config.ACCURACY_OBJETIVO else 'NO CUMPLE'}")
        print(f"  matriz conf.   : {cv.confusion_png}")
        print(f"  métricas       : {cv.metrics_path}")

    if args.solo_cv:
        print("\n[entrenar] --solo-cv: no se entrenó el modelo final. "
              "Vuelve a correr sin --solo-cv para generar el .keras.")
        return 0

    m = trainer.train(X, y, classes, cv=cv)

    print("\n===== RESULTADOS (modelo final) =====")
    print(f"  arquitectura   : {m.architecture}")
    print(f"  val_accuracy   : {m.val_accuracy:.3f}  "
          f"(objetivo MVP >= {config.ACCURACY_OBJETIVO:.2f})")
    print(f"  train_accuracy : {m.train_accuracy:.3f}")
    if m.cv is not None:
        print(f"  cross-val      : {m.cv.resumen()}")
    print(f"  modelo Keras   : {m.keras_path}")
    print(f"  matriz conf.   : {m.confusion_png}")
    print("Siguiente paso:  python exportar_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
