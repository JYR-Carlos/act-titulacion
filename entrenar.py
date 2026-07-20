"""
entrenar.py — CLI del ModelTrainer (CU-02).

Entrena el TCN sobre el dataset normalizado y reporta accuracy + matriz de
confusión.

Uso:
    python entrenar.py                                  # data/processed/dataset.npz
    python entrenar.py --dataset data/processed/dataset.npz
    python entrenar.py --sintetico                      # dataset falso (smoke test)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento del TCN (ModelTrainer)")
    ap.add_argument("--dataset", default=str(config.DATA_PROCESSED_DIR / "dataset.npz"))
    ap.add_argument("--sintetico", action="store_true",
                    help="usar un dataset sintético para smoke-testing")
    ap.add_argument("--epochs", type=int, default=config.ENTRENAMIENTO_EPOCHS)
    args = ap.parse_args()

    if args.sintetico:
        print("[entrenar] Usando dataset SINTÉTICO (solo para probar el pipeline).")
        X, y, classes = _dataset_sintetico()
    else:
        ds = Path(args.dataset)
        if not ds.exists():
            print(f"No existe el dataset {ds}. Ejecuta build_dataset.py o usa "
                  "--sintetico.")
            return 1
        X, y, classes = ModelTrainer.cargar_dataset(ds)

    print(f"[entrenar] X={X.shape}  clases={classes}")
    trainer = ModelTrainer(epochs=args.epochs)
    m = trainer.train(X, y, classes)

    print("\n===== RESULTADOS =====")
    print(f"  val_accuracy   : {m.val_accuracy:.3f}  (objetivo MVP >= 0.85)")
    print(f"  train_accuracy : {m.train_accuracy:.3f}")
    print(f"  modelo Keras   : {m.keras_path}")
    print(f"  matriz conf.   : {m.confusion_png}")
    print("Siguiente paso:  python exportar_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
