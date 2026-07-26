"""
ModelTrainer — Pipeline offline (CONTEXTO_PROYECTO.md Sección 10.2 / CU-02).

Entrena el TCN sobre el dataset normalizado y produce:
  * accuracy (train/val)
  * matriz de confusión (PNG) + classification report
  * modelo Keras entrenado (.keras) + mapa de etiquetas (labels.json)

Dos modos, con propósitos distintos (Secciones 5 y 12 del diseño):
  * `train()`      — split estratificado 80/20. Produce el **modelo final** que
                     se exporta a ONNX.
  * `evaluar_cv()` — validación cruzada estratificada k-fold. Produce la
                     **estimación de desempeño reportable** (media ± desv. est.)
                     y una matriz de confusión out-of-fold sobre todo el corpus.
                     No exporta modelo: cada fold entrena uno distinto.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .tcn import build_tcn


@dataclass
class CVMetrics:
    """Resultado de la validación cruzada estratificada (k-fold)."""
    n_splits: int
    fold_accuracies: list[float]
    mean_accuracy: float
    std_accuracy: float
    classes: list[str]
    confusion_matrix: list[list[int]]   # out-of-fold, sobre todas las muestras
    report: dict = field(default_factory=dict)
    architecture: str = config.ARQUITECTURA
    confusion_png: str = ""
    metrics_path: str = ""

    def resumen(self) -> str:
        return (f"{self.mean_accuracy:.3f} ± {self.std_accuracy:.3f} "
                f"({self.n_splits}-fold estratificado)")

    def to_dict(self) -> dict:
        return {"n_splits": self.n_splits,
                "fold_accuracies": self.fold_accuracies,
                "mean_accuracy": self.mean_accuracy,
                "std_accuracy": self.std_accuracy,
                "architecture": self.architecture,
                "confusion_matrix": self.confusion_matrix,
                "report": self.report}


@dataclass
class Metrics:
    val_accuracy: float
    train_accuracy: float
    classes: list[str]
    confusion_matrix: list[list[int]]
    report: dict = field(default_factory=dict)
    keras_path: str = ""
    labels_path: str = ""
    confusion_png: str = ""
    architecture: str = config.ARQUITECTURA
    cv: Optional[CVMetrics] = None


class ModelTrainer:
    def __init__(self,
                 epochs: int = config.ENTRENAMIENTO_EPOCHS,
                 batch_size: int = config.ENTRENAMIENTO_BATCH,
                 lr: float = config.ENTRENAMIENTO_LR,
                 val_split: float = config.ENTRENAMIENTO_VAL_SPLIT,
                 seed: int = config.SEMILLA,
                 cv_folds: int = config.ENTRENAMIENTO_CV_FOLDS) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split
        self.seed = seed
        self.cv_folds = cv_folds
        self.model = None

    @staticmethod
    def cargar_dataset(npz_path: Path):
        """Carga X (N, seq_len, F), y (N,) y la lista de clases desde un .npz."""
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype("float32")
        y = data["y"].astype("int64")
        classes = [str(c) for c in data["classes"]]
        return X, y, classes

    # -- Helpers compartidos train() / evaluar_cv() -------------------------- #
    # Ambos modos deben entrenar EXACTAMENTE la misma arquitectura y receta;
    # si no, la estimación k-fold no describiría al modelo que se exporta.
    def _nuevo_modelo(self, seq_len: int, n_features: int, n_classes: int):
        import tensorflow as tf
        modelo = build_tcn(seq_len=seq_len, n_features=n_features,
                           n_classes=n_classes)
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        return modelo

    @staticmethod
    def _callbacks():
        import tensorflow as tf
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=20,
                restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
        ]

    # -- Contrato del diseño ------------------------------------------------ #
    def train(self, X: np.ndarray, y: np.ndarray,
              classes: list[str],
              output_dir: Path = config.OUTPUTS_MODELS_DIR,
              reports_dir: Path = config.OUTPUTS_REPORTS_DIR,
              cv: Optional[CVMetrics] = None) -> Metrics:
        """Entrena el modelo final sobre un split estratificado 80/20.

        `cv` es opcional: si se pasa el resultado de `evaluar_cv()`, se adjunta
        a `metrics.json` para que el informe tenga en un solo archivo la
        estimación k-fold y el desempeño del modelo efectivamente exportado.
        """
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="int64")
        n_classes = len(classes)

        # Estratificado si cada clase tiene ≥2 muestras; si no, split simple.
        estratifica = all(np.bincount(y, minlength=n_classes) >= 2)
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.val_split, random_state=self.seed,
            stratify=y if estratifica else None)

        self.model = self._nuevo_modelo(seq_len=X.shape[1],
                                        n_features=X.shape[2],
                                        n_classes=n_classes)
        hist = self.model.fit(
            Xtr, ytr, validation_data=(Xva, yva),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=self._callbacks(), verbose=2)

        # -- Métricas ------------------------------------------------------- #
        train_acc = float(max(hist.history.get("accuracy", [0.0])))
        yva_pred = np.argmax(self.model.predict(Xva, verbose=0), axis=1)
        val_acc = float(np.mean(yva_pred == yva)) if len(yva) else 0.0
        etiquetas = list(range(n_classes))
        cm = confusion_matrix(yva, yva_pred, labels=etiquetas)
        report = classification_report(
            yva, yva_pred, labels=etiquetas, target_names=classes,
            output_dict=True, zero_division=0)

        # -- Persistencia --------------------------------------------------- #
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
        keras_path = output_dir / "tcn_lsch.keras"
        labels_path = output_dir / "labels.json"
        self.model.save(keras_path)
        labels_path.write_text(
            json.dumps({"classes": classes, "modo_manos": config.MODO_MANOS,
                        "seq_len": int(X.shape[1]), "n_features": int(X.shape[2])},
                       ensure_ascii=False, indent=2), encoding="utf-8")

        cm_png = self._plot_confusion(
            cm, classes, reports_dir / "matriz_confusion.png",
            titulo="Matriz de confusión (validación)")
        resumen = {"architecture": config.ARQUITECTURA,
                   "val_accuracy": val_acc, "train_accuracy": train_acc,
                   "report": report}
        if cv is not None:
            resumen["cross_validation"] = cv.to_dict()
        (reports_dir / "metrics.json").write_text(
            json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

        return Metrics(
            val_accuracy=val_acc, train_accuracy=train_acc, classes=classes,
            confusion_matrix=cm.tolist(), report=report,
            keras_path=str(keras_path), labels_path=str(labels_path),
            confusion_png=str(cm_png), cv=cv)

    # -- Validación cruzada (diseño, Secciones 5 y 12) ---------------------- #
    def evaluar_cv(self, X: np.ndarray, y: np.ndarray,
                   classes: list[str],
                   n_splits: Optional[int] = None,
                   reports_dir: Path = config.OUTPUTS_REPORTS_DIR,
                   verbose: int = 0) -> CVMetrics:
        """Validación cruzada estratificada k-fold sobre todo el corpus.

        Entrena un modelo por fold (misma arquitectura y receta que `train()`)
        y evalúa sobre el fold retenido. Devuelve accuracy media ± desviación
        estándar y una matriz de confusión out-of-fold: cada muestra del corpus
        es predicha exactamente una vez, por un modelo que no la vio en
        entrenamiento.

        No guarda ningún modelo — el modelo exportable lo produce `train()`.
        """
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import StratifiedKFold

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="int64")
        n_classes = len(classes)

        # k-fold estratificado exige al menos `n_splits` muestras por clase.
        n_splits = int(n_splits or self.cv_folds)
        min_por_clase = int(np.bincount(y, minlength=n_classes).min())
        if min_por_clase < 2:
            raise ValueError(
                "La validación cruzada estratificada necesita al menos 2 "
                f"muestras por clase; la clase más escasa tiene {min_por_clase}. "
                "Graba más muestras del corpus antes de evaluar con k-fold.")
        if min_por_clase < n_splits:
            print(f"[cv] AVISO: la clase más escasa tiene {min_por_clase} "
                  f"muestras; se reduce k de {n_splits} a {min_por_clase}.")
            n_splits = min_por_clase

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=self.seed)

        fold_accs: list[float] = []
        y_pred_oof = np.zeros_like(y)   # predicción out-of-fold de cada muestra

        for k, (idx_tr, idx_va) in enumerate(skf.split(X, y), start=1):
            # Semilla distinta por fold pero determinista: reproducible sin que
            # los k modelos partan de la misma inicialización.
            tf.random.set_seed(self.seed + k)
            np.random.seed(self.seed + k)

            modelo = self._nuevo_modelo(seq_len=X.shape[1],
                                        n_features=X.shape[2],
                                        n_classes=n_classes)
            modelo.fit(X[idx_tr], y[idx_tr],
                       validation_data=(X[idx_va], y[idx_va]),
                       epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=self._callbacks(), verbose=verbose)

            pred = np.argmax(modelo.predict(X[idx_va], verbose=0), axis=1)
            y_pred_oof[idx_va] = pred
            acc = float(np.mean(pred == y[idx_va]))
            fold_accs.append(acc)
            print(f"[cv] fold {k}/{n_splits}: accuracy={acc:.3f} "
                  f"(train={len(idx_tr)}, val={len(idx_va)})")

            # Liberar el grafo del fold antes de construir el siguiente.
            del modelo
            tf.keras.backend.clear_session()

        etiquetas = list(range(n_classes))
        cm = confusion_matrix(y, y_pred_oof, labels=etiquetas)
        report = classification_report(
            y, y_pred_oof, labels=etiquetas, target_names=classes,
            output_dict=True, zero_division=0)

        reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
        cm_png = self._plot_confusion(
            cm, classes, reports_dir / "matriz_confusion_cv.png",
            titulo=f"Matriz de confusión ({n_splits}-fold, out-of-fold)")

        cv = CVMetrics(
            n_splits=n_splits, fold_accuracies=fold_accs,
            mean_accuracy=float(np.mean(fold_accs)),
            std_accuracy=float(np.std(fold_accs)),
            classes=classes, confusion_matrix=cm.tolist(), report=report,
            confusion_png=str(cm_png))

        metrics_path = reports_dir / "cv_metrics.json"
        metrics_path.write_text(
            json.dumps(cv.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        cv.metrics_path = str(metrics_path)
        return cv

    @staticmethod
    def _plot_confusion(cm: np.ndarray, classes: list[str], out_png: Path,
                        titulo: str = "Matriz de confusión (validación)") -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2,
                                        1.1 * len(classes) + 2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        ax.set_title(titulo)
        umbral = cm.max() / 2.0 if cm.max() else 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > umbral else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return out_png
