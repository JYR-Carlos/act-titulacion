"""
ModelTrainer — Pipeline offline (CONTEXTO_PROYECTO.md Sección 10.2 / CU-02).

Entrena el TCN sobre el dataset normalizado y produce:
  * accuracy (train/val)
  * matriz de confusión (PNG) + classification report
  * modelo Keras entrenado (.keras) + mapa de etiquetas (labels.json)
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
class Metrics:
    val_accuracy: float
    train_accuracy: float
    classes: list[str]
    confusion_matrix: list[list[int]]
    report: dict = field(default_factory=dict)
    keras_path: str = ""
    labels_path: str = ""
    confusion_png: str = ""


class ModelTrainer:
    def __init__(self,
                 epochs: int = config.ENTRENAMIENTO_EPOCHS,
                 batch_size: int = config.ENTRENAMIENTO_BATCH,
                 lr: float = config.ENTRENAMIENTO_LR,
                 val_split: float = config.ENTRENAMIENTO_VAL_SPLIT,
                 seed: int = config.SEMILLA) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_split = val_split
        self.seed = seed
        self.model = None

    @staticmethod
    def cargar_dataset(npz_path: Path):
        """Carga X (N, seq_len, F), y (N,) y la lista de clases desde un .npz."""
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype("float32")
        y = data["y"].astype("int64")
        classes = [str(c) for c in data["classes"]]
        return X, y, classes

    # -- Contrato del diseño ------------------------------------------------ #
    def train(self, X: np.ndarray, y: np.ndarray,
              classes: list[str],
              output_dir: Path = config.OUTPUTS_MODELS_DIR,
              reports_dir: Path = config.OUTPUTS_REPORTS_DIR) -> Metrics:
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

        self.model = build_tcn(seq_len=X.shape[1], n_features=X.shape[2],
                               n_classes=n_classes)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=20,
                restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
        ]
        hist = self.model.fit(
            Xtr, ytr, validation_data=(Xva, yva),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=callbacks, verbose=2)

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

        cm_png = self._plot_confusion(cm, classes, reports_dir / "matriz_confusion.png")
        (reports_dir / "metrics.json").write_text(
            json.dumps({"val_accuracy": val_acc, "train_accuracy": train_acc,
                        "report": report}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        return Metrics(
            val_accuracy=val_acc, train_accuracy=train_acc, classes=classes,
            confusion_matrix=cm.tolist(), report=report,
            keras_path=str(keras_path), labels_path=str(labels_path),
            confusion_png=str(cm_png))

    @staticmethod
    def _plot_confusion(cm: np.ndarray, classes: list[str], out_png: Path) -> Path:
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
        ax.set_title("Matriz de confusión (validación)")
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
