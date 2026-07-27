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
    # "estratificado" reparte las muestras al azar; "por señante" mantiene a cada
    # señante entero dentro de un solo fold. Ver evaluar_cv().
    esquema: str = "estratificado"
    n_grupos: int = 0
    # Predicciones out-of-fold y su confianza, muestra a muestra. Se guardan
    # para poder calibrar CONF_THRESHOLD sin volver a entrenar.
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    confianzas: list[float] = field(default_factory=list)
    # Trazabilidad: de qué corrida salió esta cifra. Ver _metadatos_corrida().
    corrida: dict = field(default_factory=dict)

    def resumen(self) -> str:
        return (f"{self.mean_accuracy:.3f} ± {self.std_accuracy:.3f} "
                f"({self.n_splits}-fold {self.esquema})")

    def to_dict(self) -> dict:
        return {"n_splits": self.n_splits,
                "esquema": self.esquema,
                "n_grupos": self.n_grupos,
                # Primero, para que se vea al abrir el archivo: sin saber de qué
                # corrida salió, una cifra con ±0.01 de ruido no es rastreable.
                "corrida": self.corrida,
                # Se guardan para que el JSON se baste solo: `evaluar_modelo.py
                # --fuente cv` reconstruye el reporte sin tener que adivinar el
                # orden de las clases desde labels.json (que puede haber sido
                # sobrescrito por un entrenamiento posterior con otro corpus).
                "classes": self.classes,
                "fold_accuracies": self.fold_accuracies,
                "mean_accuracy": self.mean_accuracy,
                "std_accuracy": self.std_accuracy,
                "architecture": self.architecture,
                "confusion_matrix": self.confusion_matrix,
                "report": self.report,
                "out_of_fold": {"y_true": self.y_true, "y_pred": self.y_pred,
                                "confianzas": self.confianzas}}


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

    @staticmethod
    def cargar_grupos(npz_path: Path, patron: str) -> Optional[np.ndarray]:
        """Extrae el identificador de grupo (señante) de cada `sample_id`.

        `patron` admite dos formas:
          * **número de campo** (p. ej. `"2"`): toma el N-ésimo campo separado
            por `_` del `sample_id`, contando desde 1. Para nombres tipo
            `<clase>_<señante>_<repetición>` el señante es el campo 2.
          * **expresión regular** con UN grupo de captura, para convenciones de
            nombre que no sean campos separados por `_`.

        El número de campo evita tener que escribir una regex en la línea de
        comandos, donde las comillas se comportan distinto en cada shell.
        Devuelve None si el dataset no guardó `sample_ids`.
        """
        import re

        data = np.load(npz_path, allow_pickle=True)
        if "sample_ids" not in data:
            return None

        sids = [str(s) for s in data["sample_ids"]]
        grupos = []

        if patron.strip().isdigit():
            campo = int(patron.strip())
            for sid in sids:
                partes = sid.split("_")
                if len(partes) < campo:
                    raise ValueError(
                        f"El sample_id {sid!r} no tiene {campo} campos separados "
                        "por '_'. Revisa la convención de nombres del corpus.")
                grupos.append(partes[campo - 1])
        else:
            rx = re.compile(patron)
            for sid in sids:
                m = rx.search(sid)
                if not m:
                    raise ValueError(
                        f"El patrón {patron!r} no casa con el sample_id {sid!r}. "
                        "Revisa la convención de nombres del corpus.")
                grupos.append(m.group(1))

        return np.array(grupos, dtype=object)

    # -- Guardas del agrupamiento por señante -------------------------------- #
    # Pasar el número de campo equivocado a --cv-grupos NO da error por sí solo:
    # agrupa por otra cosa (la clase, el índice de repetición) y la validación
    # cruzada vuelve a dejar al mismo señante a ambos lados del fold. La métrica
    # principal del proyecto queda inflada y nada falla de forma visible. Estas
    # guardas existen para convertir ese fallo silencioso en un error ruidoso.

    # Por encima de esto es más probable que el campo enumere muestras que
    # personas. Se puede saltar con --cv-grupos-forzar.
    GRUPOS_MAX_PLAUSIBLE = 50

    @staticmethod
    def cargar_sample_ids(npz_path: Path) -> Optional[list[str]]:
        """Lista de `sample_id` del dataset, o None si el `.npz` no los guarda."""
        data = np.load(npz_path, allow_pickle=True)
        if "sample_ids" not in data:
            return None
        return [str(s) for s in data["sample_ids"]]

    @staticmethod
    def _perfil_agrupamiento(grupos: np.ndarray, y: np.ndarray) -> dict:
        """Forma del agrupamiento: cuántos grupos, de qué tamaño y qué clases cubren.

        `clases_max == 1` significa que cada grupo contiene una sola clase, o sea
        que se está agrupando por la etiqueta y no por el señante.
        """
        grupos = np.asarray(grupos)
        y = np.asarray(y, dtype="int64")
        valores, inversa = np.unique(grupos, return_inverse=True)
        inversa = np.asarray(inversa).ravel()
        tam = np.bincount(inversa, minlength=len(valores))
        clases = np.array([len(np.unique(y[inversa == i])) for i in range(len(valores))])
        return {"n_grupos": int(len(valores)),
                "ejemplos": [str(v) for v in valores[:3]],
                "tam_min": int(tam.min()), "tam_max": int(tam.max()),
                "clases_min": int(clases.min()), "clases_max": int(clases.max()),
                "clases_mediana": float(np.median(clases))}

    @staticmethod
    def _es_identidad_plausible(perfil: dict, n_muestras: int, n_clases: int) -> bool:
        """¿Puede este agrupamiento describir a *personas* que grabaron el corpus?

        El criterio que hace el trabajo es el último: un señante graba el
        vocabulario entero, así que su grupo cubre buena parte de las clases. Un
        contador de tomas (`s01_0001`) produce grupos de una o dos muestras que
        no cubren nada, y así queda descartado como candidato.
        """
        if not 2 <= perfil["n_grupos"] <= ModelTrainer.GRUPOS_MAX_PLAUSIBLE:
            return False
        if perfil["n_grupos"] >= n_muestras:
            return False
        if n_clases > 1 and perfil["clases_max"] == 1:
            return False
        return n_clases <= 1 or perfil["clases_mediana"] >= max(2, n_clases / 2)

    @staticmethod
    def verificar_grupos(sample_ids: list[str], grupos: np.ndarray, y: np.ndarray,
                         patron: str, *, forzar: bool = False,
                         imprimir: bool = True) -> dict:
        """Diagnostica el agrupamiento y aborta si es implausible.

        Imprime **siempre** con qué se agrupó, cuántos grupos salieron, tres
        identificadores de ejemplo y el tamaño mín./máx. de grupo: sin eso, un
        `--cv-grupos` mal puesto no deja ninguna huella en la salida.

        Aborta con `SystemExit` en tres casos que son estructuralmente
        imposibles —un solo grupo, un grupo por muestra, o cada grupo con una
        sola clase— y en dos heurísticos que `--cv-grupos-forzar` desactiva:
        demasiados grupos, y que otro campo del `sample_id` sea mejor candidato.

        Devuelve el perfil del agrupamiento elegido.
        """
        grupos = np.asarray(grupos)
        y = np.asarray(y, dtype="int64")
        n_muestras = int(len(grupos))
        n_clases = int(len(np.unique(y)))
        perfil = ModelTrainer._perfil_agrupamiento(grupos, y)

        campo = int(patron.strip()) if patron.strip().isdigit() else None
        origen = (f"campo {campo} del sample_id" if campo
                  else f"regex {patron!r} sobre el sample_id")
        if imprimir:
            print(f"[grupos] {origen} -> {perfil['n_grupos']} grupos")
            print("[grupos] ejemplos: " +
                  ", ".join(repr(e) for e in perfil["ejemplos"]))
            print(f"[grupos] muestras por grupo: min={perfil['tam_min']}  "
                  f"max={perfil['tam_max']}  ({n_muestras} en total)")

        pista = ("Revisa el número de campo: en LSA64 (017_001_001 = "
                 "clase_senante_repeticion) el señante es el campo 2; en un corpus "
                 "de grabar_corpus.py (s01_0001) es el campo 1.")

        if perfil["n_grupos"] < 2:
            raise SystemExit(
                f"\nERROR: {origen} produce un solo grupo, así que no queda "
                "ningún señante fuera y la validación cruzada por señante no "
                f"mide nada.\n{pista}")

        if perfil["n_grupos"] >= n_muestras:
            raise SystemExit(
                f"\nERROR: {origen} produce {perfil['n_grupos']} grupos para "
                f"{n_muestras} muestras, o sea uno por muestra. Eso agrupa por "
                "el identificador de la toma, no por la persona, y equivale a no "
                f"agrupar en absoluto.\n{pista}")

        if n_clases > 1 and perfil["clases_max"] == 1:
            raise SystemExit(
                f"\nERROR: {origen} produce {perfil['n_grupos']} grupos y cada "
                "uno contiene una sola clase: se está agrupando por la GLOSA, no "
                "por el señante. Cada fold entrenaría sin haber visto nunca las "
                f"clases que luego evalúa.\n{pista}")

        if perfil["n_grupos"] > ModelTrainer.GRUPOS_MAX_PLAUSIBLE and not forzar:
            raise SystemExit(
                f"\nERROR: {origen} produce {perfil['n_grupos']} grupos "
                f"(máximo plausible {ModelTrainer.GRUPOS_MAX_PLAUSIBLE}). Un "
                "corpus con tantos señantes es improbable; lo normal es que el "
                f"campo enumere muestras.\n{pista}\n"
                "Si el corpus de verdad tiene tantos señantes: --cv-grupos-forzar.")

        # -- Heurística del campo dominado ----------------------------------- #
        # Solo aplica al elegir por número de campo. En un corpus con cruce
        # completo (clase x señante x repetición) los campos "señante" y
        # "repetición" son indistinguibles por estructura: los dos parten el
        # corpus en grupos que cubren todas las clases. Lo único que los
        # distingue es la cardinalidad, y el señante es siempre el más fino de
        # los dos —hay más personas que repeticiones por persona—. Así que si
        # otro campo plausible produce MÁS grupos, el elegido casi seguro es el
        # contador de repeticiones.
        if campo is not None and not forzar:
            tabla = ModelTrainer._campos_candidatos(sample_ids, y, n_clases)
            mejor = max((c for c in tabla
                         if c["plausible"] and c["n_grupos"] > perfil["n_grupos"]),
                        key=lambda c: c["n_grupos"], default=None)
            if mejor is not None:
                filas = []
                for c in tabla:
                    nota = ""
                    if c["campo"] == campo:
                        nota = "  (elegido)"
                    elif c["campo"] == mejor["campo"]:
                        nota = "  <- candidato más fino"
                    elif not c["plausible"]:
                        nota = "  (no describe personas)"
                    filas.append(f"  campo {c['campo']} -> {c['n_grupos']:>3} "
                                 f"grupos{nota}")
                raise SystemExit(
                    f"\nERROR: el campo {campo} produce {perfil['n_grupos']} "
                    f"grupos, pero el campo {mejor['campo']} produce "
                    f"{mejor['n_grupos']}, también plausibles. Un campo con menos "
                    "grupos suele ser un contador de repeticiones, no una "
                    "identidad: agrupar por él infla la métrica en silencio "
                    "porque el mismo señante sigue a ambos lados del fold.\n"
                    + "\n".join(filas) +
                    f"\nUsa --cv-grupos {mejor['campo']}, o --cv-grupos-forzar si "
                    f"de verdad el señante es el campo {campo}.")

        return perfil

    @staticmethod
    def _campos_candidatos(sample_ids: list[str], y: np.ndarray,
                           n_clases: int) -> list[dict]:
        """Perfil de agrupar por cada campo del `sample_id`, con su plausibilidad."""
        sids = [str(s) for s in sample_ids]
        if not sids:
            return []
        n_campos = min(len(s.split("_")) for s in sids)
        tabla = []
        for campo in range(1, n_campos + 1):
            perfil = ModelTrainer._perfil_agrupamiento(
                np.array([s.split("_")[campo - 1] for s in sids], dtype=object), y)
            perfil["campo"] = campo
            perfil["plausible"] = ModelTrainer._es_identidad_plausible(
                perfil, len(sids), n_clases)
            tabla.append(perfil)
        return tabla

    # -- Trazabilidad de la corrida ------------------------------------------ #
    @staticmethod
    def _git_estado() -> dict:
        """SHA del commit y si el árbol tenía cambios sin commitear.

        Sin el SHA, una cifra del informe no se puede atar al código que la
        produjo. El flag `sucio` importa tanto como el SHA: una corrida hecha
        sobre cambios sin commitear no es reproducible desde el repositorio.
        """
        import subprocess
        try:
            def _git(*args):
                return subprocess.run(("git",) + args, cwd=config.RAIZ,
                                      capture_output=True, text=True,
                                      timeout=10).stdout.strip()
            sha = _git("rev-parse", "HEAD")
            if not sha:
                return {"commit": None, "sucio": None}
            return {"commit": sha,
                    "commit_corto": sha[:12],
                    "sucio": bool(_git("status", "--porcelain"))}
        except Exception:
            # Un repo sin git no debe impedir entrenar.
            return {"commit": None, "sucio": None}

    @staticmethod
    def _versiones() -> dict:
        """Versiones de las librerías que pueden mover la cifra."""
        import platform
        from importlib.metadata import PackageNotFoundError, version

        v = {"python": platform.python_version()}
        for paquete in ("numpy", "tensorflow", "keras", "scikit-learn"):
            try:
                v[paquete] = version(paquete)
            except PackageNotFoundError:
                v[paquete] = None
        return v

    def _metadatos_corrida(self, X: np.ndarray, y: np.ndarray,
                           etiqueta: str, esquema: str, n_splits: int,
                           n_grupos: int, splits: list[dict]) -> dict:
        """Todo lo que hace falta para atar esta cifra a una corrida concreta.

        El entrenamiento de Keras no es bit-determinista aunque se fije la
        semilla: con los mismos splits, la misma configuración ha dado 0.903,
        0.911, 0.915 y 0.920. El objetivo no es eliminar ese ruido —no se
        puede— sino que cada número del informe se pueda rastrear hasta la
        corrida que lo produjo, con su código, sus datos y sus splits.
        """
        import hashlib
        from datetime import datetime, timezone

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": self._git_estado(),
            "versiones": self._versiones(),
            "modo_manos": etiqueta or None,
            "esquema": esquema,
            "n_splits": n_splits,
            "n_grupos": n_grupos,
            "semilla": self.seed,
            "hiperparametros": {"epochs": self.epochs,
                                "batch_size": self.batch_size,
                                "lr": self.lr,
                                "arquitectura": config.ARQUITECTURA},
            "dataset": {
                "n_muestras": int(X.shape[0]),
                "seq_len": int(X.shape[1]),
                "n_features": int(X.shape[2]),
                "n_clases": int(len(np.unique(y))),
                # Identifica los datos exactos sin guardarlos: dos corridas con
                # el mismo sha1 vieron el mismo dataset, aunque el archivo se
                # haya renombrado o regenerado.
                "sha1_X": hashlib.sha1(
                    np.ascontiguousarray(X, dtype="float32").tobytes()).hexdigest(),
            },
            # Los splits son lo que permite repetir la evaluación sobre el mismo
            # reparto y separar el ruido de entrenamiento del ruido de partición.
            "splits": splits,
        }

    @staticmethod
    def split_indices(y: np.ndarray,
                      n_clases: int,
                      val_split: float = config.ENTRENAMIENTO_VAL_SPLIT,
                      seed: int = config.SEMILLA) -> tuple[np.ndarray, np.ndarray]:
        """Índices `(entrenamiento, validación)` del split 80/20 de `train()`.

        Existe para que **`train()` y `evaluar_modelo.py` usen literalmente el
        mismo reparto**. El script de evaluación tiene que medir el modelo
        exportado sobre las muestras que ese modelo no vio; si reconstruyera el
        split por su cuenta y algún parámetro se desincronizara (la semilla, el
        porcentaje, el criterio de estratificación), evaluaría sobre datos de
        entrenamiento y devolvería una accuracy inflada **sin dar ningún error**.
        Una sola fuente de verdad hace imposible esa divergencia.

        Devuelve índices, no arrays: `train_test_split` aplica el mismo reparto a
        todos los arrays que recibe, así que partir los índices y luego indexar
        da exactamente la misma partición que partir X e y directamente.
        """
        from sklearn.model_selection import train_test_split

        y = np.asarray(y, dtype="int64")
        # Estratificado solo si cada clase tiene >=2 muestras (una clase con 1
        # muestra no se puede repartir entre los dos lados).
        estratifica = bool(len(y)) and bool(
            np.all(np.bincount(y, minlength=n_clases) >= 2))
        idx = np.arange(len(y))
        idx_tr, idx_va = train_test_split(
            idx, test_size=val_split, random_state=seed,
            stratify=y if estratifica else None)
        return idx_tr, idx_va

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

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="int64")
        n_classes = len(classes)

        # El reparto vive en `split_indices` para que `evaluar_modelo.py` pueda
        # reproducir EXACTAMENTE este 20% de validación (ver su docstring).
        idx_tr, idx_va = self.split_indices(y, n_classes, self.val_split, self.seed)
        Xtr, Xva = X[idx_tr], X[idx_va]
        ytr, yva = y[idx_tr], y[idx_va]

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
                   verbose: int = 0,
                   groups: Optional[np.ndarray] = None,
                   etiqueta: str = "") -> CVMetrics:
        """Validación cruzada k-fold sobre todo el corpus.

        Entrena un modelo por fold (misma arquitectura y receta que `train()`)
        y evalúa sobre el fold retenido. Devuelve accuracy media ± desviación
        estándar y una matriz de confusión out-of-fold: cada muestra del corpus
        es predicha exactamente una vez, por un modelo que no la vio en
        entrenamiento.

        `groups` (opcional) asigna cada muestra a un señante. Si se pasa, los
        folds se construyen de modo que **ningún señante aparezca a la vez en
        entrenamiento y en validación**. Importa: con varias repeticiones por
        señante, un reparto al azar deja al mismo señante en ambos lados y el
        modelo puede apoyarse en idiosincrasias suyas en vez de en la seña,
        inflando la cifra. La evaluación por señante mide lo que interesa —
        generalizar a una persona nueva— y es la que debe reportarse cuando el
        corpus tiene más de un señante.

        No guarda ningún modelo — el modelo exportable lo produce `train()`.
        """
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

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

        esquema, n_grupos = "estratificado", 0
        if groups is not None:
            groups = np.asarray(groups)
            n_grupos = int(len(np.unique(groups)))
            if n_grupos < 2:
                raise ValueError(
                    f"Se pidió validación por señante pero solo hay {n_grupos} "
                    "señante(s) en el corpus. Con un único señante no se puede "
                    "medir generalización a personas nuevas.")
            if n_grupos < n_splits:
                print(f"[cv] AVISO: hay {n_grupos} señantes; se reduce k de "
                      f"{n_splits} a {n_grupos}.")
                n_splits = n_grupos
            esquema = "por señante"
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=self.seed)
            print(f"[cv] esquema: dejando señantes fuera "
                  f"({n_grupos} señantes, {n_splits} folds).")
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=self.seed)

        fold_accs: list[float] = []
        y_pred_oof = np.zeros_like(y)   # predicción out-of-fold de cada muestra
        # Confianza de la predicción out-of-fold. Es lo que permite calibrar
        # CONF_THRESHOLD sobre datos que el modelo no vio: con las confianzas
        # del propio conjunto de entrenamiento el umbral sale siempre optimista.
        conf_oof = np.zeros(len(y), dtype="float32")
        splits: list[dict] = []

        for k, (idx_tr, idx_va) in enumerate(skf.split(X, y, groups), start=1):
            # Qué se dejó fuera en este fold. Con `groups`, los identificadores
            # de señante son más legibles que los índices y son lo que hay que
            # mirar para confirmar que el fold es realmente independiente.
            splits.append({
                "fold": k,
                "n_train": int(len(idx_tr)),
                "n_val": int(len(idx_va)),
                "grupos_fuera": (sorted({str(g) for g in np.asarray(groups)[idx_va]})
                                 if groups is not None else []),
                "val_indices": [int(i) for i in idx_va],
            })

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

            probs = modelo.predict(X[idx_va], verbose=0)
            pred = np.argmax(probs, axis=1)
            y_pred_oof[idx_va] = pred
            conf_oof[idx_va] = np.max(probs, axis=1)
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

        # Sufijo distinto por esquema Y por modo de manos: los cuatro resultados
        # (2 modos x 2 esquemas) son complementarios y no deben pisarse entre sí.
        #
        # Antes el sufijo solo distinguía el esquema, así que la corrida de
        # `ambas` sobrescribía la de `dominante` y había que renombrar los JSON a
        # mano. Eso es lo que dejó cifras huérfanas en el informe: la tabla de
        # ESTADO_ACTUAL.md citaba un 0.903 cuyo JSON ya no existía en disco.
        # Con el modo en el nombre, cada corrida tiene su archivo por
        # construcción y la tabla siempre se puede rastrear hasta él.
        partes = [p for p in (etiqueta.strip(),
                              "senante" if groups is not None else "") if p]
        sufijo = ("_" + "_".join(partes)) if partes else ""
        reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
        cm_png = self._plot_confusion(
            cm, classes, reports_dir / f"matriz_confusion_cv{sufijo}.png",
            titulo=f"Matriz de confusión ({n_splits}-fold {esquema}, out-of-fold)")

        cv = CVMetrics(
            n_splits=n_splits, fold_accuracies=fold_accs,
            mean_accuracy=float(np.mean(fold_accs)),
            std_accuracy=float(np.std(fold_accs)),
            classes=classes, confusion_matrix=cm.tolist(), report=report,
            confusion_png=str(cm_png), esquema=esquema, n_grupos=n_grupos,
            y_true=y.tolist(), y_pred=y_pred_oof.tolist(),
            confianzas=[float(c) for c in conf_oof],
            corrida=self._metadatos_corrida(X, y, etiqueta, esquema, n_splits,
                                            n_grupos, splits))

        metrics_path = reports_dir / f"cv_metrics{sufijo}.json"
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
