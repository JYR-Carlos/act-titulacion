"""
Arquitectura del clasificador: Temporal Convolutional Network (TCN).

Decisión consolidada (CONTEXTO_PROYECTO.md Sección 3): TCN. LSTM y Transformer
quedan descartados; NO reabrir sin decisión explícita del equipo.

Diseño de la red (justificación del receptive field):
  * Bloques residuales de convolución 1D causal y dilatada.
  * kernel=3, dilataciones (1,2,4,8), 2 convs por bloque.
  * Receptive field = 1 + 2*(k-1)*Σdilataciones = 1 + 2*2*15 = 61 ≥ 60 frames,
    es decir, la ventana temporal completa de una seña (MAX_SEQ_FRAMES=60).

Se usan solo operadores bien soportados por Unity Sentis: Conv (1D), Relu, Add,
Pad (padding causal), Dropout (se elimina en inferencia), ReduceMean
(GlobalAveragePooling), Gemm/MatMul (Dense) y Softmax.
"""
from __future__ import annotations

from . import config


def build_tcn(seq_len: int = config.SEQ_LEN,
              n_features: int | None = None,
              n_classes: int = 10,
              filtros: int = config.TCN_FILTROS,
              kernel: int = config.TCN_KERNEL,
              dilataciones=config.TCN_DILATACIONES,
              dropout: float = config.TCN_DROPOUT):
    """Construye y devuelve el modelo Keras del TCN (sin compilar)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    n_features = n_features or config.n_features()

    def bloque_residual(x, dilation, nombre):
        prev = x
        for i in range(2):  # 2 convs causales por bloque
            x = layers.Conv1D(
                filtros, kernel, padding="causal", dilation_rate=dilation,
                name=f"{nombre}_conv{i}")(x)
            x = layers.Activation("relu", name=f"{nombre}_relu{i}")(x)
            x = layers.SpatialDropout1D(dropout, name=f"{nombre}_drop{i}")(x)
        # Ajuste de canales para la conexión residual (conv 1x1 si hace falta).
        if prev.shape[-1] != filtros:
            prev = layers.Conv1D(filtros, 1, padding="same",
                                 name=f"{nombre}_res1x1")(prev)
        x = layers.Add(name=f"{nombre}_add")([prev, x])
        return layers.Activation("relu", name=f"{nombre}_out")(x)

    entrada = layers.Input(shape=(seq_len, n_features), name="input")
    x = entrada
    for i, d in enumerate(dilataciones):
        x = bloque_residual(x, d, nombre=f"tcn{i}")

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(filtros, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout, name="fc_drop")(x)
    salida = layers.Dense(n_classes, activation="softmax", name="output")(x)

    return models.Model(entrada, salida, name="tcn_lsch")
