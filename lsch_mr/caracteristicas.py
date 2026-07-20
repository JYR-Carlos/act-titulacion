"""
Construcción de características para el clasificador.

Convierte una SignSequence cruda en el tensor de entrada de forma fija que
espera el TCN:  (SEQ_LEN, n_features).

Pasos:
  1) Normalización geométrica por frame/mano con KeypointNormalizer
     (centrado L0 + escala L0–L9).  Cada mano -> NormVector (63).
  2) Ensamblado según el modo de manos (punto abierto, Sección 6):
        "dominante" -> (T, 63)
        "ambas"     -> (T, 126)  [Left | Right]; mano ausente -> ceros
  3) Remuestreo temporal a SEQ_LEN frames (interpolación lineal en el tiempo),
     dando una longitud fija (Sentis prefiere formas estáticas) e independiente
     de la duración de la seña.
"""
from __future__ import annotations

import numpy as np

from . import config
from .keypoint_normalizer import KeypointNormalizer

_norm = KeypointNormalizer()


def _norm_o_ceros(hand_pts: np.ndarray) -> np.ndarray:
    """Normaliza una mano (21,3); si viene ausente (NaN) devuelve ceros(63)."""
    if hand_pts is None or not np.all(np.isfinite(hand_pts)):
        return np.zeros(config.NORMVECTOR_DIM, dtype=np.float32)
    return _norm.normalize(hand_pts)


def secuencia_a_features(sequence: np.ndarray, modo: str = config.MODO_MANOS
                         ) -> np.ndarray:
    """SignSequence cruda -> matriz (T, n_features) normalizada.

    Acepta:
      * modo "dominante": `sequence` con forma (T, 21, 3).
      * modo "ambas":     `sequence` con forma (T, 2, 21, 3) [Left, Right].
    """
    seq = np.asarray(sequence, dtype=np.float32)

    if modo == "ambas":
        if seq.ndim != 4 or seq.shape[1:] != (2, config.NUM_LANDMARKS, config.NUM_EJES):
            raise ValueError(
                f"Modo 'ambas' espera (T, 2, 21, 3); se recibió {seq.shape}.")
        filas = [
            np.concatenate([_norm_o_ceros(fr[0]), _norm_o_ceros(fr[1])])
            for fr in seq
        ]
    else:
        if seq.ndim != 3 or seq.shape[1:] != (config.NUM_LANDMARKS, config.NUM_EJES):
            raise ValueError(
                f"Modo 'dominante' espera (T, 21, 3); se recibió {seq.shape}.")
        filas = [_norm.normalize(fr) for fr in seq]

    return np.stack(filas, axis=0).astype(np.float32)


def remuestrear_tiempo(features: np.ndarray, seq_len: int = config.SEQ_LEN
                       ) -> np.ndarray:
    """Remuestrea (T, F) -> (seq_len, F) por interpolación lineal en el tiempo."""
    feats = np.asarray(features, dtype=np.float32)
    T, F = feats.shape
    if T == seq_len:
        return feats
    if T == 1:
        return np.repeat(feats, seq_len, axis=0)
    x_orig = np.linspace(0.0, 1.0, num=T)
    x_new = np.linspace(0.0, 1.0, num=seq_len)
    out = np.empty((seq_len, F), dtype=np.float32)
    for j in range(F):
        out[:, j] = np.interp(x_new, x_orig, feats[:, j])
    return out


def preparar_entrada(sequence: np.ndarray, modo: str = config.MODO_MANOS,
                     seq_len: int = config.SEQ_LEN) -> np.ndarray:
    """SignSequence cruda -> entrada lista para el modelo (seq_len, n_features)."""
    return remuestrear_tiempo(secuencia_a_features(sequence, modo), seq_len)
