"""Tests del RestStateDetector: segmentación por reposo, descartes."""
import numpy as np

from lsch_mr import config
from lsch_mr.rest_state_detector import RestStateDetector
from lsch_mr.tipos import SignEventType


def _mano_base():
    pts = np.zeros((config.NUM_LANDMARKS, 3), dtype=np.float32)
    # L9 a distancia 1 de L0 -> escala estable.
    pts[config.MIDDLE_MCP_IDX] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for i in range(1, config.NUM_LANDMARKS):
        if i != config.MIDDLE_MCP_IDX:
            pts[i] = np.array([0.01 * i, 0.02 * i, 0.0], dtype=np.float32)
    return pts


def _frame_movido(desplaz):
    return _mano_base() + np.array([desplaz, 0.0, 0.0], dtype=np.float32)


def test_reposo_no_emite_sena():
    det = RestStateDetector()
    base = _mano_base()
    tipos = [det.update(base).type for _ in range(30)]
    assert all(t == SignEventType.IDLE for t in tipos)


def test_ciclo_completo_emite_end():
    det = RestStateDetector()
    eventos = []
    # 10 frames en reposo
    for _ in range(10):
        eventos.append(det.update(_mano_base()))
    # 15 frames de movimiento amplio (sobre el umbral)
    for i in range(15):
        eventos.append(det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1))))
    # 12 frames de reposo -> debe cerrar la seña
    for _ in range(12):
        eventos.append(det.update(_mano_base()))

    tipos = [e.type for e in eventos]
    assert SignEventType.START in tipos
    ends = [e for e in eventos if e.type == SignEventType.END]
    assert len(ends) == 1
    assert ends[0].sequence is not None
    assert ends[0].sequence.shape[1:] == (config.NUM_LANDMARKS, config.NUM_EJES)
    assert len(ends[0].sequence) >= config.REST_MIN_FRAMES_SENA


def test_perdida_tracking_descarta():
    det = RestStateDetector()
    for _ in range(5):
        det.update(_mano_base())
    for i in range(10):  # entra en captura
        det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1)))
    assert det.capturando
    ev = det.update(None)  # pérdida de tracking
    assert ev.type == SignEventType.DISCARDED
    assert det.state == "reposo"


def test_movimiento_espurio_no_inicia():
    det = RestStateDetector()
    # Un único frame movido rodeado de reposo no debe confirmar inicio.
    det.update(_mano_base())
    ev = det.update(_frame_movido(0.3))
    det.update(_mano_base())
    # con frames_inicio>1, un solo frame no dispara START
    assert ev.type in (SignEventType.IDLE,)
