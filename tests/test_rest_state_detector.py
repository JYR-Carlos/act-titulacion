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


def test_parpadeo_breve_no_descarta():
    """running_mode="image" redetecta cada frame: un parpadeo de la mano de
    hasta REST_FRAMES_PERDIDA_MAX frames es variación normal, no pérdida real
    de tracking, y la captura en curso no debe tirarse por eso."""
    det = RestStateDetector()
    for _ in range(5):
        det.update(_mano_base())
    for i in range(10):  # entra en captura
        det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1)))
    assert det.capturando
    for _ in range(config.REST_FRAMES_PERDIDA_MAX):  # parpadeo tolerado
        ev = det.update(None)
        assert ev.type == SignEventType.CAPTURING
        assert det.capturando
    # Vuelve la mano: la captura sigue viva y puede cerrar con END normalmente.
    eventos = [det.update(_mano_base()) for _ in range(12)]
    ends = [e for e in eventos if e.type == SignEventType.END]
    assert len(ends) == 1
    assert ends[0].sequence is not None


def test_perdida_tracking_sostenida_descarta():
    det = RestStateDetector()
    for _ in range(5):
        det.update(_mano_base())
    for i in range(10):  # entra en captura
        det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1)))
    assert det.capturando
    ev = None
    for _ in range(config.REST_FRAMES_PERDIDA_MAX + 1):  # supera la tolerancia
        ev = det.update(None)
    assert ev.type == SignEventType.DISCARDED
    assert det.state == "reposo"


def test_parpadeo_en_reposo_no_mata_el_candidato_de_inicio():
    """Regresión de la sesión "noche" del 2026-07-26: con la mano detectada en
    solo la mitad de los frames, la demo pasó 104 s sin segmentar UNA sola seña
    ni reportar un descarte. Un hueco corto mientras se acumula movimiento no
    puede borrar el candidato: si lo borra, con detección inestable no se
    arranca nunca."""
    det = RestStateDetector()
    det.update(_mano_base())
    eventos = []
    # Movimiento sostenido, pero con un parpadeo intercalado cada dos frames.
    for i in range(config.REST_FRAMES_INICIO + 2):
        eventos.append(det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1))))
        eventos.append(det.update(None))          # parpadeo de 1 frame
    assert SignEventType.START in [e.type for e in eventos]


def test_parpadeo_largo_en_reposo_si_borra_el_candidato():
    """La tolerancia es acotada: superarla vuelve a foja cero, o cualquier
    movimiento aislado de hace un rato podría completar un inicio."""
    det = RestStateDetector()
    det.update(_mano_base())
    det.update(_frame_movido(0.3))                # 1 frame de movimiento
    for _ in range(config.REST_FRAMES_PERDIDA_MAX + 1):
        assert det.update(None).type == SignEventType.IDLE
    # Tras el hueco largo hacen falta otra vez `frames_inicio` frames movidos.
    eventos = [det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1)))
               for i in range(config.REST_FRAMES_INICIO - 1)]
    assert SignEventType.START not in [e.type for e in eventos]


def test_vuelta_de_parpadeo_no_cuenta_como_reposo():
    """Tras un parpadeo tolerado, el primer frame con mano debe medir su
    movimiento contra la última mano vista. Antes `_prev` se soltaba en cuanto
    faltaba un frame, ese frame devolvía movimiento 0.0 y contaba como reposo,
    adelantando el cierre de la seña."""
    det = RestStateDetector()
    for _ in range(3):
        det.update(_mano_base())
    for i in range(10):
        det.update(_frame_movido(0.3 * ((i % 2) * 2 - 1)))
    assert det.capturando

    det.update(None)                              # parpadeo tolerado
    # La mano vuelve desplazada: es movimiento, no reposo, así que la seña sigue
    # abierta y aún faltan `frames_fin` frames quietos para cerrarla.
    ev = det.update(_frame_movido(0.5))
    assert ev.type == SignEventType.CAPTURING
    eventos = [det.update(_frame_movido(0.5)) for _ in range(det.frames_fin - 1)]
    assert SignEventType.END not in [e.type for e in eventos]


def test_movimiento_espurio_no_inicia():
    det = RestStateDetector()
    # Un único frame movido rodeado de reposo no debe confirmar inicio.
    det.update(_mano_base())
    ev = det.update(_frame_movido(0.3))
    det.update(_mano_base())
    # con frames_inicio>1, un solo frame no dispara START
    assert ev.type in (SignEventType.IDLE,)
