"""Tests del MessageComposer: acumulación, maxWords, timeout y reinicio manual."""
from lsch_mr.message_composer import MessageComposer


class _Reloj:
    """Reloj inyectable: el timeout se prueba sin esperar en tiempo real."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def avanzar(self, segundos: float) -> None:
        self.t += segundos


def _composer(**kw) -> tuple[MessageComposer, _Reloj]:
    reloj = _Reloj()
    return MessageComposer(reloj=reloj, **kw), reloj


def test_acumula_palabras_en_orden():
    comp, _ = _composer()
    comp.appendWord("HOLA")
    comp.appendWord("GRACIAS")
    assert comp.texto() == "HOLA GRACIAS"
    assert comp.state == "acumulando"


def test_max_words_cierra_el_mensaje():
    comp, _ = _composer(max_words=2)
    comp.appendWord("HOLA")
    comp.appendWord("GRACIAS")
    assert comp.texto() == ""            # se desplegó y se limpió el buffer
    assert comp.state == "desplegado"


def test_tick_no_cierra_antes_del_timeout():
    comp, reloj = _composer(continuity_timeout=4.0)
    comp.appendWord("HOLA")
    reloj.avanzar(3.9)
    assert comp.tick() is False
    assert comp.texto() == "HOLA"


def test_tick_cierra_al_expirar_el_timeout():
    """El diseño exige que el mensaje se cierre al expirar el timer, aunque el
    usuario deje de señar: sin tick() solo se comprobaría en el próximo append."""
    comp, reloj = _composer(continuity_timeout=4.0)
    comp.appendWord("HOLA")
    reloj.avanzar(4.1)
    assert comp.tick() is True
    assert comp.texto() == ""
    assert comp.state == "desplegado"


def test_tick_es_idempotente_con_buffer_vacio():
    comp, reloj = _composer(continuity_timeout=1.0)
    reloj.avanzar(10.0)
    assert comp.tick() is False          # nada que cerrar, no debe fallar


def test_append_tras_timeout_inicia_mensaje_nuevo():
    comp, reloj = _composer(continuity_timeout=4.0)
    comp.appendWord("HOLA")
    reloj.avanzar(5.0)
    comp.appendWord("GRACIAS")
    assert comp.texto() == "GRACIAS"     # no arrastra la palabra anterior


def test_reiniciar_limpia_desde_cualquier_estado():
    """FA-01 de CU-03: gesto/tecla de reinicio manual."""
    comp, _ = _composer()
    comp.appendWord("HOLA")
    comp.appendWord("GRACIAS")
    comp.reiniciar()
    assert comp.texto() == ""
    assert comp.state == "vacio"
