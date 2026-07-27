"""
MessageComposer — Capa 4 (Presentación).

Concatena glosas sucesivas en un mensaje, respetando `maxWords` y
`continuityTimeout` (Sección 10.2 y máquina de estados del buffer, Sección 8).

En el sistema final el render espacial lo hace `SpatialSubtitleRenderer` en
Unity (a cargo de Tomás). Aquí se incluye MessageComposer para poder validar
en PC el flujo de CU-03 (composición por concatenación) sin la capa Unity.
"""
from __future__ import annotations

import time
from typing import Callable, Optional


class MessageComposer:
    """Acumula glosas reconocidas en un mensaje (CU-03, Sección 10.2).

    Tres formas de cerrar el mensaje, todas del diseño:
      * llegar a `max_words` glosas;
      * que venzan `continuity_timeout` segundos sin una glosa nueva — lo
        comprueban tanto `appendWord()` como `tick()`;
      * `reiniciar()`, el gesto de reinicio manual (FA-01 de CU-03).

    El reloj se inyecta (`reloj`) para poder testear el timeout sin esperar en
    tiempo real.
    """

    def __init__(self, max_words: int = 8, continuity_timeout: float = 4.0,
                 reloj: Callable[[], float] = time.monotonic) -> None:
        self.max_words = max_words
        self.continuity_timeout = continuity_timeout
        self._reloj = reloj
        self._buffer: list[str] = []
        self._ultimo_ts: Optional[float] = None
        self.state = "vacio"

    # -- Contrato del diseño ------------------------------------------------ #
    def appendWord(self, label: str) -> None:
        """Añade una glosa al mensaje en curso.

        Antes de añadirla comprueba el timeout de continuidad: si pasó
        demasiado tiempo desde la anterior, el mensaje viejo se despliega y
        esta glosa abre uno nuevo, en vez de pegarse a una frase que el usuario
        ya dio por terminada.
        """
        ahora = self._reloj()
        # Timeout de continuidad: vencido -> se despliega y se reinicia.
        if (self._ultimo_ts is not None and
                ahora - self._ultimo_ts > self.continuity_timeout):
            self._desplegar()
        self._buffer.append(label)
        self._ultimo_ts = ahora
        self.state = "acumulando"
        if len(self._buffer) >= self.max_words:
            self._desplegar()

    def tick(self) -> bool:
        """Cierra el mensaje si el timeout de continuidad ya venció.

        `appendWord()` también comprueba el timeout, pero solo cuando llega la
        siguiente palabra: sin `tick()`, un mensaje queda colgado en pantalla
        indefinidamente si el usuario deja de señar. Llamar a `tick()` una vez
        por frame hace que el cierre por expiración ocurra en el momento que
        describe el diseño ("al expirar el timer, cierra el mensaje"), sin
        necesidad de un hilo ni un temporizador activo.

        Devuelve True si el mensaje se cerró en esta llamada.
        """
        if self._ultimo_ts is None:
            return False
        if self._reloj() - self._ultimo_ts <= self.continuity_timeout:
            return False
        self._desplegar()
        return True

    def texto(self) -> str:
        """El mensaje acumulado hasta ahora, para dibujar como subtítulo."""
        return " ".join(self._buffer)

    def reiniciar(self) -> None:
        """Gesto de reinicio manual: vuelve a 'Vacío' desde cualquier estado."""
        self._buffer.clear()
        self._ultimo_ts = None
        self.state = "vacio"

    def _desplegar(self) -> None:
        self.state = "desplegado"
        # Nota: en Unity, aquí se invocaría SpatialSubtitleRenderer.render().
        self._buffer.clear()
        self._ultimo_ts = None
