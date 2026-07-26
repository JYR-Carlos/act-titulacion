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
