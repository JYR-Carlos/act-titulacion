"""
DatasetRecorder — Pipeline offline (CONTEXTO_PROYECTO.md Sección 10.2 / CU-02).

Graba y etiqueta el corpus a nivel de frame. Está pensado para el protocolo de
grabación con el profesor de lengua de señas (sesiones limitadas): se graba en
TOMAS CONTINUAS y la segmentación de cada seña la hace automáticamente el
`RestStateDetector` (Sección 10 del contexto, backlog inmediato).

Flujo interactivo (ventana OpenCV):
  * Teclas 0..9  -> selecciona la glosa activa (según Glosas_LSCh_Mappeadas.csv).
  * El operador hace la seña; al volver a reposo, la secuencia se guarda sola
    con la glosa activa.
  * Tecla ESC / q -> termina la sesión.

Fuente de video: webcam local o cámara del teléfono (ver `fuente_video.py`).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .csv_esquema import EscritorCSV
from .fuente_video import FuenteVideo, FuenteSpec
from .hand_tracking_provider import HandTrackingProvider, seleccionar_frame
from .rest_state_detector import RestStateDetector
from .tipos import SignEventType


class DatasetRecorder:
    def __init__(self, glosas: list[str],
                 fuente: FuenteSpec = 0,
                 espejo: bool = True,
                 salida_dir: Path = config.DATA_RAW_DIR,
                 senante: str = config.SENANTE_POR_DEFECTO) -> None:
        self.glosas = glosas[:10]  # teclas 0..9
        self.fuente = fuente
        self.espejo = espejo
        self.salida_dir = Path(salida_dir)
        # El identificador del señante va DENTRO del sample_id porque es lo que
        # permite después evaluar dejando señantes fuera (entrenar.py
        # --cv-grupos 1). Si no se registra al grabar, esa evaluación deja de
        # ser posible y no hay forma de reconstruirlo a posteriori.
        self.senante = str(senante).strip().replace("_", "-") or config.SENANTE_POR_DEFECTO

    # -- Contrato del diseño ------------------------------------------------ #
    def recordSession(self) -> Path:
        """Ejecuta una sesión de grabación y devuelve la ruta del CSV generado."""
        import cv2

        sesion = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.salida_dir / f"corpus_{self.senante}_{sesion}.csv"
        glosa_idx = 0
        muestra_id = 0
        guardadas = {g: 0 for g in self.glosas}

        detector = RestStateDetector()
        # running_mode="image" explícito: este grabador produce corpus, y el
        # corpus existente se extrajo en IMAGE. El default de la clase es "video"
        # y grabar con él daría muestras de otra distribución que las de LSA64.
        # Cuesta algo de estabilidad en la segmentación en vivo —IMAGE redetecta
        # la palma en cada frame, así que da más jitter y más huecos— pero esa
        # inestabilidad la va a tener igual el runtime de Unity, que también va en
        # IMAGE. Grabar en un modo más benévolo que el de servicio maquillaría el
        # problema en vez de resolverlo. Ver INTEGRACION_UNITY.md sección 7.
        with FuenteVideo(self.fuente, espejo=self.espejo).abrir() as fuente, \
                HandTrackingProvider(num_hands=2, running_mode="image") as provider, \
                EscritorCSV(csv_path) as escritor:
            print(f"[DatasetRecorder] Sesión {sesion}. Guardando en {csv_path}")
            self._imprimir_ayuda()

            for frame_bgr, ts_ms in fuente.frames():
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                multi = provider.getFrame(rgb, ts_ms)
                dom = seleccionar_frame(multi, modo="dominante")  # segmenta por dominante
                evento = detector.update(dom, payload=multi)      # acumula multi-mano

                if evento.type == SignEventType.END and evento.sequence:
                    label = self.glosas[glosa_idx]
                    sample_id = f"{self.senante}_{muestra_id:04d}"
                    n = self._guardar_secuencia(escritor, sample_id,
                                                evento.sequence, label)
                    guardadas[label] += 1
                    muestra_id += 1
                    print(f"  [OK] Guardada muestra '{label}' "
                          f"({len(evento.sequence)} frames, {n} filas) - "
                          f"total {label}: {guardadas[label]}")
                elif evento.type == SignEventType.DISCARDED:
                    pass  # secuencia demasiado corta o tracking perdido

                self._overlay(cv2, frame_bgr, self.glosas[glosa_idx],
                              detector, guardadas)
                cv2.imshow("DatasetRecorder — LSCh-MR", frame_bgr)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if ord("0") <= k <= ord("9"):
                    sel = k - ord("0")
                    if sel < len(self.glosas):
                        glosa_idx = sel
                        print(f"  -> Glosa activa: {self.glosas[glosa_idx]}")

            cv2.destroyAllWindows()

        print(f"[DatasetRecorder] Fin. Muestras por glosa: {guardadas}")
        return csv_path

    # -- Utilidades --------------------------------------------------------- #
    @staticmethod
    def _guardar_secuencia(escritor: EscritorCSV, sample_id: str,
                           secuencia_multi, label: str) -> int:
        filas = 0
        for i, multi in enumerate(secuencia_multi):
            filas += escritor.escribir_multiframe(sample_id, i, multi, label)
        return filas

    def _imprimir_ayuda(self) -> None:
        print(f"  Señante de esta sesión: {self.senante}")
        print("  Teclas: [0-9] elegir glosa | [q]/[ESC] salir")
        for i, g in enumerate(self.glosas):
            print(f"    {i} = {g}")

    @staticmethod
    def _overlay(cv2, frame, glosa_activa, detector, guardadas) -> None:
        h = frame.shape[0]
        estado = "CAPTURANDO" if detector.capturando else "reposo"
        color = (0, 0, 255) if detector.capturando else (0, 180, 0)
        cv2.putText(frame, f"Glosa: {glosa_activa}  [{guardadas.get(glosa_activa,0)}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"{estado} ({detector.n_frames_buffer})",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
