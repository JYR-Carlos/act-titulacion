// MetricsRecorder.cs — MonoBehaviour que cablea la instrumentación de métricas.
//
// Junta PipelineTelemetry (métrica 2, latencia) y FrameRateMonitor (métrica 3,
// FPS) y vuelca todo a disco al terminar la sesión:
//
//     <persistentDataPath>/metricas/sesion_<fecha>_senas.csv
//     <persistentDataPath>/metricas/sesion_<fecha>_fps.csv
//     <persistentDataPath>/metricas/sesion_<fecha>_resumen.json
//
// En Quest 3, persistentDataPath es
// /storage/emulated/0/Android/data/<package>/files — se saca con `adb pull`.
//
// Colócalo en un GameObject de la escena y llama a los cuatro Mark* desde el
// controlador de la sesión de traducción. Ver README.md de esta carpeta.

using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace LSChMR.Telemetry
{
    public sealed class MetricsRecorder : MonoBehaviour
    {
        [Header("Objetivos del MVP (Sección 11)")]
        [Tooltip("72 = Meta Quest 3. En PC, pon la tasa de refresco real del monitor.")]
        public float targetFps = 72.0f;

        [Tooltip("Presupuesto de latencia end-to-end, en ms (config.LATENCIA_MAX_MS).")]
        public double latencyBudgetMs = 500.0;

        [Tooltip("config.REST_FRAMES_FIN — debe coincidir con el valor de Python.")]
        public int restFramesEnd = 6;

        [Header("Muestreo")]
        [Tooltip("Ancho de la ventana deslizante de FPS, en segundos.")]
        public double fpsWindowSeconds = 1.0;

        [Tooltip("Cada cuántos segundos se guarda una muestra de FPS.")]
        public double fpsSampleIntervalSeconds = 5.0;

        [Tooltip("Duración prevista de la sesión (s). Solo sitúa la mitad para medir degradación.")]
        public double expectedSessionSeconds = 300.0;

        [Header("Salida")]
        [Tooltip("Escribir los CSV al salir de Play Mode / cerrar la app.")]
        public bool writeOnQuit = true;

        [Tooltip("Subcarpeta dentro de persistentDataPath.")]
        public string outputFolder = "metricas";

        public PipelineTelemetry Telemetry { get; private set; }
        public FrameRateMonitor Fps { get; private set; }

        // Periodo medio entre frames de CAPTURA. No es lo mismo que el periodo de
        // render: la cámara puede ir a 30 FPS mientras la escena se dibuja a 72,
        // y el retardo de segmentación depende del ritmo de la CÁMARA. Usar el
        // FPS de render aquí subestimaría la latencia por más del doble.
        double _captureperiodMs;
        double _lastCaptureAt = -1.0;
        int _capturePeriodSamples;

        bool _written;

        void Awake()
        {
            Telemetry = new PipelineTelemetry { RestFramesEnd = restFramesEnd };
            Fps = new FrameRateMonitor
            {
                TargetFps = targetFps,
                WindowSeconds = fpsWindowSeconds,
                SampleIntervalSeconds = fpsSampleIntervalSeconds
            };
            Fps.SetExpectedDuration(expectedSessionSeconds);
        }

        void Update()
        {
            Fps.Tick(Time.unscaledDeltaTime);
        }

        // ------------------------------------------------------------------ //
        // Ganchos que llama el controlador de la sesión de traducción
        // ------------------------------------------------------------------ //

        /// <summary>Capa 1 — una vez por frame, tras obtener los keypoints.</summary>
        public void OnKeypointsCaptured()
        {
            Telemetry.MarkCapture();

            // Media móvil exponencial del periodo de captura: se adapta si la
            // cámara cambia de tasa a mitad de sesión, sin guardar historial.
            double now = Time.realtimeSinceStartupAsDouble;
            if (_lastCaptureAt >= 0.0)
            {
                double deltaMs = (now - _lastCaptureAt) * 1000.0;
                // Se ignoran los saltos absurdos (un hitch de carga, un pause):
                // contaminarían el periodo y con él el retardo de segmentación.
                if (deltaMs > 0.0 && deltaMs < 1000.0)
                {
                    _capturePeriodSamples++;
                    _captureperiodMs = _capturePeriodSamples == 1
                        ? deltaMs
                        : _captureperiodMs * 0.9 + deltaMs * 0.1;
                }
            }
            _lastCaptureAt = now;
        }

        /// <summary>Capa 1 — el RestStateDetector confirmó FIN de seña.</summary>
        public void OnSignSegmented() => Telemetry.MarkSegmentationEnd();

        /// <summary>Capa 3 — el clasificador devolvió la glosa.</summary>
        public void OnInferenceComplete() => Telemetry.MarkInferenceEnd();

        /// <summary>Capa 1 — la seña se descartó (pérdida de tracking sostenida).</summary>
        public void OnSignDiscarded() => Telemetry.AbortSign();

        /// <summary>
        /// Capa 4 — llamar JUSTO DESPUÉS de asignar el texto del subtítulo.
        /// Espera al final del frame para sellar el instante en que el subtítulo
        /// está realmente presentado (ver el comentario de MarkRenderComplete).
        /// </summary>
        public void OnSubtitleRendered(string gloss, float confidence, bool inVocabulary)
        {
            StartCoroutine(CompleteAtEndOfFrame(gloss, confidence, inVocabulary));
        }

        IEnumerator CompleteAtEndOfFrame(string gloss, float confidence, bool inVocabulary)
        {
            // WaitForEndOfFrame cede hasta después de que la cámara ha renderizado
            // y la UI se ha compuesto: es el instante más cercano a "el usuario ya
            // puede leerlo" al que se llega desde un script.
            yield return new WaitForEndOfFrame();

            var sample = Fps.CurrentSample();
            var record = Telemetry.MarkRenderComplete(
                gloss, confidence, inVocabulary,
                _captureperiodMs, sample.Average, sample.Min);

            if (record == null) yield break;

            if (record.ExceedsThreshold)
            {
                Debug.LogWarning(string.Format(CultureInfo.InvariantCulture,
                    "[LSCh-MR] '{0}' e2e {1:F0} ms EXCEDE el presupuesto de {2:F0} ms " +
                    "(segmentación {3:F0} + proceso {4:F0})",
                    record.Gloss, record.EndToEndMs, latencyBudgetMs,
                    record.SegmentationDelayMs, record.ProcessingMs));
            }
            else
            {
                Debug.Log(string.Format(CultureInfo.InvariantCulture,
                    "[LSCh-MR] '{0}' conf={1:F2} e2e {2:F0} ms | fps {3:F0} (min {4:F0})",
                    record.Gloss, record.Confidence, record.EndToEndMs,
                    record.FpsAverage, record.FpsMin));
            }
        }

        // ------------------------------------------------------------------ //
        // Persistencia
        // ------------------------------------------------------------------ //
        void OnApplicationQuit()
        {
            if (writeOnQuit) WriteReports();
        }

        void OnApplicationPause(bool paused)
        {
            // En Android/Quest, quitarse el visor o irse a la home puede matar la
            // app sin pasar por OnApplicationQuit. Volcar en la pausa evita
            // perder una sesión de 5 minutos ya medida.
            if (paused && writeOnQuit) WriteReports();
        }

        /// <summary>Escribe los tres archivos. Idempotente dentro de una sesión.</summary>
        public string WriteReports()
        {
            if (_written) return null;
            _written = true;

            string dir = Path.Combine(Application.persistentDataPath, outputFolder);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            string baseName = Path.Combine(dir, "sesion_" + stamp);

            try
            {
                Directory.CreateDirectory(dir);
                File.WriteAllText(baseName + "_senas.csv", Telemetry.ToCsv(), Encoding.UTF8);
                File.WriteAllText(baseName + "_fps.csv", Fps.SamplesToCsv(), Encoding.UTF8);
                File.WriteAllText(baseName + "_resumen.json", BuildSummaryJson(), Encoding.UTF8);
                Debug.Log("[LSCh-MR] métricas guardadas en " + dir);
                return baseName;
            }
            catch (Exception e)
            {
                // Nunca tumbar la demo por un problema de disco o de permisos.
                Debug.LogError("[LSCh-MR] no se pudieron guardar las métricas: " + e.Message);
                return null;
            }
        }

        /// <summary>
        /// Resumen en JSON. Se construye a mano en vez de con JsonUtility porque
        /// este último no serializa propiedades ni bool anidados de forma legible,
        /// y el archivo está pensado para leerse y citarse en el informe.
        /// </summary>
        public string BuildSummaryJson()
        {
            var c = CultureInfo.InvariantCulture;
            var lat = Telemetry.Summarize();
            var fps = Fps.Summarize();
            var sb = new StringBuilder();

            sb.AppendLine("{");
            sb.AppendFormat(c, "  \"sesion\": {{ \"fecha\": \"{0}\", \"duracion_s\": {1:F1}, \"dispositivo\": \"{2}\", \"unity\": \"{3}\" }},\n",
                DateTime.Now.ToString("o", c), fps.DurationS,
                Escape(SystemInfo.deviceModel), Escape(Application.unityVersion));

            sb.AppendLine("  \"metrica_2_latencia_e2e\": {");
            sb.AppendFormat(c, "    \"umbral_ms\": {0:F1},\n", lat.ThresholdMs);
            sb.AppendFormat(c, "    \"n_senas\": {0},\n", lat.Count);
            sb.AppendFormat(c, "    \"media_ms\": {0:F2},\n", lat.MeanMs);
            sb.AppendFormat(c, "    \"mediana_ms\": {0:F2},\n", lat.MedianMs);
            sb.AppendFormat(c, "    \"min_ms\": {0:F2},\n", lat.MinMs);
            sb.AppendFormat(c, "    \"max_ms\": {0:F2},\n", lat.MaxMs);
            sb.AppendFormat(c, "    \"p95_ms\": {0:F2},\n", lat.P95Ms);
            sb.AppendFormat(c, "    \"n_excede_umbral\": {0},\n", lat.ExceededCount);
            sb.AppendFormat(c, "    \"pct_excede_umbral\": {0:F1},\n", lat.ExceededPercent);
            sb.AppendFormat(c, "    \"cumple\": {0}\n", lat.Pass ? "true" : "false");
            sb.AppendLine("  },");

            sb.AppendLine("  \"metrica_3_fps\": {");
            sb.AppendFormat(c, "    \"objetivo_fps\": {0:F1},\n", fps.TargetFps);
            sb.AppendFormat(c, "    \"duracion_s\": {0:F1},\n", fps.DurationS);
            sb.AppendFormat(c, "    \"sesion_de_5_min\": {0},\n", fps.LongEnough ? "true" : "false");
            sb.AppendFormat(c, "    \"n_frames\": {0},\n", fps.FrameCount);
            sb.AppendFormat(c, "    \"fps_promedio\": {0:F2},\n", fps.AverageFps);
            sb.AppendFormat(c, "    \"fps_min\": {0:F2},\n", fps.MinFps);
            sb.AppendFormat(c, "    \"frame_mediano_ms\": {0:F2},\n", fps.MedianFrameMs);
            sb.AppendFormat(c, "    \"frame_p99_ms\": {0:F2},\n", fps.P99FrameMs);
            sb.AppendFormat(c, "    \"frame_peor_ms\": {0:F2},\n", fps.WorstFrameMs);
            sb.AppendFormat(c, "    \"frames_bajo_objetivo\": {0},\n", fps.FramesBelowTarget);
            sb.AppendFormat(c, "    \"pct_bajo_objetivo\": {0:F2},\n", fps.PercentBelowTarget);
            sb.AppendFormat(c, "    \"n_stutters\": {0},\n", fps.StutterCount);
            sb.AppendFormat(c, "    \"umbral_stutter_ms\": {0:F2},\n", fps.StutterThresholdMs);
            sb.AppendFormat(c, "    \"fps_primera_mitad\": {0:F2},\n", fps.FirstHalfAvgFps);
            sb.AppendFormat(c, "    \"fps_segunda_mitad\": {0:F2},\n", fps.SecondHalfAvgFps);
            sb.AppendFormat(c, "    \"degradacion_pct\": {0:F2},\n", fps.DegradationPercent);
            sb.AppendFormat(c, "    \"degradacion_progresiva\": {0},\n", fps.HasProgressiveDegradation ? "true" : "false");
            sb.AppendFormat(c, "    \"cumple\": {0}\n", fps.MeetsSustainedTarget ? "true" : "false");
            sb.AppendLine("  }");
            sb.AppendLine("}");
            return sb.ToString();
        }

        static string Escape(string s)
        {
            return string.IsNullOrEmpty(s) ? "" : s.Replace("\\", "\\\\").Replace("\"", "\\\"");
        }
    }
}
