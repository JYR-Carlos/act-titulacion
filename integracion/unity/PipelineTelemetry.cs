// PipelineTelemetry.cs — MÉTRICA 2 del MVP: latencia de inferencia end-to-end.
//
// Mide el tiempo desde que la seña TERMINA de verdad hasta que el subtítulo está
// dibujado en la Spatial UI, sellando un timestamp en cada frontera de capa:
//
//     [Capture]      keypoints del frame listos (Capa 1)
//        |
//     [Segmentation] el RestStateDetector confirma FIN de seña (Capa 1)
//        |
//     [Inference]    el clasificador devuelve la glosa (Capa 3, Sentis)
//        |
//     [Render]       el subtítulo está PRESENTADO en pantalla (Capa 4)
//
// ---------------------------------------------------------------------------
//  EL RETARDO DE SEGMENTACIÓN CUENTA, Y ES LA MAYOR PARTE DEL PRESUPUESTO
// ---------------------------------------------------------------------------
// El detector no sabe que la seña terminó hasta ver REST_FRAMES_FIN frames de
// reposo sostenido. Ese retardo es inherente al diseño y el usuario lo percibe
// como parte de la espera, así que una medición honesta NO puede descontarlo:
//
//     latencia_e2e = retardo_segmentacion + tiempo_de_proceso
//     retardo_segmentacion = REST_FRAMES_FIN * periodo_medio_de_frame
//
// Es exactamente la fórmula de `demo_vivo.py` (ver el bloque `recien_clasificada`
// y `config.LATENCIA_MAX_MS`). Si el port en C# la calculara de otro modo, las
// cifras de Unity y las de Python no serían comparables y el informe estaría
// mezclando dos definiciones distintas de la misma métrica.
//
// A 30 FPS ese retardo ya son ~200 ms de los 500 ms de presupuesto. Medido en
// Python sobre 10 vídeos a 60 FPS: mediana ~239 ms, rango 171–547 ms, 1 de 9 por
// encima del umbral (ver ESTADO_ACTUAL.md). La palanca si sale corto es bajar
// REST_FRAMES_FIN de 6 a 4, a costa de cerrar las señas antes.
//
// Uso: ver MetricsRecorder.cs y el README de esta carpeta.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace LSChMR.Telemetry
{
    /// <summary>Fronteras de capa que se instrumentan.</summary>
    public enum PipelineStage
    {
        Capture = 0,
        Segmentation = 1,
        Inference = 2,
        Render = 3
    }

    /// <summary>Una seña medida de extremo a extremo.</summary>
    public sealed class SignLatencyRecord
    {
        public DateTime WallClockUtc;
        public double SessionTimeS;

        public string Gloss = "";
        public float Confidence;
        public bool InVocabulary;

        // Desglose por tramo, en milisegundos.
        public double CaptureToSegmentationMs;
        public double SegmentationToInferenceMs;
        public double InferenceToRenderMs;

        /// <summary>REST_FRAMES_FIN * periodo medio de frame.</summary>
        public double SegmentationDelayMs;

        /// <summary>Tiempo de proceso: del último frame capturado al subtítulo dibujado.</summary>
        public double ProcessingMs;

        /// <summary>SegmentationDelayMs + ProcessingMs. La métrica 2.</summary>
        public double EndToEndMs;

        // FPS vigente cuando se reconoció esta seña (ventana deslizante).
        public float FpsAverage;
        public float FpsMin;

        public bool ExceedsThreshold;
    }

    /// <summary>
    /// Acumula los timestamps de una seña en curso y produce un
    /// <see cref="SignLatencyRecord"/> cuando el subtítulo queda presentado.
    /// </summary>
    public sealed class PipelineTelemetry
    {
        /// <summary>Umbral de la Sección 11 (config.LATENCIA_MAX_MS).</summary>
        public const double LatencyBudgetMs = 500.0;

        /// <summary>config.REST_FRAMES_FIN — debe seguir al valor de Python.</summary>
        public int RestFramesEnd = 6;

        readonly List<SignLatencyRecord> _records = new List<SignLatencyRecord>();
        readonly Stopwatch _session = Stopwatch.StartNew();

        // Stopwatch en vez de Time.realtimeSinceStartup: este último es un float
        // de 32 bits que, tras unos minutos de sesión, tiene una resolución peor
        // que el propio intervalo que queremos medir. Para una métrica con un
        // presupuesto de 500 ms eso no sirve.
        long _tCapture, _tSegmentation, _tInference;
        bool _signInFlight;

        public IReadOnlyList<SignLatencyRecord> Records => _records;
        public double SessionSeconds => _session.Elapsed.TotalSeconds;

        static double ToMs(long fromTicks, long toTicks)
        {
            return (toTicks - fromTicks) * 1000.0 / Stopwatch.Frequency;
        }

        /// <summary>
        /// Capa 1 — llamar UNA VEZ POR FRAME, justo después de obtener los
        /// keypoints y ANTES de pasarlos al detector.
        /// </summary>
        /// <remarks>
        /// Se resella en cada frame a propósito: cuando el detector confirme el
        /// fin de la seña, lo que interesa es cuándo se capturó el ÚLTIMO frame,
        /// no el primero. El tramo Capture→Segmentation mide entonces el coste
        /// real de procesar ese frame, no la duración de la seña entera.
        /// </remarks>
        public void MarkCapture()
        {
            _tCapture = Stopwatch.GetTimestamp();
        }

        /// <summary>Capa 1 — el RestStateDetector acaba de emitir FIN de seña.</summary>
        public void MarkSegmentationEnd()
        {
            _tSegmentation = Stopwatch.GetTimestamp();
            _signInFlight = true;
        }

        /// <summary>Capa 3 — el clasificador (Sentis) devolvió la glosa.</summary>
        public void MarkInferenceEnd()
        {
            if (!_signInFlight) return;
            _tInference = Stopwatch.GetTimestamp();
        }

        /// <summary>
        /// Capa 4 — el subtítulo está PRESENTADO. Cierra la medición y devuelve
        /// el registro, o null si no había una seña en curso.
        /// </summary>
        /// <param name="gloss">Glosa mostrada, o "&lt;desconocida&gt;" si quedó bajo el umbral.</param>
        /// <param name="confidence">Confianza del clasificador.</param>
        /// <param name="inVocabulary">false si la confianza no alcanzó CONF_THRESHOLD (FA-01).</param>
        /// <param name="frameperiodMs">
        /// Periodo medio entre frames de CAPTURA (no de render). Convierte el
        /// retardo de segmentación, que el diseño define en frames, a los
        /// milisegundos que se comparan con el presupuesto.
        /// </param>
        /// <param name="fpsAverage">FPS medio de la ventana deslizante en este momento.</param>
        /// <param name="fpsMin">FPS mínimo de la ventana deslizante en este momento.</param>
        /// <remarks>
        /// IMPORTANTE: llamar cuando el frame se ha PRESENTADO, no cuando se hizo
        /// SetText. Asignar el texto en Update() no lo pone en pantalla: eso
        /// ocurre al final del frame. Sellar aquí el instante del SetText
        /// descontaría un frame entero (~14 ms a 72 FPS) de la métrica. Ver
        /// MetricsRecorder.CompleteAtEndOfFrame().
        /// </remarks>
        public SignLatencyRecord MarkRenderComplete(
            string gloss, float confidence, bool inVocabulary,
            double frameperiodMs, float fpsAverage, float fpsMin)
        {
            if (!_signInFlight) return null;
            _signInFlight = false;

            long tRender = Stopwatch.GetTimestamp();

            // Si nunca se selló la inferencia (p. ej. la seña se descartó antes
            // de clasificar), se colapsa el tramo en vez de reportar un valor
            // negativo sacado de un timestamp viejo.
            long tInference = _tInference > _tSegmentation ? _tInference : _tSegmentation;

            double segDelay = RestFramesEnd * Math.Max(0.0, frameperiodMs);
            double processing = ToMs(_tCapture, tRender);

            var record = new SignLatencyRecord
            {
                WallClockUtc = DateTime.UtcNow,
                SessionTimeS = _session.Elapsed.TotalSeconds,
                Gloss = gloss ?? "",
                Confidence = confidence,
                InVocabulary = inVocabulary,
                CaptureToSegmentationMs = ToMs(_tCapture, _tSegmentation),
                SegmentationToInferenceMs = ToMs(_tSegmentation, tInference),
                InferenceToRenderMs = ToMs(tInference, tRender),
                SegmentationDelayMs = segDelay,
                ProcessingMs = processing,
                EndToEndMs = segDelay + processing,
                FpsAverage = fpsAverage,
                FpsMin = fpsMin
            };
            record.ExceedsThreshold = record.EndToEndMs > LatencyBudgetMs;

            _records.Add(record);
            return record;
        }

        /// <summary>Descarta la seña en curso (p. ej. pérdida de tracking sostenida).</summary>
        public void AbortSign()
        {
            _signInFlight = false;
        }

        // ------------------------------------------------------------------ //
        // Resumen
        // ------------------------------------------------------------------ //
        public LatencySummary Summarize()
        {
            var s = new LatencySummary { ThresholdMs = LatencyBudgetMs, Count = _records.Count };
            if (_records.Count == 0) return s;

            var values = new List<double>(_records.Count);
            foreach (var r in _records) values.Add(r.EndToEndMs);
            values.Sort();

            double sum = 0.0;
            int exceeded = 0;
            foreach (var r in _records)
            {
                sum += r.EndToEndMs;
                if (r.ExceedsThreshold) exceeded++;
            }

            s.MeanMs = sum / _records.Count;
            s.MinMs = values[0];
            s.MaxMs = values[values.Count - 1];
            s.MedianMs = Percentile(values, 50.0);
            s.P95Ms = Percentile(values, 95.0);
            s.ExceededCount = exceeded;
            s.ExceededPercent = 100.0 * exceeded / _records.Count;
            s.Pass = exceeded == 0;
            return s;
        }

        /// <summary>Percentil por interpolación lineal, sobre una lista YA ordenada.</summary>
        internal static double Percentile(List<double> sorted, double p)
        {
            if (sorted.Count == 0) return 0.0;
            if (sorted.Count == 1) return sorted[0];
            double rank = (p / 100.0) * (sorted.Count - 1);
            int lo = (int)Math.Floor(rank);
            int hi = Math.Min(lo + 1, sorted.Count - 1);
            double frac = rank - lo;
            return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        }

        // ------------------------------------------------------------------ //
        // Exportación CSV
        // ------------------------------------------------------------------ //
        /// <summary>
        /// Cabecera del CSV de señas. Las cinco primeras columnas son las que
        /// pide el informe (timestamp, seña, latencia, fps medio, fps mínimo);
        /// el resto es el desglose que permite explicar un valor alto.
        /// </summary>
        public const string CsvHeader =
            "timestamp_utc,t_sesion_s,sena,confianza,en_vocabulario,latencia_ms," +
            "fps_promedio,fps_min,excede_umbral,retardo_segmentacion_ms," +
            "proceso_ms,captura_a_segmentacion_ms,segmentacion_a_inferencia_ms," +
            "inferencia_a_render_ms";

        public string ToCsv()
        {
            var sb = new StringBuilder();
            sb.AppendLine(CsvHeader);
            foreach (var r in _records) sb.AppendLine(ToCsvRow(r));
            return sb.ToString();
        }

        /// <summary>
        /// Una fila del CSV. Todo se formatea con InvariantCulture: en un equipo
        /// con locale español el separador decimal sería una coma y cada número
        /// partiría la fila en dos columnas, corrompiendo el CSV en silencio.
        /// </summary>
        public static string ToCsvRow(SignLatencyRecord r)
        {
            var c = CultureInfo.InvariantCulture;
            return string.Join(",",
                r.WallClockUtc.ToString("o", c),
                r.SessionTimeS.ToString("F2", c),
                Escape(r.Gloss),
                r.Confidence.ToString("F4", c),
                r.InVocabulary ? "1" : "0",
                r.EndToEndMs.ToString("F2", c),
                r.FpsAverage.ToString("F1", c),
                r.FpsMin.ToString("F1", c),
                r.ExceedsThreshold ? "1" : "0",
                r.SegmentationDelayMs.ToString("F2", c),
                r.ProcessingMs.ToString("F2", c),
                r.CaptureToSegmentationMs.ToString("F2", c),
                r.SegmentationToInferenceMs.ToString("F2", c),
                r.InferenceToRenderMs.ToString("F2", c));
        }

        /// <summary>Entrecomilla si el valor contiene coma, comilla o salto de línea.</summary>
        static string Escape(string value)
        {
            if (string.IsNullOrEmpty(value)) return "";
            if (value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0) return value;
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }
    }

    public sealed class LatencySummary
    {
        public int Count;
        public double ThresholdMs;
        public double MeanMs, MedianMs, MinMs, MaxMs, P95Ms;
        public int ExceededCount;
        public double ExceededPercent;

        /// <summary>
        /// true solo si NINGUNA seña superó el presupuesto. El diseño enuncia la
        /// métrica como "latencia ≤ 500 ms", no "latencia media ≤ 500 ms": una
        /// media de 300 ms con un tercio de las señas por encima del segundo
        /// sería una experiencia mala que la media escondería.
        /// </summary>
        public bool Pass;
    }
}
