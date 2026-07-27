// FrameRateMonitor.cs — MÉTRICA 3 del MVP: rendimiento gráfico (FPS).
//
// Objetivo: >= 72 FPS SOSTENIDOS durante sesiones continuas de 5+ minutos, sin
// degradación progresiva ni stuttering.
//
// Las tres palabras del enunciado piden tres mediciones distintas, y este
// monitor las separa en vez de reducirlo todo a un FPS medio:
//
//   * "72 FPS"        -> FPS medio en ventana deslizante.
//   * "sostenidos"    -> DEGRADACIÓN: se compara el primer tercio de la sesión
//                        con el último. Un sistema que arranca a 90 y termina a
//                        60 tiene un promedio de 75 y aun así falla la métrica.
//   * "sin stuttering"-> PICOS: frames individuales muy por encima del tiempo de
//                        frame mediano. Un tirón de 200 ms cada 30 s es
//                        invisible en cualquier promedio y muy visible para el
//                        usuario (y en VR, nauseabundo).
//
// ---------------------------------------------------------------------------
//  DÓNDE SE PUEDE MEDIR ESTO DE VERDAD
// ---------------------------------------------------------------------------
// El umbral de 72 FPS es el de la tasa de refresco del Meta Quest 3. En PC el
// número que importa es el del display (típicamente 60 Hz), y con VSync activo
// el FPS queda clavado en esa tasa y la métrica se vuelve trivial. Por eso
// TargetFps es configurable: mídelo en el hardware cuyo umbral vas a declarar y
// dilo explícitamente en el informe. Ver la nota de alcance en el README.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace LSChMR.Telemetry
{
    /// <summary>Una muestra agregada de FPS (una fila del CSV de la serie temporal).</summary>
    public struct FpsSample
    {
        public double TimeS;
        public float Average;
        public float Min;
        public float OnePercentLow;
        public int FramesBelowTarget;
    }

    /// <summary>
    /// Contador de FPS en ventana deslizante, con detección de caídas bajo el
    /// objetivo, degradación progresiva y stuttering.
    /// </summary>
    public sealed class FrameRateMonitor
    {
        /// <summary>Objetivo del MVP. 72 = Meta Quest 3.</summary>
        public float TargetFps = 72.0f;

        /// <summary>Ancho de la ventana deslizante, en segundos.</summary>
        public double WindowSeconds = 1.0;

        /// <summary>Cada cuánto se registra una muestra en la serie temporal.</summary>
        public double SampleIntervalSeconds = 5.0;

        /// <summary>
        /// Un frame es un "tirón" si dura más que este múltiplo del tiempo de
        /// frame mediano de la sesión. x2 = el frame tardó el doble de lo normal;
        /// a 72 FPS son 27,8 ms frente a 13,9 ms, ya perceptible como salto.
        /// </summary>
        public double StutterFactor = 2.0;

        // Ventana deslizante de duraciones de frame (segundos), con su instante.
        readonly Queue<KeyValuePair<double, double>> _window =
            new Queue<KeyValuePair<double, double>>();

        readonly List<FpsSample> _samples = new List<FpsSample>();
        readonly List<double> _allFrameTimes = new List<double>();

        double _elapsed;
        double _lastSampleAt;
        int _framesBelowTargetTotal;
        int _frameCount;

        // Suma de FPS instantáneo por mitades de sesión — la base de la
        // detección de degradación. Se acumula en vivo para no tener que guardar
        // la sesión entera en memoria (5 min a 72 FPS son 21.600 frames).
        double _sumFpsFirstHalf, _sumFpsSecondHalf;
        int _countFirstHalf, _countSecondHalf;
        double _halfwayMark = -1.0;

        public IReadOnlyList<FpsSample> Samples => _samples;
        public double ElapsedSeconds => _elapsed;
        public int FrameCount => _frameCount;
        public int StutterCount { get; private set; }
        public double WorstFrameMs { get; private set; }

        /// <summary>
        /// Duración total prevista de la sesión, en segundos. Solo se usa para
        /// situar la mitad y comparar primera vs. segunda parte. Si no se
        /// conoce, la mitad se recalcula sobre la marcha.
        /// </summary>
        public void SetExpectedDuration(double seconds)
        {
            _halfwayMark = seconds / 2.0;
        }

        /// <summary>
        /// Llamar UNA VEZ POR FRAME con Time.unscaledDeltaTime.
        /// </summary>
        /// <remarks>
        /// unscaledDeltaTime y no deltaTime: si algo toca Time.timeScale (una
        /// pausa, una animación), deltaTime deja de reflejar el tiempo real y el
        /// FPS reportado sería ficción.
        /// </remarks>
        public void Tick(double unscaledDeltaTime)
        {
            if (unscaledDeltaTime <= 0.0) return;

            _elapsed += unscaledDeltaTime;
            _frameCount++;
            _allFrameTimes.Add(unscaledDeltaTime);

            double instantFps = 1.0 / unscaledDeltaTime;
            if (instantFps < TargetFps) _framesBelowTargetTotal++;

            double frameMs = unscaledDeltaTime * 1000.0;
            if (frameMs > WorstFrameMs) WorstFrameMs = frameMs;

            // Reparto en mitades para la detección de degradación.
            double mitad = _halfwayMark > 0.0 ? _halfwayMark : _elapsed / 2.0;
            if (_elapsed <= mitad) { _sumFpsFirstHalf += instantFps; _countFirstHalf++; }
            else { _sumFpsSecondHalf += instantFps; _countSecondHalf++; }

            // Ventana deslizante.
            _window.Enqueue(new KeyValuePair<double, double>(_elapsed, unscaledDeltaTime));
            while (_window.Count > 0 && _elapsed - _window.Peek().Key > WindowSeconds)
                _window.Dequeue();

            if (_elapsed - _lastSampleAt >= SampleIntervalSeconds)
            {
                _lastSampleAt = _elapsed;
                _samples.Add(CurrentSample());
            }
        }

        /// <summary>Estado actual de la ventana deslizante.</summary>
        public FpsSample CurrentSample()
        {
            var sample = new FpsSample { TimeS = _elapsed };
            if (_window.Count == 0) return sample;

            var times = new List<double>(_window.Count);
            double total = 0.0;
            int below = 0;
            double worst = 0.0;
            foreach (var kv in _window)
            {
                times.Add(kv.Value);
                total += kv.Value;
                if (1.0 / kv.Value < TargetFps) below++;
                if (kv.Value > worst) worst = kv.Value;
            }

            sample.Average = (float)(times.Count / total);
            // El FPS "mínimo" de la ventana es el del frame MÁS LENTO: es el
            // tirón que el usuario nota, no un promedio suavizado.
            sample.Min = (float)(1.0 / worst);
            sample.FramesBelowTarget = below;

            times.Sort();
            // 1% low: media del 1% de frames más lentos. Estándar de la industria
            // para fluidez percibida; con ventanas cortas cae en el peor frame.
            int n = Math.Max(1, times.Count / 100);
            double sumWorst = 0.0;
            for (int i = times.Count - n; i < times.Count; i++) sumWorst += times[i];
            sample.OnePercentLow = (float)(n / sumWorst);
            return sample;
        }

        public float CurrentAverageFps => CurrentSample().Average;
        public float CurrentMinFps => CurrentSample().Min;

        // ------------------------------------------------------------------ //
        // Resumen de sesión
        // ------------------------------------------------------------------ //
        public FpsSummary Summarize()
        {
            var s = new FpsSummary
            {
                TargetFps = TargetFps,
                DurationS = _elapsed,
                FrameCount = _frameCount,
                FramesBelowTarget = _framesBelowTargetTotal,
                WorstFrameMs = WorstFrameMs
            };
            if (_frameCount == 0) return s;

            s.PercentBelowTarget = 100.0 * _framesBelowTargetTotal / _frameCount;
            s.AverageFps = (float)(_frameCount / Math.Max(1e-9, _elapsed));

            var sorted = new List<double>(_allFrameTimes);
            sorted.Sort();
            double median = sorted[sorted.Count / 2];
            s.MedianFrameMs = median * 1000.0;
            s.MinFps = (float)(1.0 / sorted[sorted.Count - 1]);
            // p99 del TIEMPO de frame = el 1% de frames más lentos.
            s.P99FrameMs = PipelineTelemetry.Percentile(sorted, 99.0) * 1000.0;

            int stutters = 0;
            double umbral = median * StutterFactor;
            foreach (double ft in _allFrameTimes) if (ft > umbral) stutters++;
            StutterCount = stutters;
            s.StutterCount = stutters;
            s.StutterThresholdMs = umbral * 1000.0;

            s.FirstHalfAvgFps = _countFirstHalf > 0
                ? (float)(_sumFpsFirstHalf / _countFirstHalf) : 0f;
            s.SecondHalfAvgFps = _countSecondHalf > 0
                ? (float)(_sumFpsSecondHalf / _countSecondHalf) : 0f;
            s.DegradationFps = s.FirstHalfAvgFps - s.SecondHalfAvgFps;
            s.DegradationPercent = s.FirstHalfAvgFps > 0f
                ? 100.0 * s.DegradationFps / s.FirstHalfAvgFps : 0.0;

            // Degradación "progresiva" = la segunda mitad va >5% más lenta que la
            // primera. El 5% deja fuera el ruido normal de medición sin dejar
            // pasar una caída real (a 72 FPS son ~3,6 FPS de diferencia).
            s.HasProgressiveDegradation = s.DegradationPercent > 5.0;

            s.MeetsSustainedTarget =
                s.AverageFps >= TargetFps &&
                s.PercentBelowTarget <= 1.0 &&
                !s.HasProgressiveDegradation;

            s.LongEnough = _elapsed >= 300.0;   // los 5 minutos que pide el diseño
            return s;
        }

        public const string CsvHeader =
            "t_s,fps_promedio,fps_min,fps_1pct_low,frames_bajo_objetivo";

        public string SamplesToCsv()
        {
            var c = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.AppendLine(CsvHeader);
            foreach (var s in _samples)
            {
                sb.AppendLine(string.Join(",",
                    s.TimeS.ToString("F2", c),
                    s.Average.ToString("F1", c),
                    s.Min.ToString("F1", c),
                    s.OnePercentLow.ToString("F1", c),
                    s.FramesBelowTarget.ToString(c)));
            }
            return sb.ToString();
        }
    }

    public sealed class FpsSummary
    {
        public float TargetFps;
        public double DurationS;
        public int FrameCount;
        public float AverageFps;
        public float MinFps;
        public double MedianFrameMs;
        public double P99FrameMs;
        public double WorstFrameMs;

        public int FramesBelowTarget;
        public double PercentBelowTarget;

        public int StutterCount;
        public double StutterThresholdMs;

        public float FirstHalfAvgFps;
        public float SecondHalfAvgFps;
        public float DegradationFps;
        public double DegradationPercent;
        public bool HasProgressiveDegradation;

        /// <summary>La sesión duró los 5 minutos que exige el diseño.</summary>
        public bool LongEnough;

        /// <summary>
        /// Cumple "72 FPS sostenidos": media sobre el objetivo, como mucho un 1%
        /// de frames por debajo y sin degradación progresiva. Ojo: esto NO
        /// comprueba la duración — mira también LongEnough antes de reportar.
        /// </summary>
        public bool MeetsSustainedTarget;
    }
}
