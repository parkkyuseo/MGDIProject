using System.Reflection;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showOutputs = true;
    [SerializeField] private bool showPinchAndCalibration = true;
    [SerializeField] private bool showTwist = true;
    [SerializeField] private bool showToggle = true;
    [SerializeField] private bool showState = true;

    private readonly StringBuilder _sb = new StringBuilder(512);

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input missing";
            return;
        }

        _sb.Length = 0;

        // ---- Outputs (what gameplay code actually uses) ----
        if (showOutputs)
        {
            _sb.AppendLine("[Outputs]");
            _sb.AppendLine($"Mode: {input.Mode}");
            _sb.AppendLine($"X: {input.AxisX:F2}  Y: {input.AxisY:F2}  Z: {input.AxisZ:F2}");
            _sb.AppendLine($"Slide(AxisValue): {input.AxisValue:F2}");
            _sb.AppendLine($"tUsed: {input.Debug_t:F3}   tRaw: {input.Debug_tRaw:F3}");
        }

        // ---- Pinch + Calibration (new critical debug info) ----
        if (showPinchAndCalibration)
        {
            _sb.AppendLine();
            _sb.AppendLine("[Pinch + Calibration]");

            bool isCalibrating = input.IsCalibrating;
            bool isCalibrated = input.IsCalibrated;

            _sb.AppendLine($"Calibrating: {(isCalibrating ? "YES" : "NO")}   Calibrated: {(isCalibrated ? "YES" : "NO")}");

            // These exist as private debug fields in the modified MicroThumbIndexSliderInput.
            // They are not exposed publicly, so reflection is used.
            float calibElapsed = ReadFieldOrNaN<float>(input, "debug_calibElapsed");
            int calibSamples = ReadFieldOrDefault<int>(input, "debug_calibSamples", -1);

            if (!float.IsNaN(calibElapsed) || calibSamples >= 0)
            {
                string elapsedStr = float.IsNaN(calibElapsed) ? "N/A" : $"{calibElapsed:F2}s";
                string samplesStr = (calibSamples < 0) ? "N/A" : $"{calibSamples}";
                _sb.AppendLine($"CalibWindow: {elapsedStr}   Samples: {samplesStr}");
            }

            bool thumbOn = input.Debug_thumbOnIndex;
            _sb.AppendLine($"ThumbOnIndex(stable): {(thumbOn ? "YES" : "NO")}");

            float distSm = input.Debug_thumbIndexDist; // distSm
            float distRaw = ReadFieldOrNaN<float>(input, "debug_thumbIndexDistRaw"); // distRaw (reflection)
            _sb.AppendLine($"DistSm: {distSm:F3} m   DistRaw: {(float.IsNaN(distRaw) ? "N/A" : $"{distRaw:F3} m")}");

            float distBase = ReadFieldOrNaN<float>(input, "debug_distBase");
            float onThresh = ReadFieldOrNaN<float>(input, "debug_onThresh");
            float offThresh = ReadFieldOrNaN<float>(input, "debug_offThresh");

            string baseStr = float.IsNaN(distBase) ? "N/A" : $"{distBase:F3} m";
            string onStr = float.IsNaN(onThresh) ? "N/A" : $"{onThresh:F3} m";
            string offStr = float.IsNaN(offThresh) ? "N/A" : $"{offThresh:F3} m";

            _sb.AppendLine($"DistBase: {baseStr}");
            _sb.AppendLine($"OnThresh: {onStr}   OffThresh: {offStr}");

            if (!float.IsNaN(distBase))
            {
                float delta = distSm - distBase;
                _sb.AppendLine($"Delta(distSm - base): {delta:+0.000;-0.000;0.000} m");
            }
        }

        // ---- Twist (helps verify twist-neutral + velocity) ----
        if (showTwist)
        {
            _sb.AppendLine();
            _sb.AppendLine("[Twist]");

            float twistDeg = ReadFieldOrNaN<float>(input, "debug_twistDeg");
            float twistRel = ReadFieldOrNaN<float>(input, "debug_twistRelDeg");
            float twistVel = ReadFieldOrNaN<float>(input, "debug_twistVelDegPerSec");

            _sb.AppendLine($"TwistDeg: {(float.IsNaN(twistDeg) ? "N/A" : $"{twistDeg:F1}°")}");
            _sb.AppendLine($"TwistRel: {(float.IsNaN(twistRel) ? "N/A" : $"{twistRel:F1}°")}");
            _sb.AppendLine($"TwistVel: {(float.IsNaN(twistVel) ? "N/A" : $"{twistVel:F0}°/s")}");
        }

        // ---- Mode toggle diagnostics ----
        if (showToggle)
        {
            _sb.AppendLine();
            _sb.AppendLine("[Mode Toggle]");
            _sb.AppendLine($"OffHeld: {input.Debug_offHeldSec:F2} s");
            _sb.AppendLine($"Armed: {(input.Debug_zArmed ? "YES" : "NO")}");
        }

        // ---- State string ----
        if (showState)
        {
            string st = input.Debug_state;
            if (!string.IsNullOrEmpty(st))
            {
                _sb.AppendLine();
                _sb.AppendLine("[State]");
                _sb.AppendLine(st);
            }
        }

        statusText.text = _sb.ToString();
    }

    // ---------- Reflection helpers ----------
    static T ReadFieldOrDefault<T>(object obj, string fieldName, T fallback)
    {
        if (obj == null) return fallback;
        var t = obj.GetType();
        var f = t.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (f == null) return fallback;
        object v = f.GetValue(obj);
        if (v is T tv) return tv;
        return fallback;
    }

    static float ReadFieldOrNaN<T>(object obj, string fieldName)
    {
        if (obj == null) return float.NaN;
        var t = obj.GetType();
        var f = t.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (f == null) return float.NaN;

        object v = f.GetValue(obj);
        if (v is float fv) return fv;

        // Allow numeric types that can cast to float
        if (v is int iv) return iv;
        if (v is double dv) return (float)dv;

        return float.NaN;
    }
}
