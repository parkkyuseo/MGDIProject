using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    // Reflection cache (avoid repeated reflection lookups)
    private Type _cachedType;
    private readonly Dictionary<string, MemberInfo> _memberCache = new Dictionary<string, MemberInfo>(64);
    private readonly StringBuilder _sb = new StringBuilder(768);

    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input missing";
            return;
        }

        _sb.Length = 0;

        // Always-safe public outputs
        _sb.Append("Mode: ").Append(input.Mode).Append('\n');
        _sb.Append("AxisValue: ").Append(input.AxisValue.ToString("F2")).Append('\n');

        // ---- Projection / neutral ----
        bool hasT = TryGetFloatAny(out float t, "Debug_t", "debug_t");
        bool hasT0 = TryGetFloatAny(out float t0, "Debug_tNeutral", "_tNeutral");
        if (hasT || hasT0)
        {
            _sb.Append('\n').Append("[Projection]\n");
            if (hasT) _sb.Append("t: ").Append(t.ToString("F3")).Append('\n');
            if (hasT0) _sb.Append("t0: ").Append(t0.ToString("F3")).Append('\n');
        }

        // ---- Slider <-> Tap context ----
        bool hasThumbOnIndex = TryGetBoolAny(out bool thumbOnIndex,
            "Debug_thumbOnIndex", "debug_thumbOnIndex", "_thumbOnIndex");

        bool hasThumbIndexDist = TryGetFloatAny(out float thumbIndexDist,
            "Debug_thumbIndexDist", "debug_thumbIndexDist");

        if (hasThumbOnIndex || hasThumbIndexDist)
        {
            _sb.Append('\n').Append("[Context]\n");
            if (hasThumbOnIndex)
                _sb.Append("ThumbOnIndex: ").Append(thumbOnIndex ? "YES" : "NO").Append('\n');
            if (hasThumbIndexDist)
                _sb.Append("ThumbIndexDist: ").Append(thumbIndexDist.ToString("F3")).Append(" m\n");
        }

        // ---- Touch / Tap signals ----
        bool hasTouchDist = TryGetFloatAny(out float touchDist, "Debug_touchDist", "debug_touchDist");

        bool hasDownLatched = TryGetBoolAny(out bool downLatched, "Debug_touchDownLatched");
        bool hasFsmStr = TryGetEnumString("_touchFsm", out string fsmStr);

        bool hasTapCount = TryGetIntAny(out int tapCount, "Debug_tapCount", "_tapCount");

        bool hasWindowRem = TryGetFloatAny(out float windowRem, "Debug_tapWindowRemaining");
        if (!hasWindowRem && TryGetFloatAny(out float tapWindowUntil, "_tapWindowUntil"))
        {
            windowRem = Mathf.Max(0f, tapWindowUntil - Time.time);
            hasWindowRem = true;
        }

        bool hasCooldownFlag = TryGetBoolAny(out bool inCooldown, "Debug_inTapCooldown");
        bool hasCooldownRem = false;
        float cooldownRem = 0f;
        if (!hasCooldownFlag && TryGetFloatAny(out float cooldownUntil, "_tapCooldownUntil"))
        {
            cooldownRem = Mathf.Max(0f, cooldownUntil - Time.time);
            hasCooldownRem = true;
            inCooldown = cooldownRem > 0f;
            hasCooldownFlag = true;
        }

        bool hasLastTapAge = TryGetFloatAny(out float lastTapAge, "Debug_lastTapAcceptedAge");
        if (!hasLastTapAge && TryGetFloatAny(out float lastTapTime, "_lastTapAcceptedTime"))
        {
            lastTapAge = Mathf.Max(0f, Time.time - lastTapTime);
            hasLastTapAge = true;
        }

        bool hasDebugState = TryGetStringAny(out string debugState, "Debug_state", "debug_state");

        // Section prints only if something exists
        if (hasTouchDist || hasDownLatched || hasFsmStr || hasTapCount || hasWindowRem || hasCooldownFlag || hasLastTapAge)
        {
            _sb.Append('\n').Append("[Touch/Tap]\n");

            if (hasTouchDist)
                _sb.Append("TouchDist: ").Append(touchDist.ToString("F3")).Append(" (ratio)\n");

            if (hasDownLatched)
                _sb.Append("TouchState: ").Append(downLatched ? "DOWN(latched)" : "UP(idle)").Append('\n');
            else if (hasFsmStr && !string.IsNullOrEmpty(fsmStr))
                _sb.Append("TouchFSM: ").Append(fsmStr).Append('\n');

            if (hasTapCount)
                _sb.Append("TapCount: ").Append(tapCount).Append('\n');

            if (hasWindowRem)
                _sb.Append("TapWindow: ").Append(windowRem.ToString("F2")).Append(" s\n");

            if (hasCooldownFlag)
            {
                if (hasCooldownRem)
                    _sb.Append("Cooldown: ").Append(inCooldown ? "IN" : "OUT")
                       .Append(" (").Append(cooldownRem.ToString("F2")).Append(" s)\n");
                else
                    _sb.Append("Cooldown: ").Append(inCooldown ? "IN" : "OUT").Append('\n');
            }

            if (hasLastTapAge)
                _sb.Append("LastTapAge: ").Append(lastTapAge.ToString("F2")).Append(" s\n");
        }

        // ---- Tap gate summary (clear, one-line) ----
        string gateLine = BuildTapGateSummary(
            hasDebugState ? debugState : null,
            hasThumbOnIndex ? (bool?)thumbOnIndex : null,
            hasCooldownFlag ? (bool?)inCooldown : null,
            hasCooldownRem ? (float?)cooldownRem : null,
            hasTapCount ? (int?)tapCount : null,
            hasWindowRem ? (float?)windowRem : null
        );

        if (!string.IsNullOrEmpty(gateLine))
        {
            _sb.Append('\n').Append("[Gate]\n");
            _sb.Append(gateLine).Append('\n');
        }

        // ---- Events ----
        if (input.SingleTapThisFrame || input.DoubleTapThisFrame)
        {
            _sb.Append('\n').Append("[Events]\n");
            if (input.SingleTapThisFrame) _sb.Append("EVENT: SINGLE TAP\n");
            if (input.DoubleTapThisFrame) _sb.Append("EVENT: DOUBLE TAP\n");
        }

        // ---- Raw debug state (only if present & non-empty, and only if it adds info) ----
        if (hasDebugState && !string.IsNullOrEmpty(debugState))
        {
            // If debugState already used as gate summary, still show it but label as internal
            _sb.Append('\n').Append("[Internal]\n");
            _sb.Append(debugState).Append('\n');
        }

        statusText.text = _sb.ToString();
    }

    // =========================
    // Gate summary formatting
    // =========================
    private string BuildTapGateSummary(string debugState, bool? thumbOnIndex, bool? inCooldown, float? cooldownRem, int? tapCount, float? tapWindowRem)
    {
        // 1) Prefer debugState parsing when available (most explicit)
        // Examples:
        // "Tap: blocked (thumb on index)"
        // "Tap: blocked (not in center band)"
        // "Tap: cooldown"
        // "Tap: ready"
        // "Tap: DOWN latched"
        // "Tap: SINGLE"
        // "Tap: DOUBLE"
        if (!string.IsNullOrEmpty(debugState))
        {
            string s = debugState.Trim();
            if (s.StartsWith("Tap:", StringComparison.OrdinalIgnoreCase))
            {
                string tail = s.Substring(4).Trim();

                // Normalize a bit
                if (tail.StartsWith("blocked", StringComparison.OrdinalIgnoreCase))
                {
                    // "blocked (...)" -> BLOCKED — (...)
                    int lp = tail.IndexOf('(');
                    int rp = tail.LastIndexOf(')');
                    if (lp >= 0 && rp > lp)
                    {
                        string reason = tail.Substring(lp + 1, rp - lp - 1).Trim();
                        return $"TapGate: BLOCKED — {reason}";
                    }
                    return "TapGate: BLOCKED";
                }

                if (tail.StartsWith("cooldown", StringComparison.OrdinalIgnoreCase))
                    return cooldownRem.HasValue ? $"TapGate: COOLDOWN — {cooldownRem.Value:F2}s remaining" : "TapGate: COOLDOWN";

                if (tail.StartsWith("ready", StringComparison.OrdinalIgnoreCase))
                    return "TapGate: READY";

                if (tail.StartsWith("min interval", StringComparison.OrdinalIgnoreCase))
                    return "TapGate: BLOCKED — min interval";

                if (tail.StartsWith("DOWN", StringComparison.OrdinalIgnoreCase))
                    return "TapGate: ACTIVE — touch down latched";

                if (tail.StartsWith("SINGLE", StringComparison.OrdinalIgnoreCase))
                    return "TapGate: TRIGGERED — single tap";

                if (tail.StartsWith("DOUBLE", StringComparison.OrdinalIgnoreCase))
                    return "TapGate: TRIGGERED — double tap";

                // Fallback: show Tap: ... as-is but clearer label
                return $"TapGate: {tail}";
            }
        }

        // 2) If no debugState, infer a clean summary from available signals (only if we have something)
        // Priority: thumbOnIndex -> cooldown -> waiting double tap -> ready
        bool hasAny = thumbOnIndex.HasValue || inCooldown.HasValue || tapCount.HasValue || tapWindowRem.HasValue;
        if (!hasAny) return null;

        if (thumbOnIndex.HasValue && thumbOnIndex.Value)
            return "TapGate: BLOCKED — thumb on index (sliding posture)";

        if (inCooldown.HasValue && inCooldown.Value)
        {
            if (cooldownRem.HasValue)
                return $"TapGate: COOLDOWN — {cooldownRem.Value:F2}s remaining";
            return "TapGate: COOLDOWN";
        }

        if (tapCount.HasValue && tapCount.Value == 1 && tapWindowRem.HasValue && tapWindowRem.Value > 0f)
            return $"TapGate: WAITING — second tap ({tapWindowRem.Value:F2}s left)";

        return "TapGate: READY";
    }

    // =========================
    // Reflection helpers
    // =========================
    private bool TryGetFloatAny(out float value, string a, string b = null, string c = null)
    {
        if (TryGetFloat(a, out value)) return true;
        if (!string.IsNullOrEmpty(b) && TryGetFloat(b, out value)) return true;
        if (!string.IsNullOrEmpty(c) && TryGetFloat(c, out value)) return true;
        value = 0f;
        return false;
    }

    private bool TryGetIntAny(out int value, string a, string b = null, string c = null)
    {
        if (TryGetInt(a, out value)) return true;
        if (!string.IsNullOrEmpty(b) && TryGetInt(b, out value)) return true;
        if (!string.IsNullOrEmpty(c) && TryGetInt(c, out value)) return true;
        value = 0;
        return false;
    }

    private bool TryGetBoolAny(out bool value, string a, string b = null, string c = null)
    {
        if (TryGetBool(a, out value)) return true;
        if (!string.IsNullOrEmpty(b) && TryGetBool(b, out value)) return true;
        if (!string.IsNullOrEmpty(c) && TryGetBool(c, out value)) return true;
        value = false;
        return false;
    }

    private bool TryGetStringAny(out string value, string a, string b = null, string c = null)
    {
        if (TryGetString(a, out value)) return true;
        if (!string.IsNullOrEmpty(b) && TryGetString(b, out value)) return true;
        if (!string.IsNullOrEmpty(c) && TryGetString(c, out value)) return true;
        value = null;
        return false;
    }

    private bool TryGetFloat(string name, out float value)
    {
        value = 0f;
        if (!TryGetMemberValue(name, out object obj) || obj == null) return false;

        try
        {
            if (obj is float f) { value = f; return true; }
            if (obj is double d) { value = (float)d; return true; }
            if (obj is int i) { value = i; return true; }
            if (obj is long l) { value = l; return true; }
            if (obj is string s && float.TryParse(s, out float ps)) { value = ps; return true; }
        }
        catch { }
        return false;
    }

    private bool TryGetInt(string name, out int value)
    {
        value = 0;
        if (!TryGetMemberValue(name, out object obj) || obj == null) return false;

        try
        {
            if (obj is int i) { value = i; return true; }
            if (obj is long l) { value = (int)l; return true; }
            if (obj is float f) { value = Mathf.RoundToInt(f); return true; }
            if (obj is string s && int.TryParse(s, out int ps)) { value = ps; return true; }
        }
        catch { }
        return false;
    }

    private bool TryGetBool(string name, out bool value)
    {
        value = false;
        if (!TryGetMemberValue(name, out object obj) || obj == null) return false;

        try
        {
            if (obj is bool b) { value = b; return true; }
            if (obj is string s && bool.TryParse(s, out bool ps)) { value = ps; return true; }
        }
        catch { }
        return false;
    }

    private bool TryGetString(string name, out string value)
    {
        value = null;
        if (!TryGetMemberValue(name, out object obj) || obj == null) return false;

        if (obj is string s) { value = s; return true; }
        value = obj.ToString();
        return true;
    }

    private bool TryGetEnumString(string fieldName, out string enumString)
    {
        enumString = null;
        if (!TryGetMemberValue(fieldName, out object obj) || obj == null) return false;

        if (obj is Enum)
        {
            enumString = obj.ToString();
            return true;
        }

        return false;
    }

    private bool TryGetMemberValue(string name, out object value)
    {
        value = null;
        if (input == null) return false;

        var t = input.GetType();
        if (_cachedType != t)
        {
            _cachedType = t;
            _memberCache.Clear();
        }

        if (!_memberCache.TryGetValue(name, out MemberInfo mi))
        {
            mi = (MemberInfo)t.GetProperty(name, Flags) ?? (MemberInfo)t.GetField(name, Flags);
            _memberCache[name] = mi; // can be null
        }

        if (mi == null) return false;

        try
        {
            if (mi is PropertyInfo pi)
            {
                value = pi.GetValue(input, null);
                return true;
            }

            if (mi is FieldInfo fi)
            {
                value = fi.GetValue(input);
                return true;
            }
        }
        catch
        {
            return false;
        }

        return false;
    }
}
