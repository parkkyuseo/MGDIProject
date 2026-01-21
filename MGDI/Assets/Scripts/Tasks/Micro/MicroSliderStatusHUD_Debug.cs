using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showAxis = true;
    [SerializeField] private bool showT = true;
    [SerializeField] private bool showTouch = true;
    [SerializeField] private bool showTap = true;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input missing";
            return;
        }

        string s = "";

        // Mode
        s += $"Mode: {input.Mode}\n";

        // Axis output (used for motion)
        if (showAxis)
            s += $"AxisValue: {input.AxisValue:F2}\n";

        // Raw projected t and neutral
        if (showT)
            s += $"t: {input.Debug_t:F2}  t0: {input.Debug_tNeutral:F2}\n";

        // Touch distance + state
        if (showTouch)
        {
            string touchState = input.Debug_touchDownLatched ? "DOWN(latched)" : "UP(idle)";
            string cd = input.Debug_inTapCooldown ? "cooldown" : "ready";
            s += $"TouchDist: {input.Debug_touchDist:F3} m\n";
            s += $"TouchState: {touchState}  TapGate: {cd}\n";
        }

        // Tap info + events
        if (showTap)
        {
            s += $"TapCount: {input.Debug_tapCount}  Window: {input.Debug_tapWindowRemaining:F2}s\n";
            if (input.SingleTapThisFrame) s += "EVENT: SINGLE TAP\n";
            if (input.DoubleTapThisFrame) s += "EVENT: DOUBLE TAP\n";
        }

        // Internal state string (useful when debugging)
        string st = input.Debug_state;
        if (!string.IsNullOrEmpty(st))
            s += $"State: {st}\n";

        statusText.text = s;
    }
}
