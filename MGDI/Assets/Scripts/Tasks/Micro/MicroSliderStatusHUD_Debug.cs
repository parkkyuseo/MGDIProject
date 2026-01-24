using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showContext = true;
    [SerializeField] private bool showPulse = true;
    [SerializeField] private bool showState = true;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input missing";
            return;
        }

        string s = "";

        // Always: core outputs
        s += $"Mode: {input.Mode}\n";
        s += $"Axis: {input.AxisValue:F2}\n";
        s += $"t: {input.Debug_t:F3}\n";

        // Context: thumb/index contact state drives everything now
        if (showContext)
        {
            s += "\n[Context]\n";
            s += $"ThumbOnIndex: {(input.Debug_thumbOnIndex ? "YES" : "NO")}\n";
            s += $"ThumbIndexDist: {input.Debug_thumbIndexDist:F3} m\n";
        }

        // Pulse / hold state (these are your new “discrete gesture” signals)
        if (showPulse)
        {
            s += "\n[Pulse/Hold]\n";
            s += $"PulseCount: {input.Debug_pulseCount}\n";

            // Only meaningful while OFF, but harmless to display always
            s += $"OffHeld: {input.Debug_offHeldSec:F2} s\n";
            s += $"DoubleWinRem: {input.Debug_doubleWindowRemaining:F2} s\n";
        }

        // One-line decision/diagnostic
        if (showState)
        {
            string st = input.Debug_state;
            if (!string.IsNullOrEmpty(st))
            {
                s += "\n[State]\n";
                s += $"{st}\n";
            }
        }

        statusText.text = s;
    }
}
