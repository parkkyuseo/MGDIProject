using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showContext = true;
    [SerializeField] private bool showToggle = true;
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

        // Core outputs
        s += $"Mode: {input.Mode}\n";
        s += $"X: {input.AxisX:F2}  Y: {input.AxisY:F2}  Z: {input.AxisZ:F2}\n";
        s += $"t: {input.Debug_t:F3}\n";

        if (showContext)
        {
            s += "\n[Context]\n";
            s += $"ThumbOnIndex: {(input.Debug_thumbOnIndex ? "YES" : "NO")}\n";
            s += $"ThumbIndexDist: {input.Debug_thumbIndexDist:F3} m\n";
        }

        // Mode toggle (single toggle) diagnostics
        if (showToggle)
        {
            s += "\n[Mode Toggle]\n";
            s += $"OffHeld: {input.Debug_offHeldSec:F2} s\n";
            s += $"Armed: {(input.Debug_zArmed ? "YES" : "NO")}\n";
        }

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
