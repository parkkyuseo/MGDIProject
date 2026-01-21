using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showAxisValue = true;
    [SerializeField] private bool showRawT = true;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input missing";
            return;
        }

        string mode = input.Mode.ToString(); // X / Y / Z
        string axis = showAxisValue ? $"Axis: {input.AxisValue:F2}\n" : "";

        // Raw t is only available if you enable debug in the input script.
        // If debug is off, this line will just show 0.00 (or last value).
        string tLine = "";
        if (showRawT)
        {
            // This relies on the debug field existing in MicroThumbIndexSliderInput.
            // If you removed debug fields, set showRawT=false.
            var f = typeof(MicroThumbIndexSliderInput).GetField("debug_t", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (f != null)
            {
                float t = (float)f.GetValue(input);
                tLine = $"t: {t:F2}\n";
            }
        }

        string taps = "";
        if (input.SingleTapThisFrame) taps += "Tap: SINGLE\n";
        if (input.DoubleTapThisFrame) taps += "Tap: DOUBLE\n";

        statusText.text =
            $"Micro Mode: {mode}\n" +
            axis +
            tLine +
            taps;
    }
}
