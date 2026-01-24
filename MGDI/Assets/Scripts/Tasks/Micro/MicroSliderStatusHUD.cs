using UnityEngine;
using UnityEngine.UI;

public class MicroSliderStatusHUD : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroThumbIndexSliderInput input;
    [SerializeField] private Text statusText;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input missing";
            return;
        }

        string s = "";
        s += $"Mode: {input.Mode}\n";
        s += $"Axis: {input.AxisValue:F2}\n";
        s += $"t: {input.Debug_t:F3}\n";

        // New thumb-based interaction is driven by thumb-on-index state.
        // If the property doesn't exist in this build, just omit it.
        // (Debug_thumbOnIndex exists in the new script you pasted.)
        s += $"ThumbOnIndex: {(input.Debug_thumbOnIndex ? "YES" : "NO")}\n";

        statusText.text = s;
    }
}
