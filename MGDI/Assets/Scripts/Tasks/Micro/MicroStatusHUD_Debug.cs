using UnityEngine;
using UnityEngine.UI;

public class MicroStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showDpad = true;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro HUD: input NULL\n(Assign MicroInputThumbDpadToggle)";
            return;
        }

        string engage = input.IsEngaged ? "ON" : "OFF";
        string mode = input.ZMode ? "Z" : "XY";

        string s =
            $"Micro Engage: {engage}\n" +
            $"Mode: {mode}\n" +
            $"State: {input.DebugState}\n\n" +

            $"dTI(thumb-index): {input.Debug_dTI:F3} m\n" +
            $"dTM(thumb-middle): {input.Debug_dTM:F3} m\n";

        if (showDpad)
        {
            Vector2 d = input.Dpad;
            s += $"Dpad: ({d.x:F2}, {d.y:F2})\n";
        }

        statusText.text = s;
    }
}
