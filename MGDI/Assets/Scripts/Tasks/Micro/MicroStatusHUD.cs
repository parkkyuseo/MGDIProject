using UnityEngine;
using UnityEngine.UI;

public class MicroStatusHUD : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private MicroInputThumbDpadToggle input;
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showDpadNumbers = true;

    void Update()
    {
        if (statusText == null) return;

        if (input == null)
        {
            statusText.text = "Micro: (input missing)";
            return;
        }

        string engage = input.IsEngaged ? "ON" : "OFF";
        string mode = input.ZMode ? "Z" : "XY";

        if (!showDpadNumbers)
        {
            statusText.text = $"Micro Engage: {engage}\nMode: {mode}";
            return;
        }

        Vector2 d = input.Dpad;
        statusText.text =
            $"Micro Engage: {engage}\n" +
            $"Mode: {mode}\n" +
            $"Dpad: ({d.x:F2}, {d.y:F2})";
    }
}
