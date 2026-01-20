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

            $"RemoteHand assigned: {input.DebugRemoteHandAssigned}\n" +
            $"remoteByIndex len: {input.DebugRemoteByIndexLen}\n" +
            $"Has min joints: {input.DebugHasMinJoints}\n\n" +

            $"Joints ok (0,4,5,8,9,12,13,17):\n" +
            $"Wrist(0): {input.DebugWristOk}\n" +
            $"ThumbTip(4): {input.DebugThumbTipOk}\n" +
            $"IndexMCP(5): {input.DebugIndexMcpOk}\n" +
            $"IndexTip(8): {input.DebugIndexTipOk}\n" +
            $"MiddleMCP(9): {input.DebugMiddleMcpOk}\n" +
            $"MiddleTip(12): {input.DebugMiddleTipOk}\n" +
            $"RingMCP(13): {input.DebugRingMcpOk}\n" +
            $"PinkyMCP(17): {input.DebugPinkyMcpOk}\n\n" +

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
