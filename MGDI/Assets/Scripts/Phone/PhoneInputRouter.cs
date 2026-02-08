using UnityEngine;

public class PhoneInputRouter : MonoBehaviour
{
    public enum Mode { Macro = 0, Micro = 1 }

    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;

    [Header("Mode")]
    [SerializeField] private Mode mode = Mode.Macro; // default Macro

    [Header("Debug")]
    [SerializeField] private bool logModeAndGrab = false;

    // Outputs
    public Mode CurrentMode => mode;
    public bool Grab { get; private set; }     // Macro: hold, Micro: toggle

    // One-shot swipe buffer (Micro only)
    private int _swipeBuffered = 0;            // 0 none, 1 up,2 down,3 left,4 right
    private bool _lastGrab;

    void Awake()
    {
        if (phoneRx == null)
            phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();

        // default Macro
        mode = Mode.Macro;
    }

    void Update()
    {
        if (phoneRx == null) return;

        bool hold = phoneRx.LatestHold;
        bool toggle = phoneRx.LatestToggle;
        int swipe = phoneRx.LatestSwipe;

        bool newGrab = (mode == Mode.Macro) ? hold : toggle;

        Grab = newGrab;

        if (logModeAndGrab && newGrab != _lastGrab)
        {
            _lastGrab = newGrab;
            DebugHUD.Log($"[Router] mode={mode} grab={newGrab}");
        }

        // Buffer swipes only in Micro mode
        if (mode == Mode.Micro && swipe != 0)
        {
            _swipeBuffered = swipe;
        }
    }

    // Explicit mode switch API (call from StudyFlowController later)
    public void SetModeMacro()
    {
        mode = Mode.Macro;
        if (logModeAndGrab) DebugHUD.Log("[Router] SetMode -> Macro");
    }

    public void SetModeMicro()
    {
        mode = Mode.Micro;
        if (logModeAndGrab) DebugHUD.Log("[Router] SetMode -> Micro");
    }

    public void SetMode(Mode m)
    {
        mode = m;
        if (logModeAndGrab) DebugHUD.Log($"[Router] SetMode -> {mode}");
    }

    // One-shot swipe consume (Micro only)
    public bool TryConsumeSwipe(out int dir)
    {
        dir = 0;
        if (mode != Mode.Micro) return false;

        if (_swipeBuffered == 0) return false;

        dir = _swipeBuffered;
        _swipeBuffered = 0;
        return true;
    }
}
