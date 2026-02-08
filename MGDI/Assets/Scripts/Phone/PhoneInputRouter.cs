using UnityEngine;

public class PhoneInputRouter : MonoBehaviour
{
    public enum Mode { Macro = 0, Micro = 1 }


    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;

    [Header("Mode")]
    [SerializeField] private Mode mode = Mode.Macro; // default Macro

    [Header("Debug")]
    [SerializeField] private bool logMode = false;

    public Mode CurrentMode => mode;

    public bool Grab { get; private set; }

    public Vector2 Axis { get; private set; }       // Micro only
    public bool AxisActive { get; private set; }    // Micro only

    private bool _modeToggleBuffered = false;

    void Awake()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
    }

    void Update()
    {
        if (phoneRx == null) return;

        bool hold = phoneRx.LatestHold;
        bool toggle = phoneRx.LatestToggle;

        Grab = (mode == Mode.Macro) ? hold : toggle;

        if (mode == Mode.Micro)
        {
            Axis = new Vector2(phoneRx.LatestAx, phoneRx.LatestAy);
            AxisActive = phoneRx.LatestDrag;

            if (phoneRx.LatestModeToggle)
                _modeToggleBuffered = true;
        }
        else
        {
            Axis = Vector2.zero;
            AxisActive = false;
            _modeToggleBuffered = false;
        }
    }

    public void SetModeMacro()
    {
        mode = Mode.Macro;
        if (logMode) DebugHUD.Log("[Router] Macro");
    }

    public void SetModeMicro()
    {
        mode = Mode.Micro;
        if (logMode) DebugHUD.Log("[Router] Micro");
    }

    public bool TryConsumeModeToggle()
    {
        if (mode != Mode.Micro) return false;
        if (!_modeToggleBuffered) return false;
        _modeToggleBuffered = false;
        return true;
    }
}
