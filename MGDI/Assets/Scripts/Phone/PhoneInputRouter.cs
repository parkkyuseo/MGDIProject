using System;
using UnityEngine;

public class PhoneInputRouter : MonoBehaviour
{
    public event Action OnModeToggle;

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
    private int _lastSeenModeToggleId = int.MinValue;

    void Awake()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
        _lastSeenModeToggleId = int.MinValue;
    }

    void Update()
    {
        if (phoneRx == null) return;

        bool hold = phoneRx.LatestHold;
        bool toggleState = phoneRx.LatestToggle;

        Grab = (mode == Mode.Macro) ? hold : toggleState;

        if (mode == Mode.Micro)
        {
            Axis = new Vector2(phoneRx.LatestAx, phoneRx.LatestAy);
            AxisActive = phoneRx.LatestDrag;

            if (phoneRx.HasPhonePose)
            {
                int id = phoneRx.LatestModeToggleId;

                if (_lastSeenModeToggleId == int.MinValue)
                {
                    _lastSeenModeToggleId = id; // first sync, no event
                }
                else if (id != _lastSeenModeToggleId)
                {
                    _lastSeenModeToggleId = id;
                    _modeToggleBuffered = true;
                    try { OnModeToggle?.Invoke(); } catch { }
                }
            }
        }
        else
        {
            Axis = Vector2.zero;
            AxisActive = false;
            _modeToggleBuffered = false;

            if (phoneRx.HasPhonePose && _lastSeenModeToggleId == int.MinValue)
                _lastSeenModeToggleId = phoneRx.LatestModeToggleId;
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

        if (phoneRx != null && phoneRx.HasPhonePose)
            _lastSeenModeToggleId = phoneRx.LatestModeToggleId;

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
