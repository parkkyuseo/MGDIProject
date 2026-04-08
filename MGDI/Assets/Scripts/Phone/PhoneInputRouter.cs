using System;
using UnityEngine;

public class PhoneInputRouter : MonoBehaviour
{
    public event Action OnModeToggle;
    public event Action OnTripleTap;

    public enum Mode { Macro = 0, Micro = 1 }

    [Header("Refs")]
    [SerializeField] private PhonePoseStreamReceiver phoneRx;

    [Header("Mode")]
    [SerializeField] private Mode mode = Mode.Macro; // default Macro

    [Header("Debug")]
    [SerializeField] private bool logMode = false;

    [Header("Tap Disambiguation")]
    [Tooltip("If true, mode toggle (double-tap) is emitted with a short delay and canceled when triple-tap is detected.")]
    [SerializeField] private bool suppressDoubleToggleOnTripleTap = true;
    [SerializeField] private float modeToggleEmitDelaySec = 0.20f;
    [SerializeField] private float tripleTapCancelWindowSec = 0.35f;

    public Mode CurrentMode => mode;
    public bool IsInputSuppressed => _inputSuppressed;

    public bool Grab { get; private set; }

    public Vector2 Axis { get; private set; }       // Micro only
    public bool AxisActive { get; private set; }    // Micro only
    public bool HoldActive { get; private set; }    // Micro only (screen touch)

    private bool _modeToggleBuffered = false;
    private bool _tripleTapBuffered = false;
    private int _lastSeenModeToggleId = int.MinValue;
    private int _lastSeenTripleTapId = int.MinValue;
    private bool _pendingModeToggle = false;
    private float _pendingModeToggleReadyAt = -1f;
    private float _lastTripleTapAt = float.NegativeInfinity;
    private bool _inputSuppressed = false;
    private bool _ignoreMicroGrabUntilToggleChanges = false;
    private bool _microGrabBaselineToggle = false;

    void Awake()
    {
        if (phoneRx == null) phoneRx = FindFirstObjectByType<PhonePoseStreamReceiver>();
        _lastSeenModeToggleId = int.MinValue;
        _lastSeenTripleTapId = int.MinValue;
    }

    void Update()
    {
        if (phoneRx == null) return;

        if (_inputSuppressed)
        {
            if (phoneRx.HasPhonePose)
            {
                _lastSeenModeToggleId = phoneRx.LatestModeToggleId;
                _lastSeenTripleTapId = phoneRx.LatestTripleTapId;
            }

            ClearOutputs();
            return;
        }

        bool hold = phoneRx.LatestHold;
        bool toggleState = phoneRx.LatestToggle;

        bool microGrab = toggleState;
        if (mode == Mode.Micro && _ignoreMicroGrabUntilToggleChanges)
        {
            if (toggleState == _microGrabBaselineToggle)
            {
                microGrab = false;
            }
            else
            {
                _ignoreMicroGrabUntilToggleChanges = false;
                microGrab = toggleState;
            }
        }

        Grab = (mode == Mode.Macro) ? hold : microGrab;

        if (phoneRx.HasPhonePose)
        {
            int tripleId = phoneRx.LatestTripleTapId;

            if (_lastSeenTripleTapId == int.MinValue)
            {
                _lastSeenTripleTapId = tripleId; // first sync, no event
            }
            else if (tripleId != _lastSeenTripleTapId)
            {
                _lastSeenTripleTapId = tripleId;
                _lastTripleTapAt = Time.unscaledTime;
                _tripleTapBuffered = true;
                CancelModeToggleBuffer();
                try { OnTripleTap?.Invoke(); } catch { }
            }
        }

        if (mode == Mode.Micro)
        {
            Axis = new Vector2(phoneRx.LatestAx, phoneRx.LatestAy);
            AxisActive = phoneRx.LatestDrag;
            HoldActive = hold;

            if (phoneRx.HasPhonePose)
            {
                int modeId = phoneRx.LatestModeToggleId;

                if (_lastSeenModeToggleId == int.MinValue)
                {
                    _lastSeenModeToggleId = modeId; // first sync, no event
                }
                else if (modeId != _lastSeenModeToggleId)
                {
                    _lastSeenModeToggleId = modeId;
                    QueueOrEmitModeToggle();
                }

                FlushPendingModeToggle();
            }
        }
        else
        {
            Axis = Vector2.zero;
            AxisActive = false;
            HoldActive = false;
            CancelModeToggleBuffer();

            if (phoneRx.HasPhonePose && _lastSeenModeToggleId == int.MinValue)
                _lastSeenModeToggleId = phoneRx.LatestModeToggleId;
        }
    }

    public void SetModeMacro()
    {
        mode = Mode.Macro;
        _ignoreMicroGrabUntilToggleChanges = false;
        if (logMode) DebugHUD.Log("[Router] Macro");
    }

    public void SetModeMicro()
    {
        mode = Mode.Micro;

        if (phoneRx != null && phoneRx.HasPhonePose)
        {
            _lastSeenModeToggleId = phoneRx.LatestModeToggleId;
            _lastSeenTripleTapId = phoneRx.LatestTripleTapId;
            _microGrabBaselineToggle = phoneRx.LatestToggle;
        }
        else
        {
            _microGrabBaselineToggle = false;
        }

        _ignoreMicroGrabUntilToggleChanges = true;
        Grab = false;

        if (logMode) DebugHUD.Log("[Router] Micro");
    }

    public bool TryConsumeModeToggle()
    {
        if (mode != Mode.Micro) return false;
        FlushPendingModeToggle();
        if (!_modeToggleBuffered) return false;
        _modeToggleBuffered = false;
        return true;
    }

    public bool TryConsumeTripleTap()
    {
        if (!_tripleTapBuffered) return false;
        _tripleTapBuffered = false;
        return true;
    }

    public void SetInputSuppressed(bool suppressed)
    {
        if (_inputSuppressed == suppressed)
            return;

        _inputSuppressed = suppressed;

        if (suppressed)
        {
            if (phoneRx != null && phoneRx.HasPhonePose)
            {
                _lastSeenModeToggleId = phoneRx.LatestModeToggleId;
                _lastSeenTripleTapId = phoneRx.LatestTripleTapId;
            }

            ClearOutputs();
        }
    }

    private void QueueOrEmitModeToggle()
    {
        if (!suppressDoubleToggleOnTripleTap)
        {
            _modeToggleBuffered = true;
            try { OnModeToggle?.Invoke(); } catch { }
            return;
        }

        _pendingModeToggle = true;
        _pendingModeToggleReadyAt = Time.unscaledTime + Mathf.Max(0f, modeToggleEmitDelaySec);
    }

    private void FlushPendingModeToggle()
    {
        if (!_pendingModeToggle) return;
        if (Time.unscaledTime < _pendingModeToggleReadyAt) return;

        _pendingModeToggle = false;

        float cancelWindow = Mathf.Max(0f, tripleTapCancelWindowSec);
        bool canceledByRecentTriple = (Time.unscaledTime - _lastTripleTapAt) <= cancelWindow;
        if (canceledByRecentTriple) return;

        _modeToggleBuffered = true;
        try { OnModeToggle?.Invoke(); } catch { }
    }

    private void CancelModeToggleBuffer()
    {
        _modeToggleBuffered = false;
        _pendingModeToggle = false;
        _pendingModeToggleReadyAt = -1f;
    }

    private void ClearOutputs()
    {
        Grab = false;
        Axis = Vector2.zero;
        AxisActive = false;
        HoldActive = false;
        _tripleTapBuffered = false;
        CancelModeToggleBuffer();
    }
}
