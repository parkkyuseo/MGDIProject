using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;
using UnityEngine.Windows.Speech;

public class ToolRotationTaskManager : MonoBehaviour
{
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged; // (current1Based, total)

    [Header("Roots (auto-discovered via ToolId)")]
    [SerializeField] private Transform toolsDynamicRoot;
    [SerializeField] private Transform slotsTargetsRoot;

    [Header("Grab + Twist")]
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Grabber rotation policy")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode rotationTrialGrabberMode = ProxyHandGrabber.HeldRotationMode.ExternalControl;
    [SerializeField] private ProxyHandGrabber.HeldRotationMode restoreGrabberModeOnDisable = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Progress UI (optional)")]
    [SerializeField] private UnityEngine.UI.Text progressText;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Header("Yaw Success Threshold")]
    [SerializeField] private float yawToleranceDeg = 10f;

    [Header("Twist → Yaw mapping (MACRO)")]
    [SerializeField] private float twistToYawGain = 1.5f;
    [SerializeField] private bool invertTwistToYaw = false;
    [SerializeField] private float yawMaxDegPerSec = 240f;
    [Range(0f, 1f)]
    [SerializeField] private float yawLerp = 0.15f;

    [Header("Rotation gating (MACRO)")]
    [SerializeField] private bool rotateOnlyWhenHolding = true;
    [SerializeField] private bool requireHoldingThisTool = true;

    [Header("Rotation gating (MICRO)")]
    [Tooltip("If true, micro rotation is allowed without holding (non-grasp micro).")]
    [SerializeField] private bool microAllowWithoutHolding = true;

    [Tooltip("If true, success dwell accumulates only when input is not actively driving (recommended).")]
    [SerializeField] private bool requireNotDrivingForEvaluation = true;

    [Header("Target yaw sampling")]
    [SerializeField] private float yawMinDeg = 30f;
    [SerializeField] private float yawMaxDeg = 90f;
    [SerializeField] private bool randomizeYawSign = true;

    [Header("Inter-trial behavior")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool snapYawOnSuccess = true;

    [Header("Reset")]
    [SerializeField] private bool resetToolToStartAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = false;
    [SerializeField] private string startKeyword = "start rotation";
    [SerializeField] private string restartKeyword = "restart rotation";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    // Micro controller sets this true while it is actively rotating (v != 0).
    private bool _externalDriving = false;
    public void SetExternalDriving(bool driving) => _externalDriving = driving;

    // ---------------- Runtime ----------------
    [Serializable]
    private class Item
    {
        public string id;
        public Transform tool;
        public Rigidbody toolBody;
        public Transform target;
        public Transform startParent;
        public Vector3 startPos;
        public Quaternion startRot;
    }

    private readonly List<Item> _items = new List<Item>();
    private int _trialIndex = 0;

    private float _trialTimer = 0f;
    private float _dwellTimer = 0f;

    private bool _trialRunning = false;
    private bool _inTransition = false;

    private Item _active;
    private float _startYawDeg;
    private float _twistBaselineDeg;
    private float _yawCmdDeg;
    private bool _yawCmdInit = false;

    private bool _prevMacroDriving = false;

    private KeywordRecognizer _keywordRecognizer;
    private Dictionary<string, Action> _keywordActions;

    private readonly StringBuilder _sb = new StringBuilder(512);

    public bool IsTrialRunning => _trialRunning && !_inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - _trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => _trialIndex + 1;

    // ✅ Expose active tool for micro controller
    public Transform ActiveToolTransform => _active != null ? _active.tool : null;
    public string ActiveToolId => _active != null ? _active.id : null;

    /// <summary>
    /// Micro drive permission. In non-grasp micro, this can be true even when not holding.
    /// </summary>
    public bool CanMicroDriveNow()
    {
        if (!IsTrialRunning) return false;
        if (_active == null || _active.tool == null) return false;

        if (microAllowWithoutHolding) return true;

        // If microAllowWithoutHolding is false, fall back to holding gating.
        return ShouldDriveRotationThisFrame();
    }

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        _inTransition = false;
        _trialRunning = false;
        _trialIndex = 0;

        _prevMacroDriving = false;
        _externalDriving = false;

        HideFeedbackUI();

        RebuildItemsFromScene();
        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();

        if (enableVoiceStart)
            SetupVoiceCommands();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
    }

    private void Update()
    {
        if (!_trialRunning || _inTransition) return;
        if (_active == null || _active.tool == null || _active.target == null) return;

        _trialTimer += Time.deltaTime;

        if (_trialTimer >= trialTimeoutSeconds)
        {
            StartCoroutine(EndTrialRoutine(success: false, timedOut: true));
            return;
        }

        // ---------------- MACRO rotation (twist) ----------------
        bool drive = ShouldDriveRotationThisFrame();
        bool twistReady = (remoteHand != null && remoteHand.TwistReady);

        // Macro should not overwrite when micro is actively driving.
        bool macroDriving = drive && twistReady && !_externalDriving;

        if (macroDriving)
        {
            if (!_prevMacroDriving)
                RebaselineTwistBaselineToCurrentYaw();

            UpdateYawFromTwist();
        }

        _prevMacroDriving = macroDriving;

        // ---------------- Evaluation gating ("release / stop input to evaluate") ----------------
        bool evalAllowed = true;

        if (requireNotDrivingForEvaluation)
        {
            // When macro is driving or micro is actively driving, do not accumulate dwell.
            if (macroDriving || _externalDriving)
                evalAllowed = false;
        }

        float yawErr = ComputeYawErrorDeg();

        if (progressText != null)
            UpdateProgressText(yawErr, macroDriving, evalAllowed);

        if (yawErr <= yawToleranceDeg)
        {
            if (evalAllowed)
            {
                _dwellTimer += Time.deltaTime;
                if (_dwellTimer >= dwellSeconds)
                    StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
            }
            else
            {
                _dwellTimer = 0f;
            }
        }
        else
        {
            _dwellTimer = 0f;
        }
    }

    // ---------------- Trial Flow ----------------
    private void BeginNextTrial()
    {
        if (toolsDynamicRoot == null || slotsTargetsRoot == null)
        {
            Debug.LogError("[ToolRotationTM] Missing roots (toolsDynamicRoot / slotsTargetsRoot).");
            FinishBlock();
            return;
        }

        if (remoteHand == null)
        {
            Debug.LogError("[ToolRotationTM] remoteHand is null (RemoteHandRuntime).");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && _trialIndex >= totalTrials)
        {
            if (logDebug) Debug.Log("[ToolRotationTM] Block finished.");
            FinishBlock();
            return;
        }

        if (_items.Count == 0)
        {
            RebuildItemsFromScene();
            if (_items.Count == 0)
            {
                Debug.LogError("[ToolRotationTM] No matched tool/target pairs found. Add ToolId to tools and ghosts.");
                FinishBlock();
                return;
            }
        }

        // Rotation task: manager controls yaw; grabber must NOT override rotation.
        if (grabber != null)
            grabber.SetHeldRotationMode(rotationTrialGrabberMode);

        // Active tool = round-robin through matched ids
        _active = _items[_trialIndex % _items.Count];

        EnsureActiveBody();

        // Reset active tool to its start pose at trial start (controlled start)
        ForceReleaseIfPossible();
        ResetActiveToolToStartPose();

        // Reset driving flags
        _externalDriving = false;
        _prevMacroDriving = false;

        // Record start yaw and baseline twist (valid only when TwistReady)
        _startYawDeg = _active.tool.eulerAngles.y;
        _twistBaselineDeg = remoteHand.TwistReady ? remoteHand.TwistDegrees : 0f;

        _yawCmdDeg = _startYawDeg;
        _yawCmdInit = true;

        // Sample target yaw offset and apply to target ghost (yaw-only)
        float offset = Random.Range(yawMinDeg, yawMaxDeg);
        if (randomizeYawSign && Random.value < 0.5f) offset = -offset;

        float targetYaw = _startYawDeg + offset;
        _active.target.rotation = Quaternion.Euler(0f, targetYaw, 0f);

        _trialTimer = 0f;
        _dwellTimer = 0f;
        _trialRunning = true;
        _inTransition = false;

        OnTrialChanged?.Invoke(_trialIndex + 1, totalTrials);

        if (progressText != null)
            UpdateProgressText(ComputeYawErrorDeg(), macroDriving: false, evalAllowed: true);

        if (logDebug)
        {
            Debug.Log($"[ToolRotationTM] Trial {_trialIndex + 1}/{totalTrials} tool={_active.id} targetYawOffset={offset:F1}deg tol={yawToleranceDeg:F1}deg");
        }
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (_inTransition) yield break;
        _inTransition = true;
        _trialRunning = false;

        HideFeedbackUI();

        if (success)
        {
            if (snapYawOnSuccess)
                SnapYawToTarget();

            PlaySnapSound();
            ShowStar();
            yield return new WaitForSeconds(Mathf.Max(feedbackShowSeconds, postSnapHoldSeconds));
        }
        else
        {
            ShowX();
            yield return new WaitForSeconds(feedbackShowSeconds);
        }

        HideFeedbackUI();

        if (resetToolToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            ResetActiveToolToStartPose();
        }

        _trialIndex++;
        BeginNextTrial();
    }

    // ---------------- Registry ----------------
    private void RebuildItemsFromScene()
    {
        _items.Clear();
        if (toolsDynamicRoot == null || slotsTargetsRoot == null) return;

        var toolIds = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        var toolMap = new Dictionary<string, Transform>();
        for (int i = 0; i < toolIds.Length; i++)
        {
            if (toolIds[i] == null || string.IsNullOrEmpty(toolIds[i].id)) continue;
            toolMap[toolIds[i].id] = toolIds[i].transform;
        }

        var targetIds = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        var targetMap = new Dictionary<string, Transform>();
        for (int i = 0; i < targetIds.Length; i++)
        {
            if (targetIds[i] == null || string.IsNullOrEmpty(targetIds[i].id)) continue;
            targetMap[targetIds[i].id] = targetIds[i].transform;
        }

        foreach (var kv in toolMap)
        {
            if (!targetMap.TryGetValue(kv.Key, out var tgt)) continue;

            Transform toolTf = kv.Value;

            var it = new Item
            {
                id = kv.Key,
                tool = toolTf,
                target = tgt,
                startParent = toolTf != null ? toolTf.parent : null,
                startPos = toolTf != null ? toolTf.position : Vector3.zero,
                startRot = toolTf != null ? toolTf.rotation : Quaternion.identity,
                toolBody = null
            };

            _items.Add(it);
        }

        _items.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        if (logDebug)
            Debug.Log($"[ToolRotationTM] Registry rebuilt: matched={_items.Count}");
    }

    private void EnsureActiveBody()
    {
        if (_active == null || _active.tool == null) return;
        if (_active.toolBody != null) return;

        var rb = _active.tool.GetComponent<Rigidbody>();
        if (rb == null) rb = _active.tool.GetComponentInChildren<Rigidbody>(true);
        _active.toolBody = rb;
    }

    private void ResetActiveToolToStartPose()
    {
        if (_active == null || _active.tool == null) return;

        if (_active.startParent != null)
            _active.tool.SetParent(_active.startParent, true);

        _active.tool.SetPositionAndRotation(_active.startPos, _active.startRot);
    }

    // ---------------- Drive / Gating (MACRO) ----------------
    private bool ShouldDriveRotationThisFrame()
    {
        if (!rotateOnlyWhenHolding) return true;

        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisTool) return true;

        EnsureActiveBody();
        if (_active == null || _active.toolBody == null) return false;
        if (grabber.HeldBody == null) return false;

        return grabber.HeldBody == _active.toolBody;
    }

    // ---------------- Twist->Yaw (MACRO) ----------------
    private void RebaselineTwistBaselineToCurrentYaw()
    {
        if (remoteHand == null || _active == null || _active.tool == null) return;
        if (!remoteHand.TwistReady) return;

        float twistNow = remoteHand.TwistDegrees;
        float toolYawNow = _active.tool.eulerAngles.y;

        float sign = invertTwistToYaw ? -1f : 1f;
        float denom = twistToYawGain * sign;
        if (Mathf.Abs(denom) < 1e-5f) denom = (denom >= 0f ? 1e-5f : -1e-5f);

        float yawDelta = Mathf.DeltaAngle(_startYawDeg, toolYawNow);
        _twistBaselineDeg = twistNow - (yawDelta / denom);

        _yawCmdDeg = toolYawNow;
        _yawCmdInit = true;
    }

    private void UpdateYawFromTwist()
    {
        if (remoteHand == null || _active == null || _active.tool == null) return;
        if (!remoteHand.TwistReady) return;

        float twistNow = remoteHand.TwistDegrees;
        float dTwist = Mathf.DeltaAngle(_twistBaselineDeg, twistNow);

        float sign = invertTwistToYaw ? -1f : 1f;
        float desiredYaw = _startYawDeg + dTwist * twistToYawGain * sign;

        float dt = Mathf.Max(Time.deltaTime, 1f / 120f);

        if (!_yawCmdInit)
        {
            _yawCmdDeg = desiredYaw;
            _yawCmdInit = true;
        }

        float maxStep = Mathf.Max(10f, yawMaxDegPerSec) * dt;
        float step = Mathf.DeltaAngle(_yawCmdDeg, desiredYaw);
        step = Mathf.Clamp(step, -maxStep, maxStep);

        float yawNext = _yawCmdDeg + step;

        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(yawLerp), dt * 60f);
        _yawCmdDeg = Mathf.LerpAngle(_yawCmdDeg, yawNext, k);

        _active.tool.rotation = Quaternion.Euler(0f, _yawCmdDeg, 0f);
    }

    private float ComputeYawErrorDeg()
    {
        if (_active == null || _active.tool == null || _active.target == null) return float.MaxValue;
        float toolYaw = _active.tool.eulerAngles.y;
        float targetYaw = _active.target.eulerAngles.y;
        return Mathf.Abs(Mathf.DeltaAngle(toolYaw, targetYaw));
    }

    private void SnapYawToTarget()
    {
        if (_active == null || _active.tool == null || _active.target == null) return;
        float targetYaw = _active.target.eulerAngles.y;
        _active.tool.rotation = Quaternion.Euler(0f, targetYaw, 0f);
        _yawCmdDeg = targetYaw;
        _yawCmdInit = true;
    }

    // ---------------- UI ----------------
    private void UpdateProgressText(float yawErrDeg, bool macroDriving, bool evalAllowed)
    {
        if (progressText == null) return;

        _sb.Length = 0;
        string toolName = (_active != null) ? _active.id : "N/A";

        _sb.AppendLine($"Rotation: {toolName}");
        _sb.AppendLine($"Trial: {_trialIndex + 1}/{totalTrials}");
        _sb.AppendLine($"Time: {TrialTimeRemainingSec:F1}s");
        _sb.AppendLine($"YawErr: {yawErrDeg:F1}°  (tol {yawToleranceDeg:F1}°)");

        bool holding = (grabber != null && grabber.IsHolding);
        bool twistReady = (remoteHand != null && remoteHand.TwistReady);

        _sb.AppendLine($"Holding: {(holding ? "YES" : "NO")}  TwistReady: {(twistReady ? "YES" : "NO")}");
        _sb.AppendLine($"MacroDrive: {(macroDriving ? "ON" : "OFF")}  MicroDrive: {(_externalDriving ? "ON" : "OFF")}  EvalAllowed: {(evalAllowed ? "YES" : "NO")}");

        progressText.text = _sb.ToString();
    }

    // ---------------- Feedback ----------------
    private void PlaySnapSound()
    {
        if (audioSource != null && snapClip != null)
            audioSource.PlayOneShot(snapClip);
    }

    private void ShowStar()
    {
        if (starUI != null) starUI.SetActive(true);
        if (xUI != null) xUI.SetActive(false);
    }

    private void ShowX()
    {
        if (xUI != null) xUI.SetActive(true);
        if (starUI != null) starUI.SetActive(false);
    }

    private void HideFeedbackUI()
    {
        if (starUI != null) starUI.SetActive(false);
        if (xUI != null) xUI.SetActive(false);
    }

    // ---------------- Grabber helpers ----------------
    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
            grabber.ForceRelease();
    }

    // ---------------- Voice ----------------
    private void SetupVoiceCommands()
    {
        if (_keywordRecognizer != null) return;

        _keywordActions = new Dictionary<string, Action>
        {
            { startKeyword.ToLower(), StartBlock },
            { restartKeyword.ToLower(), StartBlock }
        };

        _keywordRecognizer = new KeywordRecognizer(_keywordActions.Keys.ToArray());
        _keywordRecognizer.OnPhraseRecognized += args =>
        {
            string key = args.text.ToLower();
            if (_keywordActions.TryGetValue(key, out var action))
                action.Invoke();
        };
        _keywordRecognizer.Start();
    }

    private void FinishBlock()
    {
        _trialRunning = false;
        _inTransition = false;
        HideFeedbackUI();
        try { OnBlockFinished?.Invoke(); } catch { }
    }

    private void OnDisable()
    {
        if (grabber != null)
            grabber.SetHeldRotationMode(restoreGrabberModeOnDisable);

        if (_keywordRecognizer != null)
        {
            _keywordRecognizer.Stop();
            _keywordRecognizer.Dispose();
            _keywordRecognizer = null;
        }
    }
}
