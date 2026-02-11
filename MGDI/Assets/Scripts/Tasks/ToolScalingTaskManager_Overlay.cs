using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

public class ToolScalingTaskManager_Overlay : MonoBehaviour
{
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged; // (current1Based, total)

    public enum ControlMode
    {
        MacroWristDiagonal = 0,
        MicroThumbIndexSlider = 1
    }

    [Header("Overlay Roots (auto-matched by ToolId)")]
    [Tooltip("Root that contains the CURRENT overlays to be scaled (one per tool, with ToolId).")]
    [SerializeField] private Transform overlaysCurrentRoot;

    [Tooltip("Root that contains the TARGET overlays to show goal scale (one per tool, with ToolId).")]
    [SerializeField] private Transform overlaysTargetRoot;

    [Header("Input References")]
    [Tooltip("Grabber used for holding gating and 'release to evaluate'.")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime used for macro wrist motion.")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Tooltip("Micro slider input used for micro scaling.")]
    [SerializeField] private MicroThumbIndexSliderInput microSliderInput;

    [Header("Control Mode")]
    [SerializeField] private ControlMode controlMode = ControlMode.MacroWristDiagonal;

    [Tooltip("If true, scale changes happen only while holding.")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, scale changes only when holding THIS trial's tool rigidbody.")]
    [SerializeField] private bool requireHoldingThisTool = true;

    [Header("Evaluate on Release")]
    [Tooltip("If true, success/dwell evaluation occurs only when NOT holding (release-to-evaluate).")]
    [SerializeField] private bool requireNotHolding = true;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Header("Success Threshold (FACTOR)")]
    [Tooltip("Success if abs(currentFactor - targetFactor) <= tolerance.")]
    [SerializeField] private float scaleFactorTolerance = 0.05f;

    [Header("Target Factor Sampling (relative to baseline)")]
    [SerializeField] private float targetFactorMin = 0.70f;
    [SerializeField] private float targetFactorMax = 1.60f;

    [Header("Scale factor clamp (relative to baseline)")]
    [SerializeField] private float minScaleFactor = 0.60f;
    [SerializeField] private float maxScaleFactor = 1.80f;

    // ---------------- Macro: wrist diagonal -> scale ----------------
    [Header("Macro: Diagonal (up + side) movement -> scale")]
    [Tooltip("Axis based on camera up + side. Recommended true.")]
    [SerializeField] private bool axisFromCamera = true;

    [Tooltip("If true, left hand uses cam.left so 'up+away from body side' feels consistent.")]
    [SerializeField] private bool flipSideAxisForLeftHand = true;

    [Tooltip("Scale mapping gain for exp(gain * accumulatedMeters). Typical 3~6.")]
    [SerializeField] private float moveToScaleGain = 4.0f;

    [Tooltip("Ignore very small axis movement (meters).")]
    [SerializeField] private float moveDeadZoneMeters = 0.0025f;

    // ---------------- Micro: slider -> scale ----------------
    [Header("Micro: Slider -> scale")]
    [Tooltip("How fast slider changes accumulate along the same exp(gain * accum) mapping. Units: meters/sec (virtual).")]
    [SerializeField] private float microAccumMetersPerSec = 0.08f;

    [Tooltip("Ignore small slider output. (AxisValue in [-1..1])")]
    [SerializeField] private float microDeadZone = 0.08f;

    // ---------------- Shared smoothing ----------------
    [Range(0f, 1f)]
    [Tooltip("Extra smoothing on scale factor command (0=no smoothing, 1=very slow).")]
    [SerializeField] private float scaleLerp = 0.15f;

    [Header("Inter-trial behavior")]
    [SerializeField] private bool snapOnSuccess = false;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetScaleAfterTrial = true;
    [SerializeField] private bool forceReleaseAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    // ---- Forced active (Workflow integration) ----
    [SerializeField] private bool finishBlockAfterOneSuccessWhenForced = true;
    private string _forcedActiveId = null;

    public void SetForcedActiveId(string id) => _forcedActiveId = string.IsNullOrEmpty(id) ? null : id;
    public void ClearForcedActiveId() => _forcedActiveId = null;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Progress UI (optional)")]
    [SerializeField] private UnityEngine.UI.Text progressText;
    [SerializeField] private bool showDebugLine = true;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    // ---------------- Runtime ----------------
    [Serializable]
    private class Item
    {
        public string id;

        // The real tool (for holding gating)
        public Transform tool;
        public Rigidbody toolBody;

        // Overlays
        public Transform currentOverlay; // scaled by user
        public Transform targetOverlay;  // shows target

        // Baseline local scales (so we can apply factor without cumulative drift)
        public Vector3 baseCurrentLocalScale;
        public Vector3 baseTargetLocalScale;

        // Per-trial target factor
        public float targetFactor;

        // Drive state (per trial)
        public float axisAccum;
        public float scaleFactorCmd;
        public bool scaleFactorInit;

        // Macro wrist state
        public Vector3 wristPrev;
        public bool haveWristPrev;
    }

    private readonly List<Item> _items = new List<Item>();
    private Item _active;

    private int _trialIndex = 0;
    private float _trialTimer = 0f;
    private float _dwellTimer = 0f;
    private bool _trialRunning = false;
    private bool _inTransition = false;

    private readonly StringBuilder _sb = new StringBuilder(512);

    public bool IsTrialRunning => _trialRunning && !_inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - _trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => _trialIndex + 1;

    // --------- Public helpers (optional) ----------
    public void SetControlMode(ControlMode mode) => controlMode = mode;

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        _inTransition = false;
        _trialRunning = false;
        _trialIndex = 0;

        HideFeedbackUI();

        RebuildItemsFromScene();
        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();
        RebuildItemsFromScene();
    }

    private void Update()
    {
        if (!_trialRunning || _inTransition) return;
        if (_active == null) return;

        _trialTimer += Time.deltaTime;

        if (_trialTimer >= trialTimeoutSeconds)
        {
            StartCoroutine(EndTrialRoutine(success: false, timedOut: true));
            return;
        }

        // 1) Update scale command (while driving conditions satisfied)
        bool driving = ShouldDriveScalingThisFrame();
        bool readyMacro = HasWrist();
        bool readyMicro = (microSliderInput != null);

        bool canDriveThisFrame =
            driving &&
            ((controlMode == ControlMode.MacroWristDiagonal && readyMacro) ||
             (controlMode == ControlMode.MicroThumbIndexSlider && readyMicro));

        if (canDriveThisFrame)
        {
            if (controlMode == ControlMode.MacroWristDiagonal)
                UpdateScaleFromDiagonalWristMotion();
            else
                UpdateScaleFromMicroSlider();
        }
        else
        {
            // When not driving, clear macro-only state so next drive won't jump
            _active.haveWristPrev = false;
        }

        // 2) Evaluate only after release (if enabled)
        bool evalAllowed = true;
        if (requireNotHolding && grabber != null && grabber.IsHolding)
            evalAllowed = false;

        float curFactor = CurrentFactor();
        float err = Mathf.Abs(curFactor - _active.targetFactor);

        UpdateProgressUI(curFactor, err, canDriveThisFrame, evalAllowed);

        if (err <= scaleFactorTolerance)
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
        if (overlaysCurrentRoot == null || overlaysTargetRoot == null)
        {
            Debug.LogError("[ToolScaleTM] Missing overlay roots (overlaysCurrentRoot / overlaysTargetRoot).");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && _trialIndex >= totalTrials)
        {
            if (logDebug) Debug.Log("[ToolScaleTM] Block finished.");
            FinishBlock();
            return;
        }

        if (_items.Count == 0)
        {
            RebuildItemsFromScene();
            if (_items.Count == 0)
            {
                Debug.LogError("[ToolScaleTM] No matched overlay pairs found. Ensure ToolId on current+target overlays.");
                FinishBlock();
                return;
            }
        }

        // Active tool cycles through available ids
        _active = null;

        if (!string.IsNullOrEmpty(_forcedActiveId))
        {
            _active = _items.Find(it => it != null && it.id == _forcedActiveId);
            if (_active == null && logDebug)
                Debug.LogWarning($"[ScaleCore] ForcedActiveId '{_forcedActiveId}' not found. Falling back.");
        }

        if (_active == null)
            _active = _items[trialIndex % _items.Count];

        // Reset state
        _active.axisAccum = 0f;
        _active.scaleFactorCmd = 1f;
        _active.scaleFactorInit = true;
        _active.haveWristPrev = false;

        // Sample target factor
        float tf = Random.Range(Mathf.Min(targetFactorMin, targetFactorMax), Mathf.Max(targetFactorMin, targetFactorMax));
        tf = Mathf.Clamp(tf, minScaleFactor, maxScaleFactor);
        _active.targetFactor = tf;

        // Reset current overlay to baseline
        if (_active.currentOverlay != null)
            _active.currentOverlay.localScale = _active.baseCurrentLocalScale;

        // Set target overlay to baseline * targetFactor
        if (_active.targetOverlay != null)
            _active.targetOverlay.localScale = _active.baseTargetLocalScale * _active.targetFactor;

        // Optionally show only active overlays (recommended for clarity)
        SetOnlyActiveOverlaysVisible(_active.id);

        _trialTimer = 0f;
        _dwellTimer = 0f;
        _trialRunning = true;
        _inTransition = false;

        OnTrialChanged?.Invoke(_trialIndex + 1, totalTrials);

        if (logDebug)
        {
            Debug.Log($"[ToolScaleTM] Trial {_trialIndex + 1}/{totalTrials} id={_active.id} targetFactor={_active.targetFactor:F2} tol={scaleFactorTolerance:F3}");
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
            if (forceReleaseAfterTrial) ForceReleaseIfPossible();

            if (snapOnSuccess && _active != null && _active.currentOverlay != null)
            {
                _active.currentOverlay.localScale = _active.baseCurrentLocalScale * _active.targetFactor;
            }

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

        if (resetScaleAfterTrial && _active != null && _active.currentOverlay != null)
        {
            _active.currentOverlay.localScale = _active.baseCurrentLocalScale;
        }

        if (forceReleaseAfterTrial) ForceReleaseIfPossible();

        if (success && !string.IsNullOrEmpty(_forcedActiveId) && finishBlockAfterOneSuccessWhenForced)
        {
            FinishBlock();
            yield break;
        }
        
        _trialIndex++;
        _inTransition = false;
        BeginNextTrial();
    }

    // ---------------- Registry ----------------
    private void RebuildItemsFromScene()
    {
        _items.Clear();

        if (overlaysCurrentRoot == null || overlaysTargetRoot == null) return;

        // current overlays: id -> ToolId transform
        var curIds = overlaysCurrentRoot.GetComponentsInChildren<ToolId>(true);
        var curMap = new Dictionary<string, Transform>();
        for (int i = 0; i < curIds.Length; i++)
        {
            if (curIds[i] == null || string.IsNullOrEmpty(curIds[i].id)) continue;
            curMap[curIds[i].id] = curIds[i].transform;
        }

        // target overlays: id -> ToolId transform
        var tgtIds = overlaysTargetRoot.GetComponentsInChildren<ToolId>(true);
        var tgtMap = new Dictionary<string, Transform>();
        for (int i = 0; i < tgtIds.Length; i++)
        {
            if (tgtIds[i] == null || string.IsNullOrEmpty(tgtIds[i].id)) continue;
            tgtMap[tgtIds[i].id] = tgtIds[i].transform;
        }

        // OPTIONAL: tool bodies for holding gating (pull from Tools_Dynamic by matching id)
        Dictionary<string, Rigidbody> toolBodiesById = BuildToolBodiesById();

        foreach (var kv in curMap)
        {
            if (!tgtMap.TryGetValue(kv.Key, out var tgt)) continue;

            var cur = kv.Value;

            var it = new Item
            {
                id = kv.Key,
                currentOverlay = cur,
                targetOverlay = tgt,
                baseCurrentLocalScale = cur != null ? cur.localScale : Vector3.one,
                baseTargetLocalScale = tgt != null ? tgt.localScale : Vector3.one,
                targetFactor = 1f,
                axisAccum = 0f,
                scaleFactorCmd = 1f,
                scaleFactorInit = true,
                haveWristPrev = false,
                wristPrev = Vector3.zero,
                tool = null,
                toolBody = null
            };

            // Fill tool body if available (for requireHoldingThisTool gating)
            if (toolBodiesById != null && toolBodiesById.TryGetValue(it.id, out var rb))
            {
                it.toolBody = rb;
                it.tool = rb != null ? rb.transform : null;
            }

            _items.Add(it);
        }

        _items.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        if (logDebug)
        {
            Debug.Log($"[ToolScaleTM] Registry rebuilt: current={curMap.Count}, target={tgtMap.Count}, matched={_items.Count}");
        }
    }

    private Dictionary<string, Rigidbody> BuildToolBodiesById()
    {
        // If you want holding gating to be specific per tool, we need a mapping id -> Rigidbody.
        // This tries to find them under a standard "Tools_Dynamic" root next to overlays, but you can also assign a root.
        // If not found, requireHoldingThisTool should be turned off.
        Transform toolsRoot = null;

        // heuristic: look upward for a sibling named "Tools_Dynamic"
        if (overlaysCurrentRoot != null && overlaysCurrentRoot.parent != null)
        {
            var p = overlaysCurrentRoot.parent;
            var maybe = p.Find("Tools_Dynamic");
            if (maybe != null) toolsRoot = maybe;
        }

        if (toolsRoot == null) return null;

        var map = new Dictionary<string, Rigidbody>();
        var ids = toolsRoot.GetComponentsInChildren<ToolId>(true);
        foreach (var tid in ids)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;

            Rigidbody rb = tid.GetComponentInParent<Rigidbody>();
            if (rb == null) rb = tid.GetComponentInChildren<Rigidbody>(true);
            if (rb == null) continue;

            map[tid.id] = rb;
        }
        return map;
    }

    // ---------------- Drive / Gating ----------------
    private bool ShouldDriveScalingThisFrame()
    {
        // If not gating on holding, allow always.
        if (!scaleOnlyWhenHolding) return true;

        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisTool) return true;

        if (_active == null) return false;
        if (_active.toolBody == null) return false;
        if (grabber.HeldBody == null) return false;

        return grabber.HeldBody == _active.toolBody;
    }

    private bool HasWrist()
    {
        return remoteHand != null &&
               remoteHand.remoteByIndex != null &&
               remoteHand.remoteByIndex.Length > 0 &&
               remoteHand.remoteByIndex[0] != null;
    }

    private Vector3 GetMacroAxis()
    {
        Vector3 axis;

        if (axisFromCamera && Camera.main != null)
        {
            Transform cam = Camera.main.transform;

            Vector3 side = cam.right;
            if (flipSideAxisForLeftHand && remoteHand != null && remoteHand.isLeft)
                side = -side;

            axis = cam.up + side;
        }
        else
        {
            axis = Vector3.up + Vector3.right;
            if (flipSideAxisForLeftHand && remoteHand != null && remoteHand.isLeft)
                axis = Vector3.up + Vector3.left;
        }

        if (axis.sqrMagnitude < 1e-8f) axis = Vector3.up;
        return axis.normalized;
    }

    private void UpdateScaleFromDiagonalWristMotion()
    {
        if (_active == null || _active.currentOverlay == null) return;

        Vector3 w = remoteHand.remoteByIndex[0].position;

        if (!_active.haveWristPrev)
        {
            _active.wristPrev = w;
            _active.haveWristPrev = true;
            return;
        }

        Vector3 dp = w - _active.wristPrev;
        _active.wristPrev = w;

        Vector3 axis = GetMacroAxis();
        float delta = Vector3.Dot(dp, axis);

        if (Mathf.Abs(delta) < Mathf.Max(0f, moveDeadZoneMeters))
            return;

        _active.axisAccum += delta;
        ApplyAccumToScaleFactor();
    }

    private void UpdateScaleFromMicroSlider()
    {
        if (_active == null || _active.currentOverlay == null) return;
        if (microSliderInput == null) return;

        float v = microSliderInput.AxisValue; // [-1..1]
        if (Mathf.Abs(v) < Mathf.Max(0f, microDeadZone))
            return;

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        // Convert slider output into "virtual meters" accumulation
        float delta = v * microAccumMetersPerSec * dt;
        _active.axisAccum += delta;

        ApplyAccumToScaleFactor();
    }

    private void ApplyAccumToScaleFactor()
    {
        if (_active == null || _active.currentOverlay == null) return;

        float desiredFactor = Mathf.Exp(moveToScaleGain * _active.axisAccum);
        desiredFactor = Mathf.Clamp(desiredFactor, minScaleFactor, maxScaleFactor);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(scaleLerp), dt * 60f);

        float cur = _active.scaleFactorInit ? _active.scaleFactorCmd : 1f;
        _active.scaleFactorCmd = Mathf.Lerp(cur, desiredFactor, k);
        _active.scaleFactorInit = true;

        // Apply to overlay LOCAL scale (baseline * factor)
        _active.currentOverlay.localScale = _active.baseCurrentLocalScale * _active.scaleFactorCmd;
    }

    private float CurrentFactor()
    {
        if (_active == null || _active.currentOverlay == null) return 1f;

        float baseS = AvgAbs(_active.baseCurrentLocalScale);
        float curS = AvgAbs(_active.currentOverlay.localScale);
        if (baseS <= 1e-6f) return 1f;
        return curS / baseS;
    }

    private static float AvgAbs(Vector3 v)
    {
        return (Mathf.Abs(v.x) + Mathf.Abs(v.y) + Mathf.Abs(v.z)) / 3f;
    }

    // ---------------- Visibility helpers ----------------
    private void SetOnlyActiveOverlaysVisible(string activeId)
    {
        // Optional clarity: only show the active tool's overlays.
        // If you prefer showing all, just comment this out.
        SetVisibleById(overlaysCurrentRoot, activeId, true);
        SetVisibleById(overlaysTargetRoot, activeId, true);
    }

    private void SetVisibleById(Transform root, string activeId, bool showOthers = false)
    {
        if (root == null) return;

        var ids = root.GetComponentsInChildren<ToolId>(true);
        foreach (var tid in ids)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;

            bool on = (tid.id == activeId) || showOthers;

            // toggle renderers
            var rs = tid.GetComponentsInChildren<Renderer>(true);
            for (int i = 0; i < rs.Length; i++)
                rs[i].enabled = on;
        }
    }

    // ---------------- UI / Feedback ----------------
    private void UpdateProgressUI(float curFactor, float err, bool driving, bool evalAllowed)
    {
        if (progressText == null) return;

        _sb.Length = 0;

        string id = (_active != null) ? _active.id : "N/A";
        _sb.AppendLine($"Scaling: {id}");
        _sb.AppendLine($"Trial: {_trialIndex + 1}/{totalTrials}");
        _sb.AppendLine($"Time: {TrialTimeRemainingSec:F1}s");

        _sb.AppendLine($"curFactor: {curFactor:F2}  target: {(_active != null ? _active.targetFactor : 1f):F2}  err: {err:F2}  tol: {scaleFactorTolerance:F2}");

        bool holding = (grabber != null && grabber.IsHolding);
        _sb.AppendLine($"Holding: {(holding ? "YES" : "NO")}  Drive: {(driving ? "ON" : "OFF")}  EvalAllowed: {(evalAllowed ? "YES" : "NO")}");

        _sb.AppendLine($"Mode: {controlMode}");

        if (showDebugLine && requireHoldingThisTool && _active != null)
        {
            string heldName = grabber != null && grabber.HeldBody != null ? grabber.HeldBody.name : "null";
            string activeRb = _active.toolBody != null ? _active.toolBody.name : "null";
            _sb.AppendLine($"HeldBody: {heldName}  ActiveRB: {activeRb}");
        }

        progressText.text = _sb.ToString();
    }

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

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
            grabber.ForceRelease();
    }

    private void FinishBlock()
    {
        _trialRunning = false;
        _inTransition = false;
        HideFeedbackUI();

        try { OnBlockFinished?.Invoke(); } catch { }
    }
}
