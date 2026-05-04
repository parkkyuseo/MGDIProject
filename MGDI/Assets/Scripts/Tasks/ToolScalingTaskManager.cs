using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;
using UnityEngine.Windows.Speech;

public class ToolScalingTaskManager : MonoBehaviour
{
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged; // (current1Based, total)
    public event Action<float, bool> OnConfirmProgress; // (t01, eligible)
    public event Action OnConfirmDwellCompleted;
    public event Action<string> OnConfirmStatus;
    public event Action<bool, bool> OnTrialEnded; // (success, timedOut)

    [Header("Tools Root (auto-discovered via ToolId)")]
    [SerializeField] private Transform toolsDynamicRoot;

    [Header("Target Ghost (optional visual)")]
    [Tooltip("Targets/ghost root. Example: ContentRoot/Slots_Targets")]
    [SerializeField] private Transform slotsTargetsRoot; // ContentRoot/Slots_Targets
    [SerializeField] private bool showTargetGhostScale = true;

    [Header("Target ghost positioning (bring near active tool)")]
    [SerializeField] private bool bringTargetGhostNearActiveTool = false;

    [Tooltip("Offset near the tool when bringing target ghost close. X=right, Y=up, Z=forward (in chosen frame). Meters.")]
    [SerializeField] private Vector3 targetGhostOffsetLocal = new Vector3(0.18f, 0.03f, 0.00f);

    [Tooltip("If true, offset uses Camera frame (cam.right/up/fwd). If false, uses tool frame (tool.right/up/fwd).")]
    [SerializeField] private bool targetGhostOffsetInCameraFrame = true;

    [Header("Grab / Evaluate")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime used for MACRO wrist diagonal scaling.")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Phone (optional)")]
    [SerializeField] private PhoneInputRouter phoneRouter;

    [Header("Micro: allow scale without holding")]
    [SerializeField] private bool allowMicroScaleWithoutHolding = true;

    [Tooltip("If true, evaluation occurs only when NOT holding (release-to-evaluate).")]
    [SerializeField] private bool requireNotHolding = true;

    [Tooltip("If true, allow scaling updates only while holding.")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, allow scaling updates only while holding THIS trial's tool rigidbody (id-matched).")]
    [SerializeField] private bool requireHoldingThisTool = true;

    [Tooltip("If true, Macro mode requires holding regardless of scaleOnlyWhenHolding.")]
    [SerializeField] private bool requireHoldingInMacro = true;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;
    [SerializeField] private float confirmDwellSeconds = 3.0f;
    [SerializeField] private float stablePosSpeedMetersPerSec = 0.02f;
    [SerializeField] private float stableRotSpeedDegPerSec = 8.0f;
    [SerializeField] private float stableWarmupSeconds = 0.25f;

    [Header("Success Threshold (FACTOR)")]
    [Tooltip("Success if abs(currentFactor - targetFactor) <= tolerance.")]
    [SerializeField] private float scaleFactorTolerance = 0.05f;

    [Header("Target Factor Sampling")]
    [SerializeField] private float targetFactorMin = 0.70f;
    [SerializeField] private float targetFactorMax = 1.60f;

    [Tooltip("Avoid sampling in (1-avoidNearOne, 1+avoidNearOne) when possible.")]
    [SerializeField] private float avoidNearOne = 0.30f;

    [Tooltip("If range spans both below and above 1, ensure each side has at least this width by reducing avoidNearOne if needed.")]
    [SerializeField] private float minSideBandWidth = 0.05f;

    [Tooltip("If both smaller and larger bands exist, sample from each side ~50/50.")]
    [SerializeField] private bool balanceSmallerVsLarger = true;

    [Header("Scale factor clamp")]
    [SerializeField] private float minScaleFactor = 0.60f;
    [SerializeField] private float maxScaleFactor = 1.80f;

    // ---------------- MACRO: wrist diagonal -> scale ----------------
    [Header("MACRO Control: diagonal wrist movement -> scale")]
    [Tooltip("Axis is based on camera up + side. Recommended true.")]
    [SerializeField] private bool axisFromCamera = true;

    [Tooltip("If true, left hand uses cam.left (so 'up+away from body side' feels consistent).")]
    [SerializeField] private bool flipSideAxisForLeftHand = true;

    [Tooltip("Scale mapping gain for exp(gain * accumulatedMeters). Typical 3~6.")]
    [SerializeField] private float moveToScaleGain = 4.0f;

    [Tooltip("Ignore very small axis movement (meters).")]
    [SerializeField] private float moveDeadZoneMeters = 0.0025f;

    [Range(0f, 1f)]
    [Tooltip("Extra smoothing on scale factor command (0=no smoothing, 1=very slow).")]
    [SerializeField] private float scaleLerp = 0.15f;

    [Header("Inter-trial behavior")]
    [SerializeField] private bool snapOnSuccess = false;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetScaleAfterTrial = true;
    [SerializeField] private bool forceReleaseAfterTrial = true;

    [Header("Feedback (Audio)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip confirmClip;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Submit (HoloLens)")]
    [SerializeField] private bool enableVoiceSubmit = true;
    [SerializeField] private string submitKeyword = "next";
    [SerializeField] private bool enableTripleTapSubmit = true;

    [Header("Triple Tap Submit Safety")]
    [SerializeField] private bool blockTripleTapSubmitWhenHandNearGrabbable = true;
    [SerializeField] private float blockedTripleTapStatusSeconds = 1.0f;
    [SerializeField] private string blockedTripleTapStatus = "Move hand away,\nthen triple tap.";

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    [Header("Practice Target Randomization")]
    [SerializeField] private bool usePracticeRandomTargetScale = true;
    [SerializeField] private float practiceTargetFactorMin = 0.80f;
    [SerializeField] private float practiceTargetFactorMax = 1.25f;
    [SerializeField] private float practiceMinTargetFactorDelta = 0.10f;

    // ---- Forced active (Workflow integration) ----
    [SerializeField] private bool finishBlockAfterOneSuccessWhenForced = true;
    private string _forcedActiveId = null;

    public void SetForcedActiveId(string id) => _forcedActiveId = NormalizeToolId(id);
    public void ClearForcedActiveId() => _forcedActiveId = null;

    // ---------- runtime ----------
    [Serializable]
    private class Item
    {
        public string id;
        public Transform tool;
        public Rigidbody toolBody;
        public Transform startParent;
        public Vector3 startPos;
        public Quaternion startRot;

        // "True baseline" captured once from scene registry
        public Vector3 startLocalScale;

        public float targetFactor;

        // Current command
        public float scaleFactorCmd = 1f;

        // Macro drive state
        public float axisAccum = 0f;
        public bool haveWristPrev = false;
        public Vector3 wristPrev;
    }

    private readonly List<Item> items = new List<Item>();
    private Item active;

    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;
    private float confirmDwellTimer = 0f;
    private float stableWarmupTimer = 0f;
    private bool confirmLatched = false;
    private Vector3 confirmPrevPos = Vector3.zero;
    private Quaternion confirmPrevRot = Quaternion.identity;
    private bool confirmPrevPoseValid = false;
    private bool trialRunning = false;
    private bool inTransition = false;

    private readonly StringBuilder sb = new StringBuilder(512);
    private KeywordRecognizer _keywordRecognizer;
    private Dictionary<string, Action> _keywordActions;
    private bool _voiceSubmitRequested = false;
    private float _blockedTripleTapStatusUntil = -1f;

    // Micro controller can set this true so macro does NOT overwrite scale.
    private bool _externalDriving = false;

    // --- Ghost runtime ---
    private readonly Dictionary<string, Transform> _ghostById = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
    private Transform _activeGhost;
    private Vector3 _activeGhostBaseScale;
    private Vector3 _activeGhostBasePos;
    private Quaternion _activeGhostBaseRot = Quaternion.identity;
    private bool _activeGhostBasePoseValid = false;
    private bool _practiceGhostRandomizationEnabled = false;
    private float _lastPracticeTargetFactor = 1f;
    private bool _hasLastPracticeTargetFactor = false;
    private bool _hasStartPoseOverride = false;
    private string _startPoseOverrideId = null;
    private Vector3 _startPoseOverridePos = Vector3.zero;
    private Quaternion _startPoseOverrideRot = Quaternion.identity;

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;

    public string ActiveId => active != null ? active.id : null;
    public float ActiveTargetFactor => active != null ? active.targetFactor : 1f;
    public float ActiveCurrentFactor => GetActualScaleFactor(active);
    public float ScaleFactorTolerance => scaleFactorTolerance;
    public float ActiveScalingErrorFactor => active != null ? Mathf.Abs(ActiveCurrentFactor - active.targetFactor) : float.MaxValue;
    public Transform ActiveTargetTransform => _activeGhost;
    public bool AllowMicroScaleWithoutHolding => allowMicroScaleWithoutHolding;
    public float EffectiveMinScaleFactor => GetEffectiveMinScaleFactor();
    public float EffectiveMaxScaleFactor => GetEffectiveMaxScaleFactor();
    public bool ResetScaleAfterTrial
    {
        get => resetScaleAfterTrial;
        set => resetScaleAfterTrial = value;
    }

    public void SetPracticeGhostRandomization(bool enabled)
    {
        _practiceGhostRandomizationEnabled = enabled;
        _hasLastPracticeTargetFactor = false;
    }

    public Transform ActiveToolTransform => active != null ? active.tool : null;
    public string ActiveToolId => active != null ? active.id : null;

    public void SetStartPoseOverride(string id, Vector3 position, Quaternion rotation)
    {
        id = NormalizeToolId(id);
        if (string.IsNullOrEmpty(id))
        {
            ClearStartPoseOverride();
            return;
        }

        _hasStartPoseOverride = true;
        _startPoseOverrideId = id;
        _startPoseOverridePos = position;
        _startPoseOverrideRot = rotation;
    }

    public void ClearStartPoseOverride()
    {
        _hasStartPoseOverride = false;
        _startPoseOverrideId = null;
        _startPoseOverridePos = Vector3.zero;
        _startPoseOverrideRot = Quaternion.identity;
    }

    // ---------------- Public API for MICRO controllers ----------------
    public void SetExternalDriving(bool driving) => _externalDriving = driving;

    public bool CanDriveNow()
    {
        if (!trialRunning || inTransition) return false;
        if (active == null || active.tool == null) return false;

        bool isMicroMode = phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro;
        if (isMicroMode)
        {
            if (allowMicroScaleWithoutHolding)
                return true;
            return IsHoldingAllowedForDrive();
        }

        if (!isMicroMode && requireHoldingInMacro)
            return IsHoldingAllowedForDrive();

        if (!scaleOnlyWhenHolding) return true;

        return IsHoldingAllowedForDrive();
    }

    // Controllers (and macro) use this to apply factor to the TOOL scale
    public void ApplyScaleFactor(float factor)
    {
        if (!trialRunning || inTransition) return;
        if (active == null || active.tool == null) return;

        float minF = GetEffectiveMinScaleFactor();
        float maxF = GetEffectiveMaxScaleFactor();
        float f = Mathf.Clamp(factor, minF, maxF);
        active.scaleFactorCmd = f;

        // ✅ IMPORTANT: Always scale relative to "true baseline"
        active.tool.localScale = active.startLocalScale * f;
    }

    public float GetScaleFactorCmd() => active != null ? active.scaleFactorCmd : 1f;

    // ---------------- Flow ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        if (phoneRouter == null)
            phoneRouter = FindFirstObjectByType<PhoneInputRouter>();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;
        DrainPendingSubmitTriggers();
        _hasLastPracticeTargetFactor = false;
        _externalDriving = false;
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        ResetConfirmState();

        // Safety: in case previous run ended unexpectedly
        RestoreTargetGhostScale();

        RebuildItemsFromScene();
        RebuildGhostMap();

        BeginNextTrial();
    }

    private void Start()
    {
        if (enableVoiceSubmit)
            SetupVoiceCommands();
    }

    private void OnEnable()
    {
        if (enableVoiceSubmit)
            SetupVoiceCommands();
    }

    void Update()
    {
        if (!trialRunning || inTransition)
        {
            OnConfirmStatus?.Invoke("");
            return;
        }
        if (active == null || active.tool == null)
        {
            OnConfirmProgress?.Invoke(0f, false);
            OnConfirmStatus?.Invoke("");
            return;
        }

        float dt = Time.deltaTime;
        trialTimer += dt;

        if (trialTimer >= trialTimeoutSeconds)
        {
            OnConfirmProgress?.Invoke(0f, false);
            OnConfirmStatus?.Invoke("");
            ResetConfirmState();
            StartCoroutine(EndTrialRoutine(success: false, timedOut: true));
            return;
        }

        // ---------------- MACRO drive happens here ----------------
        if (!_externalDriving)
        {
            bool canDrive = CanDriveNow();
            bool hasWrist = HasWrist();

            if (canDrive && hasWrist)
                UpdateScaleFromDiagonalWristMotion();
            else
                active.haveWristPrev = false;
        }

        float curFactor = GetActualScaleFactor(active);
        float err = Mathf.Abs(curFactor - active.targetFactor);
        EmitScaleConfirmStatus(err, scaleFactorTolerance);
        OnConfirmProgress?.Invoke(0f, false);

        if (TryConsumeVoiceSubmit("scaling"))
        {
            confirmLatched = true;
            OnConfirmDwellCompleted?.Invoke();
            PlayConfirmSound();
            EndTrialSuccess();
        }
    }

    private void BeginNextTrial()
    {
        if (toolsDynamicRoot == null)
        {
            Debug.LogError("[ToolScaleTM] Missing toolsDynamicRoot.");
            FinishBlock();
            return;
        }

        if (string.IsNullOrEmpty(_forcedActiveId))
        {
            if (totalTrials > 0 && trialIndex >= totalTrials)
            {
                if (logDebug) Debug.Log("[ToolScaleTM] Block finished.");
                FinishBlock();
                return;
            }
        }

        if (items.Count == 0)
        {
            RebuildItemsFromScene();
            if (items.Count == 0)
            {
                Debug.LogError("[ToolScaleTM] No tools found (ToolId).");
                FinishBlock();
                return;
            }
        }

        // Safety: ensure previous ghost is restored before picking a new one
        RestoreTargetGhostScale();

        // Select active
        active = null;

        if (!string.IsNullOrEmpty(_forcedActiveId))
        {
            active = items.Find(
                it => it != null && string.Equals(it.id, _forcedActiveId, StringComparison.OrdinalIgnoreCase));
            if (active == null)
            {
                RebuildItemsFromScene();
                active = items.Find(
                    it => it != null && string.Equals(it.id, _forcedActiveId, StringComparison.OrdinalIgnoreCase));
            }

            if (active == null)
            {
                Debug.LogWarning($"[ToolScaleTM] ForcedActiveId '{_forcedActiveId}' not found. Falling back to current list order.");
            }
        }

        if (active == null)
            active = items[trialIndex % items.Count];

        EnsureActiveBody(active);
        ApplyStartPoseOverrideIfNeeded();
        ForceReleaseIfPossible();
        ResetActiveToolToStartPose();

        // Reset per-trial state
        active.axisAccum = 0f;
        active.haveWristPrev = false;

        // ✅ IMPORTANT: reset tool scale to true baseline each trial start
        active.scaleFactorCmd = 1f;
        if (active.tool != null)
            active.tool.localScale = active.startLocalScale;

        _externalDriving = false;

        // ✅ Improved target factor sampling (balanced smaller/larger when possible)
        float tf = (_practiceGhostRandomizationEnabled && usePracticeRandomTargetScale)
            ? SamplePracticeTargetFactor()
            : SampleTargetFactorBalanced();
        active.targetFactor = tf;

        // ✅ Apply target ghost visual scale (optional)
        ApplyTargetGhostScale(active.id, active.targetFactor);

        if (bringTargetGhostNearActiveTool)
            MoveTargetGhostNearActiveTool();

        trialTimer = 0f;
        dwellTimer = 0f;
        DrainPendingSubmitTriggers();
        ResetConfirmState();
        InitializeConfirmPoseFromActive();
        trialRunning = true;
        inTransition = false;
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");

        int shownTotal = string.IsNullOrEmpty(_forcedActiveId) ? totalTrials : 1;
        int shownIndex = string.IsNullOrEmpty(_forcedActiveId) ? (trialIndex + 1) : 1;

        OnTrialChanged?.Invoke(shownIndex, shownTotal);

        if (logDebug)
            DebugHUD.Log($"[ToolScaleTM] Trial {shownIndex}/{shownTotal} id={active.id} targetFactor={active.targetFactor:F2} forced={(string.IsNullOrEmpty(_forcedActiveId) ? "NO" : "YES")}");
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        ResetConfirmState();

        if (success)
        {
            if (forceReleaseAfterTrial) ForceReleaseIfPossible();

            if (snapOnSuccess && active != null && active.tool != null)
            {
                ApplyScaleFactor(active.targetFactor);
            }

            yield return new WaitForSeconds(postSnapHoldSeconds);
        }
        else
        {
            yield return null;
        }

        if (resetScaleAfterTrial && active != null && active.tool != null)
        {
            // ✅ IMPORTANT: restore true baseline
            active.tool.localScale = active.startLocalScale;
            active.scaleFactorCmd = 1f;
        }

        if (forceReleaseAfterTrial) ForceReleaseIfPossible();

        // ✅ Restore ghost scale at trial end
        RestoreTargetGhostScale();

        OnTrialEnded?.Invoke(success, timedOut);

        // Forced: finish after one success
        if (success && !string.IsNullOrEmpty(_forcedActiveId) && finishBlockAfterOneSuccessWhenForced)
        {
            inTransition = false;
            FinishBlock();
            yield break;
        }

        trialIndex++;
        inTransition = false;
        BeginNextTrial();
    }

    // ---------------- Registry ----------------
    private void RebuildItemsFromScene()
    {
        items.Clear();
        if (toolsDynamicRoot == null) return;

        var toolIds = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        var map = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < toolIds.Length; i++)
        {
            if (toolIds[i] == null) continue;
            string id = NormalizeToolId(toolIds[i].id);
            if (string.IsNullOrEmpty(id)) continue;
            map[id] = toolIds[i].transform;
        }

        foreach (var kv in map)
        {
            Transform toolTf = kv.Value;
            if (toolTf == null) continue;

            var it = new Item
            {
                id = kv.Key,
                tool = toolTf,
                toolBody = null,
                startParent = toolTf.parent,
                startPos = toolTf.position,
                startRot = toolTf.rotation,

                // ✅ IMPORTANT: capture baseline ONCE here
                startLocalScale = toolTf.localScale,

                targetFactor = 1f,
                scaleFactorCmd = 1f,
                axisAccum = 0f,
                haveWristPrev = false,
                wristPrev = Vector3.zero
            };

            items.Add(it);
        }

        items.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        if (logDebug)
            Debug.Log($"[ToolScaleTM] Registry rebuilt: matched={items.Count}");
    }

    private void EnsureActiveBody(Item it)
    {
        if (it == null || it.tool == null) return;
        if (it.toolBody != null) return;

        var rb = it.tool.GetComponent<Rigidbody>();
        if (rb == null) rb = it.tool.GetComponentInChildren<Rigidbody>(true);
        it.toolBody = rb;
    }

    // ---------------- Ghost map ----------------
    private void RebuildGhostMap()
    {
        _ghostById.Clear();
        if (slotsTargetsRoot == null) return;

        var ids = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] == null) continue;
            string id = NormalizeToolId(ids[i].id);
            if (string.IsNullOrEmpty(id)) continue;
            _ghostById[id] = ids[i].transform;
        }
    }

    private void ApplyTargetGhostScale(string id, float targetFactor)
    {
        if (!showTargetGhostScale) return;
        if (slotsTargetsRoot == null) return;
        if (string.IsNullOrEmpty(id)) return;

        if (_ghostById.Count == 0) RebuildGhostMap();

        Transform ghost = null;
        if (!_ghostById.TryGetValue(id, out ghost) || ghost == null)
        {
            // fallback scan (in case hierarchy changed after cache)
            var ids = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] != null && string.Equals(NormalizeToolId(ids[i].id), id, StringComparison.OrdinalIgnoreCase))
                {
                    ghost = ids[i].transform;
                    break;
                }
            }
            if (ghost == null) return;
        }

        _activeGhost = ghost;
        _activeGhostBaseScale = ghost.localScale;
        _activeGhostBasePos = ghost.position;
        _activeGhostBaseRot = ghost.rotation;
        _activeGhostBasePoseValid = true;

        ghost.localScale = _activeGhostBaseScale * targetFactor;
    }

    private void RestoreTargetGhostScale()
    {
        if (_activeGhost == null) return;

        _activeGhost.localScale = _activeGhostBaseScale;
        if (_activeGhostBasePoseValid)
        {
            _activeGhost.position = _activeGhostBasePos;
            _activeGhost.rotation = _activeGhostBaseRot;
        }
        _activeGhost = null;
        _activeGhostBaseScale = Vector3.one;
        _activeGhostBasePos = Vector3.zero;
        _activeGhostBaseRot = Quaternion.identity;
        _activeGhostBasePoseValid = false;
    }

    private void ResetActiveToolToStartPose()
    {
        if (active == null || active.tool == null)
            return;

        if (active.startParent != null)
            active.tool.SetParent(active.startParent, true);

        active.tool.SetPositionAndRotation(active.startPos, active.startRot);
    }

    private void ApplyStartPoseOverrideIfNeeded()
    {
        if (!_hasStartPoseOverride || active == null || active.tool == null)
            return;

        if (!string.Equals(active.id, _startPoseOverrideId, StringComparison.OrdinalIgnoreCase))
            return;

        active.startPos = _startPoseOverridePos;
        active.startRot = _startPoseOverrideRot;
    }

    private void MoveTargetGhostNearActiveTool()
    {
        if (_activeGhost == null || active == null || active.tool == null) return;

        Transform cam = Camera.main != null ? Camera.main.transform : null;

        Vector3 right, up, fwd;
        if (targetGhostOffsetInCameraFrame && cam != null)
        {
            right = cam.right;
            up = cam.up;
            fwd = cam.forward;
        }
        else
        {
            right = active.tool.right;
            up = active.tool.up;
            fwd = active.tool.forward;
        }

        _activeGhost.position =
            active.tool.position +
            right * targetGhostOffsetLocal.x +
            up * targetGhostOffsetLocal.y +
            fwd * targetGhostOffsetLocal.z;
    }

    // ---------------- Target sampling (improved) ----------------
    private float SampleTargetFactorBalanced()
    {
        float effMin = GetEffectiveMinScaleFactor();
        float effMax = GetEffectiveMaxScaleFactor();

        // Normalize range and clamp to scale clamp
        float minF = Mathf.Min(targetFactorMin, targetFactorMax);
        float maxF = Mathf.Max(targetFactorMin, targetFactorMax);

        minF = Mathf.Clamp(minF, effMin, effMax);
        maxF = Mathf.Clamp(maxF, effMin, effMax);

        if (maxF < minF)
        {
            float t = minF;
            minF = maxF;
            maxF = t;
        }

        // If degenerate range
        if (Mathf.Abs(maxF - minF) < 1e-6f)
            return Mathf.Clamp(minF, effMin, effMax);

        float avoid = Mathf.Max(0f, avoidNearOne);
        float minBand = Mathf.Max(0f, minSideBandWidth);

        bool spansOne = (minF < 1f) && (maxF > 1f);

        // When it spans 1.0, try to create both bands (smaller & larger) with minimum width.
        if (spansOne)
        {
            // We want:
            // low band:  [minF, 1-avoidEff] has width >= minBand
            // high band: [1+avoidEff, maxF] has width >= minBand
            // => avoidEff <= 1 - minF - minBand
            // => avoidEff <= maxF - 1 - minBand
            float allowedLow = (1f - minF - minBand);
            float allowedHigh = (maxF - 1f - minBand);

            if (allowedLow > 0f && allowedHigh > 0f)
            {
                float avoidEff = Mathf.Min(avoid, allowedLow, allowedHigh);
                avoidEff = Mathf.Max(0f, avoidEff);

                float lowMax = 1f - avoidEff;
                float highMin = 1f + avoidEff;

                // Safety clamp
                lowMax = Mathf.Clamp(lowMax, minF, maxF);
                highMin = Mathf.Clamp(highMin, minF, maxF);

                bool hasLow = lowMax > minF + 1e-6f;
                bool hasHigh = maxF > highMin + 1e-6f;

                if (hasLow && hasHigh)
                {
                    bool pickLow = balanceSmallerVsLarger ? (Random.value < 0.5f) : (Random.value < (lowMax - minF) / ((lowMax - minF) + (maxF - highMin)));
                    float tf = pickLow ? Random.Range(minF, lowMax) : Random.Range(highMin, maxF);
                    return Mathf.Clamp(tf, effMin, effMax);
                }

                // If one side collapses due to numeric issues, fall through to fallback below.
            }
        }

        // Fallback:
        // sample in full range; if it lands in avoid zone and there is room outside, retry a bit.
        int guard = 0;
        float outTf = 1f;

        bool hasOutside =
            (minF < (1f - avoid)) || (maxF > (1f + avoid));

        do
        {
            outTf = Random.Range(minF, maxF);
            guard++;
            if (guard >= 32) break;

            if (!hasOutside) break;
        } while (Mathf.Abs(outTf - 1f) < avoid);

        return Mathf.Clamp(outTf, effMin, effMax);
    }

    private float SamplePracticeTargetFactor()
    {
        float effMin = GetEffectiveMinScaleFactor();
        float effMax = GetEffectiveMaxScaleFactor();

        float minF = Mathf.Min(practiceTargetFactorMin, practiceTargetFactorMax);
        float maxF = Mathf.Max(practiceTargetFactorMin, practiceTargetFactorMax);

        minF = Mathf.Clamp(minF, effMin, effMax);
        maxF = Mathf.Clamp(maxF, effMin, effMax);

        if (maxF < minF)
        {
            float tmp = minF;
            minF = maxF;
            maxF = tmp;
        }

        if (Mathf.Abs(maxF - minF) < 1e-6f)
            return Mathf.Clamp(minF, effMin, effMax);

        float minDelta = Mathf.Max(0f, practiceMinTargetFactorDelta);
        float tf = minF;

        for (int i = 0; i < 16; i++)
        {
            tf = Random.Range(minF, maxF);
            if (!_hasLastPracticeTargetFactor || Mathf.Abs(tf - _lastPracticeTargetFactor) >= minDelta)
                break;
        }

        _lastPracticeTargetFactor = tf;
        _hasLastPracticeTargetFactor = true;
        return Mathf.Clamp(tf, effMin, effMax);
    }

    // ---------------- MACRO helpers ----------------
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
        if (active == null || active.tool == null) return;

        Vector3 w = remoteHand.remoteByIndex[0].position;

        if (!active.haveWristPrev)
        {
            active.wristPrev = w;
            active.haveWristPrev = true;
            return;
        }

        Vector3 dp = w - active.wristPrev;
        active.wristPrev = w;

        Vector3 axis = GetMacroAxis();
        float delta = Vector3.Dot(dp, axis);

        if (Mathf.Abs(delta) < Mathf.Max(0f, moveDeadZoneMeters))
            return;

        active.axisAccum += delta;

        float desired = Mathf.Exp(moveToScaleGain * active.axisAccum);
        desired = Mathf.Clamp(desired, GetEffectiveMinScaleFactor(), GetEffectiveMaxScaleFactor());

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(scaleLerp), dt * 60f);

        float cur = active.scaleFactorCmd;
        float next = Mathf.Lerp(cur, desired, k);

        ApplyScaleFactor(next);
    }

    private bool IsHoldingAllowedForDrive()
    {
        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisTool) return true;
        if (active == null || active.tool == null) return false;

        if (active.toolBody == null) return false;
        if (grabber.HeldBody == null) return false;
        return grabber.HeldBody == active.toolBody;
    }

    // ---------------- Scale factor measurement (SAFE) ----------------
    private float GetActualScaleFactor(Item it)
    {
        if (it == null || it.tool == null) return 1f;

        Vector3 baseS = it.startLocalScale;
        Vector3 curS = it.tool.localScale;

        float sum = 0f;
        int n = 0;

        if (Mathf.Abs(baseS.x) > 1e-6f) { sum += curS.x / baseS.x; n++; }
        if (Mathf.Abs(baseS.y) > 1e-6f) { sum += curS.y / baseS.y; n++; }
        if (Mathf.Abs(baseS.z) > 1e-6f) { sum += curS.z / baseS.z; n++; }

        if (n == 0) return 1f;

        float f = sum / n;
        // Keep it sane
        return Mathf.Clamp(f, GetEffectiveMinScaleFactor(), GetEffectiveMaxScaleFactor());
    }

    private float GetEffectiveMinScaleFactor()
    {
        float minF = minScaleFactor;
        minF = Mathf.Min(minF, targetFactorMin, targetFactorMax);
        minF = Mathf.Min(minF, practiceTargetFactorMin, practiceTargetFactorMax);
        return Mathf.Max(0.01f, minF);
    }

    private float GetEffectiveMaxScaleFactor()
    {
        float maxF = maxScaleFactor;
        maxF = Mathf.Max(maxF, targetFactorMin, targetFactorMax);
        maxF = Mathf.Max(maxF, practiceTargetFactorMin, practiceTargetFactorMax);
        return Mathf.Max(GetEffectiveMinScaleFactor(), maxF);
    }

    private bool IsMicroTouchHoldConfirmGateActive()
    {
        return phoneRouter != null &&
               phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro &&
               allowMicroScaleWithoutHolding;
    }

    private void EmitScaleConfirmStatus(float errorFactor, float toleranceFactor)
    {
        if (IsBlockedTripleTapStatusActive())
        {
            OnConfirmStatus?.Invoke(blockedTripleTapStatus);
            return;
        }
        OnConfirmStatus?.Invoke("");
    }

    private bool ComputeActiveStability(float dt)
    {
        Transform tf = ActiveToolTransform;
        if (tf == null || dt <= 0f)
            return false;

        Vector3 currentPos = tf.position;
        Quaternion currentRot = tf.rotation;

        bool stable = false;
        if (stableWarmupTimer < stableWarmupSeconds)
        {
            stableWarmupTimer += dt;
            stable = false;
        }
        else if (confirmPrevPoseValid)
        {
            float posSpeed = Vector3.Distance(currentPos, confirmPrevPos) / dt;
            float rotSpeedDeg = Quaternion.Angle(currentRot, confirmPrevRot) / dt;
            stable = posSpeed <= stablePosSpeedMetersPerSec && rotSpeedDeg <= stableRotSpeedDegPerSec;
        }

        confirmPrevPos = currentPos;
        confirmPrevRot = currentRot;
        confirmPrevPoseValid = true;
        return stable;
    }

    private bool IsNotHolding()
    {
        if (!requireNotHolding) return true;
        if (grabber == null)
        {
            // TODO: Integrate a hold-state check if a different grabber API is used.
            return true;
        }

        return !grabber.IsHolding;
    }

    private void ResetConfirmState()
    {
        confirmDwellTimer = 0f;
        stableWarmupTimer = 0f;
        confirmLatched = false;
        confirmPrevPos = Vector3.zero;
        confirmPrevRot = Quaternion.identity;
        confirmPrevPoseValid = false;
    }

    private void InitializeConfirmPoseFromActive()
    {
        Transform tf = ActiveToolTransform;
        if (tf == null) return;
        confirmPrevPos = tf.position;
        confirmPrevRot = tf.rotation;
        confirmPrevPoseValid = true;
    }

    private void EndTrialSuccess()
    {
        if (inTransition || !trialRunning) return;
        StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
    }

    // ---------------- Shared helpers ----------------
    private void PlayConfirmSound()
    {
        if (audioSource != null && confirmClip != null)
            audioSource.PlayOneShot(confirmClip);
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null) grabber.ForceRelease();
    }

    private void FinishBlock()
    {
        trialRunning = false;
        inTransition = false;
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        ResetConfirmState();

        // Safety: ensure ghost gets restored when block finishes
        RestoreTargetGhostScale();

        try { OnBlockFinished?.Invoke(); } catch { }
    }

    private void OnDisable()
    {
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        DrainPendingSubmitTriggers();
        ResetConfirmState();

        if (_keywordRecognizer != null)
        {
            _keywordRecognizer.Stop();
            _keywordRecognizer.Dispose();
            _keywordRecognizer = null;
        }
    }

    private static string NormalizeToolId(string id)
    {
        return string.IsNullOrWhiteSpace(id) ? null : id.Trim();
    }

    private void SetupVoiceCommands()
    {
        if (_keywordRecognizer != null)
            return;

        _keywordActions = new Dictionary<string, Action>();
        string submit = string.IsNullOrWhiteSpace(submitKeyword) ? "" : submitKeyword.Trim().ToLower();
        if (!string.IsNullOrEmpty(submit))
            _keywordActions[submit] = () => _voiceSubmitRequested = true;

        if (_keywordActions.Count == 0)
            return;

        _keywordRecognizer = new KeywordRecognizer(_keywordActions.Keys.ToArray());
        _keywordRecognizer.OnPhraseRecognized += args =>
        {
            string key = args.text.ToLower();
            if (_keywordActions.TryGetValue(key, out var action))
                action.Invoke();
        };
        _keywordRecognizer.Start();
    }

    private bool TryConsumeVoiceSubmit(string context)
    {
        if (_voiceSubmitRequested)
        {
            _voiceSubmitRequested = false;
            _ = context;
            return true;
        }

        if (enableTripleTapSubmit && phoneRouter != null && phoneRouter.TryConsumeTripleTap())
        {
            if (ShouldBlockTripleTapSubmit())
            {
                NotifyBlockedTripleTapSubmit();
                return false;
            }
            _ = context;
            return true;
        }

        return false;
    }

    private void DrainPendingSubmitTriggers()
    {
        _voiceSubmitRequested = false;

        if (!enableTripleTapSubmit || phoneRouter == null)
            return;

        int guard = 0;
        while (phoneRouter.TryConsumeTripleTap() && guard < 8)
            guard++;
    }

    private bool ShouldBlockTripleTapSubmit()
    {
        return blockTripleTapSubmitWhenHandNearGrabbable &&
               grabber != null &&
               grabber.IsHoldingOrHasAttachCandidateNow;
    }

    private void NotifyBlockedTripleTapSubmit()
    {
        _blockedTripleTapStatusUntil = Time.unscaledTime + Mathf.Max(0.05f, blockedTripleTapStatusSeconds);
        OnConfirmStatus?.Invoke(blockedTripleTapStatus);
    }

    private bool IsBlockedTripleTapStatusActive()
    {
        return Time.unscaledTime < _blockedTripleTapStatusUntil;
    }
}

