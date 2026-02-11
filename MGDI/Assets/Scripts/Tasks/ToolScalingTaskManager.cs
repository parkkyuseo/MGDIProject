using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

public class ToolScalingTaskManager: MonoBehaviour
{
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged; // (current1Based, total)

    [Header("Tools Root (auto-discovered via ToolId)")]
    [SerializeField] private Transform toolsDynamicRoot;

    [Header("Grab / Evaluate")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime used for MACRO wrist diagonal scaling.")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Tooltip("If true, evaluation occurs only when NOT holding (release-to-evaluate).")]
    [SerializeField] private bool requireNotHolding = true;

    [Tooltip("If true, allow scaling updates only while holding.")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, allow scaling updates only while holding THIS trial's tool rigidbody (id-matched).")]
    [SerializeField] private bool requireHoldingThisTool = true;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Header("Success Threshold (FACTOR)")]
    [Tooltip("Success if abs(currentFactor - targetFactor) <= tolerance.")]
    [SerializeField] private float scaleFactorTolerance = 0.05f;

    [Header("Target Factor Sampling")]
    [SerializeField] private float targetFactorMin = 0.70f;
    [SerializeField] private float targetFactorMax = 1.60f;

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

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    // ---- Forced active (Workflow integration) ----
    [SerializeField] private bool finishBlockAfterOneSuccessWhenForced = true;
    private string _forcedActiveId = null;

    public void SetForcedActiveId(string id) => _forcedActiveId = string.IsNullOrEmpty(id) ? null : id;
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

        public Vector3 baseLocalScale; // baseline localScale at trial start
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
    private bool trialRunning = false;
    private bool inTransition = false;

    private readonly StringBuilder sb = new StringBuilder(512);

    // Micro controller can set this true so macro does NOT overwrite scale.
    private bool _externalDriving = false;

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;

    public string ActiveId => active != null ? active.id : null;
    public float ActiveTargetFactor => active != null ? active.targetFactor : 1f;
    public float ActiveCurrentFactor => active != null ? active.scaleFactorCmd : 1f;

    // For optional UX (hand placer, etc.)
    public Transform ActiveToolTransform => active != null ? active.tool : null;

    // ---------------- Public API for MICRO controllers ----------------
    public void SetExternalDriving(bool driving) => _externalDriving = driving;

    public bool CanDriveNow()
    {
        if (!trialRunning || inTransition) return false;
        if (active == null || active.tool == null) return false;

        if (!scaleOnlyWhenHolding) return true;

        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisTool) return true;

        if (active.toolBody == null) return false;
        if (grabber.HeldBody == null) return false;

        return grabber.HeldBody == active.toolBody;
    }

    // Controllers (and macro) use this to apply factor to the TOOL scale
    public void ApplyScaleFactor(float factor)
    {
        if (!trialRunning || inTransition) return;
        if (active == null || active.tool == null) return;

        float f = Mathf.Clamp(factor, minScaleFactor, maxScaleFactor);
        active.scaleFactorCmd = f;

        // Uniform factor applied to baseline localScale
        active.tool.localScale = active.baseLocalScale * f;
    }

    public float GetScaleFactorCmd() => active != null ? active.scaleFactorCmd : 1f;

    // ---------------- Flow ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;
        _externalDriving = false;

        RebuildItemsFromScene();
        BeginNextTrial();
    }

    void Update()
    {
        if (!trialRunning || inTransition || active == null) return;

        trialTimer += Time.deltaTime;

        if (trialTimer >= trialTimeoutSeconds)
        {
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

        // ---------------- Evaluate (release-to-evaluate) ----------------
        bool evalAllowed = true;
        if (requireNotHolding && grabber != null && grabber.IsHolding)
            evalAllowed = false;

        float curFactor = active.scaleFactorCmd;
        float err = Mathf.Abs(curFactor - active.targetFactor);

        if (logDebug)
        {
            // DebugHUD.Log can be used if desired; keep console quiet by default
        }

        if (err <= scaleFactorTolerance)
        {
            if (evalAllowed)
            {
                dwellTimer += Time.deltaTime;
                if (dwellTimer >= dwellSeconds)
                    StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
            }
            else
            {
                dwellTimer = 0f;
            }
        }
        else
        {
            dwellTimer = 0f;
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

        // Select active
        active = null;

        if (!string.IsNullOrEmpty(_forcedActiveId))
        {
            active = items.Find(it => it != null && it.id == _forcedActiveId);
            if (active == null)
                Debug.LogWarning($"[ToolScaleTM] ForcedActiveId '{_forcedActiveId}' not found. Falling back.");
        }

        if (active == null)
            active = items[trialIndex % items.Count];

        EnsureActiveBody(active);

        // Reset per-trial state
        active.axisAccum = 0f;
        active.haveWristPrev = false;

        // Capture baseline scale at trial start
        active.baseLocalScale = active.tool != null ? active.tool.localScale : Vector3.one;
        active.scaleFactorCmd = 1f;
        ApplyScaleFactor(1f);

        _externalDriving = false;

        // Sample target factor
        float tf = Random.Range(Mathf.Min(targetFactorMin, targetFactorMax), Mathf.Max(targetFactorMin, targetFactorMax));
        tf = Mathf.Clamp(tf, minScaleFactor, maxScaleFactor);
        active.targetFactor = tf;

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

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
            // Restore baseline scale
            active.tool.localScale = active.baseLocalScale;
            active.scaleFactorCmd = 1f;
        }

        if (forceReleaseAfterTrial) ForceReleaseIfPossible();

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
        var map = new Dictionary<string, Transform>();
        for (int i = 0; i < toolIds.Length; i++)
        {
            if (toolIds[i] == null || string.IsNullOrEmpty(toolIds[i].id)) continue;
            map[toolIds[i].id] = toolIds[i].transform;
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
                baseLocalScale = toolTf.localScale,
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
        desired = Mathf.Clamp(desired, minScaleFactor, maxScaleFactor);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(scaleLerp), dt * 60f);

        float cur = active.scaleFactorCmd;
        float next = Mathf.Lerp(cur, desired, k);

        ApplyScaleFactor(next);
    }

    // ---------------- Shared helpers ----------------
    private void ForceReleaseIfPossible()
    {
        if (grabber != null) grabber.ForceRelease();
    }

    private void FinishBlock()
    {
        trialRunning = false;
        inTransition = false;
        try { OnBlockFinished?.Invoke(); } catch { }
    }
}
