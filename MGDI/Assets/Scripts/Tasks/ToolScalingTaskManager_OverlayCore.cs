using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

public class ToolScalingTaskManager_OverlayCore : MonoBehaviour
{
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged; // (current1Based, total)

    [Header("Overlay Roots (auto-matched by ToolId)")]
    [SerializeField] private Transform overlaysCurrentRoot;
    [SerializeField] private Transform overlaysTargetRoot;

    [Header("Tools Root (optional override)")]
    [Tooltip("If set, tool rigidbodies are searched here instead of overlaysCurrentRoot.parent/Tools_Dynamic.")]
    [SerializeField] private Transform toolsDynamicRootOverride;

    [Header("Grab / Evaluate")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("RemoteHandRuntime used for MACRO wrist diagonal scaling.")]
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Tooltip("If true, success/dwell evaluation occurs only when NOT holding (release-to-evaluate).")]
    [SerializeField] private bool requireNotHolding = true;

    [Tooltip("If true, allow scaling updates only while holding.")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, allow scaling updates only when holding THIS trial's tool rigidbody (id-matched).")]
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

    [Header("Bring overlays near active tool (like rotation task)")]
    [SerializeField] private bool bringOverlaysNearActiveTool = true;

    [Tooltip("If true, offset uses Camera frame (cam.right/up/fwd). If false, uses tool frame (tool.right/up/fwd).")]
    [SerializeField] private bool offsetInCameraFrame = true;

    [Tooltip("If true, overlay rotations are matched to the tool rotation while placed near it.")]
    [SerializeField] private bool matchOverlayRotationToTool = true;

    [Tooltip("Where to place CURRENT overlay near the tool. X=right, Y=up, Z=forward (in chosen frame).")]
    [SerializeField] private Vector3 currentOverlayOffsetLocal = new Vector3(0.18f, 0.03f, 0.00f);

    [Tooltip("Where to place TARGET overlay near the tool. Usually slightly different so they don't overlap.")]
    [SerializeField] private Vector3 targetOverlayOffsetLocal = new Vector3(0.30f, 0.03f, 0.00f);

    [Header("Inter-trial behavior")]
    [SerializeField] private bool snapOnSuccess = false;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetScaleAfterTrial = true;
    [SerializeField] private bool forceReleaseAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Progress UI (optional)")]
    [SerializeField] private UnityEngine.UI.Text progressText;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    // ---------- runtime ----------
    [Serializable]
    private class Item
    {
        public string id;

        public Transform currentOverlay;
        public Transform targetOverlay;

        public Vector3 baseCurrentLocalScale;
        public Vector3 baseTargetLocalScale;

        // Original placement pose (for restoring after trial)
        public Vector3 baseCurrentPos;
        public Quaternion baseCurrentRot;
        public Vector3 baseTargetPos;
        public Quaternion baseTargetRot;
        public bool basePoseCaptured;

        public float targetFactor;

        // For holding gating (optional)
        public Rigidbody toolBody;

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

    // Cache tool bodies by id (rebuilt when scene registry rebuilt)
    private Dictionary<string, Rigidbody> _toolBodiesById = null;

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;

    public string ActiveId => active != null ? active.id : null;
    public float ActiveTargetFactor => active != null ? active.targetFactor : 1f;
    public float ActiveCurrentFactor => active != null ? CurrentFactor(active) : 1f;

    // Expose active tool for MicroHandAutoPlacer (grab near highlighted tool)
    public Transform ActiveToolTransform => active != null && active.toolBody != null ? active.toolBody.transform : null;

    // ---------------- Public API for MICRO controllers ----------------
    public void SetExternalDriving(bool driving) => _externalDriving = driving;

    public bool CanDriveNow()
    {
        if (!trialRunning || inTransition) return false;
        if (active == null || active.currentOverlay == null) return false;

        if (!scaleOnlyWhenHolding) return true;

        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisTool) return true;

        if (active.toolBody == null) return false;
        if (grabber.HeldBody == null) return false;

        return grabber.HeldBody == active.toolBody;
    }

    // Controllers (and macro) use this to apply factor to current overlay
    public void ApplyScaleFactor(float factor)
    {
        if (!trialRunning || inTransition) return;
        if (active == null || active.currentOverlay == null) return;

        float f = Mathf.Clamp(factor, minScaleFactor, maxScaleFactor);
        active.scaleFactorCmd = f;
        active.currentOverlay.localScale = active.baseCurrentLocalScale * f;
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
        HideFeedbackUI();

        RebuildItemsFromScene();
        BeginNextTrial();
    }

    void Start()
    {
        HideFeedbackUI();
        RebuildItemsFromScene();
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

        // ---------------- MACRO drive happens here (like Lego) ----------------
        // Only when NOT externally driven by micro, and only when holding gating passes.
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

        float curFactor = CurrentFactor(active);
        float err = Mathf.Abs(curFactor - active.targetFactor);

        UpdateProgressUI(curFactor, err, evalAllowed);

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
        if (overlaysCurrentRoot == null || overlaysTargetRoot == null)
        {
            Debug.LogError("[ScaleCore] Missing overlay roots.");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            if (logDebug) Debug.Log("[ScaleCore] Block finished.");
            FinishBlock();
            return;
        }

        if (items.Count == 0)
        {
            RebuildItemsFromScene();
            if (items.Count == 0)
            {
                Debug.LogError("[ScaleCore] No matched overlay pairs. Ensure ToolId on overlays.");
                FinishBlock();
                return;
            }
        }

        active = items[trialIndex % items.Count];

        EnsureActiveToolBody(active);

        // Reset macro state per trial
        active.axisAccum = 0f;
        active.haveWristPrev = false;

        // Reset current overlay to baseline scale
        if (active.currentOverlay != null)
            active.currentOverlay.localScale = active.baseCurrentLocalScale;

        active.scaleFactorCmd = 1f;
        _externalDriving = false;

        // Sample target factor
        float tf = Random.Range(Mathf.Min(targetFactorMin, targetFactorMax), Mathf.Max(targetFactorMin, targetFactorMax));
        tf = Mathf.Clamp(tf, minScaleFactor, maxScaleFactor);
        active.targetFactor = tf;

        // Show target overlay at baseline * targetFactor
        if (active.targetOverlay != null)
            active.targetOverlay.localScale = active.baseTargetLocalScale * active.targetFactor;

        // Bring overlays near active tool (optional)
        if (bringOverlaysNearActiveTool)
            MoveOverlaysNearActiveTool(active);

        // Show only the active overlays (both roots)
        SetOnlyActiveOverlaysVisible(active.id);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        OnTrialChanged?.Invoke(trialIndex + 1, totalTrials);

        if (logDebug)
            Debug.Log($"[ScaleCore] Trial {trialIndex + 1}/{totalTrials} id={active.id} targetFactor={active.targetFactor:F2}");
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        HideFeedbackUI();

        if (success)
        {
            if (forceReleaseAfterTrial) ForceReleaseIfPossible();

            if (snapOnSuccess && active != null && active.currentOverlay != null)
            {
                active.currentOverlay.localScale = active.baseCurrentLocalScale * active.targetFactor;
                active.scaleFactorCmd = active.targetFactor;
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

        // Restore overlays to their original placement pose
        RestoreOverlayBasePose(active);

        if (resetScaleAfterTrial && active != null && active.currentOverlay != null)
        {
            active.currentOverlay.localScale = active.baseCurrentLocalScale;
            active.scaleFactorCmd = 1f;
        }

        if (forceReleaseAfterTrial) ForceReleaseIfPossible();

        trialIndex++;
        inTransition = false;
        BeginNextTrial();
    }

    // ---------------- Registry ----------------
    private void RebuildItemsFromScene()
    {
        items.Clear();
        if (overlaysCurrentRoot == null || overlaysTargetRoot == null) return;

        // Cache tool bodies by id
        _toolBodiesById = BuildToolBodiesById();

        // current overlays: id -> ToolId transform
        var curIds = overlaysCurrentRoot.GetComponentsInChildren<ToolId>(true);
        var curMap = new Dictionary<string, Transform>();
        foreach (var tid in curIds)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
            curMap[tid.id] = tid.transform;
        }

        // target overlays: id -> ToolId transform
        var tgtIds = overlaysTargetRoot.GetComponentsInChildren<ToolId>(true);
        var tgtMap = new Dictionary<string, Transform>();
        foreach (var tid in tgtIds)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
            tgtMap[tid.id] = tid.transform;
        }

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
                scaleFactorCmd = 1f,
                toolBody = null,
                axisAccum = 0f,
                haveWristPrev = false,
                wristPrev = Vector3.zero,
                basePoseCaptured = false
            };

            // Assign tool body if available
            if (_toolBodiesById != null && _toolBodiesById.TryGetValue(it.id, out var rb))
                it.toolBody = rb;

            // Capture original placement pose now (so we can restore after trial)
            CaptureOverlayBasePoseIfNeeded(it);

            items.Add(it);
        }

        items.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        if (logDebug)
            Debug.Log($"[ScaleCore] Registry rebuilt: matched={items.Count}");
    }

    private Dictionary<string, Rigidbody> BuildToolBodiesById()
    {
        Transform toolsRoot = toolsDynamicRootOverride;

        if (toolsRoot == null)
        {
            if (overlaysCurrentRoot != null && overlaysCurrentRoot.parent != null)
                toolsRoot = overlaysCurrentRoot.parent.Find("Tools_Dynamic");
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

    private void EnsureActiveToolBody(Item it)
    {
        if (it == null) return;
        if (it.toolBody != null) return;

        if (_toolBodiesById == null)
            _toolBodiesById = BuildToolBodiesById();

        if (_toolBodiesById != null && _toolBodiesById.TryGetValue(it.id, out var rb))
            it.toolBody = rb;
    }

    // ---------------- Bring overlays near tool ----------------
    private void CaptureOverlayBasePoseIfNeeded(Item it)
    {
        if (it == null || it.basePoseCaptured) return;

        if (it.currentOverlay != null)
        {
            it.baseCurrentPos = it.currentOverlay.position;
            it.baseCurrentRot = it.currentOverlay.rotation;
        }

        if (it.targetOverlay != null)
        {
            it.baseTargetPos = it.targetOverlay.position;
            it.baseTargetRot = it.targetOverlay.rotation;
        }

        it.basePoseCaptured = true;
    }

    private void RestoreOverlayBasePose(Item it)
    {
        if (it == null || !it.basePoseCaptured) return;

        if (it.currentOverlay != null)
        {
            it.currentOverlay.position = it.baseCurrentPos;
            it.currentOverlay.rotation = it.baseCurrentRot;
        }

        if (it.targetOverlay != null)
        {
            it.targetOverlay.position = it.baseTargetPos;
            it.targetOverlay.rotation = it.baseTargetRot;
        }
    }

    private void MoveOverlaysNearActiveTool(Item it)
    {
        if (it == null) return;
        if (it.toolBody == null) return;

        Transform tool = it.toolBody.transform;
        Transform cam = Camera.main != null ? Camera.main.transform : null;

        Vector3 right, up, fwd;
        if (offsetInCameraFrame && cam != null)
        {
            right = cam.right;
            up = cam.up;
            fwd = cam.forward;
        }
        else
        {
            right = tool.right;
            up = Vector3.up;
            fwd = tool.forward;
        }

        if (it.currentOverlay != null)
        {
            it.currentOverlay.position =
                tool.position +
                right * currentOverlayOffsetLocal.x +
                up * currentOverlayOffsetLocal.y +
                fwd * currentOverlayOffsetLocal.z;

            if (matchOverlayRotationToTool)
                it.currentOverlay.rotation = tool.rotation;
        }

        if (it.targetOverlay != null)
        {
            it.targetOverlay.position =
                tool.position +
                right * targetOverlayOffsetLocal.x +
                up * targetOverlayOffsetLocal.y +
                fwd * targetOverlayOffsetLocal.z;

            if (matchOverlayRotationToTool)
                it.targetOverlay.rotation = tool.rotation;
        }
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
        if (active == null || active.currentOverlay == null) return;

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
    private float CurrentFactor(Item it)
    {
        if (it == null || it.currentOverlay == null) return 1f;

        float baseS = AvgAbs(it.baseCurrentLocalScale);
        float curS = AvgAbs(it.currentOverlay.localScale);
        if (baseS <= 1e-6f) return 1f;
        return curS / baseS;
    }

    private static float AvgAbs(Vector3 v)
    {
        return (Mathf.Abs(v.x) + Mathf.Abs(v.y) + Mathf.Abs(v.z)) / 3f;
    }

    private void SetOnlyActiveOverlaysVisible(string activeId)
    {
        SetVisibleById(overlaysCurrentRoot, activeId);
        SetVisibleById(overlaysTargetRoot, activeId);
    }

    private void SetVisibleById(Transform root, string activeId)
    {
        if (root == null) return;

        var ids = root.GetComponentsInChildren<ToolId>(true);
        foreach (var tid in ids)
        {
            if (tid == null || string.IsNullOrEmpty(tid.id)) continue;
            bool on = (tid.id == activeId);

            var rs = tid.GetComponentsInChildren<Renderer>(true);
            for (int i = 0; i < rs.Length; i++)
                rs[i].enabled = on;
        }
    }

    private void UpdateProgressUI(float curFactor, float err, bool evalAllowed)
    {
        if (progressText == null) return;

        sb.Length = 0;

        string id = ActiveId ?? "N/A";
        sb.AppendLine($"Scaling: {id}");
        sb.AppendLine($"Trial: {trialIndex + 1}/{totalTrials}");
        sb.AppendLine($"Time: {TrialTimeRemainingSec:F1}s");
        sb.AppendLine($"cur: {curFactor:F2}  target: {ActiveTargetFactor:F2}  err: {err:F2}  tol: {scaleFactorTolerance:F2}");

        bool holding = (grabber != null && grabber.IsHolding);
        sb.AppendLine($"Holding: {(holding ? "YES" : "NO")}  EvalAllowed: {(evalAllowed ? "YES" : "NO")}  ExternalDriving: {(_externalDriving ? "YES" : "NO")}");

        progressText.text = sb.ToString();
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null) grabber.ForceRelease();
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

    private void FinishBlock()
    {
        trialRunning = false;
        inTransition = false;
        HideFeedbackUI();
        try { OnBlockFinished?.Invoke(); } catch { }
    }
}
