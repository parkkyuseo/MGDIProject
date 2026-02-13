using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class ToolPlacementTaskManager : MonoBehaviour
{
    // Fired when the block finishes (all trials complete).
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged;               // (current1Based, total)
    public event Action<int, int> OnProgressChanged;            // (placedCount, totalCount)
    public event Action<float, bool> OnConfirmProgress;         // (t01, eligible)
    public event Action OnConfirmDwellCompleted;

    [Header("Roots (auto-discovered via ToolId)")]
    [Tooltip("Root that contains the movable tool instances (Tools_Dynamic).")]
    [SerializeField] private Transform toolsDynamicRoot;

    [Tooltip("Root that contains the fixed target ghosts (Slots_Targets).")]
    [SerializeField] private Transform slotsTargetsRoot;

    [Header("Grab behavior per task")]
    [Tooltip("ProxyHandGrabber instance (preferred). Used for rotation lock and requireNotHolding.")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("If true, lock rotation during placement trials (translation-only).")]
    [SerializeField] private bool lockRotationDuringPlacement = true;

    [Tooltip("Rotation mode to use during placement trials when lockRotationDuringPlacement is true.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode placementGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [Tooltip("If true, restore rotation mode after each trial.")]
    [SerializeField] private bool restoreRotationModeAfterTrial = true;

    [Tooltip("Rotation mode to restore after placement trial ends.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode restoreGrabberModeAfterTrial = ProxyHandGrabber.HeldRotationMode.FollowAnchor;

    [Header("Success metric")]
    [Tooltip("If true, success uses Renderer.bounds.center distance (robust to pivot offsets). Recommended.")]
    [SerializeField] private bool useBoundsCenterForSuccess = true;

    [Tooltip("Per-tool tolerance = max(0.05 * startDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.01f;

    [Tooltip("If true, only evaluate success when not holding anything.")]
    [SerializeField] private bool requireNotHolding = true;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 35f;
    [SerializeField] private float dwellSeconds = 0.20f;
    [SerializeField] private float confirmDwellSeconds = 3.0f;
    [SerializeField] private float stablePosSpeedMetersPerSec = 0.02f;
    [SerializeField] private float stableRotSpeedDegPerSec = 8.0f;
    [SerializeField] private float stableWarmupSeconds = 0.25f;

    [Header("Snap / Reset (snap kept but default off)")]
    [SerializeField] private bool snapOnSuccess = false;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetToolsToStartAfterTrial = true;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    // ---- Forced active (Workflow integration) ----
    [SerializeField] private bool workflowSingleActiveMode = true; // workflow에서는 true 권장
    [SerializeField] private bool finishBlockAfterOneSuccessWhenForced = true;

    private string _forcedActiveId = null;
    private Item _active = null;

    public string ForcedActiveId => _forcedActiveId;

    public void SetForcedActiveId(string id)
    {
        _forcedActiveId = string.IsNullOrEmpty(id) ? null : id;
    }

    public void ClearForcedActiveId()
    {
        _forcedActiveId = null;
    }

    [Header("Progress UI (optional)")]
    [Tooltip("Optional UnityEngine.UI.Text or TMP wrapper you already use elsewhere. If null, only Debug.Log/DebugHUD is used.")]
    [SerializeField] private UnityEngine.UI.Text progressText;

    [Tooltip("If true, show per-tool distances in progressText (debug-ish).")]
    [SerializeField] private bool showPerToolLines = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 10;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = true;
    [SerializeField] private string startKeyword = "start";
    [SerializeField] private string restartKeyword = "restart";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    // ---------------- Runtime ----------------
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

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, Action> keywordActions;

    // Tools/Targets registry
    [Serializable]
    private class Item
    {
        public string id;
        public Transform tool;
        public Transform target;

        public Renderer toolR;
        public Renderer targetR;

        public Transform startParent;
        public Vector3 startPos;
        public Quaternion startRot;

        public float tolerance;   // per tool (computed at trial start)
        public bool placed;        // current frame pass/fail
        public float lastErr;      // last computed error meters
    }

    private readonly List<Item> items = new List<Item>();
    private readonly StringBuilder sb = new StringBuilder(512);

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;
    public int ToolCount => items.Count;

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;

        HideFeedbackUI();

        // rebuild once at block start (tools/targets are fixed in scene)
        RebuildItemsFromScene();

        BeginNextTrial();
    }

    // ---------------- Unity ----------------
    void Start()
    {
        if (enableVoiceStart)
            SetupVoiceCommands();

        HideFeedbackUI();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
    }

    void Update()
    {
        if (!trialRunning || inTransition) return;

        float dt = Time.deltaTime;
        trialTimer += dt;

        if (trialTimer >= trialTimeoutSeconds)
        {
            OnConfirmProgress?.Invoke(0f, false);
            ResetConfirmState();
            StartCoroutine(EndTrialRoutine(false, true));
            return;
        }

        // ---- Workflow single-active evaluation (forced id only) ----
        bool useSingleActiveEval = workflowSingleActiveMode && !string.IsNullOrEmpty(_forcedActiveId);

        if (useSingleActiveEval && _active != null)
        {
            float err = ComputeErrorMeters(_active);
            _active.lastErr = err;

            bool pass = (err <= _active.tolerance);
            _active.placed = pass;

            OnProgressChanged?.Invoke(pass ? 1 : 0, 1);
            UpdateProgressUI();

            bool stable = ComputeActiveStability(dt);
            bool eligible = IsConfirmEligible(_active, err, stable);

            if (eligible && !confirmLatched)
                confirmDwellTimer += dt;
            else
                confirmDwellTimer = 0f;

            float confirmDuration = Mathf.Max(0.0001f, confirmDwellSeconds);
            float t01 = Mathf.Clamp01(confirmDwellTimer / confirmDuration);
            OnConfirmProgress?.Invoke(t01, eligible);

            if (!confirmLatched && confirmDwellTimer >= confirmDuration)
            {
                confirmLatched = true;
                confirmDwellTimer = 0f;
                OnConfirmDwellCompleted?.Invoke();
                EndTrialSuccess(_active);
            }

            return;
        }

        OnConfirmProgress?.Invoke(0f, false);

        // ---- Fallback: original "all tools" behavior (kept for compatibility) ----
        if (requireNotHolding && grabber != null && grabber.IsHolding)
        {
            dwellTimer = 0f;
            OnConfirmProgress?.Invoke(0f, false);
            UpdateProgressUI();
            return;
        }

        int placedCount = 0;
        for (int i = 0; i < items.Count; i++)
        {
            float err = ComputeErrorMeters(items[i]);
            items[i].lastErr = err;
            bool pass = (err <= items[i].tolerance);
            items[i].placed = pass;
            if (pass) placedCount++;
        }

        OnProgressChanged?.Invoke(placedCount, items.Count);
        UpdateProgressUI();

        if (items.Count > 0 && placedCount == items.Count)
        {
            dwellTimer += dt;
            if (dwellTimer >= dwellSeconds)
                EndTrialSuccess(null);
        }
        else
        {
            dwellTimer = 0f;
        }
    }

    // ---------------- Trial Flow ----------------
    private void BeginNextTrial()
    {
        if (toolsDynamicRoot == null || slotsTargetsRoot == null)
        {
            Debug.LogError("[ToolPlacementTaskManager] Missing roots (toolsDynamicRoot / slotsTargetsRoot).");
            FinishBlock();
            return;
        }

        // If forced mode is used, we generally want a single successful trial then finish.
        // Still keep the totalTrials guard for safety when not forced.
        if (string.IsNullOrEmpty(_forcedActiveId))
        {
            if (totalTrials > 0 && trialIndex >= totalTrials)
            {
                if (logDebug) Debug.Log("[ToolPlacementTaskManager] Block finished.");
                FinishBlock();
                return;
            }
        }

        // Ensure items exist
        if (items.Count == 0)
        {
            RebuildItemsFromScene();
            if (items.Count == 0)
            {
                Debug.LogError("[ToolPlacementTaskManager] No matched tool/target pairs found. Add ToolId to tools and ghosts.");
                FinishBlock();
                return;
            }
        }

        // Translation-only: lock rotation during placement trials
        if (lockRotationDuringPlacement)
            SetGrabberRotationMode(placementGrabberMode);

        // Reset tools to their captured start poses
        /* if (resetToolsToStartAfterTrial)
         *     ResetAllToolsToStartPose(); */
        if (resetToolsToStartAfterTrial)
        {
            if (workflowSingleActiveMode && _active != null)
                ResetToolToStartPose(_active);
            else
                ResetAllToolsToStartPose();
        }

        // Refresh cached renderers each trial (safe)
        for (int i = 0; i < items.Count; i++)
        {
            items[i].toolR = items[i].tool != null ? items[i].tool.GetComponentInChildren<Renderer>(true) : null;
            items[i].targetR = items[i].target != null ? items[i].target.GetComponentInChildren<Renderer>(true) : null;
            items[i].placed = false;
            items[i].lastErr = float.MaxValue;
        }

        // ---- Select ACTIVE item (forced id preferred) ----
        _active = null;

        if (!string.IsNullOrEmpty(_forcedActiveId))
        {
            for (int i = 0; i < items.Count; i++)
            {
                if (items[i] != null && items[i].id == _forcedActiveId)
                {
                    _active = items[i];
                    break;
                }
            }

            if (_active == null)
            {
                Debug.Log($"[ToolPlacementTM] ForcedActiveId '{_forcedActiveId}' not found in registry. Falling back to first item.");
            }
        }

        // If forced id is not used, active selection is not required for fallback(all-tools) mode.
        // Keep _active null here to avoid accidental single-item behavior.
        if (!string.IsNullOrEmpty(_forcedActiveId))
        {
            if (_active == null && items.Count > 0)
                _active = items[0];
        }

        if (_active == null)
        {
            Debug.Log("[ToolPlacementTM] Active item selection failed (registry empty?).");
            FinishBlock();
            return;
        }

        // Compute tolerance for the ACTIVE item only (based on current start distance)
        float d0 = ComputeErrorMeters(_active);
        _active.tolerance = Mathf.Max(0.05f * d0, minTolMeters);

        trialTimer = 0f;
        dwellTimer = 0f;
        ResetConfirmState();
        InitializeConfirmPoseFromActive();
        trialRunning = true;
        inTransition = false;

        // For workflow, trial count UI isn't meaningful; still send something stable.
        int shownTotal = string.IsNullOrEmpty(_forcedActiveId) ? totalTrials : 1;
        int shownIndex = string.IsNullOrEmpty(_forcedActiveId) ? (trialIndex + 1) : 1;

        OnTrialChanged?.Invoke(shownIndex, shownTotal);
        OnProgressChanged?.Invoke(0, 1);
        UpdateProgressUI();

        if (logDebug)
        {
            Debug.Log($"[ToolPlacementTM] Trial {shownIndex}/{shownTotal} active={_active.id} tol={_active.tolerance:F3}m timeout={trialTimeoutSeconds:F0}s confirmDwell={confirmDwellSeconds:F2}s forced={(string.IsNullOrEmpty(_forcedActiveId) ? "NO" : "YES")}");
        }
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;
        ResetConfirmState();

        HideFeedbackUI();

        if (success)
        {
            ForceReleaseIfPossible();

            if (snapOnSuccess)
                SnapAllToolsToTargets(); // NOTE: in single-active mode, this snaps all tools; keep snapOnSuccess=false for workflow

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

        // Restore rotation policy after the trial
        if (restoreRotationModeAfterTrial)
            SetGrabberRotationMode(restoreGrabberModeAfterTrial);

        if (resetToolsToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            if (workflowSingleActiveMode && _active != null)
                ResetToolToStartPose(_active);
            else
                ResetAllToolsToStartPose();
        }

        // ---- If forced active id is set, finish after one successful trial (workflow step) ----
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

    // ---------------- Registry / Reset ----------------
    private void RebuildItemsFromScene()
    {
        items.Clear();

        if (toolsDynamicRoot == null || slotsTargetsRoot == null) return;

        // tools: id -> ToolId transform
        var toolIds = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        var toolMap = new Dictionary<string, Transform>();
        for (int i = 0; i < toolIds.Length; i++)
        {
            if (toolIds[i] == null || string.IsNullOrEmpty(toolIds[i].id)) continue;
            toolMap[toolIds[i].id] = toolIds[i].transform;
        }

        // targets: id -> ToolId transform (ghost)
        var targetIds = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        var targetMap = new Dictionary<string, Transform>();
        for (int i = 0; i < targetIds.Length; i++)
        {
            if (targetIds[i] == null || string.IsNullOrEmpty(targetIds[i].id)) continue;
            targetMap[targetIds[i].id] = targetIds[i].transform;
        }

        // Intersection only
        foreach (var kv in toolMap)
        {
            if (!targetMap.TryGetValue(kv.Key, out var tgt)) continue;
            var toolTf = kv.Value;

            var it = new Item
            {
                id = kv.Key,
                tool = toolTf,
                target = tgt,
                toolR = toolTf != null ? toolTf.GetComponentInChildren<Renderer>(true) : null,
                targetR = tgt != null ? tgt.GetComponentInChildren<Renderer>(true) : null,
                startParent = toolTf != null ? toolTf.parent : null,
                startPos = toolTf != null ? toolTf.position : Vector3.zero,
                startRot = toolTf != null ? toolTf.rotation : Quaternion.identity,
                tolerance = minTolMeters,
                placed = false,
                lastErr = float.MaxValue
            };

            items.Add(it);
        }

        // Stable ordering for UI (by id)
        items.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        if (logDebug)
            Debug.Log($"[ToolPlacementTM] Registry rebuilt: tools={toolMap.Count}, targets={targetMap.Count}, matched={items.Count}");
    }

    private void ResetAllToolsToStartPose()
    {
        for (int i = 0; i < items.Count; i++)
        {
            var it = items[i];
            if (it.tool == null) continue;

            // Ensure correct parent (grabber parents to grabAnchor while holding)
            if (it.startParent != null)
                it.tool.SetParent(it.startParent, true);

            it.tool.SetPositionAndRotation(it.startPos, it.startRot);
        }
    }

    // ---------------- Metric helpers ----------------
    private float ComputeErrorMeters(Item it)
    {
        if (it == null || it.tool == null || it.target == null)
            return float.MaxValue;

        if (!useBoundsCenterForSuccess)
        {
            return Vector3.Distance(it.tool.position, it.target.position);
        }

        if (it.toolR == null) it.toolR = it.tool.GetComponentInChildren<Renderer>(true);
        if (it.targetR == null) it.targetR = it.target.GetComponentInChildren<Renderer>(true);

        if (it.toolR == null || it.targetR == null)
            return Vector3.Distance(it.tool.position, it.target.position);

        return Vector3.Distance(it.toolR.bounds.center, it.targetR.bounds.center);
    }

    private bool ComputeActiveStability(float dt)
    {
        if (_active == null || _active.tool == null || dt <= 0f)
            return false;

        Vector3 currentPos = _active.tool.position;
        Quaternion currentRot = _active.tool.rotation;

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

    private bool IsConfirmEligible(Item it, float errorMeters, bool stable)
    {
        if (!IsTrialRunning) return false;
        if (it == null || it.tool == null) return false;
        if (errorMeters > it.tolerance) return false;
        if (!stable) return false;
        if (!IsNotHolding()) return false;
        return true;
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
        confirmPrevPoseValid = false;
        confirmPrevPos = Vector3.zero;
        confirmPrevRot = Quaternion.identity;
    }

    private void InitializeConfirmPoseFromActive()
    {
        if (_active == null || _active.tool == null) return;
        confirmPrevPos = _active.tool.position;
        confirmPrevRot = _active.tool.rotation;
        confirmPrevPoseValid = true;
    }

    private void EndTrialSuccess(Item it)
    {
        _ = it;
        if (inTransition || !trialRunning) return;
        StartCoroutine(EndTrialRoutine(true, false));
    }

    // ---------------- Optional snap (kept) ----------------
    private void SnapAllToolsToTargets()
    {
        for (int i = 0; i < items.Count; i++)
        {
            SnapToolToTarget(items[i]);
        }
    }

    private void SnapToolToTarget(Item it)
    {
        if (it == null || it.tool == null || it.target == null) return;

        if (!useBoundsCenterForSuccess)
        {
            it.tool.position = it.target.position;
            return;
        }

        if (it.toolR == null) it.toolR = it.tool.GetComponentInChildren<Renderer>(true);
        if (it.targetR == null) it.targetR = it.target.GetComponentInChildren<Renderer>(true);

        if (it.toolR == null || it.targetR == null)
        {
            it.tool.position = it.target.position;
            return;
        }

        Vector3 toolCenter = it.toolR.bounds.center;
        Vector3 targetCenter = it.targetR.bounds.center;
        Vector3 delta = targetCenter - toolCenter;

        it.tool.position += delta;
    }

    // ---------------- Feedback UI ----------------
    private void UpdateProgressUI()
    {
        int placed = 0;
        for (int i = 0; i < items.Count; i++)
            if (items[i].placed) placed++;

        if (progressText == null)
        {
            // Still allow DebugHUD if available
            try
            {
                /* sb.Length = 0;
                 * sb.Append($"[Progress] {placed}/{items.Count}");
                 * DebugHUD.Log(sb.ToString()); */
                /* if (_active != null)
                 *     DebugHUD.Log($"PLACE id={_active.id} err={_active.lastErr:F3} tol={_active.tolerance:F3} holding={(grabber!=null && grabber.IsHolding)}"); */
            }
            catch { }
            return;
        }

        sb.Length = 0;
        sb.AppendLine($"Placement: {placed}/{items.Count}");
        sb.AppendLine($"Trial: {trialIndex + 1}/{totalTrials}");
        sb.AppendLine($"Time: {TrialTimeRemainingSec:F1}s");

        if (requireNotHolding && grabber != null && grabber.IsHolding)
            sb.AppendLine("Holding: YES (eval paused)");
        else
            sb.AppendLine($"Holding: {(grabber != null && grabber.IsHolding ? "YES" : "NO")}");

        if (showPerToolLines)
        {
            sb.AppendLine();
            for (int i = 0; i < items.Count; i++)
            {
                var it = items[i];
                string mark = it.placed ? "✓" : "·";
                sb.AppendLine($"{mark} {it.id}: err={it.lastErr:F3}m tol={it.tolerance:F3}m");
            }
        }

        progressText.text = sb.ToString();
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

    // ---------------- Grabber hooks ----------------
    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
        {
            grabber.ForceRelease();
        }
    }

    private void SetGrabberRotationMode(ProxyHandGrabber.HeldRotationMode mode)
    {
        if (grabber != null)
        {
            grabber.SetHeldRotationMode(mode);
        }
    }

    // ---------------- Voice ----------------
    private void SetupVoiceCommands()
    {
        if (keywordRecognizer != null) return;

        keywordActions = new Dictionary<string, Action>
        {
            { startKeyword.ToLower(), StartBlock },
            { restartKeyword.ToLower(), StartBlock }
        };

        keywordRecognizer = new KeywordRecognizer(keywordActions.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += args =>
        {
            if (keywordActions.TryGetValue(args.text.ToLower(), out var action))
                action.Invoke();
        };
        keywordRecognizer.Start();

        if (logDebug)
            Debug.Log($"[ToolPlacementTM] Voice enabled: '{startKeyword}', '{restartKeyword}'");
    }

    private void FinishBlock()
    {
        trialRunning = false;
        inTransition = false;
        ResetConfirmState();
        HideFeedbackUI();

        try { OnBlockFinished?.Invoke(); } catch { }
    }

    private void OnDisable()
    {
        // Prevent mode leaking into other tasks if this manager is disabled mid-run
        if (restoreRotationModeAfterTrial)
            SetGrabberRotationMode(restoreGrabberModeAfterTrial);

        if (keywordRecognizer != null)
        {
            keywordRecognizer.Stop();
            keywordRecognizer.Dispose();
            keywordRecognizer = null;
        }
    }

    private void ResetToolToStartPose(Item it)
    {
        if (it == null || it.tool == null) return;
        if (it.startParent != null)
            it.tool.SetParent(it.startParent, true);
        it.tool.SetPositionAndRotation(it.startPos, it.startRot);
    }
}
