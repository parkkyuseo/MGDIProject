using System;
using System.Linq;
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
    public event Action<float, bool> OnConfirmProgress; // (t01, eligible)
    public event Action OnConfirmDwellCompleted;
    public event Action<string> OnConfirmStatus;
    public event Action<bool, bool> OnTrialEnded; // (success, timedOut)

    [Header("Roots (auto-discovered via ToolId)")]
    [SerializeField] private Transform toolsDynamicRoot;
    [SerializeField] private Transform slotsTargetsRoot;

    [Header("Grab + Phone (optional)")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("If assigned, used to detect Macro/Micro and to auto-place hand near active tool in Micro.")]
    [SerializeField] private PhoneInputRouter phoneRouter;

    [Tooltip("Optional: auto-place the proxy hand near the active tool in Micro so tap-to-grab works immediately.")]
    [SerializeField] private MicroHandAutoPlacer microAutoPlacer;

    [Header("Grabber rotation policy (Full 3D)")]
    [Tooltip("Macro: allow direct manipulation rotation (3DOF) by following grabAnchor rotation.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode macroGrabberMode = ProxyHandGrabber.HeldRotationMode.FollowAnchor;

    [Tooltip("Micro: micro controller rotates held object; grabber must not override rotation.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode microGrabberMode = ProxyHandGrabber.HeldRotationMode.ExternalControl;

    [Tooltip("Rotation mode to restore when this manager is disabled.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode restoreGrabberModeOnDisable = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private AudioClip confirmClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Progress UI (optional)")]
    [SerializeField] private UnityEngine.UI.Text progressText;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;
    [SerializeField] private float confirmDwellSeconds = 3.0f;
    [SerializeField] private float stablePosSpeedMetersPerSec = 0.02f;
    [SerializeField] private float stableRotSpeedDegPerSec = 8.0f;
    [SerializeField] private float stableWarmupSeconds = 0.25f;

    [Header("Full 3D Success Threshold")]
    [SerializeField] private float rotationToleranceDeg = 12f;

    [Header("Evaluation gating")]
    [Tooltip("If true, success dwell accumulates only when input is not actively driving (recommended for micro).")]
    [SerializeField] private bool requireNotDrivingForEvaluation = true;

    [Tooltip("If true, macro evaluation requires releasing the object (simple 'stop input to evaluate' proxy).")]
    [SerializeField] private bool requireReleaseForEvaluationInMacro = true;

    [Tooltip("If true, confirm dwell accumulates only when not holding any object.")]
    [SerializeField] private bool requireNotHolding = true;

    [Header("Target Rotation Source")]
    [SerializeField] private bool autoGenerateTargetRotation = true;

    [Header("Target rotation sampling (Full 3D)")]
    [Tooltip("Yaw random range in degrees (around world up in start-base frame).")]
    [SerializeField] private float yawMinDeg = 20f;
    [SerializeField] private float yawMaxDeg = 80f;

    [Tooltip("Pitch random range in degrees (around start-base right axis).")]
    [SerializeField] private float pitchMinDeg = 10f;
    [SerializeField] private float pitchMaxDeg = 50f;

    [Tooltip("Roll random range in degrees (around start-base forward axis).")]
    [SerializeField] private float rollMinDeg = 10f;
    [SerializeField] private float rollMaxDeg = 50f;

    [SerializeField] private bool randomizeSigns = true;

    [Header("Inter-trial behavior")]
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool snapRotationOnSuccess = true;
    [SerializeField] private bool resetToolToStartAfterTrial = true;
    [SerializeField] private bool preserveSolvedPoseForNextPhase = false;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Micro: allow rotation without holding")]
    [SerializeField] private bool allowMicroRotateWithoutHolding = true;


    public Rigidbody ActiveToolBody => _active != null ? _active.toolBody : null;
    public bool AllowMicroRotateWithoutHolding => allowMicroRotateWithoutHolding;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = false;
    [SerializeField] private string startKeyword = "start rotation";
    [SerializeField] private string restartKeyword = "restart rotation";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Target ghost positioning (bring near active tool)")]
    [SerializeField] private bool bringTargetNearActiveTool = true;

    [Tooltip("Offset near the tool when bringing target close. X=right, Y=up, Z=forward (in chosen frame). Meters.")]
    [SerializeField] private Vector3 targetOffsetLocal = new Vector3(0.18f, 0.03f, 0.00f);

    [Tooltip("If true, offset uses Camera frame (cam.right/up/fwd). If false, uses tool frame (tool.right/up/fwd).")]
    [SerializeField] private bool offsetInCameraFrame = true;

    // ---- Forced active (Workflow integration) ----
    [SerializeField] private bool finishBlockAfterOneSuccessWhenForced = true;
    private string _forcedActiveId = null;

    public void SetForcedActiveId(string id) => _forcedActiveId = NormalizeToolId(id);
    public void ClearForcedActiveId() => _forcedActiveId = null;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    [Header("Practice Target Randomization")]
    [SerializeField] private bool usePracticeRandomTargetRotation = true;
    [SerializeField] private Vector3 practiceRotationAbsEulerDeg = new Vector3(20f, 35f, 20f);
    [SerializeField] private float practiceMinRotationDeltaDeg = 10f;

    // Micro controller can set this true while it is actively rotating.
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
        public Transform targetEval;

        public Transform startParent;
        public Vector3 startPos;
        public Quaternion startRot;

        // Base pose snapshots (preserve "lying down" etc.)
        public Quaternion startBaseRot;
        public Quaternion targetBaseRot;
        public Vector3 targetBasePos;
        public Quaternion targetEvalBaseRot;
        public Vector3 targetScenePos;
        public Quaternion targetSceneRot;
        public Quaternion targetSceneEvalRot;

        // For snap on success
        public Quaternion targetDesiredRot;
    }

    private readonly List<Item> _items = new List<Item>();
    private int _trialIndex = 0;

    private float _trialTimer = 0f;
    private float _dwellTimer = 0f;
    private float _confirmDwellTimer = 0f;
    private float _stableWarmupTimer = 0f;
    private bool _confirmLatched = false;
    private Vector3 _confirmPrevPos = Vector3.zero;
    private Quaternion _confirmPrevRot = Quaternion.identity;
    private bool _confirmPrevPoseValid = false;

    private bool _trialRunning = false;
    private bool _inTransition = false;

    private Item _active;

    private KeywordRecognizer _keywordRecognizer;
    private Dictionary<string, Action> _keywordActions;

    private readonly StringBuilder _sb = new StringBuilder(512);
    private bool _practiceGhostRandomizationEnabled = false;
    private Quaternion _lastPracticeRotationOffset = Quaternion.identity;
    private bool _hasLastPracticeRotationOffset = false;
    private bool _hasCapturedCarryPose = false;
    private string _capturedCarryId = null;
    private Quaternion _capturedCarryToolRot = Quaternion.identity;
    private Quaternion _capturedCarryTargetRot = Quaternion.identity;
    private Quaternion _capturedCarryTargetEvalRot = Quaternion.identity;

    public bool IsTrialRunning => _trialRunning && !_inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - _trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => _trialIndex + 1;
    public float RotationToleranceDeg => rotationToleranceDeg;
    public float ActiveRotationErrorDeg => ComputeRotationErrorDeg();
    public bool ResetToolToStartAfterTrial
    {
        get => resetToolToStartAfterTrial;
        set => resetToolToStartAfterTrial = value;
    }
    public bool PreserveSolvedPoseForNextPhase
    {
        get => preserveSolvedPoseForNextPhase;
        set => preserveSolvedPoseForNextPhase = value;
    }

    public void SetPracticeGhostRandomization(bool enabled)
    {
        _practiceGhostRandomizationEnabled = enabled;
        _hasLastPracticeRotationOffset = false;

        if (enabled)
        {
            for (int i = 0; i < _items.Count; i++)
            {
                Item it = _items[i];
                if (it == null) continue;
                if (it.targetEval == null)
                    it.targetEval = ResolveTargetEvaluationTransform(it.target);
                if (it.targetEval != null)
                    it.targetEvalBaseRot = it.targetEval.rotation;
            }
        }
        else
        {
            RestoreAllPracticeTargetRotations();
        }
    }

    public Transform ActiveToolTransform => _active != null ? _active.tool : null;
    public string ActiveToolId => _active != null ? _active.id : null;

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        _inTransition = false;
        _trialRunning = false;
        _trialIndex = 0;
        _hasLastPracticeRotationOffset = false;

        _externalDriving = false;
        OnConfirmStatus?.Invoke("");
        OnConfirmProgress?.Invoke(0f, false);
        ResetConfirmState();
        HideFeedbackUI();

        RebuildItemsFromScene();
        BeginNextTrial();
    }

    public void CaptureActivePoseForCarry()
    {
        if (_active == null || _active.tool == null || _active.target == null) return;

        Transform eval = GetActiveEvaluationTargetTransform();
        if (eval == null) return;

        _capturedCarryId = _active.id;
        _capturedCarryToolRot = _active.tool.rotation;
        _capturedCarryTargetRot = _active.target.rotation;
        _capturedCarryTargetEvalRot = eval.rotation;
        _hasCapturedCarryPose = true;
    }

    public void ApplyCapturedPoseForId(string id)
    {
        if (!_hasCapturedCarryPose) return;
        if (string.IsNullOrEmpty(id)) return;
        if (!string.Equals(_capturedCarryId, id, StringComparison.OrdinalIgnoreCase)) return;

        if (_items.Count == 0)
            RebuildItemsFromScene();

        Item it = _items.FirstOrDefault(
            x => x != null && string.Equals(x.id, id, StringComparison.OrdinalIgnoreCase));
        if (it == null || it.tool == null || it.target == null) return;

        it.tool.rotation = _capturedCarryToolRot;
        it.target.rotation = _capturedCarryTargetRot;

        Transform eval = it.targetEval != null ? it.targetEval : ResolveTargetEvaluationTransform(it.target);
        if (eval != null)
        {
            it.targetEval = eval;
            eval.rotation = _capturedCarryTargetEvalRot;
        }
    }

    public void ClearCapturedCarryPose()
    {
        _hasCapturedCarryPose = false;
        _capturedCarryId = null;
        _capturedCarryToolRot = Quaternion.identity;
        _capturedCarryTargetRot = Quaternion.identity;
        _capturedCarryTargetEvalRot = Quaternion.identity;
    }

    public void ResetAllTargetsToSceneBaseline()
    {
        if (_items.Count == 0)
            RebuildItemsFromScene();

        for (int i = 0; i < _items.Count; i++)
        {
            Item it = _items[i];
            if (it == null || it.target == null) continue;

            it.target.position = it.targetScenePos;
            it.target.rotation = it.targetSceneRot;

            Transform eval = it.targetEval != null ? it.targetEval : ResolveTargetEvaluationTransform(it.target);
            if (eval != null)
            {
                it.targetEval = eval;
                eval.rotation = it.targetSceneEvalRot;
            }
        }
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
        if (!_trialRunning || _inTransition)
        {
            OnConfirmStatus?.Invoke("");
            return;
        }

        float dt = Time.deltaTime;
        _trialTimer += dt;

        if (_trialTimer >= trialTimeoutSeconds)
        {
            OnConfirmProgress?.Invoke(0f, false);
            OnConfirmStatus?.Invoke("");
            ResetConfirmState();
            StartCoroutine(EndTrialRoutine(success: false, timedOut: true));
            return;
        }

        if (_active == null || _active.tool == null || _active.target == null)
        {
            OnConfirmProgress?.Invoke(0f, false);
            OnConfirmStatus?.Invoke("");
            return;
        }

        // Evaluation gating ("stop input to evaluate")
        bool holding = (grabber != null && grabber.IsHolding);

        bool evalAllowed = true;
        if (requireNotDrivingForEvaluation)
        {
            bool macro = (phoneRouter == null) ? true : (phoneRouter.CurrentMode == PhoneInputRouter.Mode.Macro);

            if (macro && requireReleaseForEvaluationInMacro && holding)
                evalAllowed = false;

            if (_externalDriving)
                evalAllowed = false;
        }

        float errDeg = ComputeRotationErrorDeg();
        EmitRotationConfirmStatus(errDeg, rotationToleranceDeg);

        if (progressText != null)
            UpdateProgressText(errDeg, evalAllowed);

        bool stable = ComputeActiveStability(dt);
        bool eligible =
            IsTrialRunning &&
            ActiveToolTransform != null &&
            errDeg <= rotationToleranceDeg &&
            stable &&
            IsNotHolding() &&
            evalAllowed;

        if (eligible && !_confirmLatched)
            _confirmDwellTimer += dt;
        else
            _confirmDwellTimer = 0f;

        float confirmDuration = Mathf.Max(0.0001f, confirmDwellSeconds);
        float t01 = Mathf.Clamp01(_confirmDwellTimer / confirmDuration);
        OnConfirmProgress?.Invoke(t01, eligible);

        if (!_confirmLatched && _confirmDwellTimer >= confirmDuration)
        {
            _confirmLatched = true;
            _confirmDwellTimer = 0f;
            OnConfirmDwellCompleted?.Invoke();
            PlayConfirmSound();
            EndTrialSuccess();
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

        _active = null;
        bool hasForcedActive = !string.IsNullOrEmpty(_forcedActiveId);

        if (hasForcedActive)
        {
            _active = FindItemById(_forcedActiveId);
            if (_active == null)
            {
                RebuildItemsFromScene();
                _active = FindItemById(_forcedActiveId);
            }

            if (_active == null)
            {
                Debug.LogError($"[ToolRotationTM] ForcedActiveId '{_forcedActiveId}' not found. Trial start canceled.");
                _trialRunning = false;
                _inTransition = false;
                OnConfirmProgress?.Invoke(0f, false);
                OnConfirmStatus?.Invoke("");
                return;
            }
        }

        if (_active == null)
            _active = _items[_trialIndex % _items.Count];

        _active.targetEval = ResolveTargetEvaluationTransform(_active.target);
        EnsureActiveBody();

        // Reset active tool to its start pose at trial start
        ForceReleaseIfPossible();
        ResetActiveToolToStartPose();

        // Snapshot base poses (preserve pitch/roll shape)
        _active.startBaseRot = _active.tool.rotation;
        _active.targetBaseRot = _active.target.rotation;
        _active.targetBasePos = _active.target.position;
        Transform evalTargetAtStart = GetActiveEvaluationTargetTransform();
        if (evalTargetAtStart != null)
            _active.targetEvalBaseRot = evalTargetAtStart.rotation;

        // Move target near active tool (optional)
        if (bringTargetNearActiveTool)
            MoveTargetNearActiveTool();

        // Apply grabber mode based on technique (Macro vs Micro)
        ApplyGrabberModeForTechnique();

        // Micro UX: place proxy hand near active tool to enable immediate tap-to-grab
        if (phoneRouter != null && phoneRouter.CurrentMode == PhoneInputRouter.Mode.Micro)
        {
            if (microAutoPlacer != null)
                microAutoPlacer.PlaceHandNear(_active.tool);
        }

        if (_practiceGhostRandomizationEnabled && usePracticeRandomTargetRotation)
        {
            ApplyPracticeTargetRotation(_active);
        }
        else if (autoGenerateTargetRotation)
        {
            // Sample a full 3D target rotation offset relative to startBaseRot
            Quaternion offset = SampleRandomRotationOffset();
            _active.targetDesiredRot = offset * _active.startBaseRot; // left-multiply for "delta in world"
            _active.target.rotation = _active.targetDesiredRot;
        }

        _trialTimer = 0f;
        _dwellTimer = 0f;
        ResetConfirmState();
        InitializeConfirmPoseFromActive();
        _trialRunning = true;
        _inTransition = false;
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");

        OnTrialChanged?.Invoke(_trialIndex + 1, totalTrials);

        if (progressText != null)
            UpdateProgressText(ComputeRotationErrorDeg(), evalAllowed: true);

        if (logDebug)
            Debug.Log($"[ToolRotationTM] Trial {_trialIndex + 1}/{totalTrials} tool={_active.id} tol={rotationToleranceDeg:F1}deg");
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (_inTransition) yield break;
        _inTransition = true;
        _trialRunning = false;
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        ResetConfirmState();

        HideFeedbackUI();

        if (success)
        {
            if (snapRotationOnSuccess)
                SnapRotationToTarget();

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

        bool finishingForcedSuccess = success && !string.IsNullOrEmpty(_forcedActiveId) && finishBlockAfterOneSuccessWhenForced;
        bool keepSolvedPoseForScaling = finishingForcedSuccess && preserveSolvedPoseForNextPhase;

        if (!keepSolvedPoseForScaling)
            RestoreTargetPose();

        if (!keepSolvedPoseForScaling && resetToolToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            ResetActiveToolToStartPose();
        }

        OnTrialEnded?.Invoke(success, timedOut);

        if (finishingForcedSuccess)
        {
            FinishBlock();
            yield break;
        }

        _trialIndex++;
        BeginNextTrial();
    }

    private void RestoreTargetPose()
    {
        if (_active == null || _active.target == null) return;

        if (bringTargetNearActiveTool)
            _active.target.position = _active.targetBasePos;

        if (_practiceGhostRandomizationEnabled && usePracticeRandomTargetRotation)
        {
            RestorePracticeTargetRotation(_active);
        }
        else if (autoGenerateTargetRotation)
        {
            _active.target.rotation = _active.targetBaseRot;
        }
    }

    private void MoveTargetNearActiveTool()
    {
        if (_active == null || _active.tool == null || _active.target == null) return;

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
            right = _active.tool.right;
            up = _active.tool.up;
            fwd = _active.tool.forward;
        }

        Vector3 pos =
            _active.tool.position +
            right * targetOffsetLocal.x +
            up * targetOffsetLocal.y +
            fwd * targetOffsetLocal.z;

        _active.target.position = pos;
    }

    private void ApplyGrabberModeForTechnique()
    {
        if (grabber == null) return;

        bool isMacro = (phoneRouter == null) ? true : (phoneRouter.CurrentMode == PhoneInputRouter.Mode.Macro);
        grabber.SetHeldRotationMode(isMacro ? macroGrabberMode : microGrabberMode);
    }

    // ---------------- Registry ----------------
    private void RebuildItemsFromScene()
    {
        _items.Clear();
        if (toolsDynamicRoot == null || slotsTargetsRoot == null) return;

        var toolIds = toolsDynamicRoot.GetComponentsInChildren<ToolId>(true);
        var toolMap = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < toolIds.Length; i++)
        {
            if (toolIds[i] == null) continue;
            string id = NormalizeToolId(toolIds[i].id);
            if (string.IsNullOrEmpty(id)) continue;
            toolMap[id] = toolIds[i].transform;
        }

        var targetIds = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true);
        var targetMap = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < targetIds.Length; i++)
        {
            if (targetIds[i] == null) continue;
            string id = NormalizeToolId(targetIds[i].id);
            if (string.IsNullOrEmpty(id)) continue;
            targetMap[id] = targetIds[i].transform;
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
                targetEval = ResolveTargetEvaluationTransform(tgt),
                startParent = toolTf != null ? toolTf.parent : null,
                startPos = toolTf != null ? toolTf.position : Vector3.zero,
                startRot = toolTf != null ? toolTf.rotation : Quaternion.identity,
                toolBody = null,
                startBaseRot = Quaternion.identity,
                targetBaseRot = Quaternion.identity,
                targetBasePos = Vector3.zero,
                targetEvalBaseRot = Quaternion.identity,
                targetScenePos = tgt != null ? tgt.position : Vector3.zero,
                targetSceneRot = tgt != null ? tgt.rotation : Quaternion.identity,
                targetSceneEvalRot = Quaternion.identity,
                targetDesiredRot = Quaternion.identity
            };

            if (it.targetEval != null)
            {
                it.targetEvalBaseRot = it.targetEval.rotation;
                it.targetSceneEvalRot = it.targetEval.rotation;
            }

            _items.Add(it);
        }

        _items.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        if (logDebug)
            Debug.Log($"[ToolRotationTM] Registry rebuilt: matched={_items.Count}");
    }

    private static string NormalizeToolId(string id)
    {
        return string.IsNullOrWhiteSpace(id) ? null : id.Trim();
    }

    private Item FindItemById(string id)
    {
        if (string.IsNullOrEmpty(id)) return null;
        return _items.FirstOrDefault(
            it => it != null && string.Equals(it.id, id, StringComparison.OrdinalIgnoreCase));
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

    private float ComputeRotationErrorDeg()
    {
        if (_active == null || _active.tool == null || _active.target == null) return float.MaxValue;

        Transform evalTarget = GetActiveEvaluationTargetTransform();
        if (evalTarget == null) return float.MaxValue;

        return Quaternion.Angle(_active.tool.rotation, evalTarget.rotation);
    }

    private void EmitRotationConfirmStatus(float errorDeg, float toleranceDeg)
    {
        bool holding = requireNotHolding && grabber != null && grabber.IsHolding;
        bool withinTol = errorDeg <= toleranceDeg;

        string msg;
        if (holding)
            msg = "Release to confirm";
        else if (!withinTol)
            msg = "Align rotation";
        else
            msg = "Confirming...";

        OnConfirmStatus?.Invoke(msg);
    }

    private bool ComputeActiveStability(float dt)
    {
        Transform tf = ActiveToolTransform;
        if (tf == null || dt <= 0f)
            return false;

        Vector3 currentPos = tf.position;
        Quaternion currentRot = tf.rotation;

        bool stable = false;
        if (_stableWarmupTimer < stableWarmupSeconds)
        {
            _stableWarmupTimer += dt;
            stable = false;
        }
        else if (_confirmPrevPoseValid)
        {
            float posSpeed = Vector3.Distance(currentPos, _confirmPrevPos) / dt;
            float rotSpeedDeg = Quaternion.Angle(currentRot, _confirmPrevRot) / dt;
            stable = posSpeed <= stablePosSpeedMetersPerSec && rotSpeedDeg <= stableRotSpeedDegPerSec;
        }

        _confirmPrevPos = currentPos;
        _confirmPrevRot = currentRot;
        _confirmPrevPoseValid = true;
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
        _confirmDwellTimer = 0f;
        _stableWarmupTimer = 0f;
        _confirmLatched = false;
        _confirmPrevPos = Vector3.zero;
        _confirmPrevRot = Quaternion.identity;
        _confirmPrevPoseValid = false;
    }

    private void InitializeConfirmPoseFromActive()
    {
        Transform tf = ActiveToolTransform;
        if (tf == null) return;
        _confirmPrevPos = tf.position;
        _confirmPrevRot = tf.rotation;
        _confirmPrevPoseValid = true;
    }

    private void EndTrialSuccess()
    {
        if (_inTransition || !_trialRunning) return;
        StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
    }

    private void SnapRotationToTarget()
    {
        if (_active == null || _active.tool == null || _active.target == null) return;

        Transform evalTarget = GetActiveEvaluationTargetTransform();
        if (evalTarget == null) return;

        _active.tool.rotation = evalTarget.rotation;
    }

    private Transform GetActiveEvaluationTargetTransform()
    {
        if (_active == null || _active.target == null) return null;

        if (_active.targetEval == null)
            _active.targetEval = ResolveTargetEvaluationTransform(_active.target);

        return _active.targetEval != null ? _active.targetEval : _active.target;
    }

    private Transform ResolveTargetEvaluationTransform(Transform targetRoot)
    {
        if (targetRoot == null) return null;

        var resolver = targetRoot.GetComponent<GhostVisualResolver>();
        if (resolver != null && resolver.VisualRoot != null)
            return resolver.VisualRoot;

        Transform visualChild = targetRoot.Find("GhostVisual");
        if (visualChild != null)
            return visualChild;

        return targetRoot;
    }

    private Quaternion SampleRandomRotationOffset()
    {
        float yaw = Random.Range(yawMinDeg, yawMaxDeg);
        float pitch = Random.Range(pitchMinDeg, pitchMaxDeg);
        float roll = Random.Range(rollMinDeg, rollMaxDeg);

        if (randomizeSigns)
        {
            if (Random.value < 0.5f) yaw = -yaw;
            if (Random.value < 0.5f) pitch = -pitch;
            if (Random.value < 0.5f) roll = -roll;
        }

        // Ensure non-trivial rotation
        if (Mathf.Abs(yaw) < 1e-3f && Mathf.Abs(pitch) < 1e-3f && Mathf.Abs(roll) < 1e-3f)
            yaw = yawMinDeg;

        Quaternion qYaw = Quaternion.AngleAxis(yaw, Vector3.up);
        Quaternion qPitch = Quaternion.AngleAxis(pitch, Vector3.right);
        Quaternion qRoll = Quaternion.AngleAxis(roll, Vector3.forward);

        // Order matters; use yaw->pitch->roll
        return qYaw * qPitch * qRoll;
    }

    private void ApplyPracticeTargetRotation(Item it)
    {
        if (it == null || it.target == null) return;
        Transform evalTarget = GetActiveEvaluationTargetTransform();
        if (evalTarget == null) return;

        if (it.targetEvalBaseRot == Quaternion.identity)
            it.targetEvalBaseRot = evalTarget.rotation;

        Quaternion offset = SamplePracticeRotationOffset();
        evalTarget.rotation = it.targetEvalBaseRot * offset;
    }

    private Quaternion SamplePracticeRotationOffset()
    {
        Vector3 abs = new Vector3(
            Mathf.Abs(practiceRotationAbsEulerDeg.x),
            Mathf.Abs(practiceRotationAbsEulerDeg.y),
            Mathf.Abs(practiceRotationAbsEulerDeg.z));

        Quaternion candidate = Quaternion.identity;
        float minDelta = Mathf.Max(0f, practiceMinRotationDeltaDeg);

        for (int i = 0; i < 16; i++)
        {
            float pitch = Random.Range(-abs.x, abs.x);
            float yaw = Random.Range(-abs.y, abs.y);
            float roll = Random.Range(-abs.z, abs.z);
            candidate = Quaternion.Euler(pitch, yaw, roll);

            if (!_hasLastPracticeRotationOffset || Quaternion.Angle(candidate, _lastPracticeRotationOffset) >= minDelta)
                break;
        }

        _lastPracticeRotationOffset = candidate;
        _hasLastPracticeRotationOffset = true;
        return candidate;
    }

    private void RestorePracticeTargetRotation(Item it)
    {
        if (it == null) return;
        Transform evalTarget = GetActiveEvaluationTargetTransform();
        if (evalTarget == null) return;
        evalTarget.rotation = it.targetEvalBaseRot;
    }

    private void RestoreAllPracticeTargetRotations()
    {
        for (int i = 0; i < _items.Count; i++)
        {
            Item it = _items[i];
            if (it == null) continue;
            Transform eval = it.targetEval != null ? it.targetEval : ResolveTargetEvaluationTransform(it.target);
            if (eval == null) continue;
            if (it.targetEval == null) it.targetEval = eval;
            eval.rotation = it.targetEvalBaseRot;
        }
    }

    // ---------------- UI ----------------
    private void UpdateProgressText(float errDeg, bool evalAllowed)
    {
        if (progressText == null) return;

        _sb.Length = 0;
        string toolName = (_active != null) ? _active.id : "N/A";

        _sb.AppendLine($"Rotation (3D): {toolName}");
        _sb.AppendLine($"Trial: {_trialIndex + 1}/{totalTrials}");
        _sb.AppendLine($"Time: {TrialTimeRemainingSec:F1}s");
        _sb.AppendLine($"RotErr: {errDeg:F1}°  (tol {rotationToleranceDeg:F1}°)");
        _sb.AppendLine($"EvalAllowed: {(evalAllowed ? "YES" : "NO")}");

        bool holding = (grabber != null && grabber.IsHolding);
        _sb.AppendLine($"Holding: {(holding ? "YES" : "NO")}  MicroDriving: {(_externalDriving ? "ON" : "OFF")}");

        progressText.text = _sb.ToString();
    }

    // ---------------- Feedback ----------------
    private void PlaySnapSound()
    {
        if (audioSource != null && snapClip != null)
            audioSource.PlayOneShot(snapClip);
    }

    private void PlayConfirmSound()
    {
        if (audioSource != null && confirmClip != null)
            audioSource.PlayOneShot(confirmClip);
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
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        ResetConfirmState();
        HideFeedbackUI();
        try { OnBlockFinished?.Invoke(); } catch { }
    }

    private void OnDisable()
    {
        OnConfirmProgress?.Invoke(0f, false);
        OnConfirmStatus?.Invoke("");
        ResetConfirmState();
        RestoreAllPracticeTargetRotations();

        if (grabber != null)
            grabber.SetHeldRotationMode(restoreGrabberModeOnDisable);

        if (_keywordRecognizer != null)
        {
            _keywordRecognizer.Stop();
            _keywordRecognizer.Dispose();
            _keywordRecognizer = null;
        }
    }

    /// <summary>
    /// Returns what should be rotated by micro controller.
    /// If holding exists, prefer HeldBody; otherwise (optional) rotate active tool directly.
    /// </summary>
    public Transform GetMicroRotationTargetTransform()
    {
        if (_active == null || _active.tool == null) return null;

        EnsureActiveBody();

        if (grabber != null && grabber.IsHolding && grabber.HeldBody != null)
        {
            if (_active.toolBody == null || grabber.HeldBody == _active.toolBody)
                return grabber.HeldBody.transform;

            return allowMicroRotateWithoutHolding ? _active.tool : null;
        }

        if (allowMicroRotateWithoutHolding)
            return _active.tool;

        return null;
    }
}
