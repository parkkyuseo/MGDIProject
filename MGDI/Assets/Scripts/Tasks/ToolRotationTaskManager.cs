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
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Progress UI (optional)")]
    [SerializeField] private UnityEngine.UI.Text progressText;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Header("Full 3D Success Threshold")]
    [SerializeField] private float rotationToleranceDeg = 12f;

    [Header("Evaluation gating")]
    [Tooltip("If true, success dwell accumulates only when input is not actively driving (recommended for micro).")]
    [SerializeField] private bool requireNotDrivingForEvaluation = true;

    [Tooltip("If true, macro evaluation requires releasing the object (simple 'stop input to evaluate' proxy).")]
    [SerializeField] private bool requireReleaseForEvaluationInMacro = true;

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

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

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

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

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

        public Transform startParent;
        public Vector3 startPos;
        public Quaternion startRot;

        // Base pose snapshots (preserve "lying down" etc.)
        public Quaternion startBaseRot;
        public Quaternion targetBaseRot;
        public Vector3 targetBasePos;

        // For snap on success
        public Quaternion targetDesiredRot;
    }

    private readonly List<Item> _items = new List<Item>();
    private int _trialIndex = 0;

    private float _trialTimer = 0f;
    private float _dwellTimer = 0f;

    private bool _trialRunning = false;
    private bool _inTransition = false;

    private Item _active;

    private KeywordRecognizer _keywordRecognizer;
    private Dictionary<string, Action> _keywordActions;

    private readonly StringBuilder _sb = new StringBuilder(512);

    public bool IsTrialRunning => _trialRunning && !_inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - _trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => _trialIndex + 1;

    public Transform ActiveToolTransform => _active != null ? _active.tool : null;
    public string ActiveToolId => _active != null ? _active.id : null;

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        _inTransition = false;
        _trialRunning = false;
        _trialIndex = 0;

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

        if (progressText != null)
            UpdateProgressText(errDeg, evalAllowed);

        if (errDeg <= rotationToleranceDeg)
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

        // Active tool = round-robin through matched ids
        _active = _items[_trialIndex % _items.Count];

        EnsureActiveBody();

        // Reset active tool to its start pose at trial start
        ForceReleaseIfPossible();
        ResetActiveToolToStartPose();

        // Snapshot base poses (preserve pitch/roll shape)
        _active.startBaseRot = _active.tool.rotation;
        _active.targetBaseRot = _active.target.rotation;
        _active.targetBasePos = _active.target.position;

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

        // Sample a full 3D target rotation offset relative to startBaseRot
        Quaternion offset = SampleRandomRotationOffset();
        _active.targetDesiredRot = offset * _active.startBaseRot; // left-multiply for "delta in world"
        _active.target.rotation = _active.targetDesiredRot;

        _trialTimer = 0f;
        _dwellTimer = 0f;
        _trialRunning = true;
        _inTransition = false;

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

        // Restore target back to original placement
        RestoreTargetPose();

        if (resetToolToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            ResetActiveToolToStartPose();
        }

        _trialIndex++;
        BeginNextTrial();
    }

    private void RestoreTargetPose()
    {
        if (_active == null || _active.target == null) return;
        if (!bringTargetNearActiveTool) return;

        _active.target.position = _active.targetBasePos;
        _active.target.rotation = _active.targetBaseRot;
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
                toolBody = null,
                startBaseRot = Quaternion.identity,
                targetBaseRot = Quaternion.identity,
                targetBasePos = Vector3.zero,
                targetDesiredRot = Quaternion.identity
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

    private float ComputeRotationErrorDeg()
    {
        if (_active == null || _active.tool == null || _active.target == null) return float.MaxValue;
        return Quaternion.Angle(_active.tool.rotation, _active.target.rotation);
    }

    private void SnapRotationToTarget()
    {
        if (_active == null || _active.tool == null || _active.target == null) return;
        _active.tool.rotation = _active.target.rotation;
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
