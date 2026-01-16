using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;
using UnityEngine.Windows.Speech;

public class LegoPlacementTaskManager : MonoBehaviour
{
    // Fired when the placement block finishes (all trials complete).
    public event Action OnBlockFinished;
    public event System.Action<int, int> OnTrialChanged; // (current1Based, total)

    [Header("References")]
    [Tooltip("The transform that moves (use LegoBlockRoot).")]
    [SerializeField] private Transform blockRoot;

    [Tooltip("The target slot root transform (use TargetSlotRoot - the EMPTY root).")]
    [SerializeField] private Transform targetSlotRoot;

    [Tooltip("Target visual transform (use TargetSlotVisual - the EMPTY visual parent).")]
    [SerializeField] private Transform targetSlotVisual;

    [Tooltip("ProxyHandGrabber instance (preferred). Used for setting rotation mode and optional force release.")]
    [SerializeField] private ProxyHandGrabber grabber;

    [Tooltip("Grab controller (fallback) supporting ForceRelease() and SetHeldRotationMode(...).")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Tooltip("Reference frame for target sampling (an empty object). This will be re-positioned each trial if anchoring is enabled.")]
    [SerializeField] private Transform referenceFrame;

    [Header("Reference Frame Anchoring")]
    [Tooltip("If true, referenceFrame.position is set each trial to blockRoot.position + referenceFrameOffsetLocal (in referenceFrame local axes).")]
    [SerializeField] private bool anchorReferenceFrameToBlock = true;

    [Tooltip("Local offset applied when anchoring referenceFrame to blockRoot (meters).")]
    [SerializeField] private Vector3 referenceFrameOffsetLocal = new Vector3(0f, 0.05f, 0f);

    [Header("Grab behavior per task")]
    [Tooltip("If true, lock rotation during placement trials (translation-only).")]
    [SerializeField] private bool lockRotationDuringPlacement = true;

    [Tooltip("Rotation mode to use during placement trials when lockRotationDuringPlacement is true.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode placementGrabberMode = ProxyHandGrabber.HeldRotationMode.LockAtGrab;

    [Tooltip("If true, restore rotation mode after each trial.")]
    [SerializeField] private bool restoreRotationModeAfterTrial = true;

    [Tooltip("Rotation mode to restore after placement trial ends.")]
    [SerializeField] private ProxyHandGrabber.HeldRotationMode restoreGrabberModeAfterTrial = ProxyHandGrabber.HeldRotationMode.FollowAnchor;

    [Header("Target Visual Rotation")]
    [Tooltip("If true, copy blockRoot.rotation into targetSlotVisual.rotation each trial so the target looks aligned.")]
    [SerializeField] private bool matchTargetVisualRotationToBlock = true;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Tooltip("Tolerance = max(0.05 * targetDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.005f;

    [Header("Snap / Reset")]
    [SerializeField] private bool snapOnSuccess = true;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockToStartAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = true;
    [SerializeField] private string startKeyword = "start";
    [SerializeField] private string restartKeyword = "restart";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Target Sampling (referenceFrame local space)")]
    [Tooltip("Local X range (meters).")]
    [SerializeField] private Vector2 localXRange = new Vector2(-0.35f, 0.35f);

    [Tooltip("Local Y range (meters).")]
    [SerializeField] private Vector2 localYRange = new Vector2(0.00f, 0.15f);

    [Tooltip("Local Z range (meters).")]
    [SerializeField] private Vector2 localZRange = new Vector2(0.25f, 0.55f);

    [Header("Forbidden Center Region")]
    [Tooltip("Half-width of forbidden center strip on X (meters). abs(x) < value is rejected.")]
    [SerializeField] private float centerNoGoHalfWidthX = 0.12f;

    [Header("Distance Constraints")]
    [Tooltip("Minimum planar (XZ) distance from the anchored reference origin (meters).")]
    [SerializeField] private float minPlanarOffsetMeters = 0.30f;

    [Tooltip("Max resampling attempts.")]
    [SerializeField] private int maxResampleAttempts = 40;

    [Header("Robust success metric")]
    [Tooltip("If true, success uses Renderer.bounds.center distance (robust to pivot offsets).")]
    [SerializeField] private bool useBoundsCenterForSuccess = true;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;
    private float tolerance;

    private bool trialRunning = false;
    private bool inTransition = false;

    // Cached renderers (refreshed each trial)
    private Renderer _blockR;
    private Renderer _targetR;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;

    // ---------------- Debug helpers ----------------
    static string GetPath(Transform t)
    {
        if (t == null) return "<null>";
        string p = t.name;
        while (t.parent != null)
        {
            t = t.parent;
            p = t.name + "/" + p;
        }
        return p;
    }

    // Optional gizmo drawing
    void OnDrawGizmos()
    {
        if (blockRoot == null || targetSlotRoot == null) return;

        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(blockRoot.position, 0.05f);

        Gizmos.color = Color.green;
        Gizmos.DrawSphere(targetSlotRoot.position, 0.05f);

        var blockR = blockRoot.GetComponentInChildren<Renderer>();
        var targetR = targetSlotRoot.GetComponentInChildren<Renderer>();

        if (blockR != null)
        {
            Gizmos.color = new Color(1f, 0.5f, 0f, 1f);
            Gizmos.DrawSphere(blockR.bounds.center, 0.07f);
            Gizmos.DrawLine(blockRoot.position, blockR.bounds.center);
        }

        if (targetR != null)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawSphere(targetR.bounds.center, 0.05f);
            Gizmos.DrawLine(targetSlotRoot.position, targetR.bounds.center);
        }

        Gizmos.color = Color.magenta;
        Gizmos.DrawLine(blockRoot.position, targetSlotRoot.position);

        if (blockR != null && targetR != null)
        {
            Gizmos.color = Color.white;
            Gizmos.DrawLine(blockR.bounds.center, targetR.bounds.center);
        }
    }

    // ---------------- Public ----------------
    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;
        BeginNextTrial();
    }

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

        trialTimer += Time.deltaTime;

        if (trialTimer >= trialTimeoutSeconds)
        {
            StartCoroutine(EndTrialRoutine(false, true));
            return;
        }

        float err = ComputeErrorMeters();

        if (err <= tolerance)
        {
            dwellTimer += Time.deltaTime;
            if (dwellTimer >= dwellSeconds)
                StartCoroutine(EndTrialRoutine(true, false));
        }
        else
        {
            dwellTimer = 0f;
        }

        // if (Time.frameCount % 60 == 0)
        // {
        //     DebugHUD.Log($"blockRoot={blockRoot.name} pos={blockRoot.position}");
        //     DebugHUD.Log($"targetSlotRoot={targetSlotRoot.name} pos={targetSlotRoot.position}");
        //     DebugHUD.Log($"blockRootPath={GetPath(blockRoot)}");
        //     DebugHUD.Log($"targetSlotRootPath={GetPath(targetSlotRoot)}");
        //     DebugHUD.Log($"err={err:F4} tol={tolerance:F4} dwell={dwellTimer:F3}/{dwellSeconds:F3}");
        // }
    }

    // ---------------- Trial Flow ----------------
    private void BeginNextTrial()
    {
        if (blockRoot == null || targetSlotRoot == null || referenceFrame == null)
        {
            Debug.LogError("[LegoPlacementTaskManager] Missing references (blockRoot/targetSlotRoot/referenceFrame).");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[LegoPlacementTaskManager] Block finished.");
            FinishBlock();
            return;
        }

        // Translation-only: lock rotation (so wrist twist doesn't unintentionally spin the block)
        if (lockRotationDuringPlacement)
            SetGrabberRotationMode(placementGrabberMode);

        startPos = blockRoot.position;

        if (anchorReferenceFrameToBlock)
        {
            referenceFrame.position = blockRoot.position + referenceFrame.TransformVector(referenceFrameOffsetLocal);
        }

        Vector3 localOffset = SampleLocalOffset();
        Vector3 worldTarget = referenceFrame.TransformPoint(localOffset);

        targetSlotRoot.position = worldTarget;

        if (matchTargetVisualRotationToBlock && targetSlotVisual != null)
        {
            targetSlotVisual.rotation = blockRoot.rotation;
        }

        _blockR = blockRoot.GetComponentInChildren<Renderer>(true);
        _targetR = targetSlotRoot.GetComponentInChildren<Renderer>(true);

        float targetDistance = ComputeDistanceAtTrialStart();
        tolerance = Mathf.Max(0.05f * targetDistance, minTolMeters);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;
        OnTrialChanged?.Invoke(trialIndex + 1, totalTrials);

        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"localOffset=({localOffset.x:F2},{localOffset.y:F2},{localOffset.z:F2}) " +
                  $"targetDist={targetDistance:F2} tol={tolerance:F3}");
    }

    private Vector3 SampleLocalOffset()
    {
        for (int i = 0; i < maxResampleAttempts; i++)
        {
            float lx = Random.Range(localXRange.x, localXRange.y);
            float ly = Random.Range(localYRange.x, localYRange.y);
            float lz = Random.Range(localZRange.x, localZRange.y);

            if (Mathf.Abs(lx) < centerNoGoHalfWidthX)
                continue;

            float planar = Mathf.Sqrt(lx * lx + lz * lz);
            if (planar < minPlanarOffsetMeters)
                continue;

            return new Vector3(lx, ly, lz);
        }

        Debug.LogWarning("[LegoPlacementTaskManager] Sampling failed; using fallback.");
        float fallbackX = (Random.value < 0.5f ? -1f : 1f) * Mathf.Max(centerNoGoHalfWidthX, 0.01f);
        float fallbackY = Mathf.Clamp(localYRange.x, Mathf.Min(localYRange.x, localYRange.y), Mathf.Max(localYRange.x, localYRange.y));
        float fallbackZ = Mathf.Max(localZRange.x, minPlanarOffsetMeters);

        return new Vector3(fallbackX, fallbackY, fallbackZ);
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        HideFeedbackUI();

        if (success)
        {
            ForceReleaseIfPossible();

            if (snapOnSuccess)
                SnapBlockToTarget();

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

        // Restore rotation policy for the next task / next phase
        if (restoreRotationModeAfterTrial)
            SetGrabberRotationMode(restoreGrabberModeAfterTrial);

        if (resetBlockToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            blockRoot.position = startPos;
        }

        trialIndex++;
        BeginNextTrial();
    }

    // ---------------- Core metric helpers ----------------
    private float ComputeErrorMeters()
    {
        if (!useBoundsCenterForSuccess)
        {
            return Vector3.Distance(blockRoot.position, targetSlotRoot.position);
        }

        if (_blockR == null) _blockR = blockRoot.GetComponentInChildren<Renderer>(true);
        if (_targetR == null) _targetR = targetSlotRoot.GetComponentInChildren<Renderer>(true);

        if (_blockR == null || _targetR == null)
        {
            return Vector3.Distance(blockRoot.position, targetSlotRoot.position);
        }

        return Vector3.Distance(_blockR.bounds.center, _targetR.bounds.center);
    }

    private float ComputeDistanceAtTrialStart()
    {
        return ComputeErrorMeters();
    }

    // ---------------- Helpers ----------------
    private void SnapBlockToTarget()
    {
        if (!useBoundsCenterForSuccess)
        {
            blockRoot.position = targetSlotRoot.position;
            return;
        }

        if (_blockR == null) _blockR = blockRoot.GetComponentInChildren<Renderer>(true);
        if (_targetR == null) _targetR = targetSlotRoot.GetComponentInChildren<Renderer>(true);

        if (_blockR == null || _targetR == null)
        {
            blockRoot.position = targetSlotRoot.position;
            return;
        }

        Vector3 blockCenter = _blockR.bounds.center;
        Vector3 targetCenter = _targetR.bounds.center;

        Vector3 delta = targetCenter - blockCenter;
        blockRoot.position += delta;
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
        {
            grabber.ForceRelease();
            return;
        }

        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("ForceRelease", SendMessageOptions.DontRequireReceiver);
    }

    private void SetGrabberRotationMode(ProxyHandGrabber.HeldRotationMode mode)
    {
        if (grabber != null)
        {
            grabber.SetHeldRotationMode(mode);
            return;
        }

        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("SetHeldRotationMode", mode, SendMessageOptions.DontRequireReceiver);
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

    private void SetupVoiceCommands()
    {
        if (keywordRecognizer != null) return;

        keywordActions = new Dictionary<string, System.Action>
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

        Debug.Log($"[LegoPlacementTaskManager] Voice commands enabled: '{startKeyword}', '{restartKeyword}'");
    }

    private void FinishBlock()
    {
        trialRunning = false;
        inTransition = false;
        HideFeedbackUI();

        try { OnBlockFinished?.Invoke(); } catch { /* ignore */ }
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
}
