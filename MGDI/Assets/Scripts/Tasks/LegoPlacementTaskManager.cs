using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Windows.Speech;

public class LegoPlacementTaskManager : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The transform that moves (use LegoBlockRoot).")]
    [SerializeField] private Transform blockRoot;

    [Tooltip("The target slot root transform (e.g., TargetSlot/LegoBlockRoot).")]
    [SerializeField] private Transform targetSlotRoot;

    [Tooltip("Optional: a component that supports ForceRelease() and SetFollowHeldRotation(bool). (e.g., ProxyHandGrabber)")]
    [SerializeField] private MonoBehaviour grabReleaseComponent;

    [Tooltip("Reference frame for target sampling (e.g., an empty object at table center). If null, uses world axes.")]
    [SerializeField] private Transform referenceFrame;

    [Header("Grab behavior per task")]
    [Tooltip("If true, lock rotation follow during placement trials (translation-only).")]
    [SerializeField] private bool lockRotationDuringPlacement = true;

    [Tooltip("If true, restore rotation follow after each trial (useful when later adding rotation tasks).")]
    [SerializeField] private bool restoreRotationFollowAfterTrial = true;

    [Header("Feedback (Audio/UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Trial Parameters")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Tooltip("Tolerance = max(0.05 * targetDistance, minTolMeters).")]
    [SerializeField] private float minTolMeters = 0.005f;

    [Tooltip("If true, keep targetSlot Y equal to its current Y.")]
    [SerializeField] private bool keepTargetSlotY = true;

    [Header("Snap + Inter-trial timing")]
    [SerializeField] private bool snapOnSuccess = true;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetBlockToStartAfterTrial = true;

    [Header("Optional: Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Voice Start (HoloLens)")]
    [SerializeField] private bool enableVoiceStart = true;
    [SerializeField] private string startKeyword = "start";
    [SerializeField] private string restartKeyword = "restart";
    [SerializeField] private bool autoStartInEditor = false;

    [Header("Target Sampling: Safe Wedge (referenceFrame 기준)")]
    [Tooltip("Target distance range in meters (in XZ plane from startPos).")]
    [SerializeField] private Vector2 targetDistanceRange = new Vector2(0.35f, 0.60f);

    [Tooltip("Avoid the fragile 'straight ahead' zone by excluding |angle| < excludeCenterDeg. (deg)")]
    [SerializeField] private float excludeCenterDeg = 20f;

    [Tooltip("Maximum side angle allowed from forward direction. (deg)")]
    [SerializeField] private float maxSideDeg = 70f;

    [Tooltip("If true, sample both left and right wedges. If false, use one side only (fixed by useRightSideOnly).")]
    [SerializeField] private bool allowBothSides = true;

    [Tooltip("When allowBothSides is false, choose right wedge if true, else left wedge.")]
    [SerializeField] private bool useRightSideOnly = true;

    [Tooltip("Extra constraint: minimum planar offset magnitude. (meters)")]
    [SerializeField] private float minPlanarOffsetMeters = 0.30f;

    [Tooltip("Resample attempts for wedge sampling.")]
    [SerializeField] private int maxResampleAttempts = 30;

    // Runtime state
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private Vector3 startPos;     // start position for CURRENT trial
    private Vector3 targetPos;
    private float targetDistance;
    private float tolerance;

    private bool trialRunning = false;
    private bool inTransition = false;

    // Voice
    private KeywordRecognizer keywordRecognizer;
    private Dictionary<string, System.Action> keywordActions;

    // --- Public controls ---
    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;

        trialIndex = 0;
        BeginNextTrial();
    }

    private void Start()
    {
        if (enableVoiceStart)
            SetupVoiceCommands();

        HideFeedbackUI();

        if (autoStartInEditor && Application.isEditor)
            StartBlock();
    }

    private void Update()
    {
        if (!trialRunning || inTransition) return;

        trialTimer += Time.deltaTime;

        if (trialTimer >= trialTimeoutSeconds)
        {
            StartCoroutine(EndTrialRoutine(success: false, timedOut: true));
            return;
        }

        float err = Vector3.Distance(blockRoot.position, targetSlotRoot.position);

        if (err <= tolerance)
        {
            dwellTimer += Time.deltaTime;
            if (dwellTimer >= dwellSeconds)
            {
                StartCoroutine(EndTrialRoutine(success: true, timedOut: false));
            }
        }
        else
        {
            dwellTimer = 0f;
        }
    }

    private void BeginNextTrial()
    {
        if (blockRoot == null || targetSlotRoot == null)
        {
            Debug.LogError("[LegoPlacementTaskManager] Missing references: blockRoot or targetSlotRoot.");
            trialRunning = false;
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            trialRunning = false;
            Debug.Log("[LegoPlacementTaskManager] Block finished.");
            return;
        }

        if (lockRotationDuringPlacement)
            SetGrabberFollowRotation(false);

        // Record current block position as the startPos for THIS trial.
        startPos = blockRoot.position;

        // Sample target offset in a safe wedge relative to referenceFrame forward/right (XZ plane).
        Vector3 offset = SampleOffsetInSafeWedge_ReferenceFrame();

        float y = keepTargetSlotY ? targetSlotRoot.position.y : startPos.y;
        targetPos = new Vector3(startPos.x + offset.x, y, startPos.z + offset.z);

        targetSlotRoot.position = targetPos;

        targetDistance = Vector3.Distance(startPos, targetPos);
        tolerance = Mathf.Max(0.05f * targetDistance, minTolMeters);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1}/{totalTrials} " +
                  $"targetDist={targetDistance:F3}m tol={tolerance:F3}m " +
                  $"start=({startPos.x:F3},{startPos.y:F3},{startPos.z:F3}) " +
                  $"target=({targetPos.x:F3},{targetPos.y:F3},{targetPos.z:F3})");
    }

    private Vector3 SampleOffsetInSafeWedge_ReferenceFrame()
    {
        // Use referenceFrame forward/right projected to XZ; fallback to world axes.
        Vector3 fwd = Vector3.forward;
        Vector3 right = Vector3.right;

        if (referenceFrame != null)
        {
            fwd = referenceFrame.forward;
            right = referenceFrame.right;
        }

        fwd.y = 0f;
        right.y = 0f;

        if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
        if (right.sqrMagnitude < 1e-6f) right = Vector3.right;

        fwd.Normalize();
        right.Normalize();

        float exclude = Mathf.Clamp(excludeCenterDeg, 0f, 89f);
        float maxSide = Mathf.Clamp(maxSideDeg, exclude + 1f, 89f);

        Vector3 offset = Vector3.zero;

        for (int attempt = 0; attempt < maxResampleAttempts; attempt++)
        {
            // choose left/right side
            int sideSign;
            if (allowBothSides)
                sideSign = (Random.value < 0.5f) ? -1 : 1; // -1 left, +1 right
            else
                sideSign = useRightSideOnly ? 1 : -1;

            // angle away from center to avoid fragile zone
            float angDeg = Random.Range(exclude, maxSide);
            float angRad = angDeg * Mathf.Deg2Rad;

            float dist = Random.Range(targetDistanceRange.x, targetDistanceRange.y);

            // direction on XZ plane: rotate forward toward left/right by angDeg
            Vector3 dir = (fwd * Mathf.Cos(angRad)) + (right * (sideSign * Mathf.Sin(angRad)));
            offset = dir * dist;

            // enforce minimum planar offset
            Vector2 planar = new Vector2(offset.x, offset.z);
            if (planar.magnitude >= minPlanarOffsetMeters)
                return offset;
        }

        Debug.LogWarning("[LegoPlacementTaskManager] Safe wedge sampling failed; using fallback forward offset.");
        return fwd * Mathf.Max(minPlanarOffsetMeters, targetDistanceRange.x);
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        float finalErr = Vector3.Distance(blockRoot.position, targetSlotRoot.position);
        Debug.Log($"[LegoPlacementTaskManager] Trial {trialIndex + 1} End. " +
                  $"success={success} timedOut={timedOut} time={trialTimer:F2}s finalErr={finalErr:F3}m");

        HideFeedbackUI();

        if (success)
        {
            // Release first so grabber does not fight the snap/reset.
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
            yield return new WaitForSeconds(feedbackShowSeconds));
        }

        HideFeedbackUI();

        if (restoreRotationFollowAfterTrial)
            SetGrabberFollowRotation(true);

        if (resetBlockToStartAfterTrial)
        {
            ForceReleaseIfPossible();
            ResetBlockToStart();
        }

        trialIndex++;

        BeginNextTrial();
    }

    private void SnapBlockToTarget()
    {
        Vector3 p = targetSlotRoot.position;
        blockRoot.position = new Vector3(p.x, blockRoot.position.y, p.z);
    }

    private void ResetBlockToStart()
    {
        blockRoot.position = startPos;
    }

    private void ForceReleaseIfPossible()
    {
        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("ForceRelease", SendMessageOptions.DontRequireReceiver);
    }

    private void SetGrabberFollowRotation(bool follow)
    {
        if (grabReleaseComponent != null)
            grabReleaseComponent.SendMessage("SetFollowHeldRotation", follow, SendMessageOptions.DontRequireReceiver);
    }

    private void PlaySnapSound()
    {
        if (audioSource == null || snapClip == null) return;
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
            { startKeyword.ToLower(), () => StartBlock() },
            { restartKeyword.ToLower(), () => StartBlock() }
        };

        keywordRecognizer = new KeywordRecognizer(keywordActions.Keys.ToArray());
        keywordRecognizer.OnPhraseRecognized += args =>
        {
            string key = args.text.ToLower();
            if (keywordActions.TryGetValue(key, out var action))
                action.Invoke();
        };
        keywordRecognizer.Start();

        Debug.Log($"[LegoPlacementTaskManager] Voice commands enabled: '{startKeyword}', '{restartKeyword}'");
    }

    private void OnDisable()
    {
        if (keywordRecognizer != null)
        {
            keywordRecognizer.Stop();
            keywordRecognizer.Dispose();
            keywordRecognizer = null;
        }
    }
}
