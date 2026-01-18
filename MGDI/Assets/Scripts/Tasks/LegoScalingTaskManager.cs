using System;
using System.Collections;
using UnityEngine;
using Random = UnityEngine.Random;

public class LegoScalingTaskManager : MonoBehaviour
{
    public event Action OnBlockFinished;
    public event Action<int, int> OnTrialChanged; // (current1Based, total)

    [Header("References")]
    [SerializeField] private Transform blockRoot;
    [SerializeField] private Transform targetPivot; // target display uses LOCAL scale only.
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private RemoteHandRuntime remoteHand;

    [Header("Feedback (Audio / UI)")]
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private AudioClip snapClip;
    [SerializeField] private GameObject starUI;
    [SerializeField] private GameObject xUI;
    [SerializeField] private float feedbackShowSeconds = 0.50f;

    [Header("Trial Timing")]
    [SerializeField] private float trialTimeoutSeconds = 12f;
    [SerializeField] private float dwellSeconds = 0.20f;

    [Header("Scale Success Threshold (FACTOR)")]
    [Tooltip("Success if abs(currentFactor - targetFactor) <= tolerance.")]
    [SerializeField] private float scaleFactorTolerance = 0.05f;

    [Header("Target Factor Sampling (relative to baseline)")]
    [SerializeField] private float targetFactorMin = 0.70f;
    [SerializeField] private float targetFactorMax = 1.60f;

    [Header("Control: diagonal (up + side) movement -> scale")]
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

    [Header("Scale factor clamp (relative to baseline)")]
    [SerializeField] private float minScaleFactor = 0.60f;
    [SerializeField] private float maxScaleFactor = 1.80f;

    [Header("Scaling gating")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, drive scaling only when the grabber is holding THIS block rigidbody.")]
    [SerializeField] private bool requireHoldingThisBlock = true;

    [Header("Freeze block motion during scaling")]
    [SerializeField] private bool freezeBlockPositionWhileScaling = true;
    [SerializeField] private bool freezeBlockRotationWhileScaling = false;

    [Header("Inter-trial behavior")]
    [SerializeField] private bool snapOnSuccess = true;
    [SerializeField] private float postSnapHoldSeconds = 0.35f;
    [SerializeField] private bool resetScaleAfterTrial = true;
    [SerializeField] private bool forceReleaseAfterTrial = true;

    [Header("Trial Count")]
    [SerializeField] private int totalTrials = 20;

    [Header("Debug")]
    [SerializeField] private bool logDriveState = false;

    // Runtime
    private int trialIndex = 0;
    private float trialTimer = 0f;
    private float dwellTimer = 0f;

    private bool trialRunning = false;
    private bool inTransition = false;

    // baseline WORLD scale for block (robust to re-parenting on grab)
    private Vector3 _trialBaseWorldScale = Vector3.one;

    // target baseline LOCAL scale (captured once; prevents cumulative growth)
    private Vector3 _targetBaseLocalScale0 = Vector3.one;
    private bool _targetBaseCaptured = false;

    // target factor
    private float _targetFactor = 1f;

    // drive state
    private bool _prevEffectiveDriving = false;
    private Vector3 _wristPrev;
    private bool _haveWristPrev = false;

    private float _axisAccum = 0f;

    private float _scaleFactorCmd = 1f;
    private bool _scaleFactorInit = false;

    private Vector3 _lockPos;
    private Quaternion _lockRot;
    private bool _poseLocked = false;

    private bool _skipScaleThisFrame = false;

    private Rigidbody _blockBody;

    public bool IsTrialRunning => trialRunning && !inTransition;
    public float TrialTimeRemainingSec => Mathf.Max(0f, trialTimeoutSeconds - trialTimer);
    public int TotalTrials => totalTrials;
    public int CurrentTrialIndex1Based => trialIndex + 1;

    public void StartBlock()
    {
        StopAllCoroutines();
        inTransition = false;
        trialRunning = false;
        trialIndex = 0;

        _prevEffectiveDriving = false;
        _haveWristPrev = false;
        _scaleFactorInit = false;
        _poseLocked = false;
        _skipScaleThisFrame = false;

        EnsureBlockBody();
        HideFeedbackUI();

        CaptureTargetBaseOnce();

        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();
        EnsureBlockBody();
        CaptureTargetBaseOnce();
    }

    private void CaptureTargetBaseOnce()
    {
        if (_targetBaseCaptured) return;
        if (targetPivot == null) return;

        _targetBaseLocalScale0 = targetPivot.localScale;
        _targetBaseCaptured = true;
    }

    private void EnsureBlockBody()
    {
        if (_blockBody != null) return;
        if (blockRoot == null) return;

        _blockBody = blockRoot.GetComponent<Rigidbody>();
        if (_blockBody == null)
            _blockBody = blockRoot.GetComponentInChildren<Rigidbody>(true);
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

        bool drive = ShouldDriveScalingThisFrame();
        bool ready = HasWrist();
        bool effectiveDriving = drive && ready;

        if (logDriveState && effectiveDriving != _prevEffectiveDriving)
            Debug.Log($"[ScaleTM] effectiveDriving={effectiveDriving} drive={drive} ready={ready} requireHoldingThisBlock={requireHoldingThisBlock}");

        if (effectiveDriving)
        {
            if (!_prevEffectiveDriving)
            {
                RebaselineForScaling();
                _skipScaleThisFrame = true;
            }

            if (_skipScaleThisFrame)
                _skipScaleThisFrame = false;
            else
                UpdateScaleFromDiagonalWristMotion();
        }
        else
        {
            dwellTimer = 0f;
            _poseLocked = false;
            _skipScaleThisFrame = false;
        }

        _prevEffectiveDriving = effectiveDriving;

        float err = Mathf.Abs(CurrentFactor() - _targetFactor);
        if (err <= scaleFactorTolerance)
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

    private void LateUpdate()
    {
        if (!trialRunning || inTransition) return;
        if (!_poseLocked) return;

        if (freezeBlockPositionWhileScaling)
            blockRoot.position = _lockPos;

        if (freezeBlockRotationWhileScaling)
            blockRoot.rotation = _lockRot;

        if (_scaleFactorInit)
            SetWorldScale(blockRoot, _trialBaseWorldScale * _scaleFactorCmd);
    }

    private bool ShouldDriveScalingThisFrame()
    {
        if (!scaleOnlyWhenHolding) return true;

        if (grabber == null) return false;
        if (!grabber.IsHolding) return false;

        if (!requireHoldingThisBlock) return true;

        EnsureBlockBody();
        if (_blockBody == null) return false;
        if (grabber.HeldBody == null) return false;

        return grabber.HeldBody == _blockBody;
    }

    private void BeginNextTrial()
    {
        if (blockRoot == null)
        {
            Debug.LogError("[ScaleTM] Missing reference: blockRoot.");
            FinishBlock();
            return;
        }

        if (remoteHand == null)
        {
            Debug.LogError("[ScaleTM] remoteHand is null (RemoteHandRuntime).");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[ScaleTM] Block finished.");
            FinishBlock();
            return;
        }

        CaptureTargetBaseOnce();

        // baseline WORLD scale for block (robust to re-parenting on grab)
        _trialBaseWorldScale = blockRoot.lossyScale;

        _axisAccum = 0f;
        _scaleFactorCmd = 1f;
        _scaleFactorInit = true;

        // sample target factor
        _targetFactor = Random.Range(Mathf.Min(targetFactorMin, targetFactorMax), Mathf.Max(targetFactorMin, targetFactorMax));
        _targetFactor = Mathf.Clamp(_targetFactor, minScaleFactor, maxScaleFactor);

        // show target with FIXED base local scale (no accumulation)
        if (targetPivot != null)
            targetPivot.localScale = _targetBaseLocalScale0 * _targetFactor;

        // start block at baseline world scale
        SetWorldScale(blockRoot, _trialBaseWorldScale);

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        _prevEffectiveDriving = false;
        _haveWristPrev = false;
        _poseLocked = false;
        _skipScaleThisFrame = false;

        OnTrialChanged?.Invoke(trialIndex + 1, totalTrials);

        Debug.Log($"[ScaleTM] Trial {trialIndex + 1}/{totalTrials} targetFactor={_targetFactor:F2} tol={scaleFactorTolerance:F3}");
    }

    private bool HasWrist()
    {
        return remoteHand != null &&
               remoteHand.remoteByIndex != null &&
               remoteHand.remoteByIndex.Length > 0 &&
               remoteHand.remoteByIndex[0] != null;
    }

    private Vector3 GetAxis()
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

    private void RebaselineForScaling()
    {
        Vector3 w = remoteHand.remoteByIndex[0].position;
        _wristPrev = w;
        _haveWristPrev = true;

        _axisAccum = 0f;
        _scaleFactorCmd = 1f;
        _scaleFactorInit = true;

        _lockPos = blockRoot.position;
        _lockRot = blockRoot.rotation;
        _poseLocked = true;
    }

    private void UpdateScaleFromDiagonalWristMotion()
    {
        Vector3 w = remoteHand.remoteByIndex[0].position;

        if (!_haveWristPrev)
        {
            _wristPrev = w;
            _haveWristPrev = true;
            return;
        }

        Vector3 dp = w - _wristPrev;
        _wristPrev = w;

        Vector3 axis = GetAxis();
        float delta = Vector3.Dot(dp, axis);

        if (Mathf.Abs(delta) < Mathf.Max(0f, moveDeadZoneMeters))
            return;

        _axisAccum += delta;

        float desiredFactor = Mathf.Exp(moveToScaleGain * _axisAccum);
        desiredFactor = Mathf.Clamp(desiredFactor, minScaleFactor, maxScaleFactor);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(scaleLerp), dt * 60f);

        float cur = _scaleFactorInit ? _scaleFactorCmd : 1f;
        _scaleFactorCmd = Mathf.Lerp(cur, desiredFactor, k);
        _scaleFactorInit = true;

        SetWorldScale(blockRoot, _trialBaseWorldScale * _scaleFactorCmd);

        _poseLocked = true;
    }

    private float CurrentFactor()
    {
        float baseS = AvgAbs(_trialBaseWorldScale);
        float curS = AvgAbs(blockRoot.lossyScale);
        if (baseS <= 1e-6f) return 1f;
        return curS / baseS;
    }

    private static float AvgAbs(Vector3 v)
    {
        return (Mathf.Abs(v.x) + Mathf.Abs(v.y) + Mathf.Abs(v.z)) / 3f;
    }

    private IEnumerator EndTrialRoutine(bool success, bool timedOut)
    {
        if (inTransition) yield break;
        inTransition = true;
        trialRunning = false;

        _poseLocked = false;
        HideFeedbackUI();

        if (success)
        {
            if (forceReleaseAfterTrial) ForceReleaseIfPossible();

            if (snapOnSuccess)
            {
                SetWorldScale(blockRoot, _trialBaseWorldScale * _targetFactor);
                _scaleFactorCmd = _targetFactor;
                _scaleFactorInit = true;
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

        if (resetScaleAfterTrial)
            SetWorldScale(blockRoot, _trialBaseWorldScale);

        if (forceReleaseAfterTrial) ForceReleaseIfPossible();

        trialIndex++;
        inTransition = false;
        BeginNextTrial();
    }

    private void ForceReleaseIfPossible()
    {
        if (grabber != null)
            grabber.ForceRelease();
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
        _poseLocked = false;
        HideFeedbackUI();

        try { OnBlockFinished?.Invoke(); } catch { }
    }

    /// <summary>
    /// Sets the world-scale of a transform by compensating for parent lossyScale.
    /// Use this for the grabbed block (re-parenting causes localScale jumps).
    /// </summary>
    private static void SetWorldScale(Transform t, Vector3 desiredWorldScale)
    {
        if (t == null) return;

        Vector3 parentLossy = Vector3.one;
        if (t.parent != null)
            parentLossy = t.parent.lossyScale;

        float px = Mathf.Abs(parentLossy.x) < 1e-6f ? 1e-6f : parentLossy.x;
        float py = Mathf.Abs(parentLossy.y) < 1e-6f ? 1e-6f : parentLossy.y;
        float pz = Mathf.Abs(parentLossy.z) < 1e-6f ? 1e-6f : parentLossy.z;

        t.localScale = new Vector3(
            desiredWorldScale.x / px,
            desiredWorldScale.y / py,
            desiredWorldScale.z / pz
        );
    }
}
