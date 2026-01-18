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
    [SerializeField] private Transform targetPivot; // reused target (ghost). Scaling uses localScale only.
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

    [Header("Scale Success Threshold")]
    [SerializeField] private float scaleTolerance = 0.05f;

    [Header("Target Scale Sampling")]
    [SerializeField] private float targetScaleMin = 0.70f;
    [SerializeField] private float targetScaleMax = 1.60f;

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
    [Tooltip("Extra smoothing on scale command (0=no smoothing, 1=very slow).")]
    [SerializeField] private float scaleLerp = 0.15f;

    [Header("Scale clamp (safety)")]
    [SerializeField] private float minScale = 0.50f;
    [SerializeField] private float maxScale = 2.00f;

    [Header("Scaling gating")]
    [SerializeField] private bool scaleOnlyWhenHolding = true;

    [Tooltip("If true, drive scaling only when the grabber is holding THIS block rigidbody.")]
    [SerializeField] private bool requireHoldingThisBlock = true;

    [Header("Freeze block motion during scaling")]
    [Tooltip("If true, while scaling is active, block position is locked (recommended).")]
    [SerializeField] private bool freezeBlockPositionWhileScaling = true;

    [Tooltip("If true, while scaling is active, block rotation is locked too (optional).")]
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

    private float _targetScale = 1f;
    private Vector3 _trialStartLocalScale = Vector3.one;

    // drive state
    private bool _prevEffectiveDriving = false;

    private Vector3 _wristPrev;
    private bool _haveWristPrev = false;

    // accumulated axis motion (meters) since baseline
    private float _axisAccum = 0f;
    private float _scaleBase = 1f;

    // command smoothing
    private float _scaleCmd = 1f;
    private bool _scaleCmdInit = false;

    // freeze pose
    private Vector3 _lockPos;
    private Quaternion _lockRot;
    private bool _poseLocked = false;

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
        _scaleCmdInit = false;
        _poseLocked = false;

        EnsureBlockBody();
        HideFeedbackUI();

        BeginNextTrial();
    }

    private void Start()
    {
        HideFeedbackUI();
        EnsureBlockBody();
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
                RebaselineForScaling();

            UpdateScaleFromDiagonalWristMotion();
        }
        else
        {
            dwellTimer = 0f;
            _poseLocked = false; // release pose lock when not actively driving
        }

        _prevEffectiveDriving = effectiveDriving;

        float err = Mathf.Abs(GetUniformScale(blockRoot.localScale) - _targetScale);
        if (err <= scaleTolerance)
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
        // Override grabber-driven translation/rotation so scaling feels "size only".
        if (!trialRunning || inTransition) return;
        if (!_poseLocked) return;

        if (freezeBlockPositionWhileScaling)
            blockRoot.position = _lockPos;

        if (freezeBlockRotationWhileScaling)
            blockRoot.rotation = _lockRot;

        // Keep scale command authoritative (in case another script writes it).
        if (_scaleCmdInit)
            blockRoot.localScale = Vector3.one * Mathf.Clamp(_scaleCmd, minScale, maxScale);
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
            Debug.LogError("[LegoScalingTaskManager] Missing reference: blockRoot.");
            FinishBlock();
            return;
        }

        if (remoteHand == null)
        {
            Debug.LogError("[LegoScalingTaskManager] remoteHand is null (RemoteHandRuntime).");
            FinishBlock();
            return;
        }

        if (totalTrials > 0 && trialIndex >= totalTrials)
        {
            Debug.Log("[LegoScalingTaskManager] Block finished.");
            FinishBlock();
            return;
        }

        _trialStartLocalScale = blockRoot.localScale;

        _targetScale = Random.Range(Mathf.Min(targetScaleMin, targetScaleMax), Mathf.Max(targetScaleMin, targetScaleMax));
        if (targetPivot != null)
            targetPivot.localScale = Vector3.one * _targetScale;

        trialTimer = 0f;
        dwellTimer = 0f;
        trialRunning = true;
        inTransition = false;

        _prevEffectiveDriving = false;
        _haveWristPrev = false;
        _scaleCmdInit = false;
        _poseLocked = false;

        OnTrialChanged?.Invoke(trialIndex + 1, totalTrials);

        Debug.Log($"[ScaleTM] Trial {trialIndex + 1}/{totalTrials} targetScale={_targetScale:F2} tol={scaleTolerance:F3}");
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
        // baseline wrist
        Vector3 w = remoteHand.remoteByIndex[0].position;
        _wristPrev = w;
        _haveWristPrev = true;

        // baseline scale
        _scaleBase = Mathf.Clamp(GetUniformScale(blockRoot.localScale), minScale, maxScale);
        _axisAccum = 0f;

        _scaleCmd = _scaleBase;
        _scaleCmdInit = true;

        // lock pose so object doesn't translate while scaling
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
        float delta = Vector3.Dot(dp, axis); // meters along axis

        if (Mathf.Abs(delta) < Mathf.Max(0f, moveDeadZoneMeters))
            return;

        _axisAccum += delta;

        float desired = _scaleBase * Mathf.Exp(moveToScaleGain * _axisAccum);
        desired = Mathf.Clamp(desired, minScale, maxScale);

        float dt = Mathf.Max(Time.deltaTime, 1e-4f);
        float k = 1f - Mathf.Pow(1f - Mathf.Clamp01(scaleLerp), dt * 60f);

        float cur = _scaleCmdInit ? _scaleCmd : _scaleBase;
        _scaleCmd = Mathf.Lerp(cur, desired, k);
        _scaleCmd = Mathf.Clamp(_scaleCmd, minScale, maxScale);

        blockRoot.localScale = Vector3.one * _scaleCmd;
        _scaleCmdInit = true;

        // Keep pose lock refreshed while driving (grabber may overwrite earlier in frame)
        _lockPos = blockRoot.position;
        _lockRot = blockRoot.rotation;
        _poseLocked = true;
    }

    private static float GetUniformScale(Vector3 s)
    {
        return (s.x + s.y + s.z) / 3f;
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
                float s = Mathf.Clamp(_targetScale, minScale, maxScale);
                blockRoot.localScale = Vector3.one * s;
                _scaleCmd = s;
                _scaleCmdInit = true;
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
            blockRoot.localScale = _trialStartLocalScale;

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
}
