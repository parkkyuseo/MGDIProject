using UnityEngine;

public class PhoneTechniqueGate : MonoBehaviour
{
    public enum MicroTask { Placement = 0, Rotation = 1, Scaling = 2 }

    [Header("Refs")]
    [SerializeField] private PhoneInputRouter router;

    [Header("Macro Drivers (enabled in Macro, disabled in Micro)")]
    [Tooltip("Drives phone pose follow (position + rotation). Enabled in Macro.")]
    [SerializeField] private Behaviour macroPoseDriver;

    [Header("Micro Drivers (enabled in Micro)")]
    [Tooltip("Drives rotation follow only (no position). Enabled in Micro.")]
    [SerializeField] private Behaviour microRotationOnlyDriver;

    [Header("Micro Controllers (enabled only in Micro)")]
    [SerializeField] private Behaviour microPlacementController; 
    [SerializeField] private Behaviour microRotationController;  
    [SerializeField] private Behaviour microScalingController;   

    [Header("Micro Task (default while not wired to StudyFlowController)")]
    [SerializeField] private MicroTask microTask = MicroTask.Placement;

    [Header("Options")]
    [Tooltip("If true, calls Recenter() on macroPoseDriver when switching back to Macro (if method exists).")]
    [SerializeField] private bool recenterOnMacroEnable = true;

    [Tooltip("If true, calls RecenterRotation() on microRotationOnlyDriver when switching to Micro (if method exists).")]
    [SerializeField] private bool recenterRotationOnMicroEnable = true;

    private PhoneInputRouter.Mode _lastMode;

    void Awake()
    {
        if (router == null) router = FindFirstObjectByType<PhoneInputRouter>();
        _lastMode = (router != null) ? router.CurrentMode : PhoneInputRouter.Mode.Macro;
    }

    void Start()
    {
        ApplyMode(_lastMode, force: true);
    }

    void Update()
    {
        if (router == null) return;

        var m = router.CurrentMode;
        if (m == _lastMode) return;

        _lastMode = m;
        ApplyMode(m, force: false);
    }

    private void ApplyMode(PhoneInputRouter.Mode mode, bool force)
    {
        bool isMicro = (mode == PhoneInputRouter.Mode.Micro);

        // -------------------------
        // Driver gating
        // -------------------------
        // Macro: macroPoseDriver ON, microRotationOnlyDriver OFF
        // Micro: macroPoseDriver OFF, microRotationOnlyDriver ON
        if (macroPoseDriver != null)
        {
            macroPoseDriver.enabled = !isMicro;

            if (!isMicro && recenterOnMacroEnable && !force)
            {
                // optional: recenter baseline for macro pose driver
                macroPoseDriver.SendMessage("Recenter", SendMessageOptions.DontRequireReceiver);
            }
        }

        if (microRotationOnlyDriver != null)
        {
            microRotationOnlyDriver.enabled = isMicro;

            if (isMicro && recenterRotationOnMicroEnable && !force)
            {
                // optional: recenter rotation baseline for micro rotation driver
                microRotationOnlyDriver.SendMessage("RecenterRotation", SendMessageOptions.DontRequireReceiver);
            }
        }

        // -------------------------
        // Micro controller gating
        // -------------------------
        SetEnabled(microPlacementController, isMicro && microTask == MicroTask.Placement);
        SetEnabled(microRotationController,  isMicro && microTask == MicroTask.Rotation);
        SetEnabled(microScalingController,   isMicro && microTask == MicroTask.Scaling);
    }

    private static void SetEnabled(Behaviour b, bool on)
    {
        if (b != null) b.enabled = on;
    }

    public void SetMicroTaskPlacement()
    {
        microTask = MicroTask.Placement;
        ApplyMode(_lastMode, force: true);
    }

    public void SetMicroTaskRotation()
    {
        microTask = MicroTask.Rotation;
        ApplyMode(_lastMode, force: true);
    }

    public void SetMicroTaskScaling()
    {
        microTask = MicroTask.Scaling;
        ApplyMode(_lastMode, force: true);
    }

    public void SetMicroTask(MicroTask t)
    {
        microTask = t;
        ApplyMode(_lastMode, force: true);
    }
}
