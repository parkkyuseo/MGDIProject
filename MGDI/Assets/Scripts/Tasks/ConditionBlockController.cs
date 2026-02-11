using System;
using System.Collections.Generic;
using UnityEngine;

public class ConditionBlockController : MonoBehaviour
{
    public enum Technique { Macro, Micro }
    public enum HandLocation { NearHead, Side } // "Side" means side-of-body input, proxy stays where it is

    [Serializable]
    public class Condition
    {
        public string label = "Cond";
        public Technique technique = Technique.Macro;
        public HandLocation handLocation = HandLocation.NearHead;

        [Tooltip("Only used for Macro+Side (Side→Front remap).")]
        public bool invertSideToFront = false;
    }

    [Header("Order (edit in Inspector)")]
    public List<Condition> conditions = new List<Condition>();

    [Header("Refs")]
    [SerializeField] private WorkflowProgressionController workflow;
    [SerializeField] private PhoneInputRouter phoneRouter;
    [SerializeField] private PhoneTechniqueGate phoneTechniqueGate;
    [SerializeField] private PhoneProxyHandRootDriver phoneMacroPoseDriver;

    [Header("Debug")]
    [SerializeField] private bool logDebug = true;

    [SerializeField] private TaskContextHUD taskContextHUD;

    private int _condIndex = 0;

    void Start()
    {
        if (workflow == null) workflow = FindFirstObjectByType<WorkflowProgressionController>();
        if (phoneRouter == null) phoneRouter = FindFirstObjectByType<PhoneInputRouter>();
        if (phoneTechniqueGate == null) phoneTechniqueGate = FindFirstObjectByType<PhoneTechniqueGate>();
        if (phoneMacroPoseDriver == null) phoneMacroPoseDriver = FindFirstObjectByType<PhoneProxyHandRootDriver>();
        if (taskContextHUD == null) taskContextHUD = FindFirstObjectByType<TaskContextHUD>();

        if (workflow == null)
        {
            Debug.LogError("[Cond] WorkflowProgressionController not found.");
            return;
        }

        workflow.OnAllCompleted += HandleBlockCompleted;

        // Start first condition
        _condIndex = Mathf.Clamp(_condIndex, 0, Mathf.Max(0, conditions.Count - 1));
        ApplyCurrentConditionAndRestartWorkflow();
    }

    void OnDestroy()
    {
        if (workflow != null)
            workflow.OnAllCompleted -= HandleBlockCompleted;
    }

    private void HandleBlockCompleted()
    {
        if (conditions == null || conditions.Count == 0) return;

        _condIndex++;
        if (_condIndex >= conditions.Count)
        {
            if (logDebug) Debug.Log("[Cond] All conditions completed.");
            return;
        }

        ApplyCurrentConditionAndRestartWorkflow();
    }

    private void ApplyCurrentConditionAndRestartWorkflow()
    {
        if (conditions == null || conditions.Count == 0)
        {
            Debug.LogWarning("[Cond] No conditions defined.");
            return;
        }

        var c = conditions[_condIndex];

        // 1) Technique routing (Macro/Micro)
        if (phoneRouter != null)
        {
            if (c.technique == Technique.Micro) phoneRouter.SetModeMicro();
            else phoneRouter.SetModeMacro();
        }

        // 2) Micro task gate is still controlled by StudyFlowController_V2 per phase,
        // so we do not force a micro task here. (Optional: you can leave it out.)

        // 3) Side→Front remap: only for Macro + Side
        bool remapOn = (c.technique == Technique.Macro) && (c.handLocation == HandLocation.Side);

        if (phoneMacroPoseDriver != null)
        {
            // Avoid jumps when toggling remap
            phoneMacroPoseDriver.RebaselineKeepWorldPose();
            phoneMacroPoseDriver.SetSideToFrontRemap(remapOn, c.invertSideToFront, forceRecenter: false);
        }

        if (taskContextHUD != null)
        {
            taskContextHUD.SetVisible(true);
            taskContextHUD.SetCondition($"{c.technique} · {(c.handLocation == HandLocation.Side ? "Side" : "Near")}" +
                                         $"{((c.technique == Technique.Macro && c.handLocation == HandLocation.Side) ? (c.invertSideToFront ? " · Remap(Invert)" : " · Remap") : "")}");
        }

        if (logDebug)
        {
            Debug.Log($"[Cond] Apply #{_condIndex + 1}/{conditions.Count} '{c.label}' tech={c.technique} loc={c.handLocation} remap={remapOn} invert={c.invertSideToFront}");
        }

        // 4) Restart workflow from Tool_01 + Placement
        if (workflow != null)
            workflow.RestartFromBeginning();
    }
}
