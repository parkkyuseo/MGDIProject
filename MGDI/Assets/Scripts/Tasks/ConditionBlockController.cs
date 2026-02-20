using System;
using System.Collections.Generic;
using UnityEngine;

public class ConditionBlockController : MonoBehaviour
{
    public enum Technique { Macro, Micro }
    public enum HandLocation { NearHead, Side } // "Side" means side-of-body input.

    [Serializable]
    public class Condition
    {
        public string label = "Cond";
        public Technique technique = Technique.Macro;
        public HandLocation handLocation = HandLocation.NearHead;

        [Tooltip("Only used for Macro+Side remap.")]
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
    [SerializeField] private BasketToolResetter basketResetter;

    private int _condIndex = 0;
    private Condition _currentCondition;

    public Technique CurrentTechnique => _currentCondition != null ? _currentCondition.technique : Technique.Macro;
    public HandLocation CurrentHandLocation => _currentCondition != null ? _currentCondition.handLocation : HandLocation.NearHead;
    public bool HasCurrentCondition => _currentCondition != null;

    public string GetConditionLabel()
    {
        if (_currentCondition == null)
            return "Macro - Near Head";

        string tech = _currentCondition.technique == Technique.Micro ? "Micro" : "Macro";
        string location = _currentCondition.handLocation == HandLocation.Side ? "Side Of Body" : "Near Head";
        return $"{tech} - {location}";
    }

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
        _currentCondition = c;

        if (phoneRouter != null)
        {
            if (c.technique == Technique.Micro) phoneRouter.SetModeMicro();
            else phoneRouter.SetModeMacro();
        }

        bool remapOn = (c.technique == Technique.Macro) && (c.handLocation == HandLocation.Side);

        if (phoneMacroPoseDriver != null)
        {
            phoneMacroPoseDriver.RebaselineKeepWorldPose();
            phoneMacroPoseDriver.SetSideToFrontRemap(remapOn, c.invertSideToFront, forceRecenter: false);
        }

        if (taskContextHUD != null)
        {
            taskContextHUD.SetVisible(true);

            string condText = GetConditionLabel();
            if ((c.technique == Technique.Macro) && (c.handLocation == HandLocation.Side))
                condText += c.invertSideToFront ? " - Remap(Invert)" : " - Remap";

            taskContextHUD.SetCondition(condText);
        }

        if (logDebug)
        {
            Debug.Log($"[Cond] Apply #{_condIndex + 1}/{conditions.Count} '{c.label}' tech={c.technique} loc={c.handLocation} remap={remapOn} invert={c.invertSideToFront}");
        }

        if (workflow != null)
        {
            if (basketResetter != null)
                basketResetter.ResetAllToolsToBasket();

            workflow.RestartFromBeginning();
        }
    }
}
