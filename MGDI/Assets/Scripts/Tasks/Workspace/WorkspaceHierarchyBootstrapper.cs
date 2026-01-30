using UnityEngine;

public class WorkspaceHierarchyBootstrapper : MonoBehaviour
{
    [Header("Roots")]
    public string propsStaticName = "Props_Static";
    public string toolsDynamicName = "Tools_Dynamic";
    public string slotsTargetsName = "Slots_Targets";
    public string uiHudName = "UI_TaskHUD";

    [Header("Optional")]
    public bool createIfMissingOnStart = true;

    public Transform PropsStatic { get; private set; }
    public Transform ToolsDynamic { get; private set; }
    public Transform SlotsTargets { get; private set; }
    public Transform UiHud { get; private set; }

    void Start()
    {
        if (!createIfMissingOnStart) return;
        EnsureRoots();
    }

    [ContextMenu("Ensure Roots Now")]
    public void EnsureRoots()
    {
        PropsStatic  = EnsureChild(propsStaticName);
        ToolsDynamic = EnsureChild(toolsDynamicName);
        SlotsTargets = EnsureChild(slotsTargetsName);
        UiHud        = EnsureChild(uiHudName);
    }

    private Transform EnsureChild(string name)
    {
        var t = transform.Find(name);
        if (t != null) return t;

        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;
        return go.transform;
    }
}
