using UnityEngine;

public class TargetParentSwitcher : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Root transform that contains target objects (e.g., 'Targets').")]
    public Transform targetsRoot;

    [Tooltip("Workspace anchor (e.g., 'workspaceAnchor').")]
    public Transform workspaceAnchor;

    [Tooltip("World parent for rotation task. If null, detaches to scene root.")]
    public Transform worldParent;

    // Save original parent so it can be restored
    Transform _savedParent;
    bool _saved = false;

    void SaveIfNeeded()
    {
        if (_saved) return;
        if (targetsRoot == null) return;
        _savedParent = targetsRoot.parent;
        _saved = true;
    }

    public void AttachToWorkspace(bool keepWorld = true)
    {
        if (targetsRoot == null || workspaceAnchor == null) return;
        SaveIfNeeded();
        targetsRoot.SetParent(workspaceAnchor, keepWorld);
    }

    public void DetachToWorld(bool keepWorld = true)
    {
        if (targetsRoot == null) return;
        SaveIfNeeded();
        targetsRoot.SetParent(worldParent, keepWorld); // worldParent null => scene root
    }

    public void RestoreOriginalParent(bool keepWorld = true)
    {
        if (targetsRoot == null) return;
        if (!_saved) return;
        targetsRoot.SetParent(_savedParent, keepWorld);
    }
}
