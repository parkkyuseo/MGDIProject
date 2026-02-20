using UnityEngine;

public class GhostVisualResolver : MonoBehaviour
{
    private const string GhostVisualName = "GhostVisual";

    [SerializeField] private Transform visualRoot;

    public Transform VisualRoot => visualRoot;

    private void Awake()
    {
        EnsureVisualRoot(createIfMissing: true);
    }

    private void Reset()
    {
        EnsureVisualRoot(createIfMissing: true);
    }

    [ContextMenu("AutoSetupGhostVisual()")]
    public void AutoSetupGhostVisual()
    {
        var root = EnsureVisualRoot(createIfMissing: true);
        if (root == null) return;

        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            var child = transform.GetChild(i);
            if (child == null || child == root) continue;
            if (child.GetComponentInChildren<Renderer>(true) == null) continue;
            child.SetParent(root, true);
        }
    }

    private Transform EnsureVisualRoot(bool createIfMissing)
    {
        if (visualRoot != null && visualRoot.IsChildOf(transform))
            return visualRoot;

        var existing = transform.Find(GhostVisualName);
        if (existing != null)
        {
            visualRoot = existing;
            return visualRoot;
        }

        if (!createIfMissing) return null;

        var go = new GameObject(GhostVisualName);
        visualRoot = go.transform;
        visualRoot.SetParent(transform, false);
        visualRoot.localPosition = Vector3.zero;
        visualRoot.localRotation = Quaternion.identity;
        visualRoot.localScale = Vector3.one;
        return visualRoot;
    }
}
