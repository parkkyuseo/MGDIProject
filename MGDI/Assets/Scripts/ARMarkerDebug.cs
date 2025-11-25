using UnityEngine;
using Microsoft.MixedReality.OpenXR;

public class ARMarkerDebug : MonoBehaviour
{
    public ARMarkerManager manager;

    void OnEnable()
    {
        if (!manager) manager = GetComponent<ARMarkerManager>();
        manager.markersChanged += OnMarkersChanged;
        Debug.Log("[ARMarkerDebug] subscribed.");
        //try { DebugHUD.Log("[ARMarkerDebug] subscribed."); } catch { }
    }
    void OnDisable()
    {
        if (manager) manager.markersChanged -= OnMarkersChanged;
    }

    void OnMarkersChanged(ARMarkersChangedEventArgs args)
    {
        int a = args.added != null ? args.added.Count : 0;
        int u = args.updated != null ? args.updated.Count : 0;
        int r = args.removed != null ? args.removed.Count : 0;
        string msg = $"[ARMarkerDebug] changed: added={a} updated={u} removed={r}";
        Debug.Log(msg);
        //try { DebugHUD.Log(msg); } catch { }

        if (a > 0)
        {
            var m = args.added[0];
            var t = m.transform;
            string pose = $"pos=({t.position.x:0.00},{t.position.y:0.00},{t.position.z:0.00})";
            Debug.Log("[ARMarkerDebug] first added " + pose);
            try { DebugHUD.Log("[ARMarkerDebug] first added " + pose); } catch { }
        }
    }
}
