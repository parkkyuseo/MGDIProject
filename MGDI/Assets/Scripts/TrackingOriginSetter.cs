using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

/// <summary>
/// Ensures the tracking origin is set to Device (head-based),
/// so object positions match between Unity Editor and HoloLens runtime.
/// Attach this script to a GameObject named "TrackingOriginSetter".
/// </summary>
public class TrackingOriginSetter : MonoBehaviour
{
    void Start()
    {
        var subsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetInstances(subsystems);

        foreach (var xr in subsystems)
        {
            // Try to force Device mode (head-relative origin)
            bool success = xr.TrySetTrackingOriginMode(TrackingOriginModeFlags.Device);

            Debug.Log($"[TrackingOriginSetter] Attempted to set Device mode. Success={success}, Current={xr.GetTrackingOriginMode()}");
        }
    }
}
