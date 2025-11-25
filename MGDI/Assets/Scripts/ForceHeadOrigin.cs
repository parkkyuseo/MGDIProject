using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class ForceHeadOrigin : MonoBehaviour
{
    void OnEnable()
    {
        ApplyDeviceOrigin();
        // 일부 런타임은 초기화 후 모드를 바꿀 수 있으니 이벤트로 다시 적용
        var subsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetInstances(subsystems);
        foreach (var s in subsystems)
        {
            s.trackingOriginUpdated += OnTrackingOriginUpdated;
        }
    }

    void OnDisable()
    {
        var subsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetInstances(subsystems);
        foreach (var s in subsystems)
        {
            s.trackingOriginUpdated -= OnTrackingOriginUpdated;
        }
    }

    void OnTrackingOriginUpdated(XRInputSubsystem s)
    {
        ApplyDeviceOrigin();
    }

    void ApplyDeviceOrigin()
    {
        var subsystems = new List<XRInputSubsystem>();
        SubsystemManager.GetInstances(subsystems);
        foreach (var s in subsystems)
        {
            var supported = s.GetSupportedTrackingOriginModes();
            if ((supported & TrackingOriginModeFlags.Device) != 0)
            {
                s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Device);
            }
            // 필요하면 else에서 Floor로 폴백 등 처리
        }
    }
}
