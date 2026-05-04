using UnityEngine;
using UnityEngine.SceneManagement;

public static class DisableSpatialAwarenessToolCollisionsRuntime
{
    private static bool _sceneHookInstalled;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Initialize()
    {
        Apply();

        if (_sceneHookInstalled)
            return;

        SceneManager.sceneLoaded += HandleSceneLoaded;
        _sceneHookInstalled = true;
    }

    private static void HandleSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        _ = scene;
        _ = mode;
        Apply();
    }

    private static void Apply()
    {
        int spatialAwarenessLayer = LayerMask.NameToLayer("Spatial Awareness");
        int grabbableLayer = LayerMask.NameToLayer("Grabbable");
        int toolInactiveLayer = LayerMask.NameToLayer("ToolInactive");

        if (spatialAwarenessLayer < 0)
            return;

        if (grabbableLayer >= 0)
            Physics.IgnoreLayerCollision(grabbableLayer, spatialAwarenessLayer, true);

        if (toolInactiveLayer >= 0)
            Physics.IgnoreLayerCollision(toolInactiveLayer, spatialAwarenessLayer, true);
    }
}
