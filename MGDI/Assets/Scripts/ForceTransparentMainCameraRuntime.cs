using UnityEngine;
using UnityEngine.SceneManagement;

public sealed class ForceTransparentMainCameraRuntime : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Install()
    {
        if (FindObjectOfType<ForceTransparentMainCameraRuntime>() != null)
            return;

        var go = new GameObject(nameof(ForceTransparentMainCameraRuntime));
        DontDestroyOnLoad(go);
        go.AddComponent<ForceTransparentMainCameraRuntime>();
    }

    private void OnEnable()
    {
        SceneManager.sceneLoaded += OnSceneLoaded;
        ApplyToMainCamera();
    }

    private void OnDisable()
    {
        SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    private void LateUpdate()
    {
        ApplyToMainCamera();
    }

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        _ = scene;
        _ = mode;
        ApplyToMainCamera();
    }

    private static void ApplyToMainCamera()
    {
        Camera cam = Camera.main;
        if (cam == null)
            return;

        if (cam.clearFlags != CameraClearFlags.SolidColor)
            cam.clearFlags = CameraClearFlags.SolidColor;

        Color bg = cam.backgroundColor;
        if (bg.r != 0f || bg.g != 0f || bg.b != 0f || bg.a != 0f)
            cam.backgroundColor = Color.clear;
    }
}
