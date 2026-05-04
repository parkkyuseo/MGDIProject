using System.Collections;
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed class DisableMrtkGazeRuntime : MonoBehaviour
{
    private Coroutine disableRoutine;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Install()
    {
        if (FindObjectOfType<DisableMrtkGazeRuntime>() != null)
        {
            return;
        }

        var go = new GameObject(nameof(DisableMrtkGazeRuntime));
        DontDestroyOnLoad(go);
        go.AddComponent<DisableMrtkGazeRuntime>();
    }

    private void OnEnable()
    {
        SceneManager.sceneLoaded += OnSceneLoaded;
        QueueDisable();
    }

    private void OnDisable()
    {
        SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        QueueDisable();
    }

    private void QueueDisable()
    {
        if (disableRoutine != null)
        {
            StopCoroutine(disableRoutine);
        }

        disableRoutine = StartCoroutine(DisableWhenReady());
    }

    private IEnumerator DisableWhenReady()
    {
        for (int i = 0; i < 300; i++)
        {
            IMixedRealityGazeProvider gazeProvider = CoreServices.InputSystem?.GazeProvider;
            if (gazeProvider != null)
            {
                gazeProvider.GazeCursor?.SetVisibility(false);
                gazeProvider.Enabled = false;
                disableRoutine = null;
                yield break;
            }

            yield return null;
        }

        disableRoutine = null;
    }
}
