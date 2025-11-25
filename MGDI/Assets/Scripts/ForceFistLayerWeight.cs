using UnityEngine;

[RequireComponent(typeof(Animator))]
public class ForceFistLayerWeight : MonoBehaviour
{
    public string layerName = "FistLayer";
    public float forcedWeight = 1f;

    Animator _anim;
    int _layerIndex = -1;

    void Awake()
    {
        _anim = GetComponent<Animator>();
        _layerIndex = _anim.GetLayerIndex(layerName);
        if (_layerIndex < 0)
        {
            Debug.LogError($"[ForceFistLayerWeight] Layer '{layerName}' not found.");
        }
    }

    void LateUpdate()
    {
        if (_layerIndex < 0) return;
        _anim.SetLayerWeight(_layerIndex, forcedWeight);
    }
}
