using UnityEngine;
using System.Collections.Generic;

// 카메라에 붙여서 실행
public class HeadAxesDebug : MonoBehaviour
{
    [Header("Appearance")]
    public float len = 0.3f;           // 축 길이(m)
    public float startForward = 0.2f;  // 카메라 앞쪽으로 띄우는 오프셋(m)
    public float width = 0.003f;       // 라인 두께(m)
    public Material axisMaterial;      // 비워도 됨(내부 기본 머티리얼 사용). 안 보이면 Sprites/Default 포함

    readonly List<LineRenderer> _lrs = new List<LineRenderer>();

    void OnEnable()
    {
        CreateAxis("_AxisX", Color.red, Vector3.right);   // +X
        CreateAxis("_AxisY", Color.green, Vector3.up);      // +Y
        CreateAxis("_AxisZ", Color.blue, Vector3.forward); // +Z
    }

    void OnDisable()
    {
        // 클린업(컴포넌트 비활성/파괴 시 생성했던 축 삭제)
        foreach (var lr in _lrs) { if (lr) Destroy(lr.gameObject); }
        _lrs.Clear();
    }

    void CreateAxis(string name, Color color, Vector3 dir)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);

        var lr = go.AddComponent<LineRenderer>();
        lr.useWorldSpace = false;         // 카메라 로컬 좌표로 그린다
        lr.positionCount = 2;
        lr.widthMultiplier = width;
        lr.numCapVertices = 4;            // 끝 둥글게(가독성)

        // 머티리얼: 지정 안 하면 기본 Unlit 계열로 생성 시도
        if (axisMaterial != null)
        {
            lr.material = axisMaterial;
        }
        else
        {
            var sh = Shader.Find("Sprites/Default");
            if (sh == null) sh = Shader.Find("Unlit/Color");
            lr.material = new Material(sh);
        }

        lr.startColor = lr.endColor = color;

        // 근평면 뒤에 가려지지 않도록 '앞으로' 약간 띄운 위치에서 시작
        float near = 0.05f;
        var cam = GetComponent<Camera>();
        if (cam) near = Mathf.Max(cam.nearClipPlane, 0.01f);

        Vector3 start = Vector3.forward * Mathf.Max(startForward, near * 1.1f);
        lr.SetPosition(0, start);
        lr.SetPosition(1, start + dir.normalized * len);

        _lrs.Add(lr);
    }
}
