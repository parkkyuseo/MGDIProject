using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class WorkspaceBoundsLine : MonoBehaviour
{
    [Header("Size (meters)")]
    public float width = 0.40f;   // x 방향(40cm)
    public float depth = 0.30f;   // z 방향(30cm)

    [Header("Placement")]
    public float yOffset = 0.0f;  // 테이블 위로 아주 살짝 올리고 싶으면 0.001~0.005

    [Header("Line")]
    public float lineWidth = 0.003f; // 3mm 정도
    public bool closedLoop = true;

    private LineRenderer lr;

    void Awake()
    {
        lr = GetComponent<LineRenderer>();
        lr.useWorldSpace = false;
        lr.loop = closedLoop;
        lr.widthMultiplier = lineWidth;

        // 머티리얼이 비어있으면 기본 Unlit 머티리얼 생성(런타임)
        if (lr.material == null)
        {
            var shader = Shader.Find("Unlit/Color");
            if (shader != null)
            {
                lr.material = new Material(shader);
                lr.material.color = Color.white;
            }
        }
    }

    void OnEnable()
    {
        UpdateLine();
    }

    void OnValidate()
    {
        if (lr == null) lr = GetComponent<LineRenderer>();
        UpdateLine();
    }

    private void UpdateLine()
    {
        if (lr == null) return;

        float hx = width * 0.5f;
        float hz = depth * 0.5f;
        float y = yOffset;

        // 사각형 4점 + (루프면 자동으로 닫힘)
        Vector3[] pts = new Vector3[4]
        {
            new Vector3(-hx, y, -hz),
            new Vector3( hx, y, -hz),
            new Vector3( hx, y,  hz),
            new Vector3(-hx, y,  hz),
        };

        lr.positionCount = pts.Length;
        lr.SetPositions(pts);
    }
}
