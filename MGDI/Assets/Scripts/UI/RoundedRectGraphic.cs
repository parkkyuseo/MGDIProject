using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(CanvasRenderer))]
public class RoundedRectGraphic : MaskableGraphic
{
    [SerializeField] private float cornerRadius = 22f;
    [SerializeField] private int cornerSegments = 6;

    private static readonly List<Vector2> _points = new List<Vector2>(64);

    public float CornerRadius
    {
        get => cornerRadius;
        set
        {
            float v = Mathf.Max(0f, value);
            if (Mathf.Approximately(cornerRadius, v))
                return;

            cornerRadius = v;
            SetVerticesDirty();
        }
    }

    public int CornerSegments
    {
        get => cornerSegments;
        set
        {
            int v = Mathf.Clamp(value, 1, 24);
            if (cornerSegments == v)
                return;

            cornerSegments = v;
            SetVerticesDirty();
        }
    }

    protected override void OnPopulateMesh(VertexHelper vh)
    {
        vh.Clear();

        Rect rect = GetPixelAdjustedRect();
        if (rect.width <= 0f || rect.height <= 0f)
            return;

        float radius = Mathf.Min(cornerRadius, rect.width * 0.5f, rect.height * 0.5f);
        int segments = Mathf.Clamp(cornerSegments, 1, 24);

        if (radius <= 0.01f)
        {
            AddQuad(vh, rect, color);
            return;
        }

        _points.Clear();

        AddArc(_points, new Vector2(rect.xMin + radius, rect.yMax - radius), 180f, 90f, radius, segments, true);
        AddArc(_points, new Vector2(rect.xMax - radius, rect.yMax - radius), 90f, 0f, radius, segments, false);
        AddArc(_points, new Vector2(rect.xMax - radius, rect.yMin + radius), 0f, -90f, radius, segments, false);
        AddArc(_points, new Vector2(rect.xMin + radius, rect.yMin + radius), -90f, -180f, radius, segments, false);

        UIVertex vert = UIVertex.simpleVert;
        vert.color = color;
        vert.position = rect.center;
        vh.AddVert(vert);

        for (int i = 0; i < _points.Count; i++)
        {
            vert.position = _points[i];
            vh.AddVert(vert);
        }

        for (int i = 0; i < _points.Count; i++)
        {
            int next = (i + 1) % _points.Count;
            vh.AddTriangle(0, i + 1, next + 1);
        }
    }

    private static void AddQuad(VertexHelper vh, Rect rect, Color32 color32)
    {
        UIVertex vert = UIVertex.simpleVert;
        vert.color = color32;

        vert.position = new Vector2(rect.xMin, rect.yMin);
        vh.AddVert(vert);
        vert.position = new Vector2(rect.xMin, rect.yMax);
        vh.AddVert(vert);
        vert.position = new Vector2(rect.xMax, rect.yMax);
        vh.AddVert(vert);
        vert.position = new Vector2(rect.xMax, rect.yMin);
        vh.AddVert(vert);

        vh.AddTriangle(0, 1, 2);
        vh.AddTriangle(0, 2, 3);
    }

    private static void AddArc(
        List<Vector2> points,
        Vector2 center,
        float startDeg,
        float endDeg,
        float radius,
        int segments,
        bool includeFirst)
    {
        int steps = Mathf.Max(1, segments);
        for (int i = 0; i <= steps; i++)
        {
            if (i == 0 && !includeFirst)
                continue;

            float t = i / (float)steps;
            float deg = Mathf.Lerp(startDeg, endDeg, t);
            float rad = deg * Mathf.Deg2Rad;
            points.Add(center + new Vector2(Mathf.Cos(rad), Mathf.Sin(rad)) * radius);
        }
    }
}
