using System.Reflection;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class ToolGhostStatusHUD_Debug : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ProxyHandGrabber grabber;
    [SerializeField] private GhostHighlightController ghostHL;  // 네가 붙인 하이라이트 컨트롤러
    [SerializeField] private Transform slotsTargetsRoot;        // ContentRoot/Slots_Targets
    [SerializeField] private Text statusText;

    [Header("Options")]
    [SerializeField] private bool showGrabber = true;
    [SerializeField] private bool showHeldToolId = true;
    [SerializeField] private bool showGhostMap = true;
    [SerializeField] private bool showMaterials = true;

    private readonly StringBuilder _sb = new StringBuilder(1024);

    void Update()
    {
        if (statusText == null) return;

        _sb.Length = 0;

        // -------- Grabber ----------
        if (showGrabber)
        {
            _sb.AppendLine("[Grabber]");
            if (grabber == null)
            {
                _sb.AppendLine("grabber: MISSING");
            }
            else
            {
                _sb.AppendLine($"IsHolding: {(grabber.IsHolding ? "YES" : "NO")}");
                _sb.AppendLine($"HeldBody: {(grabber.HeldBody != null ? grabber.HeldBody.name : "null")}");
                _sb.AppendLine($"grabAnchor: {(grabber.grabAnchor != null ? grabber.grabAnchor.name : "null")}");

                // private field: _lastGripState
                string gs = ReadEnumPrivate(grabber, "_lastGripState");
                _sb.AppendLine($"GripState(private): {gs}");

                float since = ReadFieldOrNaN<float>(grabber, "_stateSinceTime");
                if (!float.IsNaN(since))
                {
                    float heldFor = Time.unscaledTime - since;
                    _sb.AppendLine($"StateSince: {since:F2}  HeldFor: {heldFor:F2}s");
                }
            }
        }

        // -------- Held ToolId resolution ----------
        if (showHeldToolId)
        {
            _sb.AppendLine();
            _sb.AppendLine("[Held ToolId]");

            if (grabber == null || grabber.HeldBody == null)
            {
                _sb.AppendLine("HeldBody: null");
            }
            else
            {
                var rb = grabber.HeldBody;

                ToolId tidChild = rb.GetComponentInChildren<ToolId>(true);
                ToolId tidParent = rb.GetComponentInParent<ToolId>();

                _sb.AppendLine($"ToolId in Children: {(tidChild != null ? tidChild.id : "null")}");
                _sb.AppendLine($"ToolId in Parent:   {(tidParent != null ? tidParent.id : "null")}");

                // Also show root object name for sanity
                Transform root = rb.transform.root;
                _sb.AppendLine($"RB root: {root.name}");
            }
        }

        // -------- Ghost map sanity ----------
        if (showGhostMap)
        {
            _sb.AppendLine();
            _sb.AppendLine("[Ghost Map]");

            if (slotsTargetsRoot == null)
            {
                _sb.AppendLine("slotsTargetsRoot: MISSING");
            }
            else
            {
                int ghostToolIdCount = slotsTargetsRoot.GetComponentsInChildren<ToolId>(true).Length;
                int ghostRendererCount = slotsTargetsRoot.GetComponentsInChildren<Renderer>(true).Length;

                _sb.AppendLine($"Ghost ToolId components: {ghostToolIdCount}");
                _sb.AppendLine($"Ghost Renderers: {ghostRendererCount}");

                // Try to read current highlighted id from GhostHighlightController if present
                if (ghostHL == null)
                {
                    _sb.AppendLine("ghostHL: MISSING");
                }
                else
                {
                    // currentId is private in my earlier suggested controller; read via reflection if exists
                    string cur = ReadFieldOrDefault<string>(ghostHL, "currentId", null);
                    _sb.AppendLine($"GhostHL.currentId(private): {(string.IsNullOrEmpty(cur) ? "null/empty" : cur)}");

                    // ghostMap is private Dictionary<...>; we can read count via reflection
                    int mapCount = ReadDictionaryCount(ghostHL, "ghostMap");
                    _sb.AppendLine($"GhostHL.ghostMap count(private): {(mapCount >= 0 ? mapCount.ToString() : "N/A")}");
                }
            }
        }

        // -------- Material diagnostics ----------
        if (showMaterials)
        {
            _sb.AppendLine();
            _sb.AppendLine("[Materials]");

            if (slotsTargetsRoot == null)
            {
                _sb.AppendLine("slotsTargetsRoot: MISSING");
            }
            else
            {
                // Sample a few renderers and report whether _Color exists
                var rs = slotsTargetsRoot.GetComponentsInChildren<Renderer>(true);
                int sample = Mathf.Min(rs.Length, 6);

                _sb.AppendLine($"SampleRenderers: {sample}/{rs.Length}");

                for (int i = 0; i < sample; i++)
                {
                    var r = rs[i];
                    if (r == null) continue;

                    var mats = r.sharedMaterials;
                    if (mats == null || mats.Length == 0)
                    {
                        _sb.AppendLine($"{r.name}: no mats");
                        continue;
                    }

                    var m0 = mats[0];
                    if (m0 == null)
                    {
                        _sb.AppendLine($"{r.name}: mat0 null");
                        continue;
                    }

                    bool hasColor = m0.HasProperty("_Color");
                    _sb.AppendLine($"{r.name}: {m0.shader.name}  _Color={(hasColor ? "YES" : "NO")}");
                }
            }
        }

        statusText.text = _sb.ToString();
    }

    // ---------- Reflection helpers ----------
    static T ReadFieldOrDefault<T>(object obj, string fieldName, T fallback)
    {
        if (obj == null) return fallback;
        var t = obj.GetType();
        var f = t.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (f == null) return fallback;
        object v = f.GetValue(obj);
        if (v is T tv) return tv;
        return fallback;
    }

    static float ReadFieldOrNaN<T>(object obj, string fieldName)
    {
        if (obj == null) return float.NaN;
        var t = obj.GetType();
        var f = t.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (f == null) return float.NaN;

        object v = f.GetValue(obj);
        if (v is float fv) return fv;
        if (v is int iv) return iv;
        if (v is double dv) return (float)dv;
        return float.NaN;
    }

    static string ReadEnumPrivate(object obj, string fieldName)
    {
        if (obj == null) return "N/A";
        var t = obj.GetType();
        var f = t.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (f == null) return "N/A";
        object v = f.GetValue(obj);
        return v != null ? v.ToString() : "N/A";
    }

    static int ReadDictionaryCount(object obj, string fieldName)
    {
        if (obj == null) return -1;
        var t = obj.GetType();
        var f = t.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (f == null) return -1;
        object v = f.GetValue(obj);
        if (v == null) return -1;

        // try property Count via reflection
        var p = v.GetType().GetProperty("Count");
        if (p == null) return -1;
        object c = p.GetValue(v);
        if (c is int ci) return ci;
        return -1;
    }
}
