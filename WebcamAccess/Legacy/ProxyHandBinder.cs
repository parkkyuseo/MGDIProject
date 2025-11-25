// ProxyHandBinder.cs
// - 씬 내 오브젝트 이름으로 찾아 ProxyHandRootFollower의 refWrist, refPalm 할당
// - 필요 시 한 번만 실행 후 비활성화해도 됨

using UnityEngine;

public class ProxyHandBinder : MonoBehaviour
{
    public string proxyFollowerObjectName = "ProxyHandRootFollower";
    public string wristObjectName = "Remote_Wrist";
    public string palmObjectName  = "Remote_Palm";

    void Start()
    {
        var followerGo = GameObject.Find(proxyFollowerObjectName);
        var wristGo = GameObject.Find(wristObjectName);
        var palmGo  = GameObject.Find(palmObjectName);

        if (followerGo == null) { Debug.LogWarning("[ProxyHandBinder] Proxy follower not found"); return; }
        var follower = followerGo.GetComponent<ProxyHandRootFollower>();
        if (follower == null) { Debug.LogWarning("[ProxyHandBinder] ProxyHandRootFollower component missing"); return; }

        if (wristGo != null) follower.refWrist = wristGo.transform;
        if (palmGo  != null) follower.refPalm  = palmGo.transform;

        Debug.Log("[ProxyHandBinder] Bound ProxyHandRootFollower refs.");
        // 필요하면 이 스크립트 비활성화 가능
        // this.enabled = false;
    }
}
