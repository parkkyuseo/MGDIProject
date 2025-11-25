using System;

public static class AnchorRuntime
{
    // Loader가 true로 올리고, 씬 종료/disable 시 false로 내리면 됨
    public static volatile bool AnchorReady = false;
}
