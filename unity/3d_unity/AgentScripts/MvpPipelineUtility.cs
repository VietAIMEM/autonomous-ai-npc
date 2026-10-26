using UnityEditor;

public static class MvpPipelineUtility
{
    public static string RefreshAssets()
    {
        AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
        return "refresh_requested";
    }
}
