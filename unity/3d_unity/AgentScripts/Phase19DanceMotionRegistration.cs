using System.Text;
using RoomAgent.GeneratedMotions;
using UnityEditor;
using UnityEngine;

/// <summary>
/// PHASE 19 support: register the production dance FBX clip in the generated
/// motion library so the DANCE plan step for "Assets/animation/dance" resolves
/// through the normal BrainActionBridge path instead of failing with
/// generated_motion_unresolved.
///
/// Run through the Unity pipeline with run_script; it never touches play mode.
/// </summary>
public static class Phase19DanceMotionRegistration
{
    private const string DanceFbxPath = "Assets/animation/dance/dance.fbx";
    private const string DanceReference = "Assets/animation/dance";
    private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";

    public static string Register()
    {
        var report = new StringBuilder();

        AnimationClip clip = null;
        foreach (var asset in AssetDatabase.LoadAllAssetsAtPath(DanceFbxPath))
        {
            if (clip != null)
                break;
            if (asset is AnimationClip candidate)
                clip = candidate;
        }

        if (clip == null)
            return "ERROR:dance_clip_missing";

        var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
        if (library == null)
            return "ERROR:library_missing";

        report.Append("clip=").Append(clip.name);
        report.Append(";length=").Append(clip.length.ToString("0.###"));
        report.Append(";persistent=").Append(AssetDatabase.Contains(clip));
        if (AssetDatabase.TryGetGUIDAndLocalFileIdentifier(clip, out var guid, out long fileId))
            report.Append(";guid=").Append(guid).Append(";fileId=").Append(fileId);

        var before = library.ResolveClip(DanceReference);
        report.Append(";resolvedBefore=").Append(before != null);

        library.Upsert(new GeneratedMotionLibrary.Motion
        {
            SourceReference = DanceReference,
            GeneratedClip = clip
        });
        EditorUtility.SetDirty(library);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        var reloaded = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
        var resolved = reloaded != null ? reloaded.ResolveClip(DanceReference) : null;
        report.Append(";entries=").Append(reloaded != null ? reloaded.Motions.Count : -1);
        report.Append(";resolvedAfterSave=").Append(resolved != null);
        report.Append(";resolvedMatchesClip=").Append(ReferenceEquals(resolved, clip));
        report.Append(";ok=").Append(resolved != null && ReferenceEquals(resolved, clip));
        return report.ToString();
    }
}
