using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RoomAgent.Editor
{
    [InitializeOnLoad]
    public static class Phase16MotionFileImporter
    {
        private const string StagedRoot = "Assets/MotionImports/Phase16";
        private const string StableRoot = "Assets/Resources/GeneratedMotions/Phase16";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string ReportPath = "Temp/Phase16MotionImport.json";
        private const string AutoImportRequestPath = "Temp/Phase16AutoImport.request";

        private static readonly string[] RequiredHumanoidMappings =
        {
            "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
            "LeftFoot", "RightFoot", "Spine", "Chest", "Neck", "Head", "LeftShoulder",
            "RightShoulder", "LeftUpperArm", "RightUpperArm", "LeftLowerArm",
            "RightLowerArm", "LeftHand", "RightHand"
        };

        static Phase16MotionFileImporter()
        {
            EditorApplication.delayCall += RunRequestedAutoImport;
        }

        [MenuItem("Tools/Phase16/Import Real Walking Motion")]
        public static void ImportRealWalking()
        {
            Run(
                "Assets/animation/walking/Walking.fbx",
                "Walk",
                "phase16_walking",
                1);
        }

        private static void RunRequestedAutoImport()
        {
            var requestPath = Path.Combine(Directory.GetCurrentDirectory(), AutoImportRequestPath);
            if (!File.Exists(requestPath))
                return;
            File.Delete(requestPath);

            ImportRealWalking();
            var reportPath = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            var import = JsonUtility.FromJson<Phase16ImportReport>(File.ReadAllText(reportPath));
            if (!import.success)
                return;

            Phase16ProductionPlaybackValidator.Run(
                import.motionId,
                import.animationReference,
                import.stableClipPath);
        }

        public static void Run()
        {
            var versionText = GetArgument("-phase16Version");
            Run(
                GetArgument("-phase16SourceFile"),
                GetArgument("-phase16ClipName"),
                GetArgument("-phase16MotionId"),
                int.TryParse(versionText, out var version) ? version : 0);
        }

        public static void Run(
            string sourceFile,
            string requestedClipName,
            string motionId,
            int version)
        {
            var report = new Phase16ImportReport();
            try
            {
                if (string.IsNullOrWhiteSpace(motionId) || !IsValidMotionId(motionId))
                    throw new InvalidOperationException("INVALID_MOTION_ID");
                if (version <= 0)
                    throw new InvalidOperationException("INVALID_MOTION_VERSION");
                if (string.IsNullOrWhiteSpace(sourceFile))
                    throw new InvalidOperationException("MOTION_SOURCE_MISSING");

                var source = Path.GetFullPath(sourceFile);
                if (!File.Exists(source) || new FileInfo(source).Length == 0)
                    throw new InvalidOperationException("MOTION_SOURCE_MISSING");
                if (!string.Equals(Path.GetExtension(source), ".fbx", StringComparison.OrdinalIgnoreCase))
                    throw new InvalidOperationException("UNSUPPORTED_MOTION_FORMAT");

                var sourceHash = ComputeSha256(source);
                var targetAvatar = LoadProductionAvatar();
                var stagedAssetPath = StageSource(source, sourceHash, motionId);
                var (sourceClip, firstFrame, lastFrame) = ImportConfiguredClip(stagedAssetPath, requestedClipName);
                var sourceAvatar = LoadSourceAvatar(stagedAssetPath);

                var sourceMappings = GetHumanoidMappings(sourceAvatar);
                var targetMappings = GetHumanoidMappings(targetAvatar);
                var sourceAvatarValid = sourceAvatar != null && sourceAvatar.isValid;
                var sourceAvatarHuman = sourceAvatar != null && sourceAvatar.isHuman;
                var targetAvatarValid = targetAvatar != null && targetAvatar.isValid;
                var targetAvatarHuman = targetAvatar != null && targetAvatar.isHuman;
                var skeletonStatus = ClassifySkeleton(
                    sourceAvatarValid,
                    sourceAvatarHuman,
                    sourceMappings,
                    targetAvatarValid,
                    targetAvatarHuman,
                    targetMappings);

                var sourceModel = AssetDatabase.LoadAssetAtPath<GameObject>(stagedAssetPath);
                var rootBoneName = GetRootBoneName(sourceAvatar, sourceModel);
                if (string.IsNullOrWhiteSpace(rootBoneName))
                    throw new InvalidOperationException("MISSING_SKELETON_INFORMATION");

                var boneCount = sourceModel != null
                    ? sourceModel.GetComponentsInChildren<Transform>(true).Length
                    : Math.Max(1, sourceMappings.Length);
                var bindings = AnimationUtility.GetCurveBindings(sourceClip);
                var objectBindings = AnimationUtility.GetObjectReferenceCurveBindings(sourceClip);
                var keyCount = bindings.Sum(binding =>
                {
                    var curve = AnimationUtility.GetEditorCurve(sourceClip, binding);
                    return curve == null ? 0 : curve.keys.Length;
                });
                var finiteValues = IsFinite(sourceClip.length) &&
                                   IsFinite(sourceClip.frameRate) &&
                                   bindings.All(binding =>
                                   {
                                       var curve = AnimationUtility.GetEditorCurve(sourceClip, binding);
                                       return curve == null || curve.keys.All(key =>
                                           IsFinite(key.time) && IsFinite(key.value));
                                   });

                if (sourceClip.length <= 0f || sourceClip.frameRate <= 0f)
                    throw new InvalidOperationException("INVALID_MOTION_DURATION");
                if (bindings.Length + objectBindings.Length <= 0 || keyCount <= 0)
                    throw new InvalidOperationException("MOTION_CURVES_MISSING");
                if (!finiteValues)
                    throw new InvalidOperationException("NON_FINITE_MOTION_VALUES");

                AnimationClip stableClip = null;
                var stableClipPath = "";
                if (skeletonStatus == "COMPATIBLE" || skeletonStatus == "RETARGETABLE")
                {
                    stableClipPath = $"{StableRoot}/{motionId}/{sourceHash}.anim";
                    stableClip = CreateStableClip(sourceClip, stableClipPath, motionId);
                    ConfigureLibrary(stagedAssetPath, stableClip);
                }

                AssetDatabase.SaveAssets();
                report = new Phase16ImportReport
                {
                    success = true,
                    motionId = motionId,
                    version = version,
                    originalSourceFile = source,
                    stagedSourceFile = stagedAssetPath,
                    sourceSha256 = sourceHash,
                    sourceFormat = "FBX",
                    clipName = sourceClip.name,
                    durationSeconds = sourceClip.length,
                    fps = sourceClip.frameRate,
                    frameCount = lastFrame - firstFrame + 1,
                    curveCount = bindings.Length + objectBindings.Length,
                    keyCount = keyCount,
                    finiteValues = finiteValues,
                    boneCount = boneCount,
                    rootBoneName = rootBoneName,
                    sourceAvatarValid = sourceAvatarValid,
                    sourceAvatarHuman = sourceAvatarHuman,
                    sourceHumanoidMappings = sourceMappings,
                    targetAvatarName = targetAvatar != null ? targetAvatar.name : "",
                    targetAvatarValid = targetAvatarValid,
                    targetAvatarHuman = targetAvatarHuman,
                    targetHumanoidMappings = targetMappings,
                    skeletonStatus = skeletonStatus,
                    importStatus = "VALIDATED",
                    animationReference = stagedAssetPath,
                    stableClipPath = stableClipPath,
                    libraryPath = LibraryPath,
                    importedAt = DateTime.UtcNow.ToString("o"),
                    metadata = new ImportMetadata
                    {
                        unityVersion = Application.unityVersion,
                        sourceClipName = sourceClip.name,
                        firstFrame = firstFrame,
                        lastFrame = lastFrame,
                        stableClipName = stableClip != null ? stableClip.name : ""
                    }
                };
                WriteReport(report);
                if (Application.isBatchMode)
                    EditorApplication.Exit(0);
            }
            catch (Exception exception)
            {
                report.success = false;
                report.error = exception.ToString();
                WriteReport(report);
                Debug.LogError("[Phase16MotionFileImporter] " + report.error);
                if (Application.isBatchMode)
                    EditorApplication.Exit(1);
            }
        }

        private static Avatar LoadProductionAvatar()
        {
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            if (!scene.isLoaded)
                throw new InvalidOperationException("PRODUCTION_SCENE_MISSING");

            var character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
            var animator = character != null ? character.GetComponent<Animator>() : null;
            if (animator == null || animator.avatar == null)
                throw new InvalidOperationException("INVALID_PRODUCTION_RIG");
            return animator.avatar;
        }

        private static string StageSource(string source, string sourceHash, string motionId)
        {
            EnsureFolder("Assets", "MotionImports");
            EnsureFolder("Assets/MotionImports", "Phase16");
            EnsureFolder(StagedRoot, motionId);

            var stagedAssetPath = $"{StagedRoot}/{motionId}/{sourceHash}.fbx";
            var stagedFile = Path.GetFullPath(stagedAssetPath);
            if (Path.GetFullPath(source).Equals(stagedFile, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("INVALID_STAGED_SOURCE_REFERENCE");
            if (File.Exists(stagedFile) && ComputeSha256(stagedFile) != sourceHash)
                throw new InvalidOperationException("STAGED_SOURCE_HASH_MISMATCH");
            if (!File.Exists(stagedFile))
                File.Copy(source, stagedFile, false);

            StageSourceImporterSettings(source, stagedAssetPath, sourceHash, motionId);
            AssetDatabase.ImportAsset(
                stagedAssetPath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
            return stagedAssetPath;
        }

        private static void StageSourceImporterSettings(
            string source,
            string stagedAssetPath,
            string sourceHash,
            string motionId)
        {
            var sourceMetaPath = source + ".meta";
            if (!File.Exists(sourceMetaPath))
                return;

            var lines = File.ReadAllLines(sourceMetaPath);
            var guidUpdated = false;
            for (var index = 0; index < lines.Length; index++)
            {
                if (!lines[index].StartsWith("guid:", StringComparison.Ordinal))
                    continue;
                lines[index] = "guid: " + BuildStagedGuid(sourceHash, motionId);
                guidUpdated = true;
                break;
            }
            if (!guidUpdated)
                throw new InvalidOperationException("SOURCE_IMPORTER_SETTINGS_INVALID");

            File.WriteAllLines(stagedAssetPath + ".meta", lines);
        }

        private static string BuildStagedGuid(string sourceHash, string motionId)
        {
            // Each staged motion needs its own GUID. Deriving it from the content
            // hash alone made every staged copy of one source share a GUID with
            // its siblings, which forced Unity to resolve the conflict with a
            // fresh GUID while the import below was still running.
            using var sha256 = SHA256.Create();
            var digest = sha256.ComputeHash(
                System.Text.Encoding.UTF8.GetBytes(sourceHash + ":" + motionId));
            return BitConverter
                .ToString(digest)
                .Replace("-", string.Empty)
                .Substring(0, 32)
                .ToLowerInvariant();
        }

        private static Tuple<AnimationClip, int, int> ImportConfiguredClip(
            string stagedAssetPath,
            string requestedClipName)
        {
            var importer = AssetImporter.GetAtPath(stagedAssetPath) as ModelImporter;
            if (importer == null)
                throw new InvalidOperationException("UNITY_IMPORT_FAILED:ModelImporter missing");

            importer.animationType = ModelImporterAnimationType.Human;
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.sourceAvatar = null;

            var configuredClips = importer.clipAnimations;
            var defaultClips = configuredClips != null && configuredClips.Length > 0
                ? configuredClips
                : importer.defaultClipAnimations;
            if (defaultClips == null || defaultClips.Length == 0)
                throw new InvalidOperationException("MOTION_CLIP_MISSING");

            ModelImporterClipAnimation selected;
            if (string.IsNullOrWhiteSpace(requestedClipName))
            {
                if (defaultClips.Length > 1)
                    throw new InvalidOperationException("AMBIGUOUS_MOTION_CLIP");
                selected = defaultClips[0];
            }
            else
            {
                var matches = defaultClips
                    .Where(clip => string.Equals(clip.name, requestedClipName.Trim(), StringComparison.Ordinal))
                    .ToArray();
                if (matches.Length == 0)
                    throw new InvalidOperationException("MOTION_CLIP_MISSING");
                if (matches.Length > 1)
                    throw new InvalidOperationException("AMBIGUOUS_MOTION_CLIP");
                selected = matches[0];
            }

            var clipName = string.IsNullOrWhiteSpace(requestedClipName)
                ? selected.name
                : requestedClipName.Trim();
            var configuredClip = BuildConfiguredClip(selected, clipName);

            importer.clipAnimations = new[] { configuredClip };
            importer.SaveAndReimport();
            if (!TryHydrateHumanoidDescription(importer, stagedAssetPath))
            {
                // Unity may only expose the auto-mapped Humanoid description
                // after the mapped import has been written. Save once so the
                // generated Avatar can be read, then continue with the clip.
                importer.SaveAndReimport();
                TryHydrateHumanoidDescription(importer, stagedAssetPath);
            }

            importer.clipAnimations = new[] { configuredClip };
            importer.SaveAndReimport();

            var clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(stagedAssetPath);
            if (clip != null && clip.name != clipName)
                clip = AssetDatabase.LoadAllAssetsAtPath(stagedAssetPath)
                    .OfType<AnimationClip>()
                    .FirstOrDefault(candidate => candidate.name == clipName);
            if (clip == null)
                throw new InvalidOperationException("MOTION_CLIP_MISSING");
            return Tuple.Create(clip, Mathf.RoundToInt(selected.firstFrame), Mathf.RoundToInt(selected.lastFrame));
        }

        private static ModelImporterClipAnimation BuildConfiguredClip(
            ModelImporterClipAnimation selected,
            string clipName)
        {
            return new ModelImporterClipAnimation
            {
                name = clipName,
                takeName = selected.takeName,
                firstFrame = selected.firstFrame,
                lastFrame = selected.lastFrame,
                loop = selected.loop,
                loopTime = selected.loopTime,
                loopPose = selected.loopPose
            };
        }

        private static bool TryHydrateHumanoidDescription(
            ModelImporter importer,
            string stagedAssetPath)
        {
            // Unity auto-maps the Humanoid rig in memory but does not always
            // serialize that mapping into the .meta, which is the contract the
            // backend validator reads. Write the detected description back so
            // the staged source keeps its skeleton information on disk.
            var avatar = AssetDatabase.LoadAllAssetsAtPath(stagedAssetPath)
                .OfType<Avatar>()
                .FirstOrDefault();
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
                return false;

            var detected = avatar.humanDescription;
            if (detected.human == null || detected.human.Length == 0)
                return false;

            var description = importer.humanDescription;
            description.human = detected.human;
            if (detected.skeleton != null && detected.skeleton.Length > 0)
                description.skeleton = detected.skeleton;
            if (!string.IsNullOrWhiteSpace(detected.rootMotionBoneName))
                description.rootMotionBoneName = detected.rootMotionBoneName;
            importer.humanDescription = description;
            return true;
        }

        private static Avatar LoadSourceAvatar(string stagedAssetPath)
        {
            return AssetDatabase.LoadAllAssetsAtPath(stagedAssetPath)
                .OfType<Avatar>()
                .FirstOrDefault();
        }

        private static string[] GetHumanoidMappings(Avatar avatar)
        {
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
                return Array.Empty<string>();
            return avatar.humanDescription.human
                .Where(bone => !string.IsNullOrWhiteSpace(bone.humanName))
                .Select(bone => bone.humanName)
                .Distinct()
                .ToArray();
        }

        private static string ClassifySkeleton(
            bool sourceAvatarValid,
            bool sourceAvatarHuman,
            string[] sourceMappings,
            bool targetAvatarValid,
            bool targetAvatarHuman,
            string[] targetMappings)
        {
            if (!targetAvatarValid || !targetAvatarHuman ||
                !RequiredHumanoidMappings.All(targetMappings.Contains))
                return "INCOMPATIBLE";
            if (!sourceAvatarValid || !sourceAvatarHuman)
                return "INCOMPATIBLE";
            if (RequiredHumanoidMappings.All(sourceMappings.Contains))
                return "COMPATIBLE";
            if (sourceMappings.Contains("Hips") && sourceMappings.Length >= 8)
                return "RETARGETABLE";
            return "UNKNOWN";
        }

        private static string GetRootBoneName(Avatar avatar, GameObject model)
        {
            if (avatar != null)
            {
                var hips = avatar.humanDescription.human
                    .FirstOrDefault(bone => bone.humanName == "Hips");
                if (!string.IsNullOrWhiteSpace(hips.boneName))
                    return hips.boneName;
                if (!string.IsNullOrWhiteSpace(avatar.humanDescription.rootMotionBoneName))
                    return avatar.humanDescription.rootMotionBoneName;
            }
            return model != null ? model.name : null;
        }

        private static AnimationClip CreateStableClip(
            AnimationClip sourceClip,
            string stableClipPath,
            string motionId)
        {
            EnsureFolder("Assets", "Resources");
            EnsureFolder("Assets/Resources", "GeneratedMotions");
            EnsureFolder("Assets/Resources/GeneratedMotions", "Phase16");

            var directory = Path.GetDirectoryName(stableClipPath)?.Replace('\\', '/');
            if (!string.IsNullOrEmpty(directory))
            {
                var parent = Path.GetDirectoryName(directory)?.Replace('\\', '/') ?? "";
                var folder = Path.GetFileName(directory);
                EnsureFolder(parent, folder);
            }

            if (AssetDatabase.LoadAssetAtPath<AnimationClip>(stableClipPath) != null)
                AssetDatabase.DeleteAsset(stableClipPath);

            var stableClip = new AnimationClip
            {
                name = motionId,
                frameRate = sourceClip.frameRate
            };
            AssetDatabase.CreateAsset(stableClip, stableClipPath);

            foreach (var binding in AnimationUtility.GetCurveBindings(sourceClip))
            {
                var curve = AnimationUtility.GetEditorCurve(sourceClip, binding);
                if (curve != null)
                    AnimationUtility.SetEditorCurve(stableClip, binding, curve);
            }

            foreach (var binding in AnimationUtility.GetObjectReferenceCurveBindings(sourceClip))
            {
                AnimationUtility.SetObjectReferenceCurve(
                    stableClip,
                    binding,
                    AnimationUtility.GetObjectReferenceCurve(sourceClip, binding));
            }

            AnimationUtility.SetAnimationClipSettings(
                stableClip,
                AnimationUtility.GetAnimationClipSettings(sourceClip));
            EditorUtility.SetDirty(stableClip);

            var savedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(stableClipPath);
            if (savedClip == null || Mathf.Abs(savedClip.length - sourceClip.length) > 0.01f)
                throw new InvalidOperationException("STABLE_CLIP_CREATION_FAILED");
            return savedClip;
        }

        private static void ConfigureLibrary(string sourceReference, AnimationClip stableClip)
        {
            EnsureFolder("Assets", "Resources");
            EnsureFolder("Assets/Resources", "GeneratedMotions");
            EnsureFolder("Assets/Resources/GeneratedMotions", "Phase16");
            var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
            if (library == null)
            {
                library = ScriptableObject.CreateInstance<GeneratedMotionLibrary>();
                AssetDatabase.CreateAsset(library, LibraryPath);
            }
            library.Upsert(new GeneratedMotionLibrary.Motion
            {
                SourceReference = sourceReference,
                GeneratedClip = stableClip
            });
            EditorUtility.SetDirty(library);
        }

        private static void EnsureFolder(string parent, string name)
        {
            var path = parent + "/" + name;
            if (AssetDatabase.IsValidFolder(path))
                return;
            if (string.IsNullOrEmpty(AssetDatabase.CreateFolder(parent, name)))
                throw new InvalidOperationException("Failed to create asset folder: " + path);
        }

        private static string ComputeSha256(string path)
        {
            using var stream = File.OpenRead(path);
            using var sha256 = SHA256.Create();
            return BitConverter
                .ToString(sha256.ComputeHash(stream))
                .Replace("-", string.Empty)
                .ToLowerInvariant();
        }

        private static bool IsValidMotionId(string motionId)
        {
            return motionId.Length <= 120 && motionId.All(character =>
                character >= 'a' && character <= 'z' ||
                character >= '0' && character <= '9' ||
                character == '_');
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static string GetArgument(string name)
        {
            var arguments = Environment.GetCommandLineArgs();
            for (var index = 0; index < arguments.Length - 1; index++)
            {
                if (string.Equals(arguments[index], name, StringComparison.OrdinalIgnoreCase))
                    return arguments[index + 1];
            }
            return null;
        }

        private static void WriteReport(Phase16ImportReport report)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase16MotionFileImporter] " + JsonUtility.ToJson(report));
        }

        [Serializable]
        private sealed class Phase16ImportReport
        {
            public bool success;
            public string error;
            public string motionId;
            public int version;
            public string originalSourceFile;
            public string stagedSourceFile;
            public string sourceSha256;
            public string sourceFormat;
            public string clipName;
            public float durationSeconds;
            public float fps;
            public int frameCount;
            public int curveCount;
            public int keyCount;
            public bool finiteValues;
            public int boneCount;
            public string rootBoneName;
            public bool sourceAvatarValid;
            public bool sourceAvatarHuman;
            public string[] sourceHumanoidMappings;
            public string targetAvatarName;
            public bool targetAvatarValid;
            public bool targetAvatarHuman;
            public string[] targetHumanoidMappings;
            public string skeletonStatus;
            public string importStatus;
            public string animationReference;
            public string stableClipPath;
            public string libraryPath;
            public string importedAt;
            public ImportMetadata metadata;
        }

        [Serializable]
        private sealed class ImportMetadata
        {
            public string unityVersion;
            public string sourceClipName;
            public int firstFrame;
            public int lastFrame;
            public string stableClipName;
        }
    }
}
