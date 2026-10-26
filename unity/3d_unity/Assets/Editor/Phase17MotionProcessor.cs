using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;

namespace RoomAgent.Editor
{
    [InitializeOnLoad]
    public static class Phase17MotionProcessor
    {
        private const string StableRoot = "Assets/Resources/GeneratedMotions/Phase17";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string ReportPath = "Temp/Phase17MotionProcessing.json";
        private const string AutoRequestPath = "Temp/Phase17MotionProcessing.request";
        private const int MaxRigSamples = 240;

        private static readonly string[] RequiredHumanoidMappings =
        {
            "Hips", "Spine", "Chest", "Neck", "Head",
            "LeftShoulder", "RightShoulder",
            "LeftUpperArm", "RightUpperArm",
            "LeftLowerArm", "RightLowerArm",
            "LeftHand", "RightHand",
            "LeftUpperLeg", "RightUpperLeg",
            "LeftLowerLeg", "RightLowerLeg",
            "LeftFoot", "RightFoot"
        };

        private static readonly HumanBodyBones[] RequiredSampleBones =
        {
            HumanBodyBones.Hips,
            HumanBodyBones.Spine,
            HumanBodyBones.Chest,
            HumanBodyBones.Neck,
            HumanBodyBones.Head,
            HumanBodyBones.LeftShoulder,
            HumanBodyBones.RightShoulder,
            HumanBodyBones.LeftUpperArm,
            HumanBodyBones.RightUpperArm,
            HumanBodyBones.LeftLowerArm,
            HumanBodyBones.RightLowerArm,
            HumanBodyBones.LeftHand,
            HumanBodyBones.RightHand,
            HumanBodyBones.LeftUpperLeg,
            HumanBodyBones.RightUpperLeg,
            HumanBodyBones.LeftLowerLeg,
            HumanBodyBones.RightLowerLeg,
            HumanBodyBones.LeftFoot,
            HumanBodyBones.RightFoot
        };

        static Phase17MotionProcessor()
        {
            EditorApplication.delayCall += RunRequestedAutoProcessing;
        }

        [MenuItem("Tools/Phase17/Process Existing Motion")]
        public static void ProcessMenuMotion()
        {
            Run(
                "phase16_walking",
                1,
                "Assets/MotionImports/Phase16/phase16_walking/2f5d893584ee508cb00bcff0c34bad4736e790494efca49179def9c80a26a827.fbx",
                "phase16_walking_processed_v1",
                1,
                true);
        }

        public static void Run()
        {
            var sourceVersionText = GetArgument("-phase17SourceVersion");
            var processingVersionText = GetArgument("-phase17ProcessingVersion");
            Run(
                GetArgument("-phase17SourceMotionId"),
                int.TryParse(sourceVersionText, out var sourceVersion) ? sourceVersion : 0,
                GetArgument("-phase17SourceReference"),
                GetArgument("-phase17ProcessedMotionId"),
                int.TryParse(processingVersionText, out var processingVersion) ? processingVersion : 0,
                false);
        }

        private static void RunRequestedAutoProcessing()
        {
            var requestPath = Path.Combine(Directory.GetCurrentDirectory(), AutoRequestPath);
            if (!File.Exists(requestPath))
                return;

            var request = JsonUtility.FromJson<AutoRunRequest>(File.ReadAllText(requestPath));
            File.Delete(requestPath);
            Run(
                request.sourceMotionId,
                request.sourceVersion,
                request.sourceReference,
                request.processedMotionId,
                request.processingVersion,
                request.runPlayback);
        }

        public static void Run(
            string sourceMotionId,
            int sourceVersion,
            string sourceReference,
            string processedMotionId,
            int processingVersion,
            bool runPlayback)
        {
            var report = new Phase17ProcessingReport
            {
                sourceMotionId = sourceMotionId ?? "",
                sourceVersion = sourceVersion,
                sourceAnimationReference = sourceReference ?? "",
                processedMotionId = processedMotionId ?? "",
                processingVersion = processingVersion,
                state = "FAILED"
            };

            var outputCreated = false;
            try
            {
                ValidateArguments(sourceMotionId, sourceVersion, sourceReference, processedMotionId, processingVersion);

                var processedClipPath =
                    $"{StableRoot}/{processedMotionId}/v{processingVersion}/{processedMotionId}.anim";
                report.processedClipPath = processedClipPath;
                report.animationReference = processedClipPath;
                report.libraryPath = LibraryPath;
                if (OutputExists(processedClipPath))
                    throw new InvalidOperationException("PROCESSED_CLIP_ALREADY_EXISTS");

                var production = LoadProductionRig();
                if (production.Animator.applyRootMotion)
                    throw new InvalidOperationException("ROOT_MOTION_AUTHORITY_INVALID");

                var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
                if (library == null)
                    throw new InvalidOperationException("GENERATED_MOTION_LIBRARY_MISSING");
                var sourceClip = library.ResolveClip(sourceReference);
                if (sourceClip == null || !AssetDatabase.Contains(sourceClip))
                    throw new InvalidOperationException("SOURCE_CLIP_UNRESOLVED");

                var sourceAvatar = LoadSourceAvatar(sourceReference);
                var sourceModel = AssetDatabase.LoadAssetAtPath<GameObject>(sourceReference);
                if (sourceAvatar == null || !sourceAvatar.isValid || !sourceAvatar.isHuman)
                    throw new InvalidOperationException("INVALID_SOURCE_AVATAR");

                var sourceMappings = GetHumanoidMappings(sourceAvatar);
                var targetMappings = GetHumanoidMappings(production.Avatar);
                var mappingEntries = BuildSkeletonMappings(
                    sourceAvatar,
                    production.Avatar,
                    out var missingMappings,
                    out var ambiguousMappings);
                var skeletonStatus = ClassifySkeleton(
                    sourceAvatar,
                    production.Avatar,
                    sourceMappings,
                    targetMappings,
                    missingMappings,
                    ambiguousMappings);

                report.sourceAvatarValid = sourceAvatar.isValid;
                report.sourceAvatarHuman = sourceAvatar.isHuman;
                report.targetAvatarValid = production.Avatar != null && production.Avatar.isValid;
                report.targetAvatarHuman = production.Avatar != null && production.Avatar.isHuman;
                report.sourceHumanoidMappings = sourceMappings;
                report.targetHumanoidMappings = targetMappings;
                report.mappingEntries = mappingEntries;
                report.missingMappings = missingMappings;
                report.ambiguousMappings = ambiguousMappings;
                report.skeletonStatus = skeletonStatus;
                report.rootBoneName = GetRootBoneName(sourceAvatar, sourceModel);
                report.boneCount = sourceModel != null
                    ? sourceModel.GetComponentsInChildren<Transform>(true).Length
                    : RequiredHumanoidMappings.Length;
                if (skeletonStatus != "COMPATIBLE")
                    throw new InvalidOperationException("MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING:" + skeletonStatus);

                var analysis = AnalyzeClip(sourceClip, production);
                ApplyAnalysis(report, analysis);
                report.rootAuthoritativeNavigation =
                    !production.Animator.applyRootMotion &&
                    production.Agent != null &&
                    production.Agent.enabled;
                report.applyRootMotion = production.Animator.applyRootMotion;
                if (analysis.HardFailures.Any() || analysis.QualityStatus == "REJECTED")
                    throw new InvalidOperationException(
                        "MOTION_QUALITY_REJECTED:" + string.Join(",", analysis.HardFailures));

                AnimationClip processedClip;
                try
                {
                    processedClip = CreateProcessedClip(sourceClip, processedClipPath, processedMotionId);
                    outputCreated = true;
                }
                catch
                {
                    if (outputCreated)
                        AssetDatabase.DeleteAsset(processedClipPath);
                    throw;
                }

                var savedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(processedClipPath);
                if (savedClip == null || !AssetDatabase.Contains(savedClip) ||
                    AssetDatabase.GetAssetPath(savedClip) != processedClipPath)
                    throw new InvalidOperationException("PROCESSED_CLIP_NOT_PERSISTENT");

                var processedAnalysis = AnalyzeClip(savedClip, production);
                ApplyAnalysis(report, processedAnalysis);
                if (processedAnalysis.HardFailures.Any() ||
                    processedAnalysis.QualityStatus == "REJECTED" ||
                    Mathf.Abs(savedClip.length - sourceClip.length) > 0.01f ||
                    Mathf.Abs(savedClip.frameRate - sourceClip.frameRate) > 0.01f)
                    throw new InvalidOperationException("PROCESSED_CLIP_VALIDATION_FAILED");

                library.Upsert(new GeneratedMotionLibrary.Motion
                {
                    SourceReference = processedClipPath,
                    GeneratedClip = savedClip
                });
                EditorUtility.SetDirty(library);
                AssetDatabase.SaveAssets();

                report.success = true;
                report.state = "READY";
                report.retargetStatus = "MAPPED";
                report.cleanupStatus = "NOT_REQUIRED";
                report.cleanupMagnitude = 0f;
                report.cleanupKeyCount = 0;
                report.processingTimestamp = DateTime.UtcNow.ToString("o");
                report.error = null;
                WriteReport(report);

                if (runPlayback && !Application.isBatchMode)
                {
                    Phase17ProductionPlaybackValidator.Run(
                        report.sourceMotionId,
                        report.sourceVersion,
                        report.processedMotionId,
                        report.processingVersion,
                        report.sourceAnimationReference,
                        report.processedClipPath);
                    return;
                }

                if (Application.isBatchMode)
                    EditorApplication.Exit(0);
            }
            catch (Exception exception)
            {
                if (outputCreated)
                    DiscardFailedOutput(report.processedClipPath);
                report.success = false;
                report.state = "FAILED";
                report.error = exception.ToString();
                report.processingTimestamp = DateTime.UtcNow.ToString("o");
                WriteReport(report);
                Debug.LogError("[Phase17MotionProcessor] " + report.error);
                if (Application.isBatchMode)
                    EditorApplication.Exit(1);
            }
        }

        private static void DiscardFailedOutput(string processedClipPath)
        {
            var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
            if (library != null && library.ResolveClip(processedClipPath) != null)
            {
                library.Configure(
                    library.Motions.Where(motion =>
                        !string.Equals(
                            NormalizeAssetReference(motion.SourceReference),
                            NormalizeAssetReference(processedClipPath),
                            StringComparison.Ordinal)));
                EditorUtility.SetDirty(library);
                AssetDatabase.SaveAssets();
            }
            AssetDatabase.DeleteAsset(processedClipPath);
        }

        private static string NormalizeAssetReference(string value)
        {
            var normalized = (value ?? string.Empty).Trim().Replace('\\', '/');
            while (normalized.EndsWith("/", StringComparison.Ordinal))
                normalized = normalized.Substring(0, normalized.Length - 1);
            return normalized;
        }

        private static void ValidateArguments(
            string sourceMotionId,
            int sourceVersion,
            string sourceReference,
            string processedMotionId,
            int processingVersion)
        {
            if (string.IsNullOrWhiteSpace(sourceMotionId) || !IsValidMotionId(sourceMotionId))
                throw new InvalidOperationException("INVALID_SOURCE_MOTION_ID");
            if (sourceVersion <= 0)
                throw new InvalidOperationException("INVALID_SOURCE_MOTION_VERSION");
            if (string.IsNullOrWhiteSpace(sourceReference) || !sourceReference.StartsWith("Assets/", StringComparison.Ordinal))
                throw new InvalidOperationException("INVALID_SOURCE_REFERENCE");
            if (sourceReference.Contains("__preview__", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("PREVIEW_ANIMATION_FORBIDDEN");
            if (string.IsNullOrWhiteSpace(processedMotionId) || !IsValidMotionId(processedMotionId))
                throw new InvalidOperationException("INVALID_PROCESSED_MOTION_ID");
            if (processingVersion <= 0)
                throw new InvalidOperationException("INVALID_PROCESSING_VERSION");
        }

        private static ProductionRig LoadProductionRig()
        {
            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            if (!scene.isLoaded)
                throw new InvalidOperationException("PRODUCTION_SCENE_MISSING");

            var character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
            var animator = character != null ? character.GetComponent<Animator>() : null;
            var agent = character != null ? character.GetComponent<NavMeshAgent>() : null;
            if (character == null || animator == null || animator.avatar == null ||
                !animator.avatar.isValid || !animator.avatar.isHuman || agent == null)
                throw new InvalidOperationException("INVALID_PRODUCTION_RIG");
            return new ProductionRig
            {
                Character = character.gameObject,
                Avatar = animator.avatar,
                Animator = animator,
                Agent = agent
            };
        }

        private static Avatar LoadSourceAvatar(string sourceReference)
        {
            return AssetDatabase.LoadAllAssetsAtPath(sourceReference)
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

        private static SkeletonMappingReport[] BuildSkeletonMappings(
            Avatar sourceAvatar,
            Avatar targetAvatar,
            out string[] missingMappings,
            out string[] ambiguousMappings)
        {
            var sourceGroups = GroupHumanBones(sourceAvatar);
            var targetGroups = GroupHumanBones(targetAvatar);
            var entries = new List<SkeletonMappingReport>();
            var missing = new List<string>();
            var ambiguous = new List<string>();
            foreach (var human in RequiredHumanoidMappings)
            {
                var sourceBones = sourceGroups.TryGetValue(human, out var source) ? source : new List<string>();
                var targetBones = targetGroups.TryGetValue(human, out var target) ? target : new List<string>();
                if (sourceBones.Count == 0 || targetBones.Count == 0)
                    missing.Add(human);
                else if (sourceBones.Count > 1 || targetBones.Count > 1)
                    ambiguous.Add(human);
                else
                {
                    entries.Add(new SkeletonMappingReport
                    {
                        humanBone = human,
                        sourceBone = sourceBones[0],
                        targetBone = targetBones[0]
                    });
                }
            }
            missingMappings = missing.ToArray();
            ambiguousMappings = ambiguous.ToArray();
            return entries.ToArray();
        }

        private static Dictionary<string, List<string>> GroupHumanBones(Avatar avatar)
        {
            var result = new Dictionary<string, List<string>>();
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
                return result;
            foreach (var bone in avatar.humanDescription.human)
            {
                if (string.IsNullOrWhiteSpace(bone.humanName) || string.IsNullOrWhiteSpace(bone.boneName))
                    continue;
                if (!result.TryGetValue(bone.humanName, out var bones))
                {
                    bones = new List<string>();
                    result[bone.humanName] = bones;
                }
                bones.Add(bone.boneName);
            }
            return result;
        }

        private static string ClassifySkeleton(
            Avatar sourceAvatar,
            Avatar targetAvatar,
            string[] sourceMappings,
            string[] targetMappings,
            string[] missingMappings,
            string[] ambiguousMappings)
        {
            if (sourceAvatar == null || !sourceAvatar.isValid || !sourceAvatar.isHuman ||
                targetAvatar == null || !targetAvatar.isValid || !targetAvatar.isHuman)
                return "INCOMPATIBLE";
            if (missingMappings.Length > 0 || ambiguousMappings.Length > 0)
                return "UNKNOWN";
            if (RequiredHumanoidMappings.All(sourceMappings.Contains) &&
                RequiredHumanoidMappings.All(targetMappings.Contains))
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
            return model != null ? model.name : "";
        }

        private static bool OutputExists(string processedClipPath)
        {
            var fullPath = Path.GetFullPath(processedClipPath);
            return File.Exists(fullPath) || File.Exists(fullPath + ".meta");
        }

        private static AnimationClip CreateProcessedClip(
            AnimationClip sourceClip,
            string processedClipPath,
            string processedMotionId)
        {
            var directory = Path.GetDirectoryName(processedClipPath)?.Replace('\\', '/');
            if (string.IsNullOrEmpty(directory))
                throw new InvalidOperationException("INVALID_PROCESSED_CLIP_PATH");
            EnsureFolder("Assets", "Resources");
            EnsureFolder("Assets/Resources", "GeneratedMotions");
            EnsureFolder("Assets/Resources/GeneratedMotions", "Phase17");
            var parts = directory.Split('/');
            for (var index = 1; index < parts.Length; index++)
                EnsureFolder(string.Join("/", parts, 0, index), parts[index]);

            var processedClip = new AnimationClip
            {
                name = processedMotionId,
                frameRate = sourceClip.frameRate
            };
            AssetDatabase.CreateAsset(processedClip, processedClipPath);

            foreach (var binding in AnimationUtility.GetCurveBindings(sourceClip))
            {
                var curve = AnimationUtility.GetEditorCurve(sourceClip, binding);
                if (curve != null)
                    AnimationUtility.SetEditorCurve(processedClip, binding, curve);
            }
            foreach (var binding in AnimationUtility.GetObjectReferenceCurveBindings(sourceClip))
            {
                AnimationUtility.SetObjectReferenceCurve(
                    processedClip,
                    binding,
                    AnimationUtility.GetObjectReferenceCurve(sourceClip, binding));
            }
            AnimationUtility.SetAnimationClipSettings(
                processedClip,
                AnimationUtility.GetAnimationClipSettings(sourceClip));
            EditorUtility.SetDirty(processedClip);

            var savedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(processedClipPath);
            if (savedClip == null || Mathf.Abs(savedClip.length - sourceClip.length) > 0.01f)
                throw new InvalidOperationException("PROCESSED_CLIP_CREATION_FAILED");
            return savedClip;
        }

        private static ClipAnalysis AnalyzeClip(AnimationClip clip, ProductionRig production)
        {
            var analysis = new ClipAnalysis
            {
                DurationSeconds = clip.length,
                Fps = clip.frameRate,
                FrameCount = Mathf.Max(1, Mathf.RoundToInt(clip.length * clip.frameRate) + 1),
                QualityStatus = "VALID",
                JointStatus = "VALID",
                RootStatus = "STABLE",
                FootStatus = "UNKNOWN",
                FootConfidence = 0f
            };

            var bindings = AnimationUtility.GetCurveBindings(clip);
            var objectBindings = AnimationUtility.GetObjectReferenceCurveBindings(clip);
            analysis.CurveCount = bindings.Length + objectBindings.Length;
            analysis.KeyCount = 0;
            var curves = new Dictionary<string, AnimationCurve>();
            foreach (var binding in bindings)
            {
                var curve = AnimationUtility.GetEditorCurve(clip, binding);
                if (curve == null)
                    continue;
                analysis.KeyCount += curve.keys.Length;
                if (curves.ContainsKey(binding.propertyName))
                    analysis.HardFailures.Add("CURVE_BINDING_AMBIGUOUS");
                else
                    curves[binding.propertyName] = curve;
            }

            var finite = IsFinite(clip.length) && IsFinite(clip.frameRate);
            var monotonic = true;
            var maxJump = 0f;
            foreach (var curve in curves.Values)
            {
                var previousTime = float.NegativeInfinity;
                var previousValue = 0f;
                foreach (var key in curve.keys)
                {
                    analysis.KeyCount += 0;
                    if (!IsFinite(key.time) || !IsFinite(key.value) ||
                        !IsFinite(key.inTangent) || !IsFinite(key.outTangent))
                        finite = false;
                    if (key.time <= previousTime)
                        monotonic = false;
                    if (previousTime > float.NegativeInfinity)
                        maxJump = Mathf.Max(maxJump, Mathf.Abs(key.value - previousValue));
                    previousTime = key.time;
                    previousValue = key.value;
                }
            }
            analysis.FiniteValues = finite;
            analysis.KeyTimesMonotonic = monotonic;
            analysis.MaxCurveJump = maxJump;
            if (!finite)
                analysis.HardFailures.Add("NON_FINITE_MOTION_VALUES");
            if (!monotonic)
                analysis.HardFailures.Add("NON_MONOTONIC_KEY_TIMES");
            if (analysis.CurveCount <= 0 || analysis.KeyCount <= 0)
                analysis.HardFailures.Add("MOTION_CURVES_MISSING");
            if (analysis.DurationSeconds <= 0f || analysis.Fps <= 0f)
                analysis.HardFailures.Add("INVALID_MOTION_TIMING");
            if (maxJump > 2.5f)
                analysis.HardFailures.Add("SEVERE_CURVE_DISCONTINUITY");
            else if (maxJump > 1.0f)
                analysis.Warnings.Add("CURVE_DISCONTINUITY_WARNING");

            var expectedKeys = Mathf.Max(1, Mathf.RoundToInt(clip.length * clip.frameRate) + 1);
            var observedKeys = curves.Values.Any()
                ? curves.Values.Max(curve => curve.keys.Length)
                : 0;
            analysis.MissingSampleCount = Mathf.Max(0, expectedKeys - observedKeys);
            analysis.MissingSampleRatio = expectedKeys > 0
                ? Mathf.Clamp01((float)analysis.MissingSampleCount / expectedKeys)
                : 1f;
            if (analysis.MissingSampleCount > 0)
                analysis.Warnings.Add("MISSING_SAMPLES");
            if (analysis.MissingSampleRatio > 0.20f)
                analysis.HardFailures.Add("EXCESSIVE_MISSING_SAMPLES");

            analysis.QuaternionCurvesFound =
                curves.ContainsKey("RootQ.x") && curves.ContainsKey("RootQ.y") &&
                curves.ContainsKey("RootQ.z") && curves.ContainsKey("RootQ.w");
            if (analysis.QuaternionCurvesFound)
            {
                var times = curves["RootQ.x"].keys.Select(key => key.time).ToArray();
                analysis.QuaternionValid = times.All(time =>
                {
                    var quaternion = new Quaternion(
                        curves["RootQ.x"].Evaluate(time),
                        curves["RootQ.y"].Evaluate(time),
                        curves["RootQ.z"].Evaluate(time),
                        curves["RootQ.w"].Evaluate(time));
                    return IsFinite(quaternion.x) && IsFinite(quaternion.y) &&
                           IsFinite(quaternion.z) && IsFinite(quaternion.w) &&
                           0.9f <= Mathf.Sqrt(Quaternion.Dot(quaternion, quaternion)) &&
                           Mathf.Sqrt(Quaternion.Dot(quaternion, quaternion)) <= 1.1f;
                });
                if (!analysis.QuaternionValid)
                    analysis.HardFailures.Add("INVALID_QUATERNION_DATA");
            }
            else
            {
                analysis.Warnings.Add("ROOT_ROTATION_UNAVAILABLE");
            }

            var samples = SampleProductionRig(production, clip);
            AnalyzeProductionSamples(analysis, samples);

            foreach (var pair in curves)
            {
                if (pair.Key.StartsWith("Root", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("T.x", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("T.y", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("T.z", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("Q.x", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("Q.y", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("Q.z", StringComparison.Ordinal) ||
                    pair.Key.EndsWith("Q.w", StringComparison.Ordinal))
                    continue;
                var warning = pair.Value.keys.Any(key => Mathf.Abs(key.value) > 1.05f);
                if (warning)
                {
                    EscalateStatus(ref analysis.JointStatus, "FLAGGED");
                    analysis.JointFlags.Add("JOINT_RANGE_WARNING:" + pair.Key);
                }
            }

            analysis.QualityStatus = analysis.HardFailures.Count > 0
                ? "REJECTED"
                : analysis.Warnings.Count > 0
                    ? "FLAGGED"
                    : "VALID";
            analysis.QualityScore = ComputeQualityScore(analysis);
            analysis.QualityFlags = analysis.HardFailures
                .Concat(analysis.Warnings)
                .Concat(analysis.JointFlags)
                .Distinct()
                .ToArray();
            return analysis;
        }

        private static List<ProductionSample> SampleProductionRig(
            ProductionRig production,
            AnimationClip clip)
        {
            if (production.Character == null || production.Avatar == null ||
                !production.Avatar.isValid || !production.Avatar.isHuman)
                throw new InvalidOperationException("PRODUCTION_RIG_SAMPLE_UNAVAILABLE");

            var clone = UnityEngine.Object.Instantiate(production.Character);
            clone.name = "Phase17_Production_Rig_Sample";
            clone.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            var agent = clone.GetComponent<NavMeshAgent>();
            if (agent != null)
                agent.enabled = false;
            var controller = clone.GetComponent<LacrimosaController>();
            if (controller != null)
                controller.enabled = false;
            var animator = clone.GetComponent<Animator>();
            if (animator == null || animator.avatar == null ||
                !animator.avatar.isValid || !animator.avatar.isHuman)
            {
                UnityEngine.Object.DestroyImmediate(clone);
                throw new InvalidOperationException("PRODUCTION_RIG_SAMPLE_UNAVAILABLE");
            }
            animator.enabled = true;
            animator.applyRootMotion = false;

            var sampleTimes = GetSampleTimes(clip.length, clip.frameRate);
            var samples = new List<ProductionSample>(sampleTimes.Length);
            AnimationMode.StartAnimationMode();
            try
            {
                foreach (var time in sampleTimes)
                {
                    AnimationMode.SampleAnimationClip(clone, clip, time);
                    var sample = new ProductionSample { Time = time };
                    foreach (var bone in RequiredSampleBones)
                    {
                        var transform = animator.GetBoneTransform(bone);
                        if (transform == null)
                            throw new InvalidOperationException("PRODUCTION_SAMPLE_BONE_MISSING:" + bone);
                        sample.Bones[bone] = new BonePose
                        {
                            Position = transform.position,
                            Rotation = transform.rotation
                        };
                    }
                    samples.Add(sample);
                }
            }
            finally
            {
                AnimationMode.StopAnimationMode();
                UnityEngine.Object.DestroyImmediate(clone);
            }
            return samples;
        }

        private static float[] GetSampleTimes(float duration, float frameRate)
        {
            if (!IsFinite(duration) || !IsFinite(frameRate) || duration <= 0f || frameRate <= 0f)
                throw new InvalidOperationException("INVALID_SAMPLE_TIMING");

            var frameCount = Mathf.Max(2, Mathf.FloorToInt(duration * frameRate) + 1);
            var sampleCount = Mathf.Min(frameCount, MaxRigSamples);
            var times = new float[sampleCount];
            if (sampleCount == frameCount)
            {
                for (var index = 0; index < sampleCount; index++)
                    times[index] = Mathf.Min(duration, index / frameRate);
                times[sampleCount - 1] = duration;
            }
            else
            {
                for (var index = 0; index < sampleCount; index++)
                    times[index] = duration * index / (sampleCount - 1);
                times[sampleCount - 1] = duration;
            }
            return times;
        }

        private static void AnalyzeProductionSamples(
            ClipAnalysis analysis,
            List<ProductionSample> samples)
        {
            if (samples.Count < 2)
            {
                analysis.FiniteValues = false;
                analysis.RootStatus = "UNKNOWN";
                analysis.FootStatus = "REJECTED";
                analysis.FootConfidence = 0.9f;
                analysis.JointStatus = "REJECTED";
                analysis.HardFailures.Add("PRODUCTION_RIG_SAMPLES_MISSING");
                return;
            }

            var productionFinite = samples.All(sample =>
                RequiredSampleBones.All(bone =>
                    IsFinite(sample.Bones[bone].Position) &&
                    IsFinite(sample.Bones[bone].Rotation)));
            if (!productionFinite)
            {
                analysis.FiniteValues = false;
                analysis.RootStatus = "REJECTED";
                analysis.FootStatus = "REJECTED";
                analysis.FootConfidence = 0.9f;
                analysis.JointStatus = "REJECTED";
                analysis.HardFailures.Add("NON_FINITE_PRODUCTION_SAMPLES");
                return;
            }

            var first = samples[0];
            var last = samples[samples.Count - 1];
            analysis.RootHorizontalDistance = HorizontalDistance(
                first.Bones[HumanBodyBones.Hips].Position,
                last.Bones[HumanBodyBones.Hips].Position);
            analysis.RootVerticalDistance = samples
                .Max(sample => sample.Bones[HumanBodyBones.Hips].Position.y) -
                samples.Min(sample => sample.Bones[HumanBodyBones.Hips].Position.y);
            analysis.RootStatus = analysis.RootHorizontalDistance > 0.05f
                ? "NAVIGATION_EXEMPT"
                : "STABLE";

            var maxRootSpeed = 0f;
            for (var index = 1; index < samples.Count; index++)
            {
                var elapsed = Mathf.Max(0.0001f, samples[index].Time - samples[index - 1].Time);
                maxRootSpeed = Mathf.Max(
                    maxRootSpeed,
                    HorizontalDistance(
                        samples[index - 1].Bones[HumanBodyBones.Hips].Position,
                        samples[index].Bones[HumanBodyBones.Hips].Position) / elapsed);
            }
            if (analysis.RootVerticalDistance > 1.5f)
                analysis.HardFailures.Add("EXTREME_ROOT_TRAJECTORY");
            else if (analysis.RootVerticalDistance > 0.8f)
                analysis.Warnings.Add("ROOT_VERTICAL_MOVEMENT");
            if (maxRootSpeed > 10f)
                analysis.HardFailures.Add("EXCESSIVE_ROOT_ACCELERATION");

            AnalyzeFootContact(analysis, samples);
            AnalyzeJointConfigurations(analysis, samples);
        }

        private static void AnalyzeFootContact(
            ClipAnalysis analysis,
            List<ProductionSample> samples)
        {
            var minimumY = Mathf.Min(
                samples.Min(sample => sample.Bones[HumanBodyBones.LeftFoot].Position.y),
                samples.Min(sample => sample.Bones[HumanBodyBones.RightFoot].Position.y));
            analysis.FootPenetrationDepth = Mathf.Max(0f, -minimumY);
            if (analysis.FootPenetrationDepth > 0.05f)
            {
                AddFootFailure(analysis, "FOOT_PENETRATION");
            }
            else if (analysis.FootPenetrationDepth > 0.02f)
            {
                AddFootWarning(analysis, "FOOT_PENETRATION_WARNING");
            }

            if (minimumY > 0.30f)
                AddFootWarning(analysis, "FOOT_FLOATING");

            analysis.FootSlidingDistance = Mathf.Max(
                RootRelativePlantedPath(samples, HumanBodyBones.LeftFoot, minimumY),
                RootRelativePlantedPath(samples, HumanBodyBones.RightFoot, minimumY));
            if (analysis.FootSlidingDistance > 0.25f)
                AddFootWarning(analysis, "FOOT_SLIDING");

            if (analysis.FootStatus != "REJECTED")
            {
                EscalateStatus(
                    ref analysis.FootStatus,
                    analysis.FootFlags.Count > 0 ? "FLAGGED" : "VALID");
                analysis.FootConfidence = analysis.FootStatus == "VALID" ? 0.9f : 0.65f;
            }
        }

        private static float RootRelativePlantedPath(
            List<ProductionSample> samples,
            HumanBodyBones foot,
            float minimumFootY)
        {
            Vector3? previous = null;
            var path = 0f;
            foreach (var sample in samples)
            {
                var footPosition = sample.Bones[foot].Position;
                if (footPosition.y > minimumFootY + 0.08f)
                {
                    previous = null;
                    continue;
                }

                var hipsPosition = sample.Bones[HumanBodyBones.Hips].Position;
                var relative = new Vector3(
                    footPosition.x - hipsPosition.x,
                    0f,
                    footPosition.z - hipsPosition.z);
                if (previous.HasValue)
                    path += Vector3.Distance(previous.Value, relative);
                previous = relative;
            }
            return path;
        }

        private static void AnalyzeJointConfigurations(
            ClipAnalysis analysis,
            List<ProductionSample> samples)
        {
            foreach (var sample in samples)
            {
                var hips = sample.Bones[HumanBodyBones.Hips].Position;
                var head = sample.Bones[HumanBodyBones.Head].Position;
                if (head.y < hips.y - 0.10f || Vector3.Distance(head, hips) < 0.10f)
                {
                    AddJointFailure(analysis, "COLLAPSED_SPINE");
                    break;
                }
            }

            foreach (var pair in samples.SelectMany(sample => LimbSegmentPairs().Select(bones =>
                     {
                         var first = sample.Bones[bones[0]].Position;
                         var second = sample.Bones[bones[1]].Position;
                         return new { Bone = bones[1], Length = Vector3.Distance(first, second) };
                     })))
            {
                if (pair.Length < 0.02f || pair.Length > 5f)
                    AddJointFailure(analysis, "INVALID_LIMB_LENGTH:" + pair.Bone);
            }

            foreach (var sample in samples.SelectMany(item => RequiredSampleBones.Select(bone => item.Bones[bone].Rotation)))
            {
                var magnitude = Mathf.Sqrt(Quaternion.Dot(sample, sample));
                if (!IsFinite(sample.x) || !IsFinite(sample.y) ||
                    !IsFinite(sample.z) || !IsFinite(sample.w) ||
                    magnitude < 0.9f || magnitude > 1.1f)
                {
                    AddJointFailure(analysis, "INVALID_JOINT_ROTATION");
                    break;
                }
            }
        }

        private static IEnumerable<HumanBodyBones[]> LimbSegmentPairs()
        {
            yield return new[] { HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm };
            yield return new[] { HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand };
            yield return new[] { HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm };
            yield return new[] { HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand };
            yield return new[] { HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg };
            yield return new[] { HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot };
            yield return new[] { HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg };
            yield return new[] { HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot };
        }

        private static void AddFootFailure(ClipAnalysis analysis, string failure)
        {
            analysis.HardFailures.Add(failure);
            analysis.FootFlags.Add(failure);
            EscalateStatus(ref analysis.FootStatus, "REJECTED");
            analysis.FootConfidence = 0.9f;
        }

        private static void AddFootWarning(ClipAnalysis analysis, string warning)
        {
            analysis.Warnings.Add(warning);
            analysis.FootFlags.Add(warning);
            EscalateStatus(ref analysis.FootStatus, "FLAGGED");
        }

        private static void AddJointFailure(ClipAnalysis analysis, string failure)
        {
            analysis.HardFailures.Add(failure);
            analysis.JointFlags.Add(failure);
            EscalateStatus(ref analysis.JointStatus, "REJECTED");
        }

        private static void EscalateStatus(ref string current, string candidate)
        {
            if (StatusRank(candidate) > StatusRank(current))
                current = candidate;
        }

        private static int StatusRank(string status)
        {
            switch (status)
            {
                case "REJECTED":
                    return 2;
                case "FLAGGED":
                    return 1;
                case "VALID":
                    return 0;
                default:
                    return -1;
            }
        }

        private static float HorizontalDistance(Vector3 first, Vector3 second)
        {
            return Mathf.Sqrt(
                Mathf.Pow(second.x - first.x, 2f) +
                Mathf.Pow(second.z - first.z, 2f));
        }

        private static float ComputeQualityScore(ClipAnalysis analysis)
        {
            if (analysis.HardFailures.Count > 0)
                return 0f;
            var score = 1f;
            score -= Mathf.Min(0.25f, Mathf.Min(analysis.Warnings.Count, 6) * 0.04f);
            score += -Mathf.Min(0.20f, analysis.MissingSampleRatio * 2f);
            if (analysis.RootStatus != "STABLE" && analysis.RootStatus != "NAVIGATION_EXEMPT")
                score -= 0.08f;
            if (analysis.FootStatus != "VALID")
                score -= 0.08f;
            if (analysis.JointStatus != "VALID")
                score -= 0.12f;
            return Mathf.Clamp01(Mathf.Round(score * 10000f) / 10000f);
        }

        private static void ApplyAnalysis(Phase17ProcessingReport report, ClipAnalysis analysis)
        {
            report.durationSeconds = analysis.DurationSeconds;
            report.fps = analysis.Fps;
            report.frameCount = analysis.FrameCount;
            report.curveCount = analysis.CurveCount;
            report.keyCount = analysis.KeyCount;
            report.qualityStatus = analysis.QualityStatus;
            report.qualityScore = analysis.QualityScore;
            report.qualityFlags = analysis.QualityFlags;
            report.hardFailures = analysis.HardFailures.ToArray();
            report.warnings = analysis.Warnings.ToArray();
            report.finiteValues = analysis.FiniteValues;
            report.quaternionValid = analysis.QuaternionValid;
            report.quaternionCurvesFound = analysis.QuaternionCurvesFound;
            report.keyTimesMonotonic = analysis.KeyTimesMonotonic;
            report.missingSampleCount = analysis.MissingSampleCount;
            report.missingSampleRatio = analysis.MissingSampleRatio;
            report.maxCurveJump = analysis.MaxCurveJump;
            report.rootStatus = analysis.RootStatus;
            report.rootHorizontalDistance = analysis.RootHorizontalDistance;
            report.rootVerticalDistance = analysis.RootVerticalDistance;
            report.footStatus = analysis.FootStatus;
            report.footConfidence = analysis.FootConfidence;
            report.footSlidingDistance = analysis.FootSlidingDistance;
            report.footPenetrationDepth = analysis.FootPenetrationDepth;
            report.footFlags = analysis.FootFlags.ToArray();
            report.jointStatus = analysis.JointStatus;
            report.jointFlags = analysis.JointFlags.ToArray();
        }

        private static void EnsureFolder(string parent, string name)
        {
            var path = parent + "/" + name;
            if (AssetDatabase.IsValidFolder(path))
                return;
            if (string.IsNullOrEmpty(AssetDatabase.CreateFolder(parent, name)))
                throw new InvalidOperationException("Failed to create asset folder: " + path);
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

        private static bool IsFinite(Vector3 value)
        {
            return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
        }

        private static bool IsFinite(Quaternion value)
        {
            return IsFinite(value.x) && IsFinite(value.y) &&
                   IsFinite(value.z) && IsFinite(value.w);
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

        private static void WriteReport(Phase17ProcessingReport report)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase17MotionProcessor] " + JsonUtility.ToJson(report));
        }

        private sealed class ProductionRig
        {
            public GameObject Character;
            public Avatar Avatar;
            public Animator Animator;
            public NavMeshAgent Agent;
        }

        private sealed class ClipAnalysis
        {
            public float DurationSeconds;
            public float Fps;
            public int FrameCount;
            public int CurveCount;
            public int KeyCount;
            public string QualityStatus;
            public float QualityScore;
            public string[] QualityFlags = Array.Empty<string>();
            public readonly List<string> HardFailures = new List<string>();
            public readonly List<string> Warnings = new List<string>();
            public readonly List<string> JointFlags = new List<string>();
            public readonly List<string> FootFlags = new List<string>();
            public bool FiniteValues;
            public bool QuaternionValid;
            public bool QuaternionCurvesFound;
            public bool KeyTimesMonotonic;
            public int MissingSampleCount;
            public float MissingSampleRatio;
            public float MaxCurveJump;
            public string RootStatus;
            public float RootHorizontalDistance;
            public float RootVerticalDistance;
            public string FootStatus;
            public float FootConfidence;
            public float FootSlidingDistance;
            public float FootPenetrationDepth;
            public string JointStatus;
        }

        private sealed class ProductionSample
        {
            public float Time;
            public readonly Dictionary<HumanBodyBones, BonePose> Bones =
                new Dictionary<HumanBodyBones, BonePose>();
        }

        private sealed class BonePose
        {
            public Vector3 Position;
            public Quaternion Rotation;
        }

        [Serializable]
        private sealed class AutoRunRequest
        {
            public string sourceMotionId;
            public int sourceVersion;
            public string sourceReference;
            public string processedMotionId;
            public int processingVersion;
            public bool runPlayback;
        }

        [Serializable]
        private sealed class Phase17ProcessingReport
        {
            public bool success;
            public string error;
            public string sourceMotionId;
            public int sourceVersion;
            public string processedMotionId;
            public int processingVersion;
            public string state;
            public string sourceAnimationReference;
            public string animationReference;
            public string processedClipPath;
            public string libraryPath;
            public float durationSeconds;
            public float fps;
            public int frameCount;
            public int curveCount;
            public int keyCount;
            public int boneCount;
            public string rootBoneName;
            public string qualityStatus;
            public float qualityScore;
            public string[] qualityFlags;
            public string[] hardFailures;
            public string[] warnings;
            public bool finiteValues;
            public bool quaternionValid;
            public bool quaternionCurvesFound;
            public bool keyTimesMonotonic;
            public int missingSampleCount;
            public float missingSampleRatio;
            public float maxCurveJump;
            public string rootStatus;
            public float rootHorizontalDistance;
            public float rootVerticalDistance;
            public bool rootAuthoritativeNavigation;
            public bool applyRootMotion;
            public string footStatus;
            public float footConfidence;
            public float footSlidingDistance;
            public float footPenetrationDepth;
            public string[] footFlags;
            public string jointStatus;
            public string[] jointFlags;
            public bool sourceAvatarValid;
            public bool sourceAvatarHuman;
            public bool targetAvatarValid;
            public bool targetAvatarHuman;
            public string[] sourceHumanoidMappings;
            public string[] targetHumanoidMappings;
            public SkeletonMappingReport[] mappingEntries;
            public string[] missingMappings;
            public string[] ambiguousMappings;
            public string skeletonStatus;
            public string retargetStatus;
            public string cleanupStatus;
            public float cleanupMagnitude;
            public int cleanupKeyCount;
            public string processingTimestamp;
        }

        [Serializable]
        private sealed class SkeletonMappingReport
        {
            public string humanBone;
            public string sourceBone;
            public string targetBone;
        }
    }
}
