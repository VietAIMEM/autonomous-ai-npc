using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RoomAgent.Brain;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace RoomAgent.Editor
{
    [InitializeOnLoad]
    public static class Phase15ProductionIntegrationValidator
    {
        private const string MotionId = "motius_1ee2c1bf116d4111";
        private const string FbxPath = "Assets/animation/phase15_motius/lacrimosa_production_motius_animated.fbx";
        private const string OriginalFbxPath = "Assets/model/lacrimosa-animated-neverness-to-everness/source/lacrimosa final.fbx";
        private const string SourceReference = "Assets/animation/phase15_motius/lacrimosa_production_motius_animated.fbx";
        private const string ClipName = "MotiusPhase15Wave";
        private const string ClipPath = "Assets/Resources/GeneratedMotions/Phase15/motius_1ee2c1bf116d4111.anim";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string ReportPath = "Temp/Phase15ProductionIntegration.json";
        private const string ActiveKey = "Phase15ProductionIntegrationValidator.Active";

        private static readonly float[] SampleTimes = { 0f, 0.5f, 1.0f };

        private enum ValidationStage
        {
            Idle,
            Starting,
            Playing,
            Cancellation,
            Finished
        }

        private static ValidationStage stage = ValidationStage.Idle;
        private static Phase15ProductionReport report = new Phase15ProductionReport();
        private static BrainActionBridge bridge;
        private static LacrimosaController character;
        private static LacrimosaBodyAnimator bodyAnimator;
        private static Animator animator;
        private static RuntimeAnimatorController originalController;
        private static AnimationClip generatedClip;
        private static AnimationClip originalDanceClip;
        private static readonly List<PlaybackSample> samples = new List<PlaybackSample>();
        private static bool callbackDone;
        private static bool callbackSuccess;
        private static bool stopRequested;
        private static float stageStartedAt;

        static Phase15ProductionIntegrationValidator()
        {
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        [MenuItem("Tools/Phase15/Validate Production Motion Integration")]
        public static void Run()
        {
            try
            {
                if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                    throw new InvalidOperationException("Unity is already in Play Mode.");

                SetupProductionAssets();
                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                if (!scene.isLoaded)
                    throw new InvalidOperationException("Production SampleScene did not load.");

                var loop = UnityEngine.Object.FindFirstObjectByType<BrainLoopController>();
                if (loop != null)
                {
                    loop.StopLoop();
                    loop.enabled = false;
                }

                SessionState.SetBool(ActiveKey, true);
                stage = ValidationStage.Idle;
                EditorApplication.isPlaying = true;
            }
            catch (Exception exception)
            {
                Fail(exception.ToString());
            }
        }

        private static void SetupProductionAssets()
        {
            var originalAvatar = AssetDatabase.LoadAssetAtPath<Avatar>(OriginalFbxPath);
            if (originalAvatar == null || !originalAvatar.isValid || !originalAvatar.isHuman)
                throw new InvalidOperationException("Original Lacrimosa FBX did not expose a valid Humanoid Avatar.");

            var importer = AssetImporter.GetAtPath(FbxPath) as ModelImporter;
            if (importer == null)
                throw new InvalidOperationException("Generated Motius FBX importer is missing.");

            importer.animationType = ModelImporterAnimationType.Human;
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.sourceAvatar = null;

            var defaultClips = importer.defaultClipAnimations;
            if (defaultClips == null || defaultClips.Length == 0)
                throw new InvalidOperationException("Motius FBX has no importable animation take.");
            importer.clipAnimations = new[]
            {
                new ModelImporterClipAnimation
                {
                    name = ClipName,
                    takeName = defaultClips[0].takeName,
                    firstFrame = 0,
                    lastFrame = 20,
                    loop = false,
                    loopTime = false,
                    loopPose = false
                }
            };
            importer.SaveAndReimport();

            var sourceModel = AssetDatabase.LoadAssetAtPath<GameObject>(FbxPath);
            var sourceClip = AssetDatabase.LoadAllAssetsAtPath(FbxPath)
                .OfType<AnimationClip>()
                .FirstOrDefault(clip => clip.name == ClipName);
            var sourceInstance = sourceModel != null ? UnityEngine.Object.Instantiate(sourceModel) : null;
            try
            {
                var sourceAnimator = sourceInstance != null
                    ? sourceInstance.GetComponentInChildren<Animator>(true)
                    : null;
                if (sourceAnimator == null || sourceAnimator.avatar == null ||
                    !sourceAnimator.avatar.isValid || !sourceAnimator.avatar.isHuman)
                    throw new InvalidOperationException(
                        "Generated Motius FBX did not produce a valid Humanoid Avatar copied from the original Lacrimosa Avatar.");
            }
            finally
            {
                if (sourceInstance != null)
                    UnityEngine.Object.DestroyImmediate(sourceInstance);
            }
            if (sourceClip == null)
                throw new InvalidOperationException("Named Motius clip was not imported.");
            if (Math.Abs(sourceClip.frameRate - 20f) > 0.01f ||
                (Math.Abs(sourceClip.length - 0.95f) > 0.01f && Math.Abs(sourceClip.length - 1f) > 0.01f))
                throw new InvalidOperationException(
                    $"Unexpected Motius clip timing: fps={sourceClip.frameRate}, length={sourceClip.length}");

            var sourceBindings = AnimationUtility.GetCurveBindings(sourceClip);
            if (sourceBindings.Length == 0)
                throw new InvalidOperationException("Named Motius clip has no curves.");

            CreateStableClip(sourceClip);
            ConfigureLibrary();
            AssetDatabase.SaveAssets();
        }

        private static void CreateStableClip(AnimationClip sourceClip)
        {
            EnsureFolder("Assets", "Resources");
            EnsureFolder("Assets/Resources", "GeneratedMotions");
            EnsureFolder("Assets/Resources/GeneratedMotions", "Phase15");

            if (AssetDatabase.LoadAssetAtPath<AnimationClip>(ClipPath) != null)
                AssetDatabase.DeleteAsset(ClipPath);

            var target = new AnimationClip();
            target.name = MotionId;
            target.frameRate = sourceClip.frameRate;

            var stableSettings = AnimationUtility.GetAnimationClipSettings(sourceClip);
            stableSettings.startTime = 0f;
            stableSettings.stopTime = 1f;

            AssetDatabase.CreateAsset(target, ClipPath);

            var bindings = AnimationUtility.GetCurveBindings(sourceClip);
            var firstCurveValue = 0f;
            var lastCurveValue = 0f;
            foreach (var binding in bindings)
            {
                var curve = AnimationUtility.GetEditorCurve(sourceClip, binding);
                if (curve == null)
                    continue;
                if (curve.keys.Length > 0)
                {
                    firstCurveValue = curve.keys.First().value;
                    lastCurveValue = curve.keys.Last().value;
                }

                if (curve.keys.Length > 0 && curve.keys.Last().time < 1f)
                    curve.AddKey(1f, curve.keys.Last().value);

                AnimationUtility.SetEditorCurve(target, binding, curve);
            }

            var objectBindings = AnimationUtility.GetObjectReferenceCurveBindings(sourceClip);
            foreach (var binding in objectBindings)
                AnimationUtility.SetObjectReferenceCurve(
                    target,
                    binding,
                    AnimationUtility.GetObjectReferenceCurve(sourceClip, binding));

            AnimationUtility.SetAnimationClipSettings(
                target,
                stableSettings);
            if (Math.Abs(target.length - 1f) > 0.01f)
                throw new InvalidOperationException(
                    $"Stable Motius clip did not normalize to 1 second: length={target.length}");
            EditorUtility.SetDirty(target);
        }

        private static void ConfigureLibrary()
        {
            var generated = AssetDatabase.LoadAssetAtPath<AnimationClip>(ClipPath);
            var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
            if (library == null)
            {
                library = ScriptableObject.CreateInstance<GeneratedMotionLibrary>();
                AssetDatabase.CreateAsset(library, LibraryPath);
            }

            library.Upsert(new GeneratedMotionLibrary.Motion
            {
                SourceReference = SourceReference,
                GeneratedClip = generated
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

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (!SessionState.GetBool(ActiveKey, false))
                return;
            if (state == PlayModeStateChange.EnteredPlayMode)
                StartPlayValidation();
            else if (state == PlayModeStateChange.ExitingPlayMode && stage != ValidationStage.Finished)
                Fail("Unity exited Play Mode before production validation completed.");
        }

        private static void StartPlayValidation()
        {
            if (stage != ValidationStage.Idle || !SessionState.GetBool(ActiveKey, false))
                return;
            stage = ValidationStage.Starting;
            try
            {
                report = new Phase15ProductionReport
                {
                    motionId = MotionId,
                    sourceReference = SourceReference,
                    fbxPath = FbxPath,
                    sourceClipName = ClipName,
                    stableClipPath = ClipPath,
                    libraryPath = LibraryPath
                };

                var loop = UnityEngine.Object.FindFirstObjectByType<BrainLoopController>();
                if (loop != null)
                {
                    loop.StopLoop();
                    loop.enabled = false;
                }

                bridge = UnityEngine.Object.FindFirstObjectByType<BrainActionBridge>();
                character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
                if (bridge == null || character == null)
                    throw new InvalidOperationException("Production BrainActionBridge or LacrimosaController is missing.");

                bodyAnimator = character.GetComponent<LacrimosaBodyAnimator>();
                if (bodyAnimator == null)
                    throw new InvalidOperationException("LacrimosaBodyAnimator is missing in the production scene.");
                animator = bodyAnimator.Animator;
                if (animator == null || animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                    throw new InvalidOperationException("Production Lacrimosa Animator does not have a valid Humanoid Avatar.");

                generatedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(ClipPath);
                var library = Resources.Load<GeneratedMotionLibrary>("GeneratedMotions/GeneratedMotionLibrary");
                if (generatedClip == null || library == null || library.ResolveClip(SourceReference) != generatedClip)
                    throw new InvalidOperationException("GeneratedMotionLibrary did not resolve the stable generated clip.");

                originalController = animator.runtimeAnimatorController;
                originalDanceClip = FindControllerClip(originalController, "Dance");
                if (originalController == null || originalDanceClip == null)
                    throw new InvalidOperationException("Production Lacrimosa controller has no built-in Dance clip to override.");
                if (generatedClip == originalDanceClip)
                    throw new InvalidOperationException("Generated clip is incorrectly the built-in Dance clip.");

                report.avatarValid = animator.avatar.isValid;
                report.avatarHuman = animator.avatar.isHuman;
                report.originalController = originalController.name;
                report.originalDanceClip = originalDanceClip.name;
                report.generatedClip = generatedClip.name;
                report.clipFrameRate = generatedClip.frameRate;
                report.clipLength = generatedClip.length;
                report.curveBindingCount = AnimationUtility.GetCurveBindings(generatedClip).Length;
                report.libraryResolved = true;

                samples.Clear();
                callbackDone = false;
                callbackSuccess = false;
                stageStartedAt = Time.realtimeSinceStartup;
                stage = ValidationStage.Playing;

                var accepted = bridge.Execute(
                    "DANCE",
                    SourceReference,
                    null,
                    null,
                    success =>
                    {
                        callbackSuccess = success;
                        callbackDone = true;
                    });
                if (!accepted)
                    throw new InvalidOperationException("BrainActionBridge rejected the generated DANCE action.");
            }
            catch (Exception exception)
            {
                Fail(exception.ToString());
            }
        }

        private static void OnEditorUpdate()
        {
            if (!SessionState.GetBool(ActiveKey, false) || !EditorApplication.isPlaying)
                return;
            if (stage == ValidationStage.Idle)
            {
                StartPlayValidation();
                return;
            }

            var elapsed = Time.realtimeSinceStartup - stageStartedAt;
            if (stage == ValidationStage.Playing)
            {
                if (samples.Count < SampleTimes.Length && elapsed >= SampleTimes[samples.Count])
                    CaptureSample(SampleTimes[samples.Count]);

                if (callbackDone)
                {
                    if (samples.Count < SampleTimes.Length)
                        CaptureSample(SampleTimes[samples.Count]);
                    FinishPlaybackStage();
                }
                else if (elapsed > 4f)
                {
                    Fail("Generated production playback did not complete within 4 seconds.");
                }
            }
            else if (stage == ValidationStage.Cancellation)
            {
                if (!stopRequested && elapsed >= 0.2f)
                {
                    character.Stop();
                    stopRequested = true;
                }

                if (callbackDone)
                    FinishCancellationStage();
                else if (elapsed > 3f)
                    Fail("Generated production cancellation did not complete within 3 seconds.");
            }
        }

        private static void CaptureSample(float sampleTime)
        {
            var hips = animator.GetBoneTransform(HumanBodyBones.Hips);
            var head = animator.GetBoneTransform(HumanBodyBones.Head);
            var leftHand = animator.GetBoneTransform(HumanBodyBones.LeftHand);
            var rightHand = animator.GetBoneTransform(HumanBodyBones.RightHand);
            if (hips == null || head == null || leftHand == null || rightHand == null)
                throw new InvalidOperationException("Production Humanoid Avatar is missing required observed bones.");

            var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
            var sample = new PlaybackSample
            {
                time = sampleTime,
                elapsed = Time.realtimeSinceStartup - stageStartedAt,
                generatedClipActive = bodyAnimator.CurrentGeneratedClip == generatedClip,
                overrideControllerActive = animator.runtimeAnimatorController is AnimatorOverrideController,
                danceStateActive = stateInfo.IsName("Dance"),
                bones = new[]
                {
                    new BoneSample { bone = "Hips", position = hips.position },
                    new BoneSample { bone = "Head", position = head.position },
                    new BoneSample { bone = "LeftHand", position = leftHand.position },
                    new BoneSample { bone = "RightHand", position = rightHand.position }
                }
            };
            samples.Add(sample);
        }

        private static void FinishPlaybackStage()
        {
            report.playbackSuccess = callbackSuccess;
            report.generatedClipActiveAtAllSamples = samples.All(sample => sample.generatedClipActive);
            report.overrideControllerActiveAtAllSamples = samples.All(sample => sample.overrideControllerActive);
            report.danceStateActiveAtAllSamples = samples.All(sample => sample.danceStateActive);
            report.finite = samples
                .SelectMany(sample => sample.bones)
                .All(sample => IsFinite(sample.position));
            report.maxBoneDisplacement = MaxBoneDisplacement(samples);
            report.nonTrivial = report.maxBoneDisplacement > 0.01f;
            report.playbackSuccess = report.playbackSuccess &&
                report.generatedClipActiveAtAllSamples &&
                report.overrideControllerActiveAtAllSamples &&
                report.danceStateActiveAtAllSamples &&
                report.finite &&
                report.nonTrivial;

            if (!report.playbackSuccess)
            {
                Fail("Generated production playback failed clip, state, finite, or displacement validation.");
                return;
            }

            callbackDone = false;
            callbackSuccess = true;
            stopRequested = false;
            stageStartedAt = Time.realtimeSinceStartup;
            stage = ValidationStage.Cancellation;
            var accepted = bridge.Execute(
                "DANCE",
                SourceReference,
                null,
                null,
                success =>
                {
                    callbackSuccess = success;
                    callbackDone = true;
                });
            if (!accepted)
                Fail("BrainActionBridge rejected the generated cancellation action.");
        }

        private static void FinishCancellationStage()
        {
            report.cancellationCallbackFailed = callbackDone && !callbackSuccess;
            report.generatedClipCleared = bodyAnimator.CurrentGeneratedClip == null;
            report.originalControllerRestored = ReferenceEquals(animator.runtimeAnimatorController, originalController);
            report.cancellationSuccess = report.cancellationCallbackFailed &&
                report.generatedClipCleared &&
                report.originalControllerRestored;

            if (!report.cancellationSuccess)
            {
                Fail("Generated production cancellation did not fail closed and restore the original controller.");
                return;
            }

            Finish();
        }

        private static AnimationClip FindControllerClip(RuntimeAnimatorController controller, string clipName)
        {
            if (controller == null)
                return null;
            var probe = new AnimatorOverrideController(controller);
            var overrides = new List<KeyValuePair<AnimationClip, AnimationClip>>();
            probe.GetOverrides(overrides);
            return overrides
                .Select(pair => pair.Key)
                .FirstOrDefault(clip => string.Equals(clip.name, clipName, StringComparison.OrdinalIgnoreCase));
        }

        private static bool IsFinite(Vector3 value)
        {
            return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
                   !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
                   !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        }

        private static float MaxBoneDisplacement(List<PlaybackSample> playbackSamples)
        {
            var maximum = 0f;
            for (var boneIndex = 0; boneIndex < 4; boneIndex++)
            {
                for (var first = 0; first < playbackSamples.Count; first++)
                {
                    for (var second = first + 1; second < playbackSamples.Count; second++)
                    {
                        maximum = Mathf.Max(
                            maximum,
                            Vector3.Distance(
                                playbackSamples[first].bones[boneIndex].position,
                                playbackSamples[second].bones[boneIndex].position));
                    }
                }
            }
            return maximum;
        }

        private static void Finish()
        {
            stage = ValidationStage.Finished;
            report.success = report.playbackSuccess && report.cancellationSuccess;
            WriteReport();
            EditorApplication.Exit(report.success ? 0 : 1);
        }

        private static void Fail(string error)
        {
            stage = ValidationStage.Finished;
            report.success = false;
            report.error = error;
            WriteReport();
            Debug.LogError("[Phase15ProductionIntegrationValidator] " + error);
            EditorApplication.Exit(1);
        }

        private static void WriteReport()
        {
            report.samples = samples.ToArray();
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase15ProductionIntegrationValidator] " + JsonUtility.ToJson(report));
        }

        [Serializable]
        private sealed class Phase15ProductionReport
        {
            public bool success;
            public string error;
            public string motionId;
            public string sourceReference;
            public string fbxPath;
            public string sourceClipName;
            public string stableClipPath;
            public string libraryPath;
            public bool avatarValid;
            public bool avatarHuman;
            public string originalController;
            public string originalDanceClip;
            public string generatedClip;
            public float clipFrameRate;
            public float clipLength;
            public int curveBindingCount;
            public bool libraryResolved;
            public bool playbackSuccess;
            public bool generatedClipActiveAtAllSamples;
            public bool overrideControllerActiveAtAllSamples;
            public bool danceStateActiveAtAllSamples;
            public bool finite;
            public bool nonTrivial;
            public float maxBoneDisplacement;
            public bool cancellationCallbackFailed;
            public bool generatedClipCleared;
            public bool originalControllerRestored;
            public bool cancellationSuccess;
            public PlaybackSample[] samples;
        }

        [Serializable]
        private sealed class PlaybackSample
        {
            public float time;
            public float elapsed;
            public bool generatedClipActive;
            public bool overrideControllerActive;
            public bool danceStateActive;
            public BoneSample[] bones;
        }

        [Serializable]
        private sealed class BoneSample
        {
            public string bone;
            public Vector3 position;
        }
    }
}
