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

namespace RoomAgent.Editor
{
    [InitializeOnLoad]
    public static class Phase16ProductionPlaybackValidator
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string ReportPath = "Temp/Phase16ProductionPlayback.json";
        private const string ActiveKey = "Phase16ProductionPlaybackValidator.Active";

        private enum ValidationStage
        {
            Idle,
            Playing,
            Cancellation,
            Finished
        }

        private static ValidationStage stage = ValidationStage.Idle;
        private static Phase16PlaybackReport report = new Phase16PlaybackReport();
        private static BrainActionBridge bridge;
        private static LacrimosaController character;
        private static LacrimosaBodyAnimator bodyAnimator;
        private static Animator animator;
        private static RuntimeAnimatorController originalController;
        private static AnimationClip originalDanceClip;
        private static AnimationClip importedClip;
        private static readonly List<PlaybackSample> samples = new List<PlaybackSample>();
        private static float[] sampleTimes = { 0.05f, 0.35f, 0.7f };
        private static bool callbackDone;
        private static bool callbackSuccess;
        private static bool stopRequested;
        private static float stageStartedAt;
        private static float playbackTimeout;

        static Phase16ProductionPlaybackValidator()
        {
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        [MenuItem("Tools/Phase16/Validate Real Walking Playback")]
        public static void ValidateRealWalking()
        {
            var reportPath = Path.Combine(
                Directory.GetCurrentDirectory(),
                "Temp/Phase16MotionImport.json");
            if (!File.Exists(reportPath))
                throw new InvalidOperationException("Phase 16 import report is missing.");
            var import = JsonUtility.FromJson<Phase16ImportReport>(File.ReadAllText(reportPath));
            Run(import.motionId, import.animationReference, import.stableClipPath);
        }

        public static void Run()
        {
            Run(
                GetArgument("-phase16MotionId"),
                GetArgument("-phase16SourceReference"),
                GetArgument("-phase16StableClipPath"));
        }

        public static void Run(
            string motionId,
            string sourceReference,
            string stableClipPath)
        {
            try
            {
                if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                    throw new InvalidOperationException("Unity is already in Play Mode.");

                report = new Phase16PlaybackReport
                {
                    motionId = motionId ?? "",
                    sourceReference = sourceReference ?? "",
                    stableClipPath = stableClipPath ?? ""
                };
                if (string.IsNullOrWhiteSpace(report.motionId) ||
                    string.IsNullOrWhiteSpace(report.sourceReference) ||
                    string.IsNullOrWhiteSpace(report.stableClipPath))
                    throw new InvalidOperationException("INVALID_PLAYBACK_ARGUMENTS");

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
            stage = ValidationStage.Playing;
            try
            {
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

                importedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(report.stableClipPath);
                var library = Resources.Load<GeneratedMotionLibrary>("GeneratedMotions/GeneratedMotionLibrary");
                if (importedClip == null || library == null || library.ResolveClip(report.sourceReference) != importedClip)
                    throw new InvalidOperationException("GeneratedMotionLibrary did not resolve the imported stable clip.");

                originalController = animator.runtimeAnimatorController;
                originalDanceClip = FindControllerClip(originalController, "Dance");
                if (originalController == null || originalDanceClip == null)
                    throw new InvalidOperationException("Production controller has no built-in Dance clip to override.");
                if (importedClip == originalDanceClip)
                    throw new InvalidOperationException("Imported clip is incorrectly the built-in Dance clip.");

                report.avatarValid = animator.avatar.isValid;
                report.avatarHuman = animator.avatar.isHuman;
                report.originalController = originalController.name;
                report.originalDanceClip = originalDanceClip.name;
                report.importedClip = importedClip.name;
                report.clipFrameRate = importedClip.frameRate;
                report.clipLength = importedClip.length;
                report.curveBindingCount = AnimationUtility.GetCurveBindings(importedClip).Length;
                report.libraryResolved = true;

                samples.Clear();
                callbackDone = false;
                callbackSuccess = false;
                stageStartedAt = Time.realtimeSinceStartup;
                playbackTimeout = Mathf.Max(2f, importedClip.length + 2f);
                sampleTimes = new[]
                {
                    0.05f,
                    Mathf.Max(0.1f, importedClip.length * 0.35f),
                    Mathf.Max(0.15f, importedClip.length * 0.7f)
                };
                stage = ValidationStage.Playing;

                var accepted = bridge.Execute(
                    "DANCE",
                    report.sourceReference,
                    null,
                    null,
                    success =>
                    {
                        callbackSuccess = success;
                        callbackDone = true;
                    });
                if (!accepted)
                    throw new InvalidOperationException("BrainActionBridge rejected the imported DANCE action.");
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
                if (samples.Count < sampleTimes.Length && elapsed >= sampleTimes[samples.Count])
                    CaptureSample(sampleTimes[samples.Count]);

                if (callbackDone)
                {
                    if (samples.Count < sampleTimes.Length)
                        CaptureSample(sampleTimes[samples.Count]);
                    FinishPlaybackStage();
                }
                else if (elapsed > playbackTimeout)
                {
                    Fail("Imported production playback did not complete within the expected clip duration.");
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
                    Fail("Imported production cancellation did not complete within 3 seconds.");
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
            samples.Add(new PlaybackSample
            {
                time = sampleTime,
                elapsed = Time.realtimeSinceStartup - stageStartedAt,
                importedClipActive = bodyAnimator.CurrentGeneratedClip == importedClip,
                overrideControllerActive = animator.runtimeAnimatorController is AnimatorOverrideController,
                danceStateActive = stateInfo.IsName("Dance"),
                bones = new[]
                {
                    new BoneSample { bone = "Hips", position = hips.position },
                    new BoneSample { bone = "Head", position = head.position },
                    new BoneSample { bone = "LeftHand", position = leftHand.position },
                    new BoneSample { bone = "RightHand", position = rightHand.position }
                }
            });
        }

        private static void FinishPlaybackStage()
        {
            report.playbackSuccess = callbackSuccess;
            report.importedClipActiveAtAllSamples = samples.All(sample => sample.importedClipActive);
            report.overrideControllerActiveAtAllSamples = samples.All(sample => sample.overrideControllerActive);
            report.danceStateActiveAtAllSamples = samples.All(sample => sample.danceStateActive);
            report.finite = samples
                .SelectMany(sample => sample.bones)
                .All(sample => IsFinite(sample.position));
            report.maxBoneDisplacement = MaxBoneDisplacement(samples);
            report.nonTrivial = report.maxBoneDisplacement > 0.01f;
            report.playbackSuccess = report.playbackSuccess &&
                report.importedClipActiveAtAllSamples &&
                report.overrideControllerActiveAtAllSamples &&
                report.danceStateActiveAtAllSamples &&
                report.finite &&
                report.nonTrivial;

            if (!report.playbackSuccess)
            {
                Fail("Imported production playback failed clip, state, finite, or displacement validation.");
                return;
            }

            callbackDone = false;
            callbackSuccess = true;
            stopRequested = false;
            stageStartedAt = Time.realtimeSinceStartup;
            stage = ValidationStage.Cancellation;
            var accepted = bridge.Execute(
                "DANCE",
                report.sourceReference,
                null,
                null,
                success =>
                {
                    callbackSuccess = success;
                    callbackDone = true;
                });
            if (!accepted)
                Fail("BrainActionBridge rejected the imported cancellation action.");
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
                Fail("Imported production cancellation did not fail closed and restore the original controller.");
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
            if (Application.isBatchMode)
                EditorApplication.Exit(report.success ? 0 : 1);
            else
            {
                SessionState.SetBool(ActiveKey, false);
                EditorApplication.isPlaying = false;
            }
        }

        private static void Fail(string error)
        {
            stage = ValidationStage.Finished;
            report.success = false;
            report.error = error;
            WriteReport();
            Debug.LogError("[Phase16ProductionPlaybackValidator] " + error);
            if (Application.isBatchMode)
                EditorApplication.Exit(1);
            else
            {
                SessionState.SetBool(ActiveKey, false);
                EditorApplication.isPlaying = false;
            }
        }

        private static void WriteReport()
        {
            report.samples = samples.ToArray();
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase16ProductionPlaybackValidator] " + JsonUtility.ToJson(report));
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

        [Serializable]
        private sealed class Phase16PlaybackReport
        {
            public bool success;
            public string error;
            public string motionId;
            public string sourceReference;
            public string stableClipPath;
            public bool avatarValid;
            public bool avatarHuman;
            public string originalController;
            public string originalDanceClip;
            public string importedClip;
            public float clipFrameRate;
            public float clipLength;
            public int curveBindingCount;
            public bool libraryResolved;
            public bool playbackSuccess;
            public bool importedClipActiveAtAllSamples;
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
        private sealed class Phase16ImportReport
        {
            public string motionId;
            public string animationReference;
            public string stableClipPath;
        }

        [Serializable]
        private sealed class PlaybackSample
        {
            public float time;
            public float elapsed;
            public bool importedClipActive;
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
