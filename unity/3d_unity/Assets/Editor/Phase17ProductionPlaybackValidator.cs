using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RoomAgent.Brain;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RoomAgent.Editor
{
    [InitializeOnLoad]
    public static class Phase17ProductionPlaybackValidator
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string ReportPath = "Temp/Phase17ProductionPlayback.json";
        private const string ActiveKey = "Phase17ProductionPlaybackValidator.Active";

        private enum ValidationStage
        {
            Idle,
            PreFlight,
            Playing,
            IdleAfter,
            Navigation,
            Finished
        }

        private static ValidationStage stage = ValidationStage.Idle;
        private static Phase17PlaybackReport report = new Phase17PlaybackReport();
        private static BrainActionBridge bridge;
        private static LacrimosaController character;
        private static LacrimosaBodyAnimator bodyAnimator;
        private static Animator animator;
        private static RuntimeAnimatorController originalController;
        private static AnimationClip originalDanceClip;
        private static AnimationClip processedClip;
        private static readonly List<PlaybackSample> samples = new List<PlaybackSample>();
        private static float[] sampleTimes = { 0.05f, 0.35f, 0.7f };
        private static bool callbackDone;
        private static bool callbackSuccess;
        private static float stageStartedAt;
        private static float idleAfterTimeout;
        private static Vector3 navigationStartPosition;

        static Phase17ProductionPlaybackValidator()
        {
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        public static void Run()
        {
            var sourceVersionText = GetArgument("-phase17SourceVersion");
            var processingVersionText = GetArgument("-phase17ProcessingVersion");
            Run(
                GetArgument("-phase17SourceMotionId"),
                int.TryParse(sourceVersionText, out var sourceVersion) ? sourceVersion : 0,
                GetArgument("-phase17ProcessedMotionId"),
                int.TryParse(processingVersionText, out var processingVersion) ? processingVersion : 0,
                GetArgument("-phase17SourceReference"),
                GetArgument("-phase17ProcessedClipPath"));
        }

        public static void Run(
            string sourceMotionId,
            int sourceVersion,
            string processedMotionId,
            int processingVersion,
            string sourceReference,
            string processedClipPath)
        {
            try
            {
                if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                    throw new InvalidOperationException("Unity is already in Play Mode.");

                report = new Phase17PlaybackReport
                {
                    sourceMotionId = sourceMotionId ?? "",
                    processedMotionId = processedMotionId ?? "",
                    processingVersion = processingVersion,
                    animationReference = processedClipPath ?? "",
                    processedClipPath = processedClipPath ?? ""
                };
                if (string.IsNullOrWhiteSpace(report.sourceMotionId) ||
                    sourceVersion <= 0 ||
                    string.IsNullOrWhiteSpace(report.processedMotionId) ||
                    processingVersion <= 0 ||
                    string.IsNullOrWhiteSpace(sourceReference) ||
                    string.IsNullOrWhiteSpace(report.processedClipPath) ||
                    !report.processedClipPath.StartsWith("Assets/", StringComparison.Ordinal))
                    throw new InvalidOperationException("INVALID_PLAYBACK_ARGUMENTS");

                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                if (!scene.isLoaded)
                    throw new InvalidOperationException("PRODUCTION_SCENE_MISSING");

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

        private static void OnPlayModeStateChanged(PlayModeStateChange stateChange)
        {
            if (!SessionState.GetBool(ActiveKey, false))
                return;
            if (stateChange == PlayModeStateChange.EnteredPlayMode)
                StartPlayValidation();
            else if (stateChange == PlayModeStateChange.ExitingPlayMode && stage != ValidationStage.Finished)
                Fail("Unity exited Play Mode before production validation completed.");
        }

        private static void StartPlayValidation()
        {
            if (stage != ValidationStage.Idle || !SessionState.GetBool(ActiveKey, false))
                return;
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
                    throw new InvalidOperationException("PRODUCTION_BRIDGE_OR_CHARACTER_MISSING");

                bodyAnimator = character.GetComponent<LacrimosaBodyAnimator>();
                animator = bodyAnimator != null ? bodyAnimator.Animator : null;
                if (bodyAnimator == null || animator == null || animator.avatar == null ||
                    !animator.avatar.isValid || !animator.avatar.isHuman)
                    throw new InvalidOperationException("INVALID_PRODUCTION_RIG");

                processedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(report.processedClipPath);
                var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
                report.clipPersistent = processedClip != null &&
                    AssetDatabase.Contains(processedClip) &&
                    AssetDatabase.GetAssetPath(processedClip) == report.processedClipPath;
                report.libraryResolved = library != null &&
                    library.ResolveClip(report.animationReference) == processedClip;
                if (processedClip == null || library == null || !report.clipPersistent || !report.libraryResolved)
                    throw new InvalidOperationException("PROCESSED_CLIP_RESOLUTION_FAILED");

                originalController = animator.runtimeAnimatorController;
                originalDanceClip = FindControllerClip(originalController, "Dance");
                if (originalController == null || originalDanceClip == null)
                    throw new InvalidOperationException("PRODUCTION_DANCE_STATE_MISSING");
                if (processedClip == originalDanceClip)
                    throw new InvalidOperationException("BUILT_IN_ANIMATION_SUBSTITUTED");

                var controllerStates = GetControllerStates(originalController);
                report.controllerStates = controllerStates;
                report.requiredControllerStatesPresent =
                    controllerStates.Contains("Idle") &&
                    controllerStates.Contains("Walk") &&
                    controllerStates.Contains("Run") &&
                    (controllerStates.Contains("Wave") || controllerStates.Contains("Waving")) &&
                    controllerStates.Contains("Dance") &&
                    controllerStates.Contains("LeftTurn") &&
                    controllerStates.Contains("RightTurn") &&
                    controllerStates.Contains("LookingBehind");
                if (!report.requiredControllerStatesPresent)
                    throw new InvalidOperationException("EXISTING_CONTROLLER_STATES_MISSING");

                report.avatarValid = animator.avatar.isValid;
                report.avatarHuman = animator.avatar.isHuman;
                report.clipDurationSeconds = processedClip.length;
                report.clipFrameRate = processedClip.frameRate;
                report.curveBindingCount = AnimationUtility.GetCurveBindings(processedClip).Length;
                if (report.clipDurationSeconds <= 0f || report.clipFrameRate <= 0f || report.curveBindingCount <= 0)
                    throw new InvalidOperationException("INVALID_PROCESSED_CLIP");

                samples.Clear();
                callbackDone = false;
                callbackSuccess = false;
                stage = ValidationStage.PreFlight;
                stageStartedAt = Time.realtimeSinceStartup;
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

            var elapsed = Time.realtimeSinceStartup - stageStartedAt;
            if (stage == ValidationStage.Idle)
            {
                StartPlayValidation();
            }
            else if (stage == ValidationStage.PreFlight)
            {
                if (elapsed < 0.2f)
                    return;
                var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
                report.idleBefore = stateInfo.IsName("Idle") && !bridge.IsBusy && !character.IsMoving;
                if (report.idleBefore)
                    StartProcessedPlayback();
                else if (elapsed > 2f)
                    Fail("Production character did not enter Idle before the processed motion.");
            }
            else if (stage == ValidationStage.Playing)
            {
                if (samples.Count < sampleTimes.Length && elapsed >= sampleTimes[samples.Count])
                    CaptureSample(sampleTimes[samples.Count]);

                if (callbackDone)
                    FinishPlaybackStage();
                else if (elapsed > Mathf.Max(2f, processedClip.length + 3f))
                    Fail("Processed production playback did not complete normally.");
            }
            else if (stage == ValidationStage.IdleAfter)
            {
                var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
                report.idleAfter = stateInfo.IsName("Idle") && !bridge.IsBusy && !character.IsMoving;
                if (report.idleAfter)
                    StartNavigationProbe();
                else if (elapsed > idleAfterTimeout)
                    Fail("Production character did not return to Idle after the processed motion.");
            }
            else if (stage == ValidationStage.Navigation)
            {
                if (callbackDone)
                    FinishNavigationStage();
                else if (elapsed > 10f)
                    Fail("Production navigation probe did not complete normally.");
            }
        }

        private static void StartProcessedPlayback()
        {
            samples.Clear();
            callbackDone = false;
            callbackSuccess = false;
            sampleTimes = new[]
            {
                0.05f,
                Mathf.Max(0.1f, processedClip.length * 0.35f),
                Mathf.Max(0.15f, processedClip.length * 0.7f)
            };
            stage = ValidationStage.Playing;
            stageStartedAt = Time.realtimeSinceStartup;

            var accepted = bridge.Execute(
                "DANCE",
                report.animationReference,
                null,
                null,
                success =>
                {
                    callbackSuccess = success;
                    callbackDone = true;
                });
            if (!accepted)
                Fail("BrainActionBridge rejected the processed DANCE action.");
        }

        private static void CaptureSample(float sampleTime)
        {
            var hips = animator.GetBoneTransform(HumanBodyBones.Hips);
            var head = animator.GetBoneTransform(HumanBodyBones.Head);
            var leftHand = animator.GetBoneTransform(HumanBodyBones.LeftHand);
            var rightHand = animator.GetBoneTransform(HumanBodyBones.RightHand);
            if (hips == null || head == null || leftHand == null || rightHand == null)
                throw new InvalidOperationException("PRODUCTION_BONES_MISSING");

            var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
            samples.Add(new PlaybackSample
            {
                time = sampleTime,
                elapsed = Time.realtimeSinceStartup - stageStartedAt,
                processedClipActive = bodyAnimator.CurrentGeneratedClip == processedClip,
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
            report.processedClipActiveAtAllSamples = samples.All(sample => sample.processedClipActive);
            report.danceStateActiveAtAllSamples = samples.All(sample => sample.danceStateActive);
            report.finiteSamples = samples
                .SelectMany(sample => sample.bones)
                .All(sample => IsFinite(sample.position));
            report.maxBoneDisplacement = MaxBoneDisplacement(samples);
            report.nontrivialMovement = report.maxBoneDisplacement > 0.01f;
            report.generatedClipCleared = bodyAnimator.CurrentGeneratedClip == null;
            report.originalControllerRestored = ReferenceEquals(animator.runtimeAnimatorController, originalController);
            report.builtInAnimationSubstituted =
                processedClip == originalDanceClip ||
                !report.processedClipActiveAtAllSamples;
            report.playbackSuccess = report.playbackSuccess &&
                report.processedClipActiveAtAllSamples &&
                report.danceStateActiveAtAllSamples &&
                report.finiteSamples &&
                report.nontrivialMovement &&
                report.generatedClipCleared &&
                report.originalControllerRestored;

            if (!report.playbackSuccess)
            {
                Fail("Processed production playback failed normal completion validation.");
                return;
            }

            stage = ValidationStage.IdleAfter;
            stageStartedAt = Time.realtimeSinceStartup;
            idleAfterTimeout = Mathf.Max(8f, originalDanceClip.length + 3f);
        }

        private static void StartNavigationProbe()
        {
            navigationStartPosition = character.transform.position;
            callbackDone = false;
            callbackSuccess = false;
            stage = ValidationStage.Navigation;
            stageStartedAt = Time.realtimeSinceStartup;

            var accepted = bridge.Execute(
                "MOVE_TO",
                "TABLE",
                null,
                null,
                success =>
                {
                    callbackSuccess = success;
                    callbackDone = true;
                });
            if (!accepted)
                Fail("BrainActionBridge rejected the production navigation probe.");
        }

        private static void FinishNavigationStage()
        {
            report.navigationDisplacement = Vector3.Distance(
                navigationStartPosition,
                character.transform.position);
            report.navigationSuccess = callbackSuccess && report.navigationDisplacement > 0.05f;
            report.applyRootMotionDisabled = !animator.applyRootMotion;
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

        private static string[] GetControllerStates(RuntimeAnimatorController controller)
        {
            if (controller is AnimatorController animatorController)
            {
                return animatorController.layers
                    .SelectMany(layer => layer.stateMachine.states)
                    .Select(state => state.state.name)
                    .Distinct()
                    .OrderBy(name => name)
                    .ToArray();
            }
            return controller.animationClips
                .Select(clip => clip.name)
                .Distinct()
                .OrderBy(name => name)
                .ToArray();
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
            report.success =
                report.clipPersistent &&
                report.libraryResolved &&
                report.playbackSuccess &&
                report.processedClipActiveAtAllSamples &&
                report.danceStateActiveAtAllSamples &&
                report.finiteSamples &&
                report.nontrivialMovement &&
                !report.builtInAnimationSubstituted &&
                report.generatedClipCleared &&
                report.originalControllerRestored &&
                report.idleBefore &&
                report.idleAfter &&
                report.navigationSuccess &&
                report.applyRootMotionDisabled &&
                report.requiredControllerStatesPresent;
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
            Debug.LogError("[Phase17ProductionPlaybackValidator] " + error);
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
            Debug.Log("[Phase17ProductionPlaybackValidator] " + JsonUtility.ToJson(report));
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
        private sealed class Phase17PlaybackReport
        {
            public bool success;
            public string error;
            public string sourceMotionId;
            public string processedMotionId;
            public int processingVersion;
            public string animationReference;
            public string processedClipPath;
            public bool avatarValid;
            public bool avatarHuman;
            public bool clipPersistent;
            public float clipDurationSeconds;
            public float clipFrameRate;
            public int curveBindingCount;
            public bool libraryResolved;
            public bool playbackSuccess;
            public bool processedClipActiveAtAllSamples;
            public bool danceStateActiveAtAllSamples;
            public bool finiteSamples;
            public bool nontrivialMovement;
            public float maxBoneDisplacement;
            public bool generatedClipCleared;
            public bool originalControllerRestored;
            public bool idleBefore;
            public bool idleAfter;
            public bool navigationSuccess;
            public float navigationDisplacement;
            public bool applyRootMotionDisabled;
            public bool builtInAnimationSubstituted;
            public bool requiredControllerStatesPresent;
            public string[] controllerStates;
            public PlaybackSample[] samples;
        }

        [Serializable]
        private sealed class PlaybackSample
        {
            public float time;
            public float elapsed;
            public bool processedClipActive;
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
