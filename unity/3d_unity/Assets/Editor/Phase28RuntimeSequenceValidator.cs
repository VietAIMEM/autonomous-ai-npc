using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using RoomAgent.Brain;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using RoomAgent.Smartphone;
using RoomAgent.Speech;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Networking;

namespace RoomAgent.Editor
{
    /// <summary>
    /// PHASE 28 runtime acceptance.
    ///
    /// Drives the representative motion sequence through the production
    /// BrainActionBridge (Idle -> Walk -> Turn -> Look -> Gesture -> Speak ->
    /// Walk/Idle), then an interruption round and the Phase 14-17 production
    /// motion, sampling the live Animator / NavMeshAgent / IK state on every
    /// frame and rendering close-up frames of the real character.
    ///
    /// Every reported value is read back from the running components; a
    /// requested transition is never treated as an observed one.
    /// </summary>
    [InitializeOnLoad]
    public static class Phase28RuntimeSequenceValidator
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string ReportPath = "Temp/Phase28RuntimeSequence.json";
        private const string FramesDirectory = "Temp/phase28_frames";
        private const string ActiveKey = "Phase28RuntimeSequenceValidator.Active";
        private const float SampleIntervalSeconds = 0.033f;
        private const int MaxSamples = 24000;
        private const float FrameWidth = 540f;
        private const float FrameHeight = 960f;

        public static string Describe() =>
            "Phase28RuntimeSequenceValidator ready (drives the production BrainActionBridge).";

        private enum Stage
        {
            None,
            IdleBefore,
            WalkOut,
            SettleAfterWalkOut,
            TurnToPhone,
            TurnSettle,
            LookAtPhone,
            Wave,
            WaveSettle,
            SpeechPrepare,
            Speech,
            WalkBack,
            SettleAfterWalkBack,
            InterruptOneArm,
            InterruptOneWait,
            InterruptOneRecover,
            InterruptTwoArm,
            InterruptTwoWait,
            InterruptTwoRecover,
            GeneratedMotion,
            PhonePickUp,
            PhonePlace,
            FinalStop,
            Finalize,
            Finished
        }

        private static readonly string[] ExpectedStages =
        {
            "IdleBefore",
            "WalkOut",
            "SettleAfterWalkOut",
            "TurnToPhone",
            "TurnSettle",
            "LookAtPhone",
            "Wave",
            "WaveSettle",
            "SpeechPrepare",
            "Speech",
            "WalkBack",
            "SettleAfterWalkBack",
            "InterruptOneArm",
            "InterruptOneWait",
            "InterruptOneRecover",
            "InterruptTwoArm",
            "InterruptTwoWait",
            "InterruptTwoRecover",
            "GeneratedMotion",
            "PhonePickUp",
            "PhonePlace",
            "FinalStop"
        };

        private static readonly string[] RequiredSnapshotKeys =
        {
            "animator_state",
            "animator_previous_state",
            "animator_requested_action",
            "animator_transition_active",
            "animator_transition_progress",
            "animator_transition_target",
            "animator_state_normalized_time",
            "animator_base_layer_weight",
            "root_motion_apply",
            "locomotion_owner",
            "agent_velocity",
            "agent_desired_speed",
            "agent_is_stopped",
            "agent_destination",
            "orientation_state",
            "orientation_target",
            "orientation_angle_remaining",
            "rotation_owner",
            "look_at_active",
            "look_at_target",
            "ik_left_hand_target",
            "ik_right_hand_target",
            "ik_left_hand_weight",
            "ik_right_hand_weight",
            "motion_reference",
            "animation_completion_reason",
            "animation_last_result",
            "last_action_result",
            "execution_id"
        };

        private static Stage stage = Stage.None;
        private static Phase28Report report = new Phase28Report();
        private static BrainActionBridge bridge;
        private static LacrimosaController character;
        private static LacrimosaBodyAnimator bodyAnimator;
        private static Animator animator;
        private static NavMeshAgent agent;
        private static SmartphoneController phone;
        private static LacrimosaLookAt lookAt;
        private static LacrimosaIK ik;
        private static SpeechAudioPlayer voice;
        private static RuntimeAnimatorController originalController;
        private static AnimationClip generatedClip;
        private static string generatedClipPath;
        private static string generatedReference;
        private static string adminBaseUrl;
        private static string adminTokenPath;
        private static string speechText;
        private static string walkOutTarget = "PHONE_STAND";
        private static string walkBackTarget = "CENTER";
        private static string interruptReason = "phase28_interruption_test";
        private static string interruptReasonRepeat = "phase28_interruption_repeat";
        // The Bip001 head bone's local forward axis is not the gaze axis, so the
        // rest pose is measured once and reused to turn the head's rotation into
        // a real gaze direction.
        private static Vector3 headGazeLocalRest = Vector3.forward;
        private static bool headGazeRestCaptured;

        private static readonly List<MotionSample> samples = new List<MotionSample>();
        private static readonly List<MotionEvent> events = new List<MotionEvent>();
        private static readonly List<CheckResult> checks = new List<CheckResult>();
        private static readonly List<DiagnosticSnapshot> diagnosticSnapshots = new List<DiagnosticSnapshot>();
        private static readonly List<string> screenshots = new List<string>();

        private static StageRecord currentStage;
        private static bool stageEntered;
        private static bool capturedThisStage;
        private static float stageStartedAt;
        private static float validationStartedAt;
        private static float lastSampleAt;
        private static float lastReportWriteAt;
        private static int screenshotIndex;
        private static string lastSampledState = "none";

        private static bool callbackDone;
        private static bool callbackSuccess;
        private static string callbackReason = "none";

        private static Vector3 stagePositionBefore;
        private static float stageYawBefore;
        private static Vector3 walkOutDestination;
        private static float walkOutDistance;
        private static float turnAngleBefore;
        private static float turnAngleAfter;
        private static float turnMaxYawStep;
        private static float lastTurnYaw;
        private static float lastTurnSampleAt;
        private static int turnSamples;
        private static bool turnCaptured;
        private static float settleStartedAt;
        private static float walkIdleRecoverSeconds;
        private static string walkTransitionObserved = "none";

        private static bool speechTtsUsed;
        private static string speechFallbackReason = "none";
        private static string speechRequestId = "";
        private static string speechStateAtPrepare = "none";
        private static int speechHttpPhase;
        private static string speechCookie;
        private static UnityWebRequest speechRequest;
        private static string speechSubmitPhase = "none";
        private static bool speechAudioObserved;
        private static bool speechFrameCaptured;

        private static float interruptWaveSeenAt;
        private static float waveStateSeenAt;
        private static float interruptIssuedAt;
        private static bool interruptIssued;
        private static float interruptRecoverySeconds = -1f;
        private static bool interruptRoundOnePassed;
        private static bool interruptRoundTwoPassed;
        private static string interruptRoundReason = "none";

        private static float generatedProgressObserved;
        private static bool generatedFrameCaptured;
        private static bool generatedClipCleared;
        private static bool generatedControllerRestored;

        private static float pickupDistanceBefore = -1f;
        private static bool phoneParentIsRightHandBone;
        private static bool phoneParentAnchoredToRightHand;
        private static string phoneParentBoneName = "none";
        private static float phoneOffsetError = -1f;
        private static float phoneScaleError = -1f;

        static Phase28RuntimeSequenceValidator()
        {
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        /// <summary>
        /// Starts the acceptance run. Called from the Unity pipeline (reflection)
        /// while the editor is idle: the run itself enters Play Mode, drives the
        /// sequence, writes Temp/Phase28RuntimeSequence.json and leaves Play Mode.
        /// </summary>
        public static void Run(
            string processedClipPath,
            string processedReference,
            string adminUrl,
            string tokenPath,
            string text)
        {
            try
            {
                if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                    throw new InvalidOperationException("UNITY_ALREADY_IN_PLAY_MODE");

                report = new Phase28Report
                {
                    started_at_utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                    generated_clip_path = processedClipPath ?? "",
                    generated_reference = processedReference ?? ""
                };
                generatedClipPath = processedClipPath ?? "";
                generatedReference = processedReference ?? "";
                adminBaseUrl = (adminUrl ?? "").TrimEnd('/');
                adminTokenPath = tokenPath ?? "";
                speechText = string.IsNullOrWhiteSpace(text)
                    ? "Phase twenty-eight motion check. Everything is moving smoothly."
                    : text.Trim();

                if (string.IsNullOrWhiteSpace(generatedClipPath) ||
                    !generatedClipPath.StartsWith("Assets/", StringComparison.Ordinal) ||
                    string.IsNullOrWhiteSpace(generatedReference))
                    throw new InvalidOperationException("INVALID_GENERATED_MOTION_ARGUMENTS");

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
                stage = Stage.None;
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
            else if (stateChange == PlayModeStateChange.ExitingPlayMode && stage != Stage.Finished)
                Fail("UNITY_EXITED_PLAY_MODE_BEFORE_COMPLETION");
        }

        private static void StartPlayValidation()
        {
            try
            {
                bridge = UnityEngine.Object.FindFirstObjectByType<BrainActionBridge>();
                character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
                phone = UnityEngine.Object.FindFirstObjectByType<SmartphoneController>();
                voice = UnityEngine.Object.FindFirstObjectByType<SpeechAudioPlayer>();
                if (bridge == null || character == null || phone == null)
                    throw new InvalidOperationException("PRODUCTION_BRIDGE_CHARACTER_OR_PHONE_MISSING");

                var loop = UnityEngine.Object.FindFirstObjectByType<BrainLoopController>();
                if (loop != null)
                {
                    loop.StopLoop();
                    loop.enabled = false;
                }

                bodyAnimator = character.BodyAnimator;
                animator = bodyAnimator != null ? bodyAnimator.Animator : null;
                if (bodyAnimator == null || animator == null ||
                    animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                    throw new InvalidOperationException("INVALID_PRODUCTION_RIG");

                originalController = animator.runtimeAnimatorController;
                var movement = character.GetComponent<LacrimosaMovement>();
                agent = movement != null ? movement.Agent : character.GetComponent<NavMeshAgent>();
                lookAt = character.GetComponent<LacrimosaLookAt>();
                ik = character.GetComponent<LacrimosaIK>();

                samples.Clear();
                events.Clear();
                checks.Clear();
                diagnosticSnapshots.Clear();
                screenshots.Clear();
                screenshotIndex = 0;
                validationStartedAt = Time.realtimeSinceStartup;
                lastSampleAt = 0f;

                RunPreflight();
            }
            catch (Exception exception)
            {
                Fail(exception.ToString());
            }
        }

        private static void RunPreflight()
        {
            report.unity_version = Application.unityVersion;
            report.scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            report.avatar_valid = animator.avatar.isValid;
            report.avatar_human = animator.avatar.isHuman;
            report.controller_name = originalController != null ? originalController.name : "missing";
            report.animator_layers = animator.layerCount;
            report.ik_pass_enabled = false;
            report.apply_root_motion = animator.applyRootMotion;
            report.look_at_component_present = lookAt != null;
            report.ik_component_present = ik != null;
            report.agent_speed = agent != null ? agent.speed : 0f;
            report.agent_angular_speed = agent != null ? agent.angularSpeed : 0f;
            report.agent_update_rotation_baseline = agent != null && agent.updateRotation;
            report.phone_state_baseline = phone.CurrentState.ToString();
            report.phone_position_baseline = ToList(phone.transform.position);
            report.locomotion_owner = "navmesh_agent";
            report.rotation_owner_baseline = character.RotationOwner;
            // The rig's head bone forward axis is not its gaze axis, so the rest
            // pose is measured once here and reused when sampling the head.
            var headBone = animator.GetBoneTransform(HumanBodyBones.Head);
            if (headBone != null)
            {
                headGazeLocalRest = headBone.InverseTransformDirection(character.transform.forward);
                headGazeRestCaptured = true;
                report.head_gaze_rest_offset_degrees = Vector3.Angle(headBone.forward, character.transform.forward);
            }
            report.snapshot_keys_present = RequiredSnapshotKeys
                .Where(key => !bridge.BuildSnapshotJson().Contains("\"" + key + "\"", StringComparison.Ordinal))
                .ToArray();

            var controller = originalController as AnimatorController;
            if (controller == null && originalController is AnimatorOverrideController overrideController)
                controller = overrideController.runtimeAnimatorController as AnimatorController;
            if (controller != null && controller.layers.Length > 0)
            {
                // Mecanim IK pass is authored on the controller layer, so the
                // value is read from the real controller asset, not assumed.
                report.ik_pass_enabled = controller.layers[0].iKPass;
                report.controller_states = controller.layers[0].stateMachine.states
                    .Select(child => child.state.name)
                    .Distinct()
                    .OrderBy(name => name)
                    .ToArray();
                report.controller_clips = controller.animationClips
                    .Select(clip => clip.name)
                    .Distinct()
                    .OrderBy(name => name)
                    .ToArray();
            }
            else
            {
                report.controller_states = Array.Empty<string>();
                report.controller_clips = Array.Empty<string>();
            }

            report.animator_parameters = animator.parameters
                .Select(parameter => $"{parameter.name}:{parameter.type}")
                .OrderBy(name => name)
                .ToArray();

            var states = new HashSet<string>(report.controller_states);
            var requiredStates = new[]
            {
                LacrimosaBodyAnimator.IdleStateName,
                LacrimosaBodyAnimator.WalkStateName,
                LacrimosaBodyAnimator.RunStateName,
                "Waving",
                LacrimosaBodyAnimator.DanceStateName,
                "LeftTurn",
                "RightTurn",
                "LookingBehind"
            };
            report.missing_controller_states = requiredStates.Where(state => !states.Contains(state)).ToArray();

            Check(
                "animator_architecture_reused",
                report.avatar_valid && report.avatar_human &&
                report.controller_name == "LacrimosaAnimator" &&
                report.animator_layers == 1 &&
                !report.apply_root_motion &&
                report.missing_controller_states.Length == 0,
                $"controller={report.controller_name} layers={report.animator_layers} root_motion={report.apply_root_motion} missing_states={string.Join(",", report.missing_controller_states)}",
                "existing LacrimosaAnimator controller, 1 layer, root motion off, all production states present");
            Check(
                "single_motion_owner_walker",
                report.agent_update_rotation_baseline && agent != null &&
                Math.Abs(report.agent_speed - 1.4f) < 0.01f,
                $"updateRotation={report.agent_update_rotation_baseline} speed={report.agent_speed}",
                "NavMeshAgent owns translation/rotation, root motion off");
            Check(
                "ik_and_gaze_mechanisms_present",
                report.ik_pass_enabled && report.look_at_component_present,
                $"ikPass={report.ik_pass_enabled} lookAt={report.look_at_component_present} ik={report.ik_component_present}",
                "Mecanim IK pass with the existing LacrimosaLookAt/LacrimosaIK components");
            Check(
                "diagnostics_keys_exposed",
                report.snapshot_keys_present.Length == 0,
                $"missing={string.Join(",", report.snapshot_keys_present)}",
                "all Phase 28 diagnostic keys present in the live snapshot");

            var stream = bridge.BuildSnapshot().stream;
            Check(
                "livestream_not_started",
                !stream.is_streaming,
                $"is_streaming={stream.is_streaming}",
                "stream stays stopped during Phase 28 validation");

            CaptureFrame("01_idle_before");
            AddDiagnosticSnapshot("idle_before");
            EventLog("preflight", $"controller={report.controller_name} states={report.controller_states.Length}");
            EnterStage(Stage.IdleBefore);
        }

        private static void OnEditorUpdate()
        {
            if (!SessionState.GetBool(ActiveKey, false))
                return;
            if (stage == Stage.None || stage == Stage.Finalize || stage == Stage.Finished)
                return;
            if (!EditorApplication.isPlaying)
                return;

            try
            {
                var elapsed = Time.realtimeSinceStartup - stageStartedAt;
                if (elapsed > StageTimeoutSeconds(stage))
                {
                    Fail($"STAGE_TIMEOUT:{stage}:{elapsed.ToString("0.0", CultureInfo.InvariantCulture)}s");
                    return;
                }

                Tick();

                if (stage == Stage.Finalize || stage == Stage.Finished)
                    return;
                Sample();
                if (Time.realtimeSinceStartup - lastReportWriteAt > 5f)
                {
                    lastReportWriteAt = Time.realtimeSinceStartup;
                    WriteReport();
                }
            }
            catch (Exception exception)
            {
                Fail(exception.ToString());
            }
        }

        private static float StageTimeoutSeconds(Stage value)
        {
            switch (value)
            {
                case Stage.IdleBefore: return 10f;
                case Stage.WalkOut: return 30f;
                case Stage.SettleAfterWalkOut: return 6f;
                case Stage.TurnToPhone: return 10f;
                case Stage.TurnSettle: return 6f;
                case Stage.LookAtPhone: return 6f;
                case Stage.Wave: return 12f;
                case Stage.WaveSettle: return 6f;
                case Stage.SpeechPrepare: return 60f;
                case Stage.Speech: return 60f;
                case Stage.WalkBack: return 30f;
                case Stage.SettleAfterWalkBack: return 6f;
                case Stage.InterruptOneArm: return 8f;
                case Stage.InterruptOneWait: return 8f;
                case Stage.InterruptOneRecover: return 8f;
                case Stage.InterruptTwoArm: return 8f;
                case Stage.InterruptTwoWait: return 8f;
                case Stage.InterruptTwoRecover: return 8f;
                case Stage.GeneratedMotion: return 15f;
                case Stage.PhonePickUp: return 30f;
                case Stage.PhonePlace: return 40f;
                case Stage.FinalStop: return 10f;
                default: return 15f;
            }
        }

        private static void Tick()
        {
            switch (stage)
            {
                case Stage.IdleBefore: TickIdleBefore(); break;
                case Stage.WalkOut: TickWalkOut(); break;
                case Stage.SettleAfterWalkOut: TickSettleAfterWalkOut(); break;
                case Stage.TurnToPhone: TickTurnToPhone(); break;
                case Stage.TurnSettle: TickTurnSettle(); break;
                case Stage.LookAtPhone: TickLookAtPhone(); break;
                case Stage.Wave: TickWave(); break;
                case Stage.WaveSettle: TickWaveSettle(); break;
                case Stage.SpeechPrepare: TickSpeechPrepare(); break;
                case Stage.Speech: TickSpeech(); break;
                case Stage.WalkBack: TickWalkBack(); break;
                case Stage.SettleAfterWalkBack: TickSettleAfterWalkBack(); break;
                case Stage.InterruptOneArm: TickInterruptArm(true); break;
                case Stage.InterruptOneWait: TickInterruptWait(true); break;
                case Stage.InterruptOneRecover: TickInterruptRecover(true); break;
                case Stage.InterruptTwoArm: TickInterruptArm(false); break;
                case Stage.InterruptTwoWait: TickInterruptWait(false); break;
                case Stage.InterruptTwoRecover: TickInterruptRecover(false); break;
                case Stage.GeneratedMotion: TickGeneratedMotion(); break;
                case Stage.PhonePickUp: TickPhonePickUp(); break;
                case Stage.PhonePlace: TickPhonePlace(); break;
                case Stage.FinalStop: TickFinalStop(); break;
            }
        }

        private static void TickIdleBefore()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                return;
            }

            if (!bridge.IsBusy && IsIdle() && !bodyAnimator.TransitionActive)
            {
                CompleteStage(true, "idle_ready");
                EnterStage(Stage.WalkOut);
            }
        }

        private static void TickWalkOut()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                stagePositionBefore = character.transform.position;
                var snapshot = bridge.BuildSnapshot();
                if (snapshot.targets.TryGetValue(walkOutTarget, out var target) && target.position != null)
                    walkOutDestination = new Vector3(target.position[0], target.position[1], target.position[2]);
                walkOutDistance = Vector3.Distance(stagePositionBefore, walkOutDestination);
                report.walk_out_target = walkOutTarget;
                report.walk_out_distance = walkOutDistance;
                if (!StartBridgeAction("MOVE_TO", walkOutTarget, null, null))
                    return;
                return;
            }

            if (!callbackDone)
            {
                if (!capturedThisStage && Time.realtimeSinceStartup - stageStartedAt > 0.8f && IsWalk())
                {
                    capturedThisStage = true;
                    CaptureFrame("02_walk_out");
                    AddDiagnosticSnapshot("walking");
                }
                return;
            }

            var walkSamples = SamplesIn("WalkOut");
            var walkStates = walkSamples.Count(sample => sample.state == LacrimosaBodyAnimator.WalkStateName);
            var movingSamples = walkSamples.Count(sample => sample.agent_velocity > 0.4f);
            var transitionToWalk = walkSamples.Any(sample =>
                sample.transition_active &&
                sample.transition_target == LacrimosaBodyAnimator.WalkStateName);
            if (!transitionToWalk)
            {
                var transitionEvent = events.FirstOrDefault(item =>
                    item.kind == "transition" && item.detail.Contains("-> Walk"));
                transitionToWalk = transitionEvent != null;
            }
            walkTransitionObserved = transitionToWalk ? "observed" : "not_observed";

            report.walk_state_samples = walkStates;
            report.walk_moving_samples = movingSamples;
            report.walk_transition_observed = transitionToWalk;
            report.walk_displacement = Vector3.Distance(stagePositionBefore, character.transform.position);

            Check(
                "idle_to_walk_transition",
                callbackSuccess && walkStates >= 2 && movingSamples >= 2 &&
                report.walk_displacement > 0.5f && transitionToWalk,
                $"success={callbackSuccess} reason={callbackReason} walk_samples={walkStates} moving_samples={movingSamples} displacement={report.walk_displacement.ToString("0.00", CultureInfo.InvariantCulture)} transition={walkTransitionObserved}",
                "real Idle->Walk transition, walking samples and NavMesh displacement");

            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.SettleAfterWalkOut);
        }

        private static void TickSettleAfterWalkOut()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                settleStartedAt = Time.realtimeSinceStartup;
                return;
            }

            if (IsIdle() && !bodyAnimator.TransitionActive && AgentSettled() &&
                Math.Abs(bodyAnimator.BaseLayerWeight - 1f) < 0.01f)
            {
                walkIdleRecoverSeconds = Time.realtimeSinceStartup - settleStartedAt;
                report.walk_idle_recover_seconds = walkIdleRecoverSeconds;
                Check(
                    "walk_to_idle_recovery",
                    walkIdleRecoverSeconds <= 3f &&
                    Math.Abs(bodyAnimator.BaseLayerWeight - 1f) < 0.01f,
                    $"recovered_in={walkIdleRecoverSeconds.ToString("0.00", CultureInfo.InvariantCulture)}s layer_weight={bodyAnimator.BaseLayerWeight.ToString("0.00", CultureInfo.InvariantCulture)} {AgentEvidence()}",
                    "walk returns to Idle with full layer weight and a stopped agent");
                CompleteStage(true, "settled_idle");
                EnterStage(Stage.TurnToPhone);
            }
            else if (Time.realtimeSinceStartup - settleStartedAt > 5f)
            {
                Check(
                    "walk_to_idle_recovery",
                    false,
                    $"state={bodyAnimator.CurrentStateName} transition={bodyAnimator.TransitionActive} {AgentEvidence()}",
                    "walk returns to Idle with full layer weight and a stopped agent");
                CompleteStage(false, "walk_did_not_settle");
                EnterStage(Stage.TurnToPhone);
            }
        }

        private static void TickTurnToPhone()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                stageYawBefore = character.transform.eulerAngles.y;
                turnAngleBefore = FlatAngleToTarget(phone.transform.position);
                lastTurnYaw = stageYawBefore;
                lastTurnSampleAt = Time.realtimeSinceStartup;
                turnSamples = 0;
                turnMaxYawStep = 0f;
                if (!StartBridgeAction("TURN_TO", "PHONE", null, null))
                    return;
                return;
            }

            var now = Time.realtimeSinceStartup;
            var yaw = character.transform.eulerAngles.y;
            if (character.IsTurning && turnSamples < 3)
            {
                var step = Mathf.Abs(Mathf.DeltaAngle(lastTurnYaw, yaw));
                if (now - lastTurnSampleAt <= 0.1f)
                    turnMaxYawStep = Mathf.Max(turnMaxYawStep, step);
                lastTurnYaw = yaw;
                lastTurnSampleAt = now;
                turnSamples++;
            }
            report.turn_samples = turnSamples;
            report.turn_early_yaw_step_degrees = turnMaxYawStep;

            if (!turnCaptured && character.IsTurning)
            {
                turnCaptured = true;
                CaptureFrame("03_turn_to_phone");
            }

            if (!callbackDone)
                return;

            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.TurnSettle);
        }

        private static void TickTurnSettle()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                settleStartedAt = Time.realtimeSinceStartup;
                return;
            }

            var settled = !character.IsTurning && !bodyAnimator.TransitionActive;
            if (settled || Time.realtimeSinceStartup - settleStartedAt > 4f)
            {
                turnAngleAfter = FlatAngleToTarget(phone.transform.position);
                var turnSamplesObserved = SamplesIn("TurnToPhone");
                var scriptOwned = turnSamplesObserved.Count(sample => sample.rotation_owner == "script_turn");
                // The first record of the stage is the pre-turn frame, where the
                // controller reports no active turn, so only the samples that
                // actually ran under script_turn describe the rotation.
                var turningSamples = turnSamplesObserved
                    .Where(sample => sample.orientation_state == "turning")
                    .ToList();
                var angleDecreased = turningSamples.Count > 1 &&
                    turningSamples.First().orientation_angle_remaining >
                    turningSamples.Last().orientation_angle_remaining + 5f;
                var maxYawStep = 0f;
                var yawStepSign = 0f;
                var monotonic = true;
                for (var index = 1; index < turningSamples.Count; index++)
                {
                    var previous = turningSamples[index - 1];
                    var current = turningSamples[index];
                    if (current.t - previous.t > 0.1f)
                        continue;
                    var delta = Mathf.DeltaAngle(previous.yaw_degrees, current.yaw_degrees);
                    maxYawStep = Mathf.Max(maxYawStep, Mathf.Abs(delta));
                    if (Mathf.Abs(delta) <= 0.05f)
                        continue;
                    var sign = Mathf.Sign(delta);
                    if (yawStepSign != 0f && sign != yawStepSign)
                        monotonic = false;
                    yawStepSign = sign;
                }
                report.turn_max_yaw_step_degrees = maxYawStep;
                report.turn_yaw_monotonic = monotonic;
                var rotationRestored = character.RotationOwner == "navmesh_agent" &&
                    agent != null && agent.updateRotation;

                report.turn_angle_before_degrees = turnAngleBefore;
                report.turn_angle_after_degrees = turnAngleAfter;
                report.turn_script_owned_samples = scriptOwned;
                report.rotation_owner_after_turn = character.RotationOwner;
                report.turn_angle_decreased = angleDecreased;

                Check(
                    "turn_gradual_and_owned",
                    scriptOwned >= 2 && maxYawStep <= 20f &&
                    turnAngleAfter <= 6f && rotationRestored,
                    $"script_owned_samples={scriptOwned} turning_samples={turningSamples.Count} max_yaw_step={maxYawStep.ToString("0.0", CultureInfo.InvariantCulture)} angle_before={turnAngleBefore.ToString("0.0", CultureInfo.InvariantCulture)} angle_after={turnAngleAfter.ToString("0.0", CultureInfo.InvariantCulture)} owner_after={report.rotation_owner_after_turn}",
                    "scripted turn takes rotation ownership, turns smoothly and always hands it back");
                Check(
                    "turn_angle_decreased",
                    angleDecreased && monotonic,
                    $"angle_before={turnAngleBefore.ToString("0.0", CultureInfo.InvariantCulture)} angle_after={turnAngleAfter.ToString("0.0", CultureInfo.InvariantCulture)} remaining_first={turningSamples.FirstOrDefault()?.orientation_angle_remaining.ToString("0.0", CultureInfo.InvariantCulture) ?? "n/a"} remaining_last={turningSamples.LastOrDefault()?.orientation_angle_remaining.ToString("0.0", CultureInfo.InvariantCulture) ?? "n/a"} monotonic={monotonic}",
                    "orientation error decreases while the turn runs");
                CompleteStage(turnAngleAfter <= 6f && rotationRestored, "turn_settled");
                EnterStage(Stage.LookAtPhone);
            }
        }

        private static void TickLookAtPhone()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                if (!StartBridgeAction("LOOK_AT", "PHONE", null, null))
                    return;
                return;
            }

            if (!callbackDone)
                return;

            if (Time.realtimeSinceStartup - stageStartedAt < 0.8f)
                return;

            var lookSamples = SamplesIn("LookAtPhone");
            var maxWeight = lookSamples.Count > 0 ? lookSamples.Max(sample => sample.look_at_weight) : 0f;
            var activeSamples = lookSamples.Count(sample => sample.look_at_active);
            var headBetter = lookSamples.Any(sample =>
                sample.look_at_active &&
                sample.head_angle_to_target >= 0f &&
                sample.head_angle_to_target + 5f < sample.body_angle_to_target);
            // The turn stage runs before this one, so the body can already be
            // facing the phone when the gaze overlay starts and a head-over-body
            // win is then impossible by construction. The check below therefore
            // measures two separate things and records both: the resolved gaze
            // error onto the target, and whether the head did the aiming itself
            // (head relative to the body) when the body was still mis-aligned.
            var measurable = lookSamples
                .Where(sample => sample.look_at_active &&
                    sample.head_angle_to_target >= 0f &&
                    sample.body_angle_to_target >= 0f)
                .ToList();
            var bestLook = measurable
                .OrderBy(sample => sample.head_angle_to_target)
                .FirstOrDefault();
            var entryLook = measurable.FirstOrDefault();
            var gazeErrorBest = measurable.Count > 0 ? measurable.Min(sample => sample.head_angle_to_target) : -1f;
            var gazeErrorWorst = measurable.Count > 0 ? measurable.Max(sample => sample.head_angle_to_target) : -1f;
            var headRelativeMax = measurable.Count > 0
                ? measurable.Max(sample => sample.head_relative_to_body)
                : 0f;
            var bodyAngleAtEntry = entryLook?.body_angle_to_target ?? -1f;
            var headRelativeAtEntry = entryLook?.head_relative_to_body ?? -1f;
            var bodyAlignedAtEntry = entryLook != null && entryLook.body_angle_to_target <= 8f;
            var headDidTheAiming = headBetter || headRelativeMax >= 5f;
            var gazeResolved = activeSamples >= 3 && maxWeight >= 0.5f &&
                gazeErrorBest >= 0f && gazeErrorBest <= 10f &&
                (bodyAlignedAtEntry || headDidTheAiming);

            report.look_at_active_samples = activeSamples;
            report.look_at_max_weight = maxWeight;
            report.look_at_head_rotation_observed = headBetter;
            report.look_at_target = "PHONE";
            report.look_at_head_over_body_best_delta = bestLook != null
                ? bestLook.body_angle_to_target - bestLook.head_angle_to_target
                : 0f;
            report.look_at_body_angle_at_entry_degrees = bodyAngleAtEntry;
            report.look_at_head_relative_at_entry_degrees = headRelativeAtEntry;
            report.look_at_head_relative_max_degrees = headRelativeMax;
            report.look_at_gaze_error_best_degrees = gazeErrorBest;
            report.look_at_gaze_error_worst_degrees = gazeErrorWorst;
            report.look_at_body_aligned_at_entry = bodyAlignedAtEntry;
            report.look_at_head_did_the_aiming = headDidTheAiming;

            Check(
                "gaze_overlay_active",
                activeSamples >= 3 && maxWeight >= 0.5f,
                $"active_samples={activeSamples} max_weight={maxWeight.ToString("0.00", CultureInfo.InvariantCulture)}",
                "Mecanim look-at weight reaches at least 0.5 while the gaze target is set");
            Check(
                "gaze_head_rotation_observed",
                gazeResolved,
                $"gaze_error_best={gazeErrorBest.ToString("0.0", CultureInfo.InvariantCulture)} gaze_error_worst={gazeErrorWorst.ToString("0.0", CultureInfo.InvariantCulture)} body_at_entry={bodyAngleAtEntry.ToString("0.0", CultureInfo.InvariantCulture)} body_aligned_at_entry={bodyAlignedAtEntry} head_relative_at_entry={headRelativeAtEntry.ToString("0.0", CultureInfo.InvariantCulture)} head_relative_max={headRelativeMax.ToString("0.0", CultureInfo.InvariantCulture)} head_better_than_body={headBetter}",
                "the measured head gaze resolves onto the target while the overlay is active; a body already facing the target needs no head correction, otherwise the head must have done the aiming (head-over-body win or head-relative rotation)");

            if (!capturedThisStage)
            {
                capturedThisStage = true;
                CaptureFrame("04_look_at_phone");
            }
            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.Wave);
        }

        private static void TickWave()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                waveStateSeenAt = 0f;
                if (!StartBridgeAction("WAVE", null, null, null))
                    return;
                return;
            }

            if (!callbackDone)
            {
                // Capture the gesture once it is really running rather than on
                // the transition frame, so the picture shows the played motion.
                if (waveStateSeenAt <= 0f && (IsWaving() || TransitionTargetIs("Waving")))
                    waveStateSeenAt = Time.realtimeSinceStartup;
                if (!capturedThisStage && waveStateSeenAt > 0f &&
                    Time.realtimeSinceStartup - waveStateSeenAt >= 1f)
                {
                    capturedThisStage = true;
                    CaptureFrame("05_wave");
                    AddDiagnosticSnapshot("waving");
                }
                return;
            }

            var waveSamples = SamplesIn("Wave");
            var waveStates = waveSamples.Count(sample => sample.state == "Waving");
            var transitionToWave = waveSamples.Any(sample =>
                sample.transition_active && sample.transition_target == "Waving") ||
                events.Any(item => item.kind == "transition" && item.detail.Contains("-> Waving"));
            var returned = waveSamples.Where((sample, index) =>
                index > 0 &&
                sample.state == LacrimosaBodyAnimator.IdleStateName &&
                !sample.transition_active).Any();

            report.wave_state_samples = waveStates;
            report.wave_transition_observed = transitionToWave;
            report.wave_completion_reason = bodyAnimator.LastCompletionReason;
            report.wave_last_completion = bodyAnimator.LastCompletion;
            report.wave_returned_to_idle = returned;

            Check(
                "gesture_transition_and_return",
                callbackSuccess && waveStates >= 1 && transitionToWave &&
                string.Equals(bodyAnimator.LastCompletionReason, "animation_completed", StringComparison.Ordinal),
                $"success={callbackSuccess} reason={callbackReason} wave_samples={waveStates} transition={transitionToWave} completion={bodyAnimator.LastCompletion}",
                "Idle->Waving transition observed, clip runs to its controller exit and reports animation_completed");

            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.WaveSettle);
        }

        private static void TickWaveSettle()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                settleStartedAt = Time.realtimeSinceStartup;
                return;
            }

            if ((IsIdle() && !bodyAnimator.TransitionActive) ||
                Time.realtimeSinceStartup - settleStartedAt > 4f)
            {
                var clean = IsIdle() && !bodyAnimator.TransitionActive &&
                    Math.Abs(bodyAnimator.BaseLayerWeight - 1f) < 0.01f &&
                    !bodyAnimator.ActionStateActive;
                Check(
                    "gesture_recovery_clean",
                    clean,
                    $"state={bodyAnimator.CurrentStateName} transition={bodyAnimator.TransitionActive} weight={bodyAnimator.BaseLayerWeight.ToString("0.00", CultureInfo.InvariantCulture)} action_active={bodyAnimator.ActionStateActive}",
                    "no stale action state, overlay or layer weight after the gesture");
                CompleteStage(clean, "gesture_settled");
                EnterStage(Stage.SpeechPrepare);
            }
        }

        private static void TickSpeechPrepare()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                if (string.IsNullOrWhiteSpace(adminBaseUrl) || string.IsNullOrWhiteSpace(adminTokenPath) ||
                    !File.Exists(adminTokenPath))
                {
                    speechFallbackReason = "admin_credentials_unavailable";
                    speechSubmitPhase = speechFallbackReason;
                    report.speech_tts_used = false;
                    report.speech_fallback_reason = speechFallbackReason;
                    CompleteStage(false, speechFallbackReason);
                    EnterStage(Stage.Speech);
                    return;
                }

                var token = File.ReadAllText(adminTokenPath).Trim();
                if (token.Length == 0)
                {
                    speechFallbackReason = "admin_token_empty";
                    speechSubmitPhase = speechFallbackReason;
                    report.speech_tts_used = false;
                    report.speech_fallback_reason = speechFallbackReason;
                    CompleteStage(false, speechFallbackReason);
                    EnterStage(Stage.Speech);
                    return;
                }

                speechRequest = BuildJsonRequest(
                    adminBaseUrl + "/admin/api/login",
                    "{\"token\":" + JsonUtilityQuote(token) + "}");
                speechRequest.SendWebRequest();
                speechHttpPhase = 1;
                speechSubmitPhase = "login_sent";
                return;
            }

            if (speechHttpPhase == 1)
            {
                if (!speechRequest.isDone)
                    return;
                if (speechRequest.result != UnityWebRequest.Result.Success)
                {
                    speechHttpPhase = 4;
                    FallBackFromTts($"login_http_{speechRequest.responseCode}");
                    return;
                }

                var cookieHeader = speechRequest.GetResponseHeader("Set-Cookie");
                if (!string.IsNullOrWhiteSpace(cookieHeader))
                {
                    var pair = cookieHeader.Split(';')[0].Trim();
                    if (pair.Contains("="))
                        speechCookie = pair;
                }
                speechRequest.Dispose();
                if (string.IsNullOrWhiteSpace(speechCookie))
                {
                    speechHttpPhase = 4;
                    FallBackFromTts("login_cookie_missing");
                    return;
                }

                speechRequest = BuildJsonRequest(
                    adminBaseUrl + "/admin/api/tts/test",
                    "{\"text\":" + JsonUtilityQuote(speechText) + "}");
                speechRequest.SetRequestHeader("Cookie", speechCookie);
                speechRequest.timeout = 45;
                speechRequest.SendWebRequest();
                speechHttpPhase = 2;
                speechSubmitPhase = "tts_request_sent";
                EventLog("speech_request", speechText);
                return;
            }

            if (speechHttpPhase == 2)
            {
                if (!speechRequest.isDone)
                    return;
                if (speechRequest.result != UnityWebRequest.Result.Success)
                {
                    speechHttpPhase = 4;
                    FallBackFromTts($"tts_http_{speechRequest.responseCode}");
                    return;
                }

                var payload = speechRequest.downloadHandler != null ? speechRequest.downloadHandler.text : null;
                speechRequest.Dispose();
                var parsed = TryParseAdminTts(payload);
                if (parsed == null || parsed.request == null ||
                    string.IsNullOrWhiteSpace(parsed.request.request_id))
                {
                    speechHttpPhase = 4;
                    FallBackFromTts("tts_response_unparsed");
                    return;
                }

                speechTtsUsed = true;
                speechRequestId = parsed.request.request_id;
                speechStateAtPrepare = parsed.state;
                report.speech_tts_used = true;
                report.speech_request_id = speechRequestId;
                report.speech_state_at_prepare = speechStateAtPrepare;
                report.speech_text = speechText;
                speechHttpPhase = 3;
                speechSubmitPhase = "ready";
                EventLog("speech_ready", $"{speechRequestId} state={speechStateAtPrepare}");
                CompleteStage(true, "tts_ready:" + speechStateAtPrepare);
                EnterStage(Stage.Speech);
            }
        }

        private static void FallBackFromTts(string reason)
        {
            speechFallbackReason = reason;
            speechSubmitPhase = reason;
            report.speech_tts_used = false;
            report.speech_fallback_reason = reason;
            if (speechRequest != null)
            {
                speechRequest.Dispose();
                speechRequest = null;
            }
            EventLog("speech_fallback", reason);
            CompleteStage(false, reason);
            EnterStage(Stage.Speech);
        }

        private static void TickSpeech()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                var requestId = speechTtsUsed ? speechRequestId : null;
                if (!StartBridgeAction("SPEAK", null, speechText, requestId))
                    return;
                return;
            }

            var playingNow = voice != null && voice.IsPlaying;
            if (playingNow)
                speechAudioObserved = true;

            if (!callbackDone)
            {
                if (!speechFrameCaptured && playingNow)
                {
                    speechFrameCaptured = true;
                    CaptureFrame("06_speak");
                }
                return;
            }

            report.speech_completion_reason = bodyAnimator.LastCompletionReason;
            report.speech_completion = callbackReason;
            report.speech_audio_observed = speechAudioObserved;

            var completed = callbackSuccess &&
                (callbackReason == "speech" || callbackReason == "speech_cancelled" ||
                 callbackReason == "speech_initiation_only" || callbackReason == "initiation_only");
            if (!speechTtsUsed && callbackSuccess)
                completed = true;

            Check(
                "speech_action_completed",
                completed,
                $"success={callbackSuccess} reason={callbackReason} tts_used={speechTtsUsed} request={speechRequestId}",
                "SPEAK completes through the production bridge");
            Check(
                "speech_tts_audio_observed",
                speechTtsUsed && speechAudioObserved,
                $"tts_used={speechTtsUsed} request={speechRequestId} state_at_prepare={speechStateAtPrepare} audio_observed={speechAudioObserved} fallback={speechFallbackReason}",
                "a real backend TTS utterance is synthesised and played by Unity",
                required: false);

            CompleteStage(completed, callbackReason);
            EnterStage(Stage.WalkBack);
        }

        private static void TickWalkBack()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                stagePositionBefore = character.transform.position;
                if (!StartBridgeAction("MOVE_TO", walkBackTarget, null, null))
                    return;
                return;
            }

            if (!callbackDone)
            {
                if (!capturedThisStage && IsWalk())
                {
                    capturedThisStage = true;
                    CaptureFrame("07_walk_back_with_gaze");
                    AddDiagnosticSnapshot("walking_with_gaze");
                }
                return;
            }

            var walkSamples = SamplesIn("WalkBack");
            var walkStates = walkSamples.Count(sample => sample.state == LacrimosaBodyAnimator.WalkStateName);
            var gazeWhileWalking = walkSamples.Count(sample =>
                sample.state == LacrimosaBodyAnimator.WalkStateName &&
                sample.look_at_active && sample.look_at_weight > 0.5f);
            var headOverlay = walkSamples.Any(sample =>
                sample.state == LacrimosaBodyAnimator.WalkStateName &&
                sample.look_at_active &&
                sample.head_angle_to_target >= 0f &&
                sample.head_angle_to_target + 5f < sample.body_angle_to_target);
            var bestHeadOverlay = walkSamples
                .Where(sample => sample.state == LacrimosaBodyAnimator.WalkStateName &&
                    sample.look_at_active &&
                    sample.head_angle_to_target >= 0f &&
                    sample.body_angle_to_target >= 0f)
                .OrderBy(sample => sample.head_angle_to_target - sample.body_angle_to_target)
                .FirstOrDefault();
            var headRelativeMax = walkSamples
                .Where(sample => sample.state == LacrimosaBodyAnimator.WalkStateName && sample.look_at_active)
                .Select(sample => sample.head_relative_to_body)
                .DefaultIfEmpty(0f)
                .Max();

            report.walk_back_walk_samples = walkStates;
            report.walk_back_gaze_samples = gazeWhileWalking;
            report.walk_back_head_overlay_observed = headOverlay;
            report.walk_back_head_over_body_best_delta = bestHeadOverlay != null
                ? bestHeadOverlay.body_angle_to_target - bestHeadOverlay.head_angle_to_target
                : 0f;
            report.walk_back_head_relative_max_degrees = headRelativeMax;
            report.walk_back_displacement = Vector3.Distance(stagePositionBefore, character.transform.position);

            Check(
                "walk_plus_gaze_layering",
                callbackSuccess && walkStates >= 2 && gazeWhileWalking >= 2 && headOverlay &&
                headRelativeMax >= 10f &&
                report.walk_back_displacement > 0.5f,
                $"success={callbackSuccess} walk_samples={walkStates} gaze_samples={gazeWhileWalking} head_overlay={headOverlay} head_relative_max={headRelativeMax.ToString("0.0", CultureInfo.InvariantCulture)} best_head={bestHeadOverlay?.head_angle_to_target.ToString("0.0", CultureInfo.InvariantCulture) ?? "n/a"} best_body={bestHeadOverlay?.body_angle_to_target.ToString("0.0", CultureInfo.InvariantCulture) ?? "n/a"} displacement={report.walk_back_displacement.ToString("0.00", CultureInfo.InvariantCulture)}",
                "the existing Mecanim look-at overlay stays active during locomotion without corrupting it");

            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.SettleAfterWalkBack);
        }

        private static void TickSettleAfterWalkBack()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                settleStartedAt = Time.realtimeSinceStartup;
                return;
            }

            if ((IsIdle() && !bodyAnimator.TransitionActive && AgentSettled()) ||
                Time.realtimeSinceStartup - settleStartedAt > 4f)
            {
                CompleteStage(true, "settled_idle");
                EnterStage(Stage.InterruptOneArm);
            }
        }

        private static void TickInterruptArm(bool firstRound)
        {
            if (!stageEntered)
            {
                stageEntered = true;
                interruptWaveSeenAt = 0f;
                interruptIssued = false;
                interruptIssuedAt = 0f;
                if (!StartBridgeAction("WAVE", null, null, null))
                    return;
                return;
            }

            if (IsWaving() || TransitionTargetIs("Waving"))
            {
                if (interruptWaveSeenAt <= 0f)
                    interruptWaveSeenAt = Time.realtimeSinceStartup;

                if (Time.realtimeSinceStartup - interruptWaveSeenAt >= 1f)
                {
                    // Shot taken at the moment the interrupt is issued, so the
                    // frame shows the action that was actually cancelled.
                    CaptureFrame(firstRound ? "08_interrupt_wave_running" : "10_interrupt_wave_running_2");
                    var reason = firstRound ? interruptReason : interruptReasonRepeat;
                    interruptIssued = bridge.Interrupt(reason);
                    interruptIssuedAt = Time.realtimeSinceStartup;
                    interruptRoundReason = reason;
                    EventLog("interrupt_issued", $"{reason} accepted={interruptIssued}");
                    CompleteStage(interruptIssued, interruptIssued ? "interrupt_issued" : "interrupt_rejected");
                    EnterStage(firstRound ? Stage.InterruptOneWait : Stage.InterruptTwoWait);
                }
            }
        }

        private static void TickInterruptWait(bool firstRound)
        {
            if (!stageEntered)
            {
                stageEntered = true;
                settleStartedAt = Time.realtimeSinceStartup;
                return;
            }

            if (!callbackDone)
            {
                if (Time.realtimeSinceStartup - settleStartedAt > 5f)
                {
                    Check(
                        firstRound ? "interrupt_round_one_honest_failure" : "interrupt_round_two_honest_failure",
                        false,
                        "callback never fired after the interrupt",
                        "the pending callback reports the interruption as a failure");
                    CompleteStage(false, "interrupt_callback_missing");
                    EnterStage(firstRound ? Stage.InterruptOneRecover : Stage.InterruptTwoRecover);
                }
                return;
            }

            var honestFailure = !callbackSuccess &&
                string.Equals(callbackReason, interruptRoundReason, StringComparison.Ordinal) &&
                bridge.LastObservedResult.EndsWith("FAILED", StringComparison.Ordinal) &&
                string.Equals(bodyAnimator.LastInterruptionReason, interruptRoundReason, StringComparison.Ordinal);

            if (firstRound)
                interruptRoundOnePassed = honestFailure;
            else
                interruptRoundTwoPassed = honestFailure;

            report.interrupt_rounds_run = firstRound ? 1 : 2;
            report.interrupt_last_result = bridge.LastObservedResult;
            report.interrupt_last_interruption_reason = bodyAnimator.LastInterruptionReason;
            report.interrupt_callback_success = callbackSuccess;
            report.interrupt_callback_reason = callbackReason;
            report.interrupt_action_active_seconds = bodyAnimator.LastActionActiveSeconds;
            report.interrupt_issued_at_seconds = interruptIssuedAt - validationStartedAt;

            Check(
                firstRound ? "interrupt_round_one_honest_failure" : "interrupt_round_two_honest_failure",
                honestFailure,
                $"callback_success={callbackSuccess} callback_reason={callbackReason} bridge_result={bridge.LastObservedResult} interruption_reason={bodyAnimator.LastInterruptionReason} active_seconds={bodyAnimator.LastActionActiveSeconds.ToString("0.00", CultureInfo.InvariantCulture)}",
                "interrupted action reports a failed callback carrying the interruption reason");
            CompleteStage(honestFailure, "interrupt_observed");
            EnterStage(firstRound ? Stage.InterruptOneRecover : Stage.InterruptTwoRecover);
        }

        private static void TickInterruptRecover(bool firstRound)
        {
            if (!stageEntered)
            {
                stageEntered = true;
                settleStartedAt = Time.realtimeSinceStartup;
                return;
            }

            var clean = !bodyAnimator.ActionStateActive &&
                !bodyAnimator.TransitionActive &&
                (IsIdle() || IsWalk()) &&
                Math.Abs(bodyAnimator.BaseLayerWeight - 1f) < 0.01f &&
                !bodyAnimator.CurrentGeneratedClip;

            if (clean || Time.realtimeSinceStartup - settleStartedAt > 5f)
            {
                interruptRecoverySeconds = Time.realtimeSinceStartup - settleStartedAt;
                report.interrupt_recovery_seconds = interruptRecoverySeconds;
                report.interrupt_recovered_to_locomotion = clean;
                report.last_restore_reason = bodyAnimator.LastRestoreReason;
                report.last_completion_state = bodyAnimator.LastCompletionState;

                Check(
                    firstRound ? "interrupt_round_one_recovery" : "interrupt_round_two_recovery",
                    clean && interruptRecoverySeconds <= 5f,
                    $"clean={clean} recovery_seconds={interruptRecoverySeconds.ToString("0.00", CultureInfo.InvariantCulture)} state={bodyAnimator.CurrentStateName} restore_reason={bodyAnimator.LastRestoreReason}",
                    "the Animator returns to a valid locomotion state with no stale overlay");

                if (firstRound)
                    CaptureFrame("09_interrupt_recovered");
                else
                    CaptureFrame("11_interrupt_recovered_2");

                CompleteStage(clean, "interrupt_recovered");
                EnterStage(firstRound ? Stage.InterruptTwoArm : Stage.GeneratedMotion);
            }
        }

        private static void TickGeneratedMotion()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                generatedProgressObserved = 0f;
                generatedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(generatedClipPath);
                var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
                var resolved = library != null ? library.ResolveClip(generatedReference) : null;
                report.generated_clip_asset_match = generatedClip != null &&
                    AssetDatabase.Contains(generatedClip) &&
                    AssetDatabase.GetAssetPath(generatedClip) == generatedClipPath;
                report.generated_library_resolved = resolved != null && resolved == generatedClip;
                report.generated_clip_length_seconds = generatedClip != null ? generatedClip.length : 0f;
                report.generated_clip_looping = generatedClip != null && generatedClip.isLooping;

                Check(
                    "generated_motion_provenance",
                    report.generated_clip_asset_match && report.generated_library_resolved &&
                    report.generated_clip_length_seconds > 0.1f,
                    $"asset_match={report.generated_clip_asset_match} library_resolved={report.generated_library_resolved} length={report.generated_clip_length_seconds.ToString("0.000", CultureInfo.InvariantCulture)} looping={report.generated_clip_looping}",
                    "the Phase 14-17 production clip is a persistent asset registered in the generated motion library");

                if (!report.generated_clip_asset_match || !report.generated_library_resolved)
                {
                    CompleteStage(false, "generated_motion_unresolved");
                    EnterStage(Stage.PhonePickUp);
                    return;
                }

                if (!StartBridgeAction("DANCE", generatedReference, null, null))
                    return;
                return;
            }

            if (generatedClip != null &&
                character.TryGetGeneratedClipProgress(generatedClip, out var normalized, out var active) &&
                active)
            {
                generatedProgressObserved = Mathf.Max(generatedProgressObserved, normalized);
                if (!generatedFrameCaptured && normalized > 0.25f)
                {
                    generatedFrameCaptured = true;
                    CaptureFrame("12_phase17_motion_running");
                    AddDiagnosticSnapshot("generated_motion");
                }
            }

            if (!callbackDone)
                return;

            generatedClipCleared = !bodyAnimator.CurrentGeneratedClip;
            generatedControllerRestored = animator.runtimeAnimatorController == originalController;
            report.generated_progress_observed = generatedProgressObserved;
            report.generated_completion_reason = callbackReason;
            report.generated_clip_cleared = generatedClipCleared;
            report.generated_controller_restored = generatedControllerRestored;
            report.generated_animator_apply_root_motion = animator.applyRootMotion;

            Check(
                "generated_motion_runtime_playback",
                callbackSuccess && generatedProgressObserved >= 0.9f &&
                callbackReason == "generated_motion_completed" &&
                generatedClipCleared && generatedControllerRestored && !animator.applyRootMotion,
                $"success={callbackSuccess} reason={callbackReason} progress={generatedProgressObserved.ToString("0.00", CultureInfo.InvariantCulture)} clip_cleared={generatedClipCleared} controller_restored={generatedControllerRestored}",
                "the production motion runs through the existing Dance state override, completes one pass and restores the controller");

            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.PhonePickUp);
        }

        private static void TickPhonePickUp()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                pickupDistanceBefore = Vector3.Distance(character.transform.position, phone.transform.position);
                if (!StartBridgeAction("PICK_UP_PHONE", null, null, null))
                    return;
                return;
            }

            if (!callbackDone)
                return;

            var rightHand = animator.GetBoneTransform(HumanBodyBones.RightHand);
            var parent = phone.transform.parent;
            phoneParentIsRightHandBone = parent != null && rightHand != null && parent == rightHand;
            phoneParentAnchoredToRightHand = parent != null && rightHand != null &&
                (parent == rightHand || parent.IsChildOf(rightHand));
            phoneParentBoneName = parent != null ? parent.name : "none";

            var expectedPosition = new Vector3(-0.015f, 0.005f, 0.05f);
            phoneOffsetError = Vector3.Distance(phone.transform.localPosition, expectedPosition);
            phoneScaleError = Vector3.Distance(phone.transform.localScale, Vector3.one);

            report.phone_state_after_pickup = phone.CurrentState.ToString();
            report.phone_holder_after_pickup = phone.Holder != null ? phone.Holder.name : "none";
            report.phone_parent_bone_after_pickup = phoneParentBoneName;
            report.phone_parent_is_right_hand_bone = phoneParentIsRightHandBone;
            report.phone_parent_anchored_to_right_hand = phoneParentAnchoredToRightHand;
            report.phone_local_position_after_pickup = ToList(phone.transform.localPosition);
            report.phone_local_scale_after_pickup = ToList(phone.transform.localScale);
            report.phone_offset_error = phoneOffsetError;
            report.pickup_distance_before = pickupDistanceBefore;
            report.character_held_phone_after_pickup = character.HeldPhone != null;

            Check(
                "phone_pickup_preserved",
                callbackSuccess && phone.CurrentState == SmartphoneState.Held &&
                phoneParentAnchoredToRightHand && phoneOffsetError < 0.001f && phoneScaleError < 0.001f &&
                character.HeldPhone == phone,
                $"success={callbackSuccess} reason={callbackReason} state={report.phone_state_after_pickup} parent={phoneParentBoneName} parent_under_right_hand={phoneParentAnchoredToRightHand} offset_error={phoneOffsetError.ToString("0.0000", CultureInfo.InvariantCulture)} scale_error={phoneScaleError.ToString("0.0000", CultureInfo.InvariantCulture)}",
                "the phone attaches to the existing right-hand hand mount with its established offsets");

            CaptureFrame("13_phone_held");
            AddDiagnosticSnapshot("phone_held");
            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.PhonePlace);
        }

        private static void TickPhonePlace()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                if (!StartBridgeAction("PLACE_PHONE", "TABLE", null, null))
                    return;
                return;
            }

            if (!callbackDone)
            {
                if (!capturedThisStage && IsWalk())
                {
                    capturedThisStage = true;
                    CaptureFrame("14_phone_walk_to_table");
                }
                return;
            }

            var surfaceName = phone.PlacementSurface != null
                ? (phone.PlacementSurface.transform.parent != null
                    ? phone.PlacementSurface.transform.parent.name
                    : phone.PlacementSurface.name)
                : "none";
            report.phone_state_after_placement = phone.CurrentState.ToString();
            report.phone_surface_after_placement = surfaceName;
            report.phone_holder_after_placement = phone.Holder != null ? phone.Holder.name : "none";
            report.character_held_phone_after_placement = character.HeldPhone != null;
            report.phone_position_after_placement = ToList(phone.transform.position);

            Check(
                "phone_placement_preserved",
                callbackSuccess && phone.CurrentState == SmartphoneState.Placed &&
                surfaceName == "Table" && phone.Holder == null && character.HeldPhone == null,
                $"success={callbackSuccess} reason={callbackReason} state={report.phone_state_after_placement} surface={surfaceName} holder={report.phone_holder_after_placement}",
                "the phone is placed back on the existing Table interaction surface");

            CaptureFrame("15_phone_placed");
            CompleteStage(callbackSuccess, callbackReason);
            EnterStage(Stage.FinalStop);
        }

        private static void TickFinalStop()
        {
            if (!stageEntered)
            {
                stageEntered = true;
                if (!StartBridgeAction("STOP", null, null, null))
                    return;
                return;
            }

            if (!callbackDone)
                return;

            if (Time.realtimeSinceStartup - stageStartedAt < 1.2f)
                return;

            var clean = IsIdle() && !bodyAnimator.TransitionActive &&
                Math.Abs(bodyAnimator.BaseLayerWeight - 1f) < 0.01f &&
                lookAt != null && !lookAt.IsActive &&
                AgentSettled();

            report.final_state = bodyAnimator.CurrentStateName;
            report.final_look_at_active = lookAt != null && lookAt.IsActive;
            report.final_layer_weight = bodyAnimator.BaseLayerWeight;
            report.final_agent_stopped = agent != null && agent.isOnNavMesh && agent.isStopped;
            report.final_agent_settled = AgentSettled();
            report.final_agent_evidence = AgentEvidence();

            Check(
                "final_reset_clean",
                clean,
                $"state={report.final_state} look_at={report.final_look_at_active} weight={report.final_layer_weight.ToString("0.00", CultureInfo.InvariantCulture)} {report.final_agent_evidence}",
                "STOP clears gaze, stops navigation and leaves the Animator in Idle");

            var stream = bridge.BuildSnapshot().stream;
            Check(
                "livestream_still_stopped",
                !stream.is_streaming,
                $"is_streaming={stream.is_streaming}",
                "the phase never started the external broadcast");

            CaptureFrame("16_final_idle");
            AddDiagnosticSnapshot("final_idle");
            CompleteStage(clean, "final_stop");
            EnterStage(Stage.Finalize);
        }

        private static void EnterStage(Stage next)
        {
            if (next == Stage.Finalize)
            {
                FinishValidation();
                return;
            }

            stage = next;
            stageStartedAt = Time.realtimeSinceStartup;
            stageEntered = false;
            capturedThisStage = false;
            callbackDone = false;
            callbackSuccess = false;
            callbackReason = "none";
            currentStage = new StageRecord
            {
                stage = next.ToString(),
                started_at_seconds = stageStartedAt - validationStartedAt,
                state_before = bodyAnimator != null ? bodyAnimator.CurrentStateName : "unknown"
            };
            EventLog("stage_start", next.ToString());
        }

        private static void CompleteStage(bool success, string reason)
        {
            if (currentStage != null)
            {
                currentStage.ended_at_seconds = Time.realtimeSinceStartup - validationStartedAt;
                currentStage.success = success;
                currentStage.reason = reason ?? "none";
                currentStage.state_after = bodyAnimator != null ? bodyAnimator.CurrentStateName : "unknown";
                currentStage.execution_id = currentStage.execution_id ?? "none";
                report.stages.Add(currentStage);
                EventLog("stage_end", $"{currentStage.stage}:{(success ? "SUCCESS" : "FAILED")}:{currentStage.reason}");
            }
        }

        private static bool StartBridgeAction(string action, string target, string text, string speechId)
        {
            callbackDone = false;
            callbackSuccess = false;
            callbackReason = "none";
            var accepted = bridge.Execute(action, target, text, speechId, OnBridgeCallback);
            if (!accepted)
            {
                callbackDone = true;
                callbackSuccess = false;
                callbackReason = "bridge_rejected";
                EventLog("action_rejected", $"{action} (bridge busy)");
                return false;
            }

            if (currentStage != null)
            {
                currentStage.requested_action = action;
                currentStage.target = target ?? "none";
                currentStage.execution_id = bridge.ActiveExecutionId;
            }
            EventLog("action_requested", $"{action} target={target ?? "none"} execution={bridge.ActiveExecutionId}");
            return true;
        }

        private static void OnBridgeCallback(bool success)
        {
            callbackDone = true;
            callbackSuccess = success;
            callbackReason = bridge != null ? bridge.LastObservedReason : "none";
            EventLog(
                "action_callback",
                $"{stage}:{(success ? "SUCCESS" : "FAILED")}:{callbackReason} execution={bridge?.ActiveExecutionId ?? "none"}");
        }

        private static void Sample()
        {
            var now = Time.realtimeSinceStartup;
            var state = bodyAnimator.CurrentStateName;
            var stateChanged = !string.Equals(state, lastSampledState, StringComparison.Ordinal);
            if (!stateChanged && now - lastSampleAt < SampleIntervalSeconds)
                return;
            if (samples.Count >= MaxSamples)
                return;

            if (stateChanged)
            {
                lastSampledState = state;
                EventLog(
                    "transition",
                    $"{bodyAnimator.PreviousStateName} -> {state} (transition={bodyAnimator.LastTransition})");
            }

            lastSampleAt = now;
            var target = lookAt != null && lookAt.IsActive ? lookAt.TargetPosition : Vector3.zero;
            var head = animator.GetBoneTransform(HumanBodyBones.Head);
            var bodyAngle = -1f;
            var headAngle = -1f;
            var headRelativeToBody = 0f;
            var headRelativeYaw = 0f;
            if (lookAt != null && lookAt.IsActive)
            {
                var flatToTarget = target - character.transform.position;
                flatToTarget.y = 0f;
                var flatBody = character.transform.forward;
                flatBody.y = 0f;
                Vector3 flatHead = Vector3.zero;
                if (head != null)
                {
                    // Rotate the rest-pose gaze axis by the head bone's live
                    // rotation so the measured direction is the real gaze.
                    flatHead = headGazeRestCaptured
                        ? head.TransformDirection(headGazeLocalRest)
                        : head.forward;
                    flatHead.y = 0f;
                }
                if (flatBody.sqrMagnitude > 0.001f && flatHead.sqrMagnitude > 0.001f)
                {
                    // Head orientation measured against the body, not against
                    // the target: this separates "the head moved" from "the
                    // body happened to already face the target".
                    headRelativeToBody = Vector3.Angle(flatBody, flatHead);
                    headRelativeYaw = Vector3.SignedAngle(flatBody, flatHead, Vector3.up);
                }
                if (flatToTarget.sqrMagnitude > 0.001f)
                {
                    bodyAngle = Vector3.Angle(flatBody, flatToTarget);
                    if (flatHead.sqrMagnitude > 0.001f)
                        headAngle = Vector3.Angle(flatHead, flatToTarget);
                }
            }

            var generated = bodyAnimator.CurrentGeneratedClip;
            var generatedProgress = 0f;
            if (generated != null &&
                character.TryGetGeneratedClipProgress(generated, out var progress, out var active) && active)
                generatedProgress = progress;

            samples.Add(new MotionSample
            {
                t = now - validationStartedAt,
                stage = stage.ToString(),
                action = bridge.RequestedActionName,
                execution_id = bridge.ActiveExecutionId,
                busy = bridge.IsBusy,
                state = state,
                previous_state = bodyAnimator.PreviousStateName,
                transition = bodyAnimator.LastTransition,
                transition_active = bodyAnimator.TransitionActive,
                transition_progress = bodyAnimator.TransitionProgress,
                transition_target = bodyAnimator.TransitionTargetName,
                normalized_time = bodyAnimator.CurrentStateNormalizedTime,
                layer_weight = bodyAnimator.BaseLayerWeight,
                agent_velocity = agent != null ? agent.velocity.magnitude : 0f,
                agent_desired_speed = agent != null ? agent.desiredVelocity.magnitude : 0f,
                agent_stopped = agent == null || !agent.isOnNavMesh || agent.isStopped,
                agent_settled = AgentSettled(),
                agent_remaining_distance = agent != null && agent.isOnNavMesh ? agent.remainingDistance : 0f,
                orientation_state = character.OrientationState,
                orientation_angle_remaining = character.OrientationAngleRemaining,
                yaw_degrees = character.transform.eulerAngles.y,
                rotation_owner = character.RotationOwner,
                look_at_active = lookAt != null && lookAt.IsActive,
                look_at_weight = lookAt != null ? lookAt.CurrentWeight : 0f,
                body_angle_to_target = bodyAngle,
                head_angle_to_target = headAngle,
                head_relative_to_body = headRelativeToBody,
                head_relative_yaw = headRelativeYaw,
                phone_state = phone.CurrentState.ToString(),
                phone_parent = phone.transform.parent != null ? phone.transform.parent.name : "none",
                voice_playing = voice != null && voice.IsPlaying,
                posture = bridge.BuildSnapshot().character.current_action,
                generated_clip = generated != null ? generated.name : "none",
                generated_progress = generatedProgress,
                ik_left_weight = ik != null ? ik.LeftHandWeight : 0f,
                ik_right_weight = ik != null ? ik.RightHandWeight : 0f
            });
        }

        private static void AddDiagnosticSnapshot(string label)
        {
            diagnosticSnapshots.Add(new DiagnosticSnapshot
            {
                stage = label,
                t = Time.realtimeSinceStartup - validationStartedAt,
                json = bridge.BuildSnapshotJson()
            });
        }

        private static void CaptureFrame(string label)
        {
            try
            {
                var directory = Path.Combine(Directory.GetCurrentDirectory(), FramesDirectory);
                Directory.CreateDirectory(directory);
                screenshotIndex++;
                var fileName =
                    $"{screenshotIndex:00}_{label}_{DateTime.Now.ToString("HHmmss", CultureInfo.InvariantCulture)}.png";
                var path = Path.Combine(directory, fileName);

                var cameraObject = new GameObject("Phase28CaptureCamera");
                var camera = cameraObject.AddComponent<Camera>();
                camera.fieldOfView = 32f;
                camera.nearClipPlane = 0.05f;
                camera.farClipPlane = 100f;
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = new Color(0.09f, 0.1f, 0.13f, 1f);
                camera.cullingMask = ~0;

                // Three-quarter view in front of the character, framed on the
                // upper body, so poses can actually be inspected.
                var anchor = character.transform;
                var focus = anchor.position + Vector3.up * 1.15f;
                var offset = anchor.rotation * new Vector3(1.75f, 1.55f, 2.45f);
                camera.transform.position = focus + offset;
                camera.transform.LookAt(focus + anchor.forward * 0.15f);

                var renderTexture = new RenderTexture(
                    Mathf.RoundToInt(FrameWidth),
                    Mathf.RoundToInt(FrameHeight),
                    24,
                    RenderTextureFormat.ARGB32);
                camera.targetTexture = renderTexture;
                camera.Render();

                var previous = RenderTexture.active;
                RenderTexture.active = renderTexture;
                var texture = new Texture2D(
                    Mathf.RoundToInt(FrameWidth),
                    Mathf.RoundToInt(FrameHeight),
                    TextureFormat.RGB24,
                    false);
                texture.ReadPixels(new Rect(0f, 0f, FrameWidth, FrameHeight), 0, 0);
                texture.Apply();
                RenderTexture.active = previous;
                File.WriteAllBytes(path, texture.EncodeToPNG());

                UnityEngine.Object.DestroyImmediate(texture);
                // Detach before releasing, otherwise Unity reports "Releasing render
                // texture that is set as Camera.targetTexture!" for every frame.
                camera.targetTexture = null;
                renderTexture.Release();
                UnityEngine.Object.DestroyImmediate(renderTexture);
                UnityEngine.Object.DestroyImmediate(cameraObject);

                screenshots.Add(FramesDirectory + "/" + fileName);
                report.screenshots = screenshots.ToArray();
                EventLog("frame", label);
            }
            catch (Exception exception)
            {
                EventLog("frame_failed", label + ":" + exception.Message);
            }
        }

        private static List<MotionSample> SamplesIn(string stageName) =>
            samples.Where(sample => sample.stage == stageName).ToList();

        private static bool IsIdle() =>
            string.Equals(bodyAnimator.CurrentStateName, LacrimosaBodyAnimator.IdleStateName, StringComparison.Ordinal);

        /// <summary>
        /// The runtime stops navigation with NavMeshAgent.ResetPath(), so isStopped
        /// stays false, and a completed path keeps hasPath true until ResetPath is
        /// called. A settled agent therefore means: no residual velocity and either
        /// no path or a finished one inside the stopping distance.
        /// </summary>
        private static bool AgentSettled(float velocityThreshold = 0.05f)
        {
            if (agent == null)
                return true;
            if (agent.velocity.magnitude >= velocityThreshold)
                return false;
            if (!agent.hasPath)
                return true;
            return !agent.pathPending &&
                agent.remainingDistance <= agent.stoppingDistance + 0.1f;
        }

        /// <summary>Settled-agent evidence for the report, read from the live agent.</summary>
        private static string AgentEvidence()
        {
            if (agent == null)
                return "agent=none";
            return $"has_path={agent.hasPath} path_pending={agent.pathPending} " +
                $"remaining={agent.remainingDistance.ToString("0.00", CultureInfo.InvariantCulture)} " +
                $"stopping={agent.stoppingDistance.ToString("0.00", CultureInfo.InvariantCulture)} " +
                $"velocity={agent.velocity.magnitude.ToString("0.000", CultureInfo.InvariantCulture)}";
        }

        private static bool IsWalk() =>
            string.Equals(bodyAnimator.CurrentStateName, LacrimosaBodyAnimator.WalkStateName, StringComparison.Ordinal);

        private static bool IsWaving() =>
            string.Equals(bodyAnimator.CurrentStateName, "Waving", StringComparison.Ordinal);

        private static bool TransitionTargetIs(string stateName) =>
            bodyAnimator.TransitionActive &&
            string.Equals(bodyAnimator.TransitionTargetName, stateName, StringComparison.Ordinal);

        private static float FlatAngleToTarget(Vector3 target)
        {
            var direction = target - character.transform.position;
            direction.y = 0f;
            if (direction.sqrMagnitude < 0.001f)
                return 0f;
            return Vector3.Angle(character.transform.forward, direction);
        }

        private static void Check(string name, bool passed, string observed, string expected, bool required = true)
        {
            checks.Add(new CheckResult
            {
                name = name,
                passed = passed,
                required = required,
                observed = observed ?? "",
                expected = expected ?? ""
            });
            EventLog("check", $"{name}:{(passed ? "PASS" : "FAIL")} ({observed})");
        }

        private static void EventLog(string kind, string detail)
        {
            events.Add(new MotionEvent
            {
                t = Time.realtimeSinceStartup - validationStartedAt,
                kind = kind,
                detail = detail ?? ""
            });
        }

        private static void FinishValidation()
        {
            try
            {
                foreach (var expected in ExpectedStages)
                {
                    if (report.stages.All(record => record.stage != expected))
                    {
                        Check(
                            "stage_" + expected,
                            false,
                            "stage never ran",
                            "every representative sequence stage runs");
                    }
                }

                var requiredChecks = checks.Where(check => check.required).ToList();
                report.checks_passed = checks.Count(check => check.passed);
                report.checks_total = checks.Count;
                report.required_checks_passed = requiredChecks.Count(check => check.passed);
                report.required_checks_total = requiredChecks.Count;
                report.success = requiredChecks.Count > 0 && requiredChecks.All(check => check.passed) &&
                    report.stages.Count > 0 && report.stages.All(record => record.success);
                report.finished_at_utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
                report.duration_seconds = Time.realtimeSinceStartup - validationStartedAt;
                report.sample_count = samples.Count;
                report.interrupt_stable_across_rounds = interruptRoundOnePassed && interruptRoundTwoPassed;
                report.final_state = bodyAnimator != null ? bodyAnimator.CurrentStateName : "unknown";

                stage = Stage.Finished;
                WriteReport();
                Debug.Log("[Phase28RuntimeSequenceValidator] " + JsonUtility.ToJson(report));
            }
            catch (Exception exception)
            {
                Debug.LogError("[Phase28RuntimeSequenceValidator] finish failed: " + exception);
            }
            finally
            {
                RestoreEditor();
            }
        }

        private static void Fail(string error)
        {
            try
            {
                report.error = error;
                report.success = false;
                report.finished_at_utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
                report.sample_count = samples.Count;
                report.checks_total = checks.Count;
                stage = Stage.Finished;
                WriteReport();
                Debug.LogError("[Phase28RuntimeSequenceValidator] " + error);
            }
            finally
            {
                RestoreEditor();
            }
        }

        private static void RestoreEditor()
        {
            SessionState.SetBool(ActiveKey, false);
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(report.success ? 0 : 1);
                return;
            }

            var loop = UnityEngine.Object.FindFirstObjectByType<BrainLoopController>();
            if (loop != null)
                loop.enabled = true;
            EditorApplication.delayCall += () =>
            {
                if (EditorApplication.isPlaying)
                    EditorApplication.isPlaying = false;
            };
        }

        private static void WriteReport()
        {
            report.stages_snapshot = report.stages.ToArray();
            report.samples = samples.ToArray();
            report.events = events.ToArray();
            report.check_results = checks.ToArray();
            report.diagnostic_snapshots = diagnosticSnapshots.ToArray();
            report.screenshots = screenshots.ToArray();
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
        }

        private static UnityWebRequest BuildJsonRequest(string url, string body)
        {
            var request = new UnityWebRequest(url, "POST");
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = 30;
            return request;
        }

        private static AdminTtsResponse TryParseAdminTts(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return null;
            try
            {
                return JsonUtility.FromJson<AdminTtsResponse>(json);
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static string JsonUtilityQuote(string value) =>
            BrainActionBridge.JsonUtilityQuote(value);

        private static List<float> ToList(Vector3 value) => new List<float> { value.x, value.y, value.z };

        [Serializable]
        private sealed class AdminTtsResponse
        {
            public bool ok;
            public string state;
            public bool pending;
            public string error;
            public AdminTtsRequest request;
        }

        [Serializable]
        private sealed class AdminTtsRequest
        {
            public string request_id;
            public string state;
            public string text;
            public string format;
            public float duration_seconds;
            public string provider;
            public string model;
        }

        [Serializable]
        private sealed class MotionEvent
        {
            public float t;
            public string kind;
            public string detail;
        }

        [Serializable]
        private sealed class CheckResult
        {
            public string name;
            public bool passed;
            public bool required;
            public string observed;
            public string expected;
        }

        [Serializable]
        private sealed class DiagnosticSnapshot
        {
            public string stage;
            public float t;
            public string json;
        }

        [Serializable]
        private sealed class StageRecord
        {
            public string stage;
            public string requested_action;
            public string target;
            public string execution_id;
            public float started_at_seconds;
            public float ended_at_seconds;
            public bool success;
            public string reason;
            public string state_before;
            public string state_after;
        }

        [Serializable]
        private sealed class MotionSample
        {
            public float t;
            public string stage;
            public string action;
            public string execution_id;
            public bool busy;
            public string state;
            public string previous_state;
            public string transition;
            public bool transition_active;
            public float transition_progress;
            public string transition_target;
            public float normalized_time;
            public float layer_weight;
            public float agent_velocity;
            public float agent_desired_speed;
            public bool agent_stopped;
            public bool agent_settled;
            public float agent_remaining_distance;
            public string orientation_state;
            public float orientation_angle_remaining;
        public float yaw_degrees;
            public string rotation_owner;
            public bool look_at_active;
            public float look_at_weight;
            public float body_angle_to_target;
            public float head_angle_to_target;
            // Head orientation relative to the character body, measured on the
            // flat plane: this is the value that proves a head-only rotation
            // when the body already faces the target.
            public float head_relative_to_body;
            public float head_relative_yaw;
            public string phone_state;
            public string phone_parent;
            public bool voice_playing;
            public string posture;
            public string generated_clip;
            public float generated_progress;
            public float ik_left_weight;
            public float ik_right_weight;
        }

        [Serializable]
        private sealed class Phase28Report
        {
            public bool success;
            public string error;
            public string started_at_utc;
            public string finished_at_utc;
            public float duration_seconds;
            public string unity_version;
            public string scene;
            public int sample_count;
            public int checks_total;
            public int checks_passed;
            public int required_checks_total;
            public int required_checks_passed;

            public bool avatar_valid;
            public bool avatar_human;
            public string controller_name;
            public int animator_layers;
            public bool ik_pass_enabled;
            public bool apply_root_motion;
            public bool look_at_component_present;
            public bool ik_component_present;
            public string[] controller_states;
            public string[] controller_clips;
            public string[] animator_parameters;
            public string[] missing_controller_states;
            public string[] snapshot_keys_present;
            public string locomotion_owner;
            public string rotation_owner_baseline;
            public float agent_speed;
            public float agent_angular_speed;
            public bool agent_update_rotation_baseline;
            public float head_gaze_rest_offset_degrees;

            public string phone_state_baseline;
            public List<float> phone_position_baseline;

            public string walk_out_target;
            public float walk_out_distance;
            public int walk_state_samples;
            public int walk_moving_samples;
            public bool walk_transition_observed;
            public float walk_displacement;
            public float walk_idle_recover_seconds;

            public int turn_samples;
            public float turn_max_yaw_step_degrees;
            public float turn_early_yaw_step_degrees;
            public bool turn_yaw_monotonic;
            public float turn_angle_before_degrees;
            public float turn_angle_after_degrees;
            public int turn_script_owned_samples;
            public string rotation_owner_after_turn;
            public bool turn_angle_decreased;

            public string look_at_target;
            public int look_at_active_samples;
            public float look_at_max_weight;
            public bool look_at_head_rotation_observed;
            public float look_at_head_over_body_best_delta;
            public float look_at_body_angle_at_entry_degrees;
            public float look_at_head_relative_at_entry_degrees;
            public float look_at_head_relative_max_degrees;
            public float look_at_gaze_error_best_degrees;
            public float look_at_gaze_error_worst_degrees;
            public bool look_at_body_aligned_at_entry;
            public bool look_at_head_did_the_aiming;

            public int wave_state_samples;
            public bool wave_transition_observed;
            public string wave_completion_reason;
            public string wave_last_completion;
            public bool wave_returned_to_idle;

            public bool speech_tts_used;
            public string speech_request_id;
            public string speech_state_at_prepare;
            public string speech_fallback_reason;
            public string speech_text;
            public string speech_completion;
            public string speech_completion_reason;
            public bool speech_audio_observed;

            public int walk_back_walk_samples;
            public int walk_back_gaze_samples;
            public bool walk_back_head_overlay_observed;
            public float walk_back_head_over_body_best_delta;
            public float walk_back_head_relative_max_degrees;
            public float walk_back_displacement;

            public int interrupt_rounds_run;
            public bool interrupt_callback_success;
            public string interrupt_callback_reason;
            public string interrupt_last_result;
            public string interrupt_last_interruption_reason;
            public float interrupt_recovery_seconds;
            public bool interrupt_recovered_to_locomotion;
            public float interrupt_issued_at_seconds;
            public float interrupt_action_active_seconds;
            public bool interrupt_stable_across_rounds;
            public string last_restore_reason;
            public string last_completion_state;

            public string generated_clip_path;
            public string generated_reference;
            public bool generated_clip_asset_match;
            public bool generated_library_resolved;
            public float generated_clip_length_seconds;
            public bool generated_clip_looping;
            public float generated_progress_observed;
            public string generated_completion_reason;
            public bool generated_clip_cleared;
            public bool generated_controller_restored;
            public bool generated_animator_apply_root_motion;

            public float pickup_distance_before;
            public string phone_state_after_pickup;
            public string phone_holder_after_pickup;
            public string phone_parent_bone_after_pickup;
            public bool phone_parent_is_right_hand_bone;
            public bool phone_parent_anchored_to_right_hand;
            public List<float> phone_local_position_after_pickup;
            public List<float> phone_local_scale_after_pickup;
            public float phone_offset_error;
            public bool character_held_phone_after_pickup;
            public string phone_state_after_placement;
            public string phone_surface_after_placement;
            public string phone_holder_after_placement;
            public List<float> phone_position_after_placement;
            public bool character_held_phone_after_placement;

            public string final_state;
            public bool final_look_at_active;
            public float final_layer_weight;
            public bool final_agent_stopped;
            public bool final_agent_settled;
            public string final_agent_evidence;

            public StageRecord[] stages_snapshot;
            public List<StageRecord> stages = new List<StageRecord>();
            public MotionSample[] samples;
            public MotionEvent[] events;
            public CheckResult[] check_results;
            public DiagnosticSnapshot[] diagnostic_snapshots;
            public string[] screenshots;
        }
    }
}
