using System;
using System.Collections;
using System.Collections.Generic;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using RoomAgent.Livestream;
using RoomAgent.Smartphone;
using RoomAgent.Speech;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Networking;

namespace RoomAgent.Brain
{
    [DisallowMultipleComponent]
    public sealed class BrainActionBridge : MonoBehaviour
    {
        [SerializeField] private LacrimosaController character;
        [SerializeField] private SmartphoneController phone;
        [SerializeField] private LivestreamSystem livestream;
        [SerializeField] private float actionTimeoutSeconds = 20f;
        [SerializeField] private string cameraTargetName = "Camera";
        [SerializeField] private string speechBaseUrl = SpeechServiceClient.DefaultBaseUrl;
        [SerializeField] private int speechTimeoutSeconds = 30;
        [SerializeField] private string streamBaseUrl = StreamIngestClient.DefaultBaseUrl;
        [SerializeField] private bool e2eIsolationEnabled;
        [SerializeField] private string heldSelfieMotionReference =
            "Assets/animation/phase15_held_selfie/held_phone_selfie.fbx";
        [SerializeField] private float heldSelfieCameraDistance = 0.30f;
        [SerializeField] private float heldSelfieTransitionSeconds = 0.25f;

        // Budget for the real transport to reach LIVE. The encoder only reports
        // live once genuine pictures are flowing, so this covers ffmpeg startup.
        [SerializeField] private float streamStartWaitSeconds = 15f;
        [SerializeField] private float streamStopWaitSeconds = 5f;

        // After a failed real start, refuse another attempt for a while so the
        // Brain's stream-maintenance goal cannot respawn the encoder per tick.
        [SerializeField] private float streamRetryCooldownSeconds = 20f;

        // The backend transport is the source of truth for whether a stream is
        // really live. A runtime restart can leave this local flag latched on,
        // which starves the Brain's stream-maintenance goal, so the bridge
        // re-reads /stream/state on a slow clock and repairs the mismatch.
        [SerializeField] private float streamReconcileIntervalSeconds = 5f;
        private const float StreamReconcileTimeoutSeconds = 2f;

        private float nextStreamAttemptAt;
        private float nextStreamReconcileAt;
        private bool streamReconcileInFlight;
        private Transform heldSelfieHandTarget;
        private bool heldSelfiePoseActive;
        private float heldSelfiePoseWeight;

        // How often a playing utterance asks the backend whether it was cancelled.
        private const float SpeechCancelPollSeconds = 0.5f;

        private Transform cameraTarget;
        private SpeechAudioPlayer voice;

        // PHASE 26B: observed gesture evidence for the world snapshot. These
        // are written only where the animation is actually played, so the
        // backend can verify a gesture happened instead of trusting the plan.
        private string lastGesture;
        private string lastGestureAt;
        private bool gestureActive;

        // PHASE 28: one correlation id per executed action plus the honest
        // observed outcome, so a requested transition can never be mistaken for
        // one the runtime really performed.
        private int executionSequence;
        private string activeExecutionId = "none";
        private string requestedActionName = "none";
        private string lastObservedResult = "none";
        private string lastObservedReason = "none";
        private bool interruptRequested;
        private string interruptReason;

        public bool IsBusy { get; private set; }

        public bool HeldSelfiePoseActive => heldSelfiePoseActive;

        public float HeldSelfiePoseWeight => heldSelfiePoseWeight;

        /// <summary>Correlation id of the action currently being executed.</summary>
        public string ActiveExecutionId => activeExecutionId;

        /// <summary>The last observed outcome, as reason:SUCCESS/FAILED.</summary>
        public string LastObservedResult => lastObservedResult;

        public string LastObservedReason => lastObservedReason;

        public string RequestedActionName => requestedActionName;

        public bool IsCharacterMoving =>
            character != null &&
            character.GetComponent<LacrimosaMovement>() is { } movement &&
            movement.IsMoving;

        public event Action<string> Log;

        public static BrainActionBridge FindOrCreate()
        {
            var existing = FindFirstObjectByType<BrainActionBridge>();
            if (existing != null)
                return existing;

            var host = new GameObject("BrainActionBridge");
            var bridge = host.AddComponent<BrainActionBridge>();
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(host, scene);
            return bridge;
        }

        public void ResolveReferences()
        {
            character = character != null ? character : FindFirstObjectByType<LacrimosaController>();
            phone = phone != null ? phone : FindFirstObjectByType<SmartphoneController>();
            livestream = livestream != null ? livestream : FindFirstObjectByType<LivestreamSystem>();
            cameraTarget = GameObject.Find(cameraTargetName)?.transform;
            if (livestream != null && livestream.ActiveCamera != null)
                cameraTarget = livestream.ActiveCamera.transform;
            voice = voice != null ? voice : SpeechAudioPlayer.GetOrCreate(
                character != null ? character.transform : null);
        }

        public bool Execute(string action, string target, string text, Action<bool> callback)
            => Execute(action, target, text, null, callback);

        public bool Execute(
            string action,
            string target,
            string text,
            string speechRequestId,
            Action<bool> callback)
        {
            if (IsBusy)
                return false;

            if (string.IsNullOrWhiteSpace(action))
            {
                Complete(callback, false, "empty_action");
                return true;
            }

            // E2E isolation must reject before reference resolution or any
            // action switch can run. In test mode an empty, malformed, or
            // production-equivalent stream URL fails closed, and stream-control
            // actions are denied until a safe test dependency exists.
            if (!BrainIsolationPolicy.TryAuthorize(
                    e2eIsolationEnabled,
                    streamBaseUrl,
                    action,
                    out var isolationReason))
            {
                Complete(callback, false, isolationReason);
                return true;
            }

            ResolveReferences();
            if (character == null || phone == null || livestream == null)
            {
                Complete(callback, false, "references_missing");
                return true;
            }

            // PHASE 28: the bridge owns exactly one action at a time. The flag
            // guards the Brain loop and is what Interrupt() reports on, so every
            // path below either clears it on the spot or through Complete().
            IsBusy = true;

            executionSequence++;
            activeExecutionId = $"exec{executionSequence:0000}";
            requestedActionName = action.ToUpperInvariant();
            interruptRequested = false;
            interruptReason = null;

            switch (action.ToUpperInvariant())
            {
                case "MOVE_TO":
                case "TURN_TO":
                case "LOOK_AT":
                    var point = ResolveTargetPoint(target);
                    if (!point.HasValue)
                    {
                        Complete(callback, false, "target_unresolved");
                        return true;
                    }

                    if (action.Equals("MOVE_TO", StringComparison.OrdinalIgnoreCase))
                    {
                        if (!character.MoveTo(point.Value))
                        {
                            Complete(callback, false, "move_unreachable");
                            return true;
                        }
                        StartCoroutine(WaitUntilIdle(callback));
                    }
                    else if (action.Equals("TURN_TO", StringComparison.OrdinalIgnoreCase))
                    {
                        if (!character.TurnTo(point.Value))
                        {
                            Complete(callback, false, "turn_unreachable");
                            return true;
                        }
                        StartCoroutine(WaitUntilNotTurning(callback));
                    }
                    else
                    {
                        character.LookAt(point.Value);
                        Complete(callback, true, "look_at");
                    }
                    return true;

                case "STOP":
                    character.Stop();
                    character.ClearLookAt();
                    Complete(callback, true, "stop");
                    return true;

                case "SPEAK":
                    var safeText = string.IsNullOrWhiteSpace(text) ? "..." : text.Trim();
                    if (string.IsNullOrWhiteSpace(speechRequestId))
                    {
                        // No speech request attached: the text still counts as
                        // delivered so TTS being off can never stall the Brain.
                        Debug.Log($"[BrainActionBridge] SPEAK: {safeText}");
                        Complete(callback, true, "speak_text_only");
                        return true;
                    }

                    StartCoroutine(RunSpeech(speechRequestId, safeText, callback));
                    return true;

                case "WAVE":
                    if (!character.PlayWave())
                    {
                        Complete(callback, false, "wave_unavailable");
                        return true;
                    }
                    MarkGesture("wave");
                    StartCoroutine(WaitForActionState("Wave", WithGesture("wave", callback)));
                    return true;

                case "DANCE":
                    if (string.IsNullOrWhiteSpace(target))
                    {
                        if (!character.PlayDance())
                        {
                            Complete(callback, false, "dance_unavailable");
                            return true;
                        }
                        MarkGesture("dance");
                        StartCoroutine(WaitForActionState("Dance", WithGesture("dance", callback)));
                        return true;
                    }

                    // A non-empty DANCE target is a motion reference, never a
                    // spatial name. Every GeneratedMotionLibrary entry is an
                    // Assets/ path, so anything else is an invalid reference
                    // and must not be reported as an unresolved generated
                    // motion.
                    if (!LooksLikeMotionReference(target))
                    {
                        Complete(callback, false, "motion_reference_invalid");
                        return true;
                    }

                    var generatedClip = ResolveGeneratedClip(target);
                    if (generatedClip == null)
                    {
                        Complete(callback, false, "generated_motion_unresolved");
                        return true;
                    }
                    var heldSelfie = IsHeldSelfieReference(target);
                    if (heldSelfie && phone.CurrentState != SmartphoneState.Held)
                    {
                        Complete(callback, false, "selfie_phone_not_held");
                        return true;
                    }
                    if (heldSelfie)
                        BeginHeldSelfiePose();
                    if (!character.PlayGeneratedClip(generatedClip))
                    {
                        if (heldSelfie)
                            EndHeldSelfiePose();
                        Complete(callback, false, "generated_motion_playback_failed");
                        return true;
                    }
                    MarkGesture("dance");
                    var completed = WithGesture("dance", callback);
                    if (heldSelfie)
                        completed = WithHeldSelfiePose(completed);
                    StartCoroutine(WaitForGeneratedAction(generatedClip, completed));
                    return true;

                case "RUN":
                    var runPoint = ResolveTargetPoint(target);
                    if (runPoint.HasValue && character.MoveTo(runPoint.Value, true))
                        StartCoroutine(WaitUntilIdle(callback));
                    else
                        Complete(callback, false, "run_unreachable");
                    return true;

                case "PICK_UP_PHONE":
                    StartCoroutine(WalkThenPickUp(callback));
                    return true;

                case "PLACE_PHONE":
                    StartCoroutine(MoveThenPlace(target, callback));
                    return true;

                case "START_STREAM":
                    if (!livestream.StartStream())
                    {
                        Complete(callback, false, "start_stream_unavailable");
                        return true;
                    }
                    StartCoroutine(RunRealStreamStart(callback));
                    return true;

                case "STOP_STREAM":
                    var stopped = livestream.StopStream();
                    // The logical stop is what arms the Brain's stop latch, so it is
                    // reported immediately; the transport teardown continues quietly
                    // in the background and can never undo that latch.
                    Complete(callback, stopped, "stop_stream_logical");
                    StartCoroutine(RunRealStreamStop());
                    return true;

                case "SET_CAMERA":
                    Complete(callback, livestream.SetCamera(livestream.ActiveCamera), "set_camera");
                    return true;

                default:
                    Complete(callback, false, "unsupported_action");
                    return true;
            }
        }

        private void Update()
        {
            UpdateHeldSelfiePose();

            if (streamReconcileInFlight || Time.time < nextStreamReconcileAt)
                return;

            nextStreamReconcileAt = Time.time + Mathf.Max(1f, streamReconcileIntervalSeconds);
            if (IsBusy)
                return;

            // While a real start/stop handshake owns the transport the two states
            // are mid-transition, and that path already reports its own outcome.
            // A stale capture pump from before a restart keeps IsCapturing true
            // without any handshake, so that flag is deliberately not a guard.
            var ingest = StreamIngestClient.Instance;
            if (ingest != null && ingest.ControlPending)
                return;

            StartCoroutine(ReconcileStreamState());
        }

        /// <summary>
        /// Repairs a local stream flag that went stale while nobody was driving
        /// the transport, such as after a backend restart. Only the "transport is
        /// not live" direction is applied: a live transport with a stopped local
        /// flag is left to the Brain's stream-maintenance goal, so Unity never
        /// claims a stream it is not actually feeding.
        /// </summary>
        private IEnumerator ReconcileStreamState()
        {
            streamReconcileInFlight = true;
            try
            {
                ResolveReferences();
                if (livestream == null || string.IsNullOrEmpty(streamBaseUrl))
                    yield break;

                using (var request = UnityWebRequest.Get($"{streamBaseUrl}/stream/state"))
                {
                    request.timeout = Mathf.CeilToInt(StreamReconcileTimeoutSeconds);
                    yield return request.SendWebRequest();
                    if (request.result != UnityWebRequest.Result.Success)
                        yield break;

                    var transport = ParseTransportState(request.downloadHandler?.text);
                    if (transport == null || string.IsNullOrWhiteSpace(transport.state))
                        yield break;

                    // Simulated livestream mode (PHASE 04-12 contract) reports
                    // STOPPED with no real transport configured; reconciling that
                    // would tear down a stream that is behaving correctly.
                    if (!transport.enabled || !transport.transport_enabled)
                        yield break;

                    var backendLive = string.Equals(
                        transport.state,
                        "LIVE",
                        StringComparison.OrdinalIgnoreCase);
                    if (livestream.ApplyTransportState(backendLive))
                    {
                        // The backend is not live, so the stale capture pump is
                        // only feeding refused frames. Stopping the pump is local;
                        // /stream/stop is deliberately not called because it would
                        // arm the intentional-stop latch and suppress the Brain's
                        // stream-maintenance goal.
                        var stalePump = StreamIngestClient.Instance;
                        if (stalePump != null && !IsBusy && !stalePump.ControlPending)
                            stalePump.StopPump();

                        EmitLog(
                            $"stream state reconciled: backend={transport.state} " +
                            "is_streaming=false");
                    }
                }
            }
            finally
            {
                streamReconcileInFlight = false;
            }
        }

        private static StreamTransportState ParseTransportState(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                return null;

            try
            {
                return JsonUtility.FromJson<StreamTransportState>(json);
            }
            catch (Exception)
            {
                return null;
            }
        }

        public WorldSnapshot BuildSnapshot()
        {
            ResolveReferences();
            var stream = livestream.GetStreamState();
            var cameraTransform = livestream.ActiveCamera != null ? livestream.ActiveCamera.transform : null;
            var targets = new Dictionary<string, TargetSnapshot>();
            AddTarget(targets, "CENTER", FindScenePoint("Center"), true, true);
            AddTarget(targets, "TABLE", FindColliderPoint("Table"), true, true);
            AddTarget(targets, "BENCH", FindColliderPoint("Seat"), true, true);
            AddTarget(targets, "FLOOR", new Vector3(0f, 0f, 0f), true, true);
            AddTarget(targets, "PHONE", phone.transform.position, false, true);
            AddTarget(
                targets,
                "CAMERA",
                cameraTransform != null ? cameraTransform.position : Vector3.zero,
                false,
                true);
            // PHASE 25: the anchor-relative standing points the backend plans
            // MOVE_TO against. They are only published when the navmesh actually
            // accepts them, so a request for one can never invent a position the
            // character cannot walk to.
            AddStandTarget(
                targets,
                "CAMERA_STAND",
                cameraTransform != null ? cameraTransform.position : Vector3.zero,
                cameraTransform != null ? cameraTransform.forward : Vector3.forward,
                Vector3.forward);
            AddStandTarget(
                targets,
                "PHONE_STAND",
                phone.transform.position,
                phone.transform.forward,
                cameraTransform != null ? cameraTransform.forward : Vector3.forward);

            return new WorldSnapshot
            {
                character = BuildCharacterSnapshot(),
                phone = new PhoneSnapshot
                {
                    state = phone.CurrentState.ToString(),
                    position = ToList(phone.transform.position),
                    holder = phone.Holder != null ? phone.Holder.name : null,
                    placement_surface = PlacementSurfaceName(phone.PlacementSurface)
                },
                stream = new StreamSnapshot
                {
                    is_streaming = stream.IsStreaming,
                    camera_available = stream.HasCamera,
                    camera_active = stream.IsCameraActive,
                    duration_seconds = stream.DurationSeconds,
                    viewer_count = stream.ViewerCount,
                    camera_position = stream.HasCamera ? ToList(stream.CameraPosition) : null,
                    camera_rotation = stream.HasCamera ? ToList(stream.CameraRotationEuler) : null,
                    camera_forward = stream.HasCamera && cameraTransform != null
                        ? ToList(cameraTransform.forward)
                        : (stream.HasCamera ? ToList(ForwardFromEuler(stream.CameraRotationEuler)) : null),
                    camera_field_of_view = stream.HasCamera ? stream.FieldOfView : 0f
                },
                targets = targets
            };
        }

        /// <summary>
        /// PHASE 28: the character read-model. The motion fields are read back
        /// from the Animator, NavMeshAgent and IK components every time, so the
        /// diagnostic snapshot describes what the runtime did rather than what
        /// the backend asked for.
        /// </summary>
        private CharacterSnapshot BuildCharacterSnapshot()
        {
            var bodyAnimator = character.BodyAnimator;
            var animator = bodyAnimator != null ? bodyAnimator.Animator : null;
            var movement = character.GetComponent<LacrimosaMovement>();
            var agent = movement != null ? movement.Agent : null;
            var lookAt = character.GetComponent<LacrimosaLookAt>();
            var ik = character.GetComponent<LacrimosaIK>();

            return new CharacterSnapshot
            {
                position = ToList(character.transform.position),
                forward = ToList(character.transform.forward),
                rotation_y = character.transform.eulerAngles.y,
                posture = "standing",
                current_action = DescribeAction(bodyAnimator, agent),
                last_gesture = lastGesture,
                last_gesture_at = lastGestureAt,
                gesture_active = gestureActive,
                animator_state = bodyAnimator != null ? bodyAnimator.CurrentStateName : "unknown",
                animator_previous_state = bodyAnimator != null ? bodyAnimator.PreviousStateName : "unknown",
                animator_requested_action = bodyAnimator != null ? bodyAnimator.RequestedAction : "none",
                animator_transition_active = bodyAnimator != null && bodyAnimator.TransitionActive,
                animator_transition_progress = bodyAnimator != null ? bodyAnimator.TransitionProgress : 0f,
                animator_transition_target = bodyAnimator != null ? bodyAnimator.TransitionTargetName : "none",
                animator_state_normalized_time = bodyAnimator != null ? bodyAnimator.CurrentStateNormalizedTime : 0f,
                animator_base_layer_weight = bodyAnimator != null ? bodyAnimator.BaseLayerWeight : 0f,
                // PHASE 28 root-motion ownership: the Animator never drives
                // translation, the NavMeshAgent does. Reported, not assumed.
                root_motion_apply = animator != null && animator.applyRootMotion,
                locomotion_owner = "navmesh_agent",
                agent_velocity = agent != null ? agent.velocity.magnitude : 0f,
                agent_desired_speed = agent != null ? agent.desiredVelocity.magnitude : 0f,
                // NavMeshAgent.isStopped logs an error and returns false while the
                // agent is off the NavMesh (spawn/teardown frames), so read it only
                // when the agent is actually placed. Off-mesh counts as not moving.
                agent_is_stopped = agent == null || !agent.isOnNavMesh || agent.isStopped,
                agent_destination = agent != null && agent.hasPath ? ToList(agent.destination) : null,
                orientation_state = character.OrientationState,
                orientation_target = character.IsTurning ? ToList(character.OrientationTarget) : null,
                orientation_angle_remaining = character.OrientationAngleRemaining,
                rotation_owner = character.RotationOwner,
                look_at_active = lookAt != null && lookAt.IsActive,
                look_at_target = lookAt != null && lookAt.IsActive ? ToList(lookAt.TargetPosition) : null,
                ik_left_hand_target = ik != null && ik.LeftHandTarget != null ? ik.LeftHandTarget.name : null,
                ik_right_hand_target = ik != null && ik.RightHandTarget != null ? ik.RightHandTarget.name : null,
                ik_left_hand_weight = ik != null ? ik.LeftHandWeight : 0f,
                ik_right_hand_weight = ik != null ? ik.RightHandWeight : 0f,
                motion_reference = bodyAnimator == null
                    ? "unknown"
                    : bodyAnimator.CurrentGeneratedClip != null
                        ? bodyAnimator.GeneratedClipName
                        : bodyAnimator.CurrentStateName,
                animation_completion_reason = bodyAnimator != null ? bodyAnimator.LastCompletionReason : "none",
                animation_last_result = bodyAnimator != null ? bodyAnimator.LastCompletion : "none",
                last_action_result = lastObservedResult,
                execution_id = IsBusy ? activeExecutionId : "idle"
            };
        }

        /// <summary>
        /// What the character is really doing, using the vocabulary the backend
        /// already reads (social attention treats these as busy). An action that
        /// the Animator actually entered wins over the movement summary.
        /// </summary>
        private string DescribeAction(LacrimosaBodyAnimator bodyAnimator, NavMeshAgent agent)
        {
            var observation = bodyAnimator != null ? bodyAnimator.ActiveAction : null;
            if (observation != null && !observation.finished)
            {
                switch (observation.trigger)
                {
                    case "Wave":
                        return "waving";
                    case "Dance":
                        return "dancing";
                    case "TurnLeft":
                    case "TurnRight":
                        return "turning";
                    case "Look":
                        return "looking";
                }
            }

            if (character.IsTurning)
                return "turning";

            var speed = agent != null ? agent.velocity.magnitude : 0f;
            if (speed > 0.1f)
                return speed >= 1.8f ? "running" : "walking";

            return character.HeldPhone != null ? "holding_phone" : "idle";
        }

        public string BuildSnapshotJson()
        {
            var snapshot = BuildSnapshot();
            var targetBuilder = new System.Text.StringBuilder();
            foreach (var pair in snapshot.targets)
            {
                if (targetBuilder.Length > 0)
                    targetBuilder.Append(',');
                var target = JsonUtility.ToJson(pair.Value);
                targetBuilder.Append(JsonUtilityQuote(pair.Key)).Append(':').Append(target);
            }

            var holder = snapshot.phone.holder != null
                ? JsonUtilityQuote(snapshot.phone.holder)
                : "null";
            var surface = snapshot.phone.placement_surface != null
                ? JsonUtilityQuote(snapshot.phone.placement_surface)
                : "null";
            return $"{{\"character\":{JsonUtility.ToJson(snapshot.character)}," +
                   $"\"phone\":{{\"state\":{JsonUtilityQuote(snapshot.phone.state)}," +
                   $"\"position\":{FloatListToJson(snapshot.phone.position)}," +
                   $"\"holder\":{holder},\"placement_surface\":{surface}}}," +
                   $"\"stream\":{JsonUtility.ToJson(snapshot.stream)}," +
                   $"\"targets\":{{{targetBuilder}}}}}";
        }

        public static string JsonUtilityQuote(string value)
        {
            var safeValue = value ?? string.Empty;
            return "\"" + safeValue
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n")
                .Replace("\t", "\\t") + "\"";
        }

        private static string FloatListToJson(List<float> values)
        {
            var safeValues = values ?? new List<float>();
            var builder = new System.Text.StringBuilder("[");
            for (var index = 0; index < safeValues.Count; index++)
            {
                if (index > 0)
                    builder.Append(',');
                builder.Append(safeValues[index].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            }
            return builder.Append(']').ToString();
        }

        public bool SubmitChat(string viewerId, string text)
        {
            return BrainChatSubmitter.Submit(viewerId, text);
        }

        /// <summary>
        /// Waits for the backend synthesiser, then plays the returned audio.
        /// Guarantees exactly one callback so the Brain loop can never hang.
        /// </summary>
        private IEnumerator RunSpeech(string requestId, string safeText, Action<bool> callback)
        {
            var deadline = Time.realtimeSinceStartup + Mathf.Max(1, speechTimeoutSeconds);
            SpeechStatePayload payload = null;
            while (Time.realtimeSinceStartup < deadline)
            {
                if (interruptRequested)
                {
                    EmitLog($"SPEAK:{requestId} interrupted before playback");
                    Complete(callback, false, interruptReason);
                    yield break;
                }
                SpeechStatePayload fetched = null;
                yield return SpeechServiceClient.FetchState(
                    speechBaseUrl, requestId, 5, result => fetched = result);

                if (fetched == null)
                {
                    yield return new WaitForSecondsRealtime(0.25f);
                    continue;
                }

                if (fetched.state == "READY")
                {
                    payload = fetched;
                    break;
                }

                if (fetched.state != "SPEECH_REQUESTED" &&
                    fetched.state != "SYNTHESIZING" &&
                    fetched.state != "PLAYING" &&
                    fetched.state != "COMPLETED")
                {
                    EmitLog($"SPEAK:{requestId} {fetched.state.ToLowerInvariant()}:{fetched.error ?? "-"}");
                    Complete(callback, false, "speech_unavailable");
                    yield break;
                }

                yield return new WaitForSecondsRealtime(0.25f);
            }

            if (payload == null)
            {
                EmitLog($"SPEAK:{requestId} not ready within {speechTimeoutSeconds}s");
                Complete(callback, false, "speech_timeout");
                yield break;
            }

            var playable = false;
            yield return SpeechServiceClient.PostFlag(
                speechBaseUrl, "playing", requestId, 5, accepted => playable = accepted);
            if (!playable)
            {
                // A refused claim means the utterance already left the playable
                // states, so stay silent instead of speaking a cancelled line.
                SpeechStatePayload live = null;
                yield return SpeechServiceClient.FetchState(
                    speechBaseUrl, requestId, 5, result => live = result);
                var state = live != null ? live.state : "UNKNOWN";
                EmitLog($"SPEAK:{requestId} not playable:{state}");
                var intentional = state == "CANCELLED";
                Complete(callback, intentional,
                    intentional ? "speech_cancelled" : "speech_not_playable");
                yield break;
            }

            AudioClip clip = null;
            yield return SpeechServiceClient.DownloadClip(
                speechBaseUrl,
                requestId,
                SpeechServiceClient.ResolveAudioType(payload.format),
                speechTimeoutSeconds,
                result => clip = result);

            voice = voice != null ? voice : SpeechAudioPlayer.GetOrCreate(
                character != null ? character.transform : null);
            if (voice == null || !voice.Play(clip))
            {
                EmitLog($"SPEAK:{requestId} decode failed:{SpeechAudioPlayer.Describe(clip)}");
                yield return SpeechServiceClient.PostFlag(
                    speechBaseUrl, "cancel", requestId, 5, accepted => { });
                Complete(callback, false, "speech_playback_failed");
                yield break;
            }

            var duration = Mathf.Max(0.2f, clip.length);
            var guard = duration + 2f;
            var played = 0f;
            var nextCancelCheck = 0f;
            var cancelled = false;
            while (voice.IsPlaying && played < guard)
            {
                played += Time.unscaledDeltaTime;
                if (interruptRequested)
                {
                    voice.Stop();
                    EmitLog($"SPEAK:{requestId} interrupted during playback");
                    Complete(callback, false, interruptReason);
                    yield break;
                }
                if (Time.realtimeSinceStartup >= nextCancelCheck)
                {
                    // Cooperative stop: a cancel asked of the backend only
                    // silences Lacrimosa if the player ever goes looking for it.
                    nextCancelCheck = Time.realtimeSinceStartup + SpeechCancelPollSeconds;
                    SpeechStatePayload live = null;
                    yield return SpeechServiceClient.FetchState(
                        speechBaseUrl, requestId, 5, result => live = result);
                    if (live != null && live.state == "CANCELLED")
                    {
                        cancelled = true;
                        break;
                    }
                }

                yield return null;
            }

            if (cancelled || played >= guard)
                voice.Stop();

            if (cancelled)
            {
                // An interruption is an intentional stop, not a fault, so the
                // Brain is told it ran and moves on to the next step.
                EmitLog($"SPEAK:{requestId} cancelled during playback");
                Complete(callback, true, "speech_cancelled");
                yield break;
            }

            yield return SpeechServiceClient.PostFlag(
                speechBaseUrl, "completed", requestId, 5, accepted => { });
            EmitLog($"SPEAK:{requestId} \"{safeText}\" {SpeechAudioPlayer.Describe(clip)}");
            Complete(callback, true, "speech");
        }

        /// <summary>
        /// Brings the PHASE 13 transport up on top of a stream that already made
        /// sense logically. When no real transport is configured the action keeps
        /// the PHASE 04 behaviour: the virtual stream alone is a success.
        /// </summary>
        private IEnumerator RunRealStreamStart(Action<bool> callback)
        {
            if (Time.realtimeSinceStartup < nextStreamAttemptAt)
            {
                livestream.StopStream();
                EmitLog("START_STREAM: transport cooling down");
                Complete(callback, false, "start_stream_backoff");
                yield break;
            }

            if (!StreamIngestClient.TryHost(streamBaseUrl, out _, out _))
            {
                EmitLog($"START_STREAM: unusable stream endpoint {streamBaseUrl}");
                Complete(callback, true, "start_stream");
                yield break;
            }

            var ingest = StreamIngestClient.GetOrCreate();
            if (!ingest.BeginPump(livestream.ActiveCamera, streamBaseUrl))
            {
                livestream.StopStream();
                Complete(callback, false, "start_stream_capture_unavailable");
                yield break;
            }

            var profileDeadline = Time.realtimeSinceStartup + StreamProbeSeconds;
            while (ingest.ControlPending && Time.realtimeSinceStartup < profileDeadline)
                yield return null;

            if (ingest.ControlPending)
            {
                FailStart(ingest, "profile_timeout");
                Complete(callback, false, "start_stream");
                yield break;
            }

            if (!ingest.ProfileReady || !ingest.TransportConfigured)
            {
                // Simulated livestream only: exactly the PHASE 04-12 contract.
                ingest.StopPump();
                StreamIngestClient.DestroyInstance();
                EmitLog("START_STREAM: simulated only (no real transport)");
                Complete(callback, true, "start_stream");
                yield break;
            }

            ingest.RequestStart(streamStartWaitSeconds);
            var liveDeadline = Time.realtimeSinceStartup + streamStartWaitSeconds + StreamProbeSeconds;
            while (ingest.ControlPending && Time.realtimeSinceStartup < liveDeadline)
            {
                // Capture stays armed while the transport negotiates.
                ingest.CaptureNow();
                yield return null;
            }

            if (!ingest.ControlPending && ingest.ControlOk)
            {
                EmitLog(
                    $"START_STREAM: LIVE {ingest.Width}x{ingest.Height}@{ingest.Fps:0.#} " +
                    $"frames={ingest.FramesPosted} audio={ingest.AudioPosted}");
                nextStreamAttemptAt = 0f;
                Complete(callback, true, "start_stream_live");
                yield break;
            }

            FailStart(ingest, ingest.ControlReason);
            Complete(callback, false, "start_stream");
        }

        private IEnumerator RunRealStreamStop()
        {
            var ingest = StreamIngestClient.Instance;
            if (ingest == null)
                yield break;

            ingest.RequestStop(streamStopWaitSeconds);
            var deadline = Time.realtimeSinceStartup + streamStopWaitSeconds + StreamProbeSeconds;
            while (ingest != null && ingest.ControlPending && Time.realtimeSinceStartup < deadline)
                yield return null;

            if (ingest == null)
                yield break;

            EmitLog($"STOP_STREAM: transport={ingest.ControlState} frames={ingest.FramesPosted}");
            ingest.Shutdown();
        }

        /// <summary>
        /// Tears a failed start down completely: no encoder, no capture pump, no
        /// logical stream, plus a cooldown so a retry cannot become a respawn loop.
        /// </summary>
        private void FailStart(StreamIngestClient ingest, string reason)
        {
            EmitLog($"START_STREAM failed: {reason ?? "-"}");
            nextStreamAttemptAt = Time.realtimeSinceStartup + streamRetryCooldownSeconds;
            if (ingest != null)
            {
                ingest.RequestStop(2f);
                ingest.StopPump();
                ingest.Shutdown();
            }
            livestream.StopStream();
        }

        private const float StreamProbeSeconds = 6f;

        private IEnumerator WalkThenPickUp(Action<bool> callback)
        {
            if (phone.CurrentState == SmartphoneState.Held)
            {
                Complete(callback, true, "phone_already_held");
                yield break;
            }

            if (!character.MoveTo(phone.transform.position))
            {
                Complete(callback, false, "phone_unreachable");
                yield break;
            }

            // PHASE 25: same one-frame settle as WaitUntilIdle so the pickup
            // is not attempted before the walk has registered.
            yield return null;
            var elapsed = 0f;
            while (IsCharacterMoving && elapsed < actionTimeoutSeconds)
            {
                if (interruptRequested)
                {
                    character.Stop(interruptReason);
                    Complete(callback, false, interruptReason);
                    yield break;
                }
                elapsed += Time.deltaTime;
                yield return null;
            }

            if (elapsed >= actionTimeoutSeconds)
            {
                character.Stop();
                Complete(callback, false, "pickup_timeout");
                yield break;
            }

            Complete(callback, character.PickUp(phone), "pickup");
        }

        private IEnumerator MoveThenPlace(string target, Action<bool> callback)
        {
            if (!character.HeldPhone)
            {
                Complete(callback, false, "phone_not_held");
                yield break;
            }

            var placementDirection = ResolvePlacementDirection();
            var destination = ResolvePlacementPoint(target, placementDirection);
            if (!destination.HasValue)
            {
                Complete(callback, false, "placement_target_invalid");
                yield break;
            }

            if (!character.MoveTo(destination.Value))
            {
                Complete(callback, false, "placement_unreachable");
                yield break;
            }

            // PHASE 25: same one-frame settle as WaitUntilIdle so the placement
            // is not attempted before the walk has registered.
            yield return null;
            var elapsed = 0f;
            while (IsCharacterMoving && elapsed < actionTimeoutSeconds)
            {
                if (interruptRequested)
                {
                    character.Stop(interruptReason);
                    Complete(callback, false, interruptReason);
                    yield break;
                }
                elapsed += Time.deltaTime;
                yield return null;
            }

            if (elapsed >= actionTimeoutSeconds)
            {
                character.Stop();
                Complete(callback, false, "placement_timeout");
                yield break;
            }

            Complete(
                callback,
                character.PlaceAt(destination.Value, ResolvePlacementYaw(placementDirection)),
                "placement");
        }

        private IEnumerator WaitUntilIdle(Action<bool> callback)
        {
            // PHASE 25: MoveTo registers the destination on the next agent
            // update, so IsMoving still reads false in this frame; without this
            // settle the walk reports success before it has started.
            yield return null;
            var elapsed = 0f;
            while (IsCharacterMoving && elapsed < actionTimeoutSeconds)
            {
                if (interruptRequested)
                {
                    character.Stop(interruptReason);
                    Complete(callback, false, interruptReason);
                    yield break;
                }
                elapsed += Time.deltaTime;
                yield return null;
            }

            if (elapsed >= actionTimeoutSeconds)
                character.Stop();

            Complete(callback, elapsed < actionTimeoutSeconds, "move");
        }

        private IEnumerator WaitUntilNotTurning(Action<bool> callback)
        {
            var elapsed = 0f;
            while (character.IsTurning && elapsed < actionTimeoutSeconds)
            {
                if (interruptRequested)
                {
                    character.Stop(interruptReason);
                    Complete(callback, false, interruptReason);
                    yield break;
                }
                elapsed += Time.deltaTime;
                yield return null;
            }
            Complete(callback, elapsed < actionTimeoutSeconds, "turn");
        }

        /// <summary>
        /// PHASE 28: waits for the Animator to really end a controller-authored
        /// action instead of trusting a fixed timer. The budget comes from the
        /// clip the production controller actually contains, and a looping
        /// action clip (which has no natural end) is closed after one observed
        /// full pass. Interruptions are reported as failures, never as success.
        /// </summary>
        private IEnumerator WaitForActionState(string trigger, Action<bool> callback)
        {
            var clipLength = character.ResolveActionStateLength(trigger);
            var budget = Mathf.Min(
                clipLength > 0f ? clipLength + 1f : 5f,
                Mathf.Max(1f, actionTimeoutSeconds));
            // Only a state that is still active this long past its clip length
            // can be a looping clip; a controller-authored exit wins the race.
            var passCompleteAt = clipLength + 0.25f;
            var elapsed = 0f;

            while (elapsed < budget)
            {
                if (interruptRequested)
                {
                    character.Stop(interruptReason);
                    Complete(callback, false, interruptReason);
                    yield break;
                }

                var observation = character.ActiveAction;
                if (observation == null ||
                    !string.Equals(observation.trigger, trigger, StringComparison.Ordinal))
                {
                    Complete(callback, false, "action_observation_lost");
                    yield break;
                }

                if (observation.finished)
                {
                    var completed = observation.completed &&
                        string.Equals(observation.reason, "animation_completed", StringComparison.Ordinal);
                    if (!completed)
                        character.RestoreLocomotion(observation.reason);
                    Complete(callback, completed, observation.reason);
                    yield break;
                }

                var activeFor = observation.started ? Time.time - observation.startedAt : 0f;
                if (clipLength > 0f && activeFor >= passCompleteAt)
                {
                    character.CompleteAction("action_clip_pass_completed");
                    Complete(callback, true, "action_clip_pass_completed");
                    yield break;
                }

                elapsed += Time.deltaTime;
                yield return null;
            }

            character.Stop("action_timeout");
            Complete(callback, false, "action_timeout");
        }

        private IEnumerator WaitForGeneratedAction(AnimationClip generatedClip, Action<bool> callback)
        {
            var clipLength = Mathf.Max(0.05f, generatedClip.length);
            var budget = Mathf.Min(clipLength + 1f, Mathf.Max(1f, actionTimeoutSeconds));
            var elapsed = 0f;
            var sawState = false;

            while (elapsed < budget)
            {
                if (interruptRequested)
                {
                    character.Stop(interruptReason);
                    Complete(callback, false, interruptReason);
                    yield break;
                }

                var tracking = character.TryGetGeneratedClipProgress(
                    generatedClip,
                    out var normalizedTime,
                    out var stateActive);
                if (tracking && stateActive)
                {
                    sawState = true;
                    // The registered clip loops, so one full observed pass is
                    // the honest end of this production motion.
                    if (normalizedTime >= 1f)
                    {
                        character.StopGeneratedClip("completed");
                        character.RestoreLocomotion("generated_motion_completed");
                        Complete(callback, true, "generated_motion_completed");
                        yield break;
                    }
                }
                else if (sawState)
                {
                    character.RestoreLocomotion("generated_motion_interrupted");
                    Complete(callback, false, "generated_motion_interrupted");
                    yield break;
                }
                else if (elapsed > 2f)
                {
                    character.Stop("generated_motion_never_started");
                    Complete(callback, false, "generated_motion_never_started");
                    yield break;
                }

                elapsed += Time.deltaTime;
                yield return null;
            }

            character.Stop("generated_motion_timeout");
            Complete(callback, false, "generated_motion_timeout");
        }

        /// <summary>
        /// PHASE 28: cancels the action currently being executed. The running
        /// coroutine observes the request, the Animator is returned to a valid
        /// locomotion state and the pending callback fires exactly once with a
        /// failure carrying the interruption reason.
        /// </summary>
        public bool Interrupt(string reason)
        {
            if (!IsBusy)
                return false;

            interruptRequested = true;
            interruptReason = string.IsNullOrWhiteSpace(reason) ? "interrupted" : reason;
            ResolveReferences();
            if (character != null)
                character.Stop(interruptReason);
            EmitLog($"INTERRUPT requested: {interruptReason}");
            return true;
        }

        /// <summary>Only a gesture that really ran may clear its own evidence.</summary>
        private Action<bool> WithGesture(string gesture, Action<bool> callback)
        {
            return success =>
            {
                ClearGesture(gesture);
                callback?.Invoke(success);
            };
        }

        private Action<bool> WithHeldSelfiePose(Action<bool> callback)
        {
            return success =>
            {
                EndHeldSelfiePose();
                callback?.Invoke(success);
            };
        }

        private bool IsHeldSelfieReference(string reference)
        {
            return string.Equals(
                (reference ?? string.Empty).Trim().Replace('\\', '/'),
                heldSelfieMotionReference.Trim().Replace('\\', '/'),
                StringComparison.OrdinalIgnoreCase);
        }

        private void BeginHeldSelfiePose()
        {
            if (heldSelfieHandTarget == null)
            {
                var targetObject = new GameObject("HeldPhoneSelfieHandTarget");
                targetObject.hideFlags = HideFlags.DontSave;
                heldSelfieHandTarget = targetObject.transform;
            }
            heldSelfiePoseActive = true;
            EmitLog("held selfie pose active");
        }

        private void EndHeldSelfiePose()
        {
            if (heldSelfiePoseActive)
                EmitLog("held selfie pose ending");
            heldSelfiePoseActive = false;
        }

        private void UpdateHeldSelfiePose()
        {
            var ik = character != null ? character.GetComponent<LacrimosaIK>() : null;
            if (ik == null || heldSelfieHandTarget == null)
                return;

            var targetWeight = heldSelfiePoseActive ? 1f : 0f;
            var transition = Mathf.Max(0.05f, heldSelfieTransitionSeconds);
            heldSelfiePoseWeight = Mathf.MoveTowards(
                heldSelfiePoseWeight,
                targetWeight,
                Time.deltaTime / transition);

            if (!heldSelfiePoseActive && heldSelfiePoseWeight <= 0f)
            {
                ik.SetRightHandTarget(null, 0f);
                ik.SetRightHandRotationTarget(null, 0f);
                Destroy(heldSelfieHandTarget.gameObject);
                heldSelfieHandTarget = null;
                return;
            }

            if (!UpdateHeldSelfieTarget())
                return;

            ik.SetRightHandTarget(heldSelfieHandTarget, heldSelfiePoseWeight);
            ik.SetRightHandRotationTarget(heldSelfieHandTarget, heldSelfiePoseWeight);
        }

        private bool UpdateHeldSelfieTarget()
        {
            var animator = character != null ? character.BodyAnimator.Animator : null;
            var cameraTransform = livestream != null && livestream.PhysicalCamera != null
                ? livestream.PhysicalCamera.transform
                : null;
            if (animator == null || cameraTransform == null || heldSelfieHandTarget == null)
                return false;

            var head = animator.GetBoneTransform(HumanBodyBones.Head);
            var shoulder = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            var elbow = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);
            var hand = animator.GetBoneTransform(HumanBodyBones.RightHand);
            if (head == null || shoulder == null || elbow == null || hand == null)
                return false;

            var forward = character.transform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 0.0001f)
                forward = Vector3.forward;
            forward.Normalize();

            var cameraLocalRelative =
                Quaternion.Inverse(hand.rotation) * cameraTransform.rotation;
            var cameraOffsetLocal =
                Quaternion.Inverse(hand.rotation) *
                (cameraTransform.position - hand.position);
            var maximumReach =
                Vector3.Distance(shoulder.position, elbow.position) +
                Vector3.Distance(elbow.position, hand.position);

            var desiredHandPosition = hand.position;
            var desiredHandRotation = hand.rotation;
            var desiredCameraRotation = cameraTransform.rotation;
            for (var pass = 0; pass < 3; pass++)
            {
                var desiredCameraPosition =
                    head.position +
                    forward * Mathf.Max(0.2f, heldSelfieCameraDistance);
                desiredHandPosition =
                    desiredCameraPosition - desiredHandRotation * cameraOffsetLocal;

                var shoulderToHand = desiredHandPosition - shoulder.position;
                if (shoulderToHand.sqrMagnitude > maximumReach * maximumReach &&
                    shoulderToHand.sqrMagnitude > 0.0001f)
                {
                    desiredHandPosition =
                        shoulder.position + shoulderToHand.normalized * maximumReach;
                }

                // The hand may have been clamped, so aim the phone from the
                // camera position the clamped hand actually produces.
                var achievedCameraPosition =
                    desiredHandPosition + desiredHandRotation * cameraOffsetLocal;
                desiredCameraRotation = Quaternion.LookRotation(
                    head.position - achievedCameraPosition,
                    Vector3.up);
                desiredHandRotation =
                    desiredCameraRotation * Quaternion.Inverse(cameraLocalRelative);
            }

            heldSelfieHandTarget.SetPositionAndRotation(
                desiredHandPosition,
                desiredHandRotation);
            return true;
        }

        private static AnimationClip ResolveGeneratedClip(string sourceReference)
        {
            var library = Resources.Load<GeneratedMotionLibrary>("GeneratedMotions/GeneratedMotionLibrary");
            return library != null ? library.ResolveClip(sourceReference) : null;
        }

        /// <summary>
        /// Motion references are project asset paths; target names such as
        /// BENCH are not. This mirrors the generated-motion registry contract.
        /// </summary>
        private static bool LooksLikeMotionReference(string reference)
        {
            var normalized = (reference ?? string.Empty).Trim().Replace('\\', '/');
            const string prefix = "Assets/";
            return normalized.Length > prefix.Length &&
                normalized.StartsWith(prefix, System.StringComparison.OrdinalIgnoreCase);
        }

        private void Complete(Action<bool> callback, bool success, string operation)
        {
            IsBusy = false;
            lastObservedReason = operation ?? "none";
            lastObservedResult = $"{(string.IsNullOrWhiteSpace(operation) ? "unknown" : operation)}:" +
                (success ? "SUCCESS" : "FAILED");
            interruptRequested = false;
            interruptReason = null;
            EmitLog($"{operation}:{(success ? "SUCCESS" : "FAILED")}");
            callback?.Invoke(success);
        }

        private void MarkGesture(string gesture)
        {
            lastGesture = gesture;
            lastGestureAt = System.DateTime.UtcNow.ToString("o");
            gestureActive = true;
        }

        private void ClearGesture(string gesture)
        {
            // Only the gesture that is finishing may clear the observation; a
            // newer gesture never loses its evidence to a late callback.
            if (string.Equals(lastGesture, gesture, System.StringComparison.OrdinalIgnoreCase))
                gestureActive = false;
        }

        private void EmitLog(string message)
        {
            Log?.Invoke(message);
            Debug.Log($"[BrainActionBridge] {message}");
        }

        private Vector3? ResolveTargetPoint(string target)
        {
            var key = (target ?? string.Empty).Trim().ToUpperInvariant();
            var snapshot = BuildSnapshot();
            if (snapshot.targets.TryGetValue(key, out var configured))
                return new Vector3(configured.position[0], configured.position[1], configured.position[2]);
            return null;
        }

        private Vector3 ResolvePlacementDirection()
        {
            var cameraTransform = livestream.ActiveCamera != null
                ? livestream.ActiveCamera.transform
                : null;
            if (cameraTransform == null)
                return FlatForward(phone.transform.forward, Vector3.forward);

            var snapshot = BuildSnapshot();
            if (snapshot.targets.TryGetValue("CENTER", out var center))
            {
                var centerPosition = new Vector3(
                    center.position[0],
                    center.position[1],
                    center.position[2]);
                var direction = centerPosition - FindColliderPoint("Table");
                direction.y = 0f;
                if (direction.sqrMagnitude >= 0.01f)
                    return direction.normalized;
            }

            if (snapshot.targets.TryGetValue("CAMERA_STAND", out var stand))
            {
                var standPosition = new Vector3(
                    stand.position[0],
                    stand.position[1],
                    stand.position[2]);
                var direction = standPosition - cameraTransform.position;
                direction.y = 0f;
                if (direction.sqrMagnitude >= 0.01f)
                    return direction.normalized;
            }

            return FlatForward(cameraTransform.forward, phone.transform.forward);
        }

        private float ResolvePlacementYaw(Vector3 placementDirection)
        {
            var cameraTransform = livestream.ActiveCamera != null
                ? livestream.ActiveCamera.transform
                : null;
            if (cameraTransform == null)
                return phone.transform.eulerAngles.y;

            var localCameraRotation = Quaternion.Inverse(phone.transform.rotation) * cameraTransform.rotation;
            var localYaw = localCameraRotation.eulerAngles.y;
            if (localYaw > 180f)
                localYaw -= 360f;
            var worldYaw = Mathf.Atan2(placementDirection.x, placementDirection.z) * Mathf.Rad2Deg;
            return Mathf.Repeat(worldYaw - localYaw, 360f);
        }

        private Vector3? ResolvePlacementPoint(string target, Vector3 placementDirection)
        {
            var key = (target ?? string.Empty).Trim().ToUpperInvariant();
            switch (key)
            {
                case "TABLE":
                    var tablePoint = FindColliderPoint("Table");
                    var placement = phone.GetComponent<SmartphonePlacement>();
                    if (placement != null &&
                        placement.TryResolveEdgePlacementPoint(
                            tablePoint,
                            placementDirection,
                            out var edgePoint))
                    {
                        return edgePoint;
                    }
                    return tablePoint;
                case "BENCH":
                    return FindColliderPoint("Seat");
                case "FLOOR":
                    return Vector3.zero;
                default:
                    return null;
            }
        }

        private static void AddTarget(
            Dictionary<string, TargetSnapshot> targets,
            string name,
            Vector3 position,
            bool navigable,
            bool lookable)
        {
            targets[name] = new TargetSnapshot
            {
                position = ToList(position),
                navigation_target = navigable,
                look_at_target = lookable,
                // PHASE 25: the same navmesh probe NavMeshWalker.MoveTo uses, so
                // "reachable" means exactly "the walker will accept this point".
                reachable = IsOnNavMesh(position)
            };
        }

        /// <summary>
        /// PHASE 25: publishes a navmesh-sampled standing point this many metres
        /// in front of a spatial anchor (the stream camera or the phone).
        /// </summary>
        private const float StandDistanceMeters = 3.5f;

        /// <summary>Mirrors NavMeshWalker's projection radius, so a published
        /// standing point is one the walker itself would accept.</summary>
        private const float StandProbeRadiusMeters = 1.5f;

        private static void AddStandTarget(
            Dictionary<string, TargetSnapshot> targets,
            string name,
            Vector3 anchor,
            Vector3 anchorForward,
            Vector3 fallbackForward)
        {
            var forward = FlatForward(anchorForward, fallbackForward);
            var candidate = anchor + forward * StandDistanceMeters;
            if (!NavMesh.SamplePosition(candidate, out var hit, StandProbeRadiusMeters, NavMesh.AllAreas))
                return;

            // Stand points are movement destinations only; the backend turns
            // toward the anchor itself, never toward the standing point.
            AddTarget(targets, name, hit.position, true, false);
        }

        /// <summary>Horizontal, unit-length forward; falls back when degenerate.</summary>
        private static Vector3 FlatForward(Vector3 forward, Vector3 fallback)
        {
            forward.y = 0f;
            if (forward.sqrMagnitude >= 0.01f)
                return forward.normalized;

            fallback.y = 0f;
            return fallback.sqrMagnitude >= 0.01f ? fallback.normalized : Vector3.forward;
        }

        /// <summary>
        /// Unity euler angles (applied Z, then X, then Y) to a forward vector.
        /// Only used when no camera transform is available to read directly.
        /// </summary>
        private static Vector3 ForwardFromEuler(Vector3 eulerDegrees)
        {
            var pitch = eulerDegrees.x * Mathf.Deg2Rad;
            var yaw = eulerDegrees.y * Mathf.Deg2Rad;
            return new Vector3(
                Mathf.Sin(yaw) * Mathf.Cos(pitch),
                -Mathf.Sin(pitch),
                Mathf.Cos(yaw) * Mathf.Cos(pitch));
        }

        private static bool IsOnNavMesh(Vector3 position)
        {
            return NavMesh.SamplePosition(position, out _, StandProbeRadiusMeters, NavMesh.AllAreas);
        }

        private static Vector3 FindScenePoint(string objectName)
        {
            var target = GameObject.Find(objectName);
            return target != null ? target.transform.position : Vector3.zero;
        }

        private static Vector3 FindColliderPoint(string objectName)
        {
            var target = GameObject.Find(objectName);
            var collider = target != null ? target.GetComponentInChildren<Collider>() : null;
            return collider != null ? collider.bounds.center : (target != null ? target.transform.position : Vector3.zero);
        }

        private static string PlacementSurfaceName(Collider surface)
        {
            if (surface == null)
                return null;

            // Surface colliders are often child meshes such as Table/Top;
            // learning context needs the stable semantic parent name.
            return surface.transform.parent != null ? surface.transform.parent.name : surface.name;
        }

        private static List<float> ToList(Vector3 value) => new List<float> { value.x, value.y, value.z };
    }
}
