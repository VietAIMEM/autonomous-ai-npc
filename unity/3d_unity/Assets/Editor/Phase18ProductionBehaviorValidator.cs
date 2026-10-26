using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RoomAgent.Brain;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using RoomAgent.Livestream;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Networking;

namespace RoomAgent.Editor
{
    /// <summary>
    /// PHASE 18 acceptance: one real contextual event must travel the production
    /// path World -> Social -> Memory -> Goal -> Behavior -> learned Skill ->
    /// SkillExecutor -> BrainActionBridge -> Unity -> outcome -> SkillLearning.
    ///
    /// Nothing here is mocked: the authoritative Brain loop, the chat transport,
    /// the real TTS provider and the generated motion library all run as they do
    /// in production. The validator only observes and asserts.
    /// </summary>
    [InitializeOnLoad]
    public static class Phase18ProductionBehaviorValidator
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string ClipPath = "Assets/Resources/GeneratedMotions/Phase15/motius_1ee2c1bf116d4111.anim";
        private const string AnimationReference =
            "Assets/animation/phase15_motius/lacrimosa_production_motius_animated.fbx";
        private const string MotionId = "motius_1ee2c1bf116d4111";
        private const string ViewerId = "phase18_production_viewer";
        private const string GreetingText = "hello lacrimosa";
        private const string FollowUpText = "what are you doing?";
        private const string BrainBaseUrl = "http://127.0.0.1:8766";
        private const string StateUrl = BrainBaseUrl + "/brain/state";
        private const string SpeechStateUrl = BrainBaseUrl + "/tts/state";
        private const string LearningSkillUrl = BrainBaseUrl + "/skills/learning/skill/" + MotionId;
        private const string ViewerMemoryUrl = BrainBaseUrl + "/memory/viewer/" + ViewerId;
        private const string ReportPath = "Temp/Phase18ProductionBehavior.json";
        private const string ActiveKey = "Phase18ProductionBehaviorValidator.Active";

        private const float MinimumDwellWaitSeconds = 10.5f;
        private const float DwellDeadlineSeconds = 150f;
        private const float PlaybackDeadlineSeconds = 30f;
        private const float OutcomeDeadlineSeconds = 120f;
        private const float ScenarioDeadlineSeconds = 300f;

        private enum ValidationStage
        {
            Idle,
            AwaitCurious,
            AwaitHappy,
            AwaitPlayback,
            AwaitOutcome,
            PostFlight,
            Finished
        }

        private static ValidationStage stage = ValidationStage.Idle;
        private static Phase18BehaviorReport report = new Phase18BehaviorReport();
        private static readonly List<PlaybackSample> samples = new List<PlaybackSample>();
        private static readonly List<string> timeline = new List<string>();

        private static BrainLoopController loop;
        private static BrainActionBridge bridge;
        private static LacrimosaController character;
        private static LacrimosaBodyAnimator bodyAnimator;
        private static Animator animator;
        private static LivestreamSystem livestream;
        private static RuntimeAnimatorController originalController;
        private static AnimationClip expectedClip;

        private static readonly List<KeyValuePair<string, string>> requestQueue =
            new List<KeyValuePair<string, string>>();
        private static UnityWebRequest activeRequest;
        private static string activeRequestTag;
        private static float activeRequestStartedAt;

        private static float scenarioStartedAt;
        private static float curiousObservedAt;
        private static float playbackFinishedAt;
        private static float playbackStageStartedAt;
        private static float nextStateRequestAt;
        private static float lastSampleAt;
        private static bool followUpSubmitted;
        private static bool awaitingPostFlightChain;
        private static string happyWithoutMotion;

        static Phase18ProductionBehaviorValidator()
        {
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        public static void Run()
        {
            try
            {
                if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                    throw new InvalidOperationException("Unity is already in Play Mode.");

                report = new Phase18BehaviorReport
                {
                    motionId = MotionId,
                    animationReference = AnimationReference,
                    clipPath = ClipPath,
                    viewerId = ViewerId
                };
                timeline.Clear();
                samples.Clear();

                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                if (!scene.isLoaded)
                    throw new InvalidOperationException("PRODUCTION_SCENE_MISSING");
                report.sceneLoaded = true;

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
                Fail("Unity exited Play Mode before production behavior validation completed.");
        }

        private static void StartPlayValidation()
        {
            if (stage != ValidationStage.Idle || !SessionState.GetBool(ActiveKey, false))
                return;
            try
            {
                loop = UnityEngine.Object.FindFirstObjectByType<BrainLoopController>();
                bridge = UnityEngine.Object.FindFirstObjectByType<BrainActionBridge>();
                character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
                livestream = UnityEngine.Object.FindFirstObjectByType<LivestreamSystem>();
                if (loop == null || bridge == null || character == null || livestream == null)
                    throw new InvalidOperationException("PRODUCTION_SCENE_COMPONENTS_MISSING");

                bodyAnimator = character.GetComponent<LacrimosaBodyAnimator>();
                animator = bodyAnimator != null ? bodyAnimator.Animator : null;
                if (bodyAnimator == null || animator == null || animator.avatar == null ||
                    !animator.avatar.isValid || !animator.avatar.isHuman)
                    throw new InvalidOperationException("INVALID_PRODUCTION_RIG");

                originalController = animator.runtimeAnimatorController;
                if (originalController == null)
                    throw new InvalidOperationException("PRODUCTION_ANIMATOR_CONTROLLER_MISSING");
                report.controllerStatesPresent = GetControllerStates(originalController)
                    .Contains("Dance");
                if (!report.controllerStatesPresent)
                    throw new InvalidOperationException("EXISTING_DANCE_STATE_MISSING");

                expectedClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(ClipPath);
                var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
                report.clipPersistent = expectedClip != null &&
                    AssetDatabase.Contains(expectedClip) &&
                    AssetDatabase.GetAssetPath(expectedClip) == ClipPath;
                report.libraryResolved = library != null &&
                    library.ResolveClip(AnimationReference) == expectedClip;
                if (!report.clipPersistent || !report.libraryResolved)
                    throw new InvalidOperationException("GENERATED_MOTION_RESOLUTION_FAILED");

                requestQueue.Clear();
                requestQueue.Add(new KeyValuePair<string, string>("baseline_learning", LearningSkillUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_memory", ViewerMemoryUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_speech", SpeechStateUrl));
                stage = ValidationStage.AwaitCurious;
                scenarioStartedAt = Time.realtimeSinceStartup;
                nextStateRequestAt = Time.realtimeSinceStartup;
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
            if (stage == ValidationStage.Finished)
                return;

            PumpRequests();

            if (stage != ValidationStage.AwaitCurious && stage != ValidationStage.AwaitHappy &&
                stage != ValidationStage.AwaitPlayback && stage != ValidationStage.AwaitOutcome)
                return;

            var now = Time.realtimeSinceStartup;
            if (now - scenarioStartedAt > ScenarioDeadlineSeconds)
            {
                Fail("SCENARIO_TIMEOUT at stage " + stage + " " + CompactState());
                return;
            }

            if (stage != ValidationStage.AwaitPlayback &&
                activeRequest == null &&
                requestQueue.Count == 0 &&
                now >= nextStateRequestAt)
            {
                nextStateRequestAt = now + 0.2f;
                requestQueue.Add(new KeyValuePair<string, string>("state", StateUrl));
            }

            if (stage == ValidationStage.AwaitPlayback)
                PumpPlayback(now);
        }

        private static void BeginScenario()
        {
            try
            {
                if (!livestream.IsStreaming && !livestream.StartStream())
                    throw new InvalidOperationException("STREAM_START_FAILED");
                report.streamStarted = livestream.IsStreaming;

                var viewers = livestream.Viewers;
                viewers.ClearViewers();
                report.viewerAdded = viewers.AddViewer(ViewerId, "Phase18 Production Viewer");
                if (!report.viewerAdded)
                    throw new InvalidOperationException("VIEWER_ADD_FAILED");

                loop.enabled = true;
                loop.StartLoop();
                report.loopRunningAtStart = loop.IsRunning;
                if (!report.loopRunningAtStart)
                    throw new InvalidOperationException("BRAIN_LOOP_NOT_RUNNING");

                loop.SetViewerResponse("positive");
                report.greetingSubmitted = bridge.SubmitChat(ViewerId, GreetingText);
                AddTimeline("chat:" + GreetingText);
                stage = ValidationStage.AwaitCurious;
                curiousObservedAt = 0f;
            }
            catch (Exception exception)
            {
                Fail(exception.ToString());
            }
        }

        private static void HandleState(BrainStateSnapshot snapshot)
        {
            if (snapshot == null || snapshot.behavior == null)
                return;

            if (stage == ValidationStage.AwaitCurious)
            {
                var state = snapshot.behavior.current_state;
                var reason = snapshot.behavior.transition_reason;
                if (state == "CURIOUS")
                {
                    report.curiousObserved = true;
                    report.curiousReason = reason;
                    curiousObservedAt = Time.realtimeSinceStartup;
                    AddTimeline("CURIOUS:" + reason);
                    stage = ValidationStage.AwaitHappy;
                }
                else if (state == "HAPPY" && MatchesLearnedMotion(snapshot))
                {
                    report.curiousObserved = false;
                    report.curiousSkipReason = "already_happy";
                    CaptureHappy(snapshot);
                    stage = ValidationStage.AwaitPlayback;
                    lastSampleAt = 0f;
                    playbackStageStartedAt = Time.realtimeSinceStartup;
                }
                else if (Time.realtimeSinceStartup - scenarioStartedAt > DwellDeadlineSeconds)
                {
                    Fail("CURIOUS_TRANSITION_TIMEOUT " + CompactState() + " " + DescribeState(snapshot));
                }
                return;
            }

            if (stage == ValidationStage.AwaitHappy)
            {
                if (snapshot.behavior.current_state == "HAPPY")
                {
                    if (MatchesLearnedMotion(snapshot))
                    {
                        CaptureHappy(snapshot);
                        stage = ValidationStage.AwaitPlayback;
                        lastSampleAt = 0f;
                        playbackStageStartedAt = Time.realtimeSinceStartup;
                        return;
                    }
                    happyWithoutMotion = DescribeState(snapshot);
                }

                if (!followUpSubmitted &&
                    Time.realtimeSinceStartup - curiousObservedAt >= MinimumDwellWaitSeconds)
                {
                    report.followUpSubmitted = bridge.SubmitChat(ViewerId, FollowUpText);
                    followUpSubmitted = true;
                    AddTimeline("chat:" + FollowUpText);
                }

                if (Time.realtimeSinceStartup - curiousObservedAt > DwellDeadlineSeconds)
                {
                    Fail("HAPPY_TRANSITION_TIMEOUT " + CompactState() + " " + DescribeState(snapshot) +
                         " happy_without_motion=" + (happyWithoutMotion ?? "never"));
                }
                return;
            }

            if (stage == ValidationStage.AwaitOutcome)
            {
                CaptureOutcome(snapshot);
                var completed = snapshot.status == "IDLE";
                if (report.outcomeObserved && report.evaluationObserved &&
                    (completed || Time.realtimeSinceStartup - playbackFinishedAt > 45f))
                {
                    report.planCompleted = completed;
                    StartPostFlight();
                }
                else if (Time.realtimeSinceStartup - playbackFinishedAt > OutcomeDeadlineSeconds)
                {
                    Fail("OUTCOME_TIMEOUT " + DescribeState(snapshot));
                }
            }
        }

        private static void CaptureHappy(BrainStateSnapshot snapshot)
        {
            report.happyObserved = true;
            report.happyReason = snapshot.behavior.transition_reason;
            AddTimeline("HAPPY:" + snapshot.behavior.transition_reason);
            report.goalType = snapshot.goal != null ? snapshot.goal.goal_type : null;
            report.goalMatched = report.goalType == "RESPOND_TO_VIEWER";
            report.activeSkillId = snapshot.skills != null && snapshot.skills.runtime != null &&
                snapshot.skills.runtime.active != null
                ? snapshot.skills.runtime.active.skill_id
                : null;
            report.skillMatched = report.activeSkillId == MotionId;
            report.planSummary = DescribePlan(snapshot.plan);
            var firstStep = snapshot.plan != null && snapshot.plan.steps != null && snapshot.plan.steps.Length > 0
                ? snapshot.plan.steps[0]
                : null;
            report.planMatched = firstStep != null &&
                firstStep.action == "DANCE" &&
                firstStep.target == AnimationReference;
            if (snapshot.pending_action != null)
            {
                report.pendingAction = snapshot.pending_action.action + ":" +
                    (snapshot.pending_action.target ?? "-");
                report.pendingDanceObserved = snapshot.pending_action.action == "DANCE" &&
                    snapshot.pending_action.target == AnimationReference;
            }
        }

        private static void CaptureOutcome(BrainStateSnapshot snapshot)
        {
            var outcomes = snapshot.behavior != null ? snapshot.behavior.recent_outcomes : null;
            if (outcomes != null)
            {
                foreach (var outcome in outcomes)
                {
                    if (outcome == null || outcome.action != "DANCE")
                        continue;
                    report.outcomeObserved = true;
                    report.outcomeAction = outcome.action;
                    report.outcomeSkillId = outcome.skill_id;
                    report.outcomeSuccess = outcome.success;
                    report.outcomeReason = outcome.reason;
                }
            }

            var evaluations = snapshot.skills != null && snapshot.skills.runtime != null
                ? snapshot.skills.runtime.recent_evaluations
                : null;
            if (evaluations != null)
            {
                foreach (var evaluation in evaluations)
                {
                    if (evaluation == null || evaluation.skill_id != MotionId)
                        continue;
                    report.evaluationObserved = true;
                    report.evaluationStatus = evaluation.status;
                    report.evaluationViewerResponse = evaluation.viewer_response;
                    break;
                }
            }
        }

        private static bool MatchesLearnedMotion(BrainStateSnapshot snapshot)
        {
            if (snapshot.plan == null || snapshot.plan.steps == null || snapshot.plan.steps.Length == 0)
                return false;
            var firstStep = snapshot.plan.steps[0];
            if (firstStep == null || firstStep.action != "DANCE" || firstStep.target != AnimationReference)
                return false;
            var active = snapshot.skills != null && snapshot.skills.runtime != null
                ? snapshot.skills.runtime.active
                : null;
            return active != null && active.skill_id == MotionId;
        }

        private static void PumpPlayback(float now)
        {
            var clipActive = ReferenceEquals(bodyAnimator.CurrentGeneratedClip, expectedClip);
            if (clipActive)
            {
                report.clipPlayed = true;
                if (samples.Count < 3 && now - lastSampleAt >= 0.18f)
                {
                    lastSampleAt = now;
                    CaptureSample(now);
                }
                return;
            }

            if (report.clipPlayed)
            {
                playbackFinishedAt = now;
                FinishPlayback();
                return;
            }

            if (now - playbackStageStartedAt > PlaybackDeadlineSeconds)
                Fail("GENERATED_MOTION_NEVER_PLAYED " + CompactState());
        }

        private static void CaptureSample(float now)
        {
            var hips = animator.GetBoneTransform(HumanBodyBones.Hips);
            var head = animator.GetBoneTransform(HumanBodyBones.Head);
            var leftHand = animator.GetBoneTransform(HumanBodyBones.LeftHand);
            var rightHand = animator.GetBoneTransform(HumanBodyBones.RightHand);
            if (hips == null || head == null || leftHand == null || rightHand == null)
            {
                Fail("PRODUCTION_BONES_MISSING");
                return;
            }

            var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
            samples.Add(new PlaybackSample
            {
                realtime = now - scenarioStartedAt,
                generatedClipActive = true,
                danceStateActive = stateInfo.IsName("Dance"),
                bonePositions = new[]
                {
                    hips.position,
                    head.position,
                    leftHand.position,
                    rightHand.position
                }
            });
        }

        private static void FinishPlayback()
        {
            report.sampleCount = samples.Count;
            report.danceStateActiveAtAllSamples = samples.Count > 0 &&
                samples.All(sample => sample.danceStateActive);
            report.finiteSamples = samples
                .SelectMany(sample => sample.bonePositions)
                .All(IsFinite);
            report.maxBoneDisplacement = MaxBoneDisplacement(samples);
            report.nontrivialMovement = report.maxBoneDisplacement > 0.01f;
            report.clipCleared = bodyAnimator.CurrentGeneratedClip == null;
            report.controllerRestored = ReferenceEquals(animator.runtimeAnimatorController, originalController);
            report.playbackSuccess = report.clipPlayed &&
                report.sampleCount >= 2 &&
                report.danceStateActiveAtAllSamples &&
                report.finiteSamples &&
                report.nontrivialMovement &&
                report.clipCleared &&
                report.controllerRestored;
            AddTimeline("playback:samples=" + report.sampleCount +
                        " displacement=" + report.maxBoneDisplacement.ToString("0.000"));
            if (!report.playbackSuccess)
            {
                Fail("GENERATED_MOTION_PLAYBACK_INVALID " + JsonUtility.ToJson(report));
                return;
            }

            stage = ValidationStage.AwaitOutcome;
            playbackFinishedAt = Time.realtimeSinceStartup;
        }

        private static void StartPostFlight()
        {
            stage = ValidationStage.PostFlight;
            requestQueue.Clear();
            requestQueue.Add(new KeyValuePair<string, string>("after_learning", LearningSkillUrl));
            requestQueue.Add(new KeyValuePair<string, string>("after_memory", ViewerMemoryUrl));
            requestQueue.Add(new KeyValuePair<string, string>("after_speech", SpeechStateUrl));
            awaitingPostFlightChain = true;
        }

        private static void PumpRequests()
        {
            if (activeRequest != null)
            {
                if (!activeRequest.isDone)
                {
                    if (Time.realtimeSinceStartup - activeRequestStartedAt > 20f)
                    {
                        var stuckTag = activeRequestTag;
                        DisposeRequest();
                        Fail("HTTP_TIMEOUT:" + stuckTag);
                    }
                    return;
                }

                var tag = activeRequestTag;
                var succeeded = activeRequest.result == UnityWebRequest.Result.Success;
                var text = activeRequest.downloadHandler != null ? activeRequest.downloadHandler.text : "";
                var error = activeRequest.error;
                DisposeRequest();
                if (succeeded)
                    HandleResponse(tag, text);
                else
                    Fail("HTTP_FAILED:" + tag + ":" + error);
                return;
            }

            if (requestQueue.Count == 0)
                return;

            var next = requestQueue[0];
            requestQueue.RemoveAt(0);
            var request = UnityWebRequest.Get(next.Value);
            request.timeout = 15;
            activeRequest = request;
            activeRequestTag = next.Key;
            activeRequestStartedAt = Time.realtimeSinceStartup;
            request.SendWebRequest();
        }

        private static void DisposeRequest()
        {
            if (activeRequest == null)
                return;
            activeRequest.Dispose();
            activeRequest = null;
            activeRequestTag = null;
        }

        private static void HandleResponse(string tag, string body)
        {
            switch (tag)
            {
                case "baseline_learning":
                    var learning = Parse<LearningSkillSnapshot>(body);
                    report.learningContextsBefore = learning != null && learning.contexts != null
                        ? learning.contexts.Length
                        : 0;
                    break;
                case "baseline_memory":
                    var memory = Parse<ViewerMemorySnapshot>(body);
                    report.viewerInteractionsBefore = memory != null && memory.viewer != null
                        ? memory.viewer.interaction_count
                        : 0;
                    break;
                case "baseline_speech":
                    var speech = Parse<TtsStateSnapshot>(body);
                    report.synthesizedBefore = speech != null && speech.counters != null
                        ? speech.counters.synthesized
                        : 0;
                    report.failedBefore = speech != null && speech.counters != null
                        ? speech.counters.failed
                        : 0;
                    report.speechProvider = speech != null ? speech.provider : null;
                    BeginScenario();
                    break;
                case "state":
                    HandleState(Parse<BrainStateSnapshot>(body));
                    break;
                case "after_learning":
                    var finalLearning = Parse<LearningSkillSnapshot>(body);
                    report.learningContextsAfter = finalLearning != null && finalLearning.contexts != null
                        ? finalLearning.contexts.Length
                        : 0;
                    report.learningContextKey = finalLearning != null && finalLearning.contexts != null &&
                        finalLearning.contexts.Length > 0
                        ? finalLearning.contexts[0].context_key
                        : null;
                    report.learningSuccesses = finalLearning != null && finalLearning.contexts != null &&
                        finalLearning.contexts.Length > 0
                        ? finalLearning.contexts[0].successes
                        : 0;
                    break;
                case "after_memory":
                    var finalMemory = Parse<ViewerMemorySnapshot>(body);
                    report.viewerInteractionsAfter = finalMemory != null && finalMemory.viewer != null
                        ? finalMemory.viewer.interaction_count
                        : 0;
                    report.viewerMemoryCreated = finalMemory != null && finalMemory.viewer != null;
                    report.viewerRelationshipScore = finalMemory != null && finalMemory.viewer != null
                        ? finalMemory.viewer.relationship_score
                        : 0f;
                    break;
                case "after_speech":
                    var finalSpeech = Parse<TtsStateSnapshot>(body);
                    report.synthesizedAfter = finalSpeech != null && finalSpeech.counters != null
                        ? finalSpeech.counters.synthesized
                        : 0;
                    report.speechFailures = finalSpeech != null && finalSpeech.counters != null
                        ? finalSpeech.counters.failed
                        : 0;
                    report.speechLastState = finalSpeech != null && finalSpeech.last_result != null
                        ? finalSpeech.last_result.state
                        : null;
                    report.speechByteLength = finalSpeech != null && finalSpeech.last_result != null
                        ? finalSpeech.last_result.byte_length
                        : 0;
                    Finish();
                    break;
            }
        }

        private static T Parse<T>(string body) where T : class
        {
            try
            {
                return JsonUtility.FromJson<T>(body);
            }
            catch (Exception exception)
            {
                Debug.LogWarning("[Phase18ProductionBehaviorValidator] parse failure: " + exception.Message);
                return null;
            }
        }

        private static void Finish()
        {
            stage = ValidationStage.Finished;
            report.loopStillRunning = loop != null && loop.IsRunning;
            report.characterIdle = character != null && !character.IsMoving && !character.IsTurning;
            report.generatedClipClearedAtEnd = bodyAnimator != null && bodyAnimator.CurrentGeneratedClip == null;
            report.speechSynthesized = report.synthesizedAfter > report.synthesizedBefore &&
                report.speechFailures == report.failedBefore;
            report.learningRecorded = report.learningContextsAfter > report.learningContextsBefore;
            report.memoryRecorded = report.viewerMemoryCreated &&
                report.viewerInteractionsAfter > report.viewerInteractionsBefore;

            report.success =
                report.sceneLoaded &&
                report.clipPersistent &&
                report.libraryResolved &&
                report.controllerStatesPresent &&
                report.streamStarted &&
                report.viewerAdded &&
                report.loopRunningAtStart &&
                report.greetingSubmitted &&
                report.curiousObserved &&
                report.curiousReason == "new_viewer_or_topic" &&
                report.happyObserved &&
                report.happyReason == "positive_viewer_reaction" &&
                report.goalMatched &&
                report.skillMatched &&
                report.planMatched &&
                report.playbackSuccess &&
                report.outcomeObserved &&
                report.outcomeSuccess &&
                report.outcomeSkillId == MotionId &&
                report.evaluationObserved &&
                report.evaluationStatus == "SUCCESS" &&
                report.learningRecorded &&
                report.memoryRecorded &&
                report.speechSynthesized &&
                report.loopStillRunning &&
                report.characterIdle &&
                report.generatedClipClearedAtEnd;

            WriteReport();
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(report.success ? 0 : 1);
                return;
            }

            SessionState.SetBool(ActiveKey, false);
            EditorApplication.isPlaying = false;
        }

        private static void Fail(string error)
        {
            if (stage == ValidationStage.Finished)
                return;
            stage = ValidationStage.Finished;
            report.success = false;
            report.error = error;
            WriteReport();
            Debug.LogError("[Phase18ProductionBehaviorValidator] " + error);
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(1);
                return;
            }

            SessionState.SetBool(ActiveKey, false);
            EditorApplication.isPlaying = false;
        }

        private static void WriteReport()
        {
            report.timeline = timeline.ToArray();
            report.samples = samples.ToArray();
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase18ProductionBehaviorValidator] " + JsonUtility.ToJson(report));
        }

        private static void AddTimeline(string entry)
        {
            if (timeline.Count < 40)
                timeline.Add(entry);
        }

        private static string CompactState()
        {
            return "loop=" + (loop != null && loop.IsRunning) + " stage=" + stage;
        }

        private static string DescribeState(BrainStateSnapshot snapshot)
        {
            if (snapshot == null)
                return "no_state";
            return "behavior=" + (snapshot.behavior != null ? snapshot.behavior.current_state : "null") +
                   " reason=" + (snapshot.behavior != null ? snapshot.behavior.transition_reason : "null") +
                   " goal=" + (snapshot.goal != null ? snapshot.goal.goal_type : "null") +
                   " plan=" + DescribePlan(snapshot.plan) +
                   " skill=" + (snapshot.skills != null && snapshot.skills.runtime != null &&
                                 snapshot.skills.runtime.active != null
                       ? snapshot.skills.runtime.active.skill_id
                       : "null");
        }

        private static string DescribePlan(PlanSnapshot plan)
        {
            if (plan == null || plan.steps == null)
                return "null";
            return string.Join("|", plan.steps.Select(step =>
                step == null ? "-" : step.action + ":" + (step.target ?? "-")));
        }

        private static string[] GetControllerStates(RuntimeAnimatorController controller)
        {
            if (controller is UnityEditor.Animations.AnimatorController animatorController)
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
                        var a = playbackSamples[first].bonePositions;
                        var b = playbackSamples[second].bonePositions;
                        if (a == null || b == null || boneIndex >= a.Length || boneIndex >= b.Length)
                            continue;
                        maximum = Mathf.Max(maximum, Vector3.Distance(a[boneIndex], b[boneIndex]));
                    }
                }
            }
            return maximum;
        }

        [Serializable]
        private sealed class PlaybackSample
        {
            public float realtime;
            public bool generatedClipActive;
            public bool danceStateActive;
            public Vector3[] bonePositions;
        }

        [Serializable]
        private sealed class Phase18BehaviorReport
        {
            public bool success;
            public string error;
            public string motionId;
            public string animationReference;
            public string clipPath;
            public string viewerId;
            public bool sceneLoaded;
            public bool clipPersistent;
            public bool libraryResolved;
            public bool controllerStatesPresent;
            public bool streamStarted;
            public bool viewerAdded;
            public bool loopRunningAtStart;
            public bool greetingSubmitted;
            public bool followUpSubmitted;
            public bool curiousObserved;
            public string curiousReason;
            public string curiousSkipReason;
            public bool happyObserved;
            public string happyReason;
            public bool goalMatched;
            public string goalType;
            public bool skillMatched;
            public string activeSkillId;
            public bool planMatched;
            public string planSummary;
            public bool pendingDanceObserved;
            public string pendingAction;
            public bool clipPlayed;
            public int sampleCount;
            public bool danceStateActiveAtAllSamples;
            public bool finiteSamples;
            public float maxBoneDisplacement;
            public bool nontrivialMovement;
            public bool clipCleared;
            public bool controllerRestored;
            public bool playbackSuccess;
            public bool outcomeObserved;
            public bool outcomeSuccess;
            public string outcomeAction;
            public string outcomeSkillId;
            public string outcomeReason;
            public bool evaluationObserved;
            public string evaluationStatus;
            public string evaluationViewerResponse;
            public bool planCompleted;
            public int learningContextsBefore;
            public int learningContextsAfter;
            public string learningContextKey;
            public int learningSuccesses;
            public bool learningRecorded;
            public bool viewerMemoryCreated;
            public int viewerInteractionsBefore;
            public int viewerInteractionsAfter;
            public float viewerRelationshipScore;
            public bool memoryRecorded;
            public string speechProvider;
            public int synthesizedBefore;
            public int synthesizedAfter;
            public int failedBefore;
            public int speechFailures;
            public string speechLastState;
            public int speechByteLength;
            public bool speechSynthesized;
            public bool loopStillRunning;
            public bool characterIdle;
            public bool generatedClipClearedAtEnd;
            public string[] timeline;
            public PlaybackSample[] samples;
        }

        [Serializable]
        private sealed class BrainStateSnapshot
        {
            public string status;
            public GoalSnapshot goal;
            public PlanSnapshot plan;
            public int step_index;
            public ActionSnapshot pending_action;
            public BehaviorSnapshot behavior;
            public SkillsSnapshot skills;
        }

        [Serializable]
        private sealed class GoalSnapshot
        {
            public string goal_id;
            public string goal_type;
            public float priority;
            public string status;
            public string reason;
        }

        [Serializable]
        private sealed class PlanSnapshot
        {
            public string plan_id;
            public string goal_id;
            public string reason;
            public int version;
            public StepSnapshot[] steps;
        }

        [Serializable]
        private sealed class StepSnapshot
        {
            public string action;
            public string target;
            public string text;
        }

        [Serializable]
        private sealed class ActionSnapshot
        {
            public string action;
            public string target;
            public string text;
            public string speech_request_id;
        }

        [Serializable]
        private sealed class BehaviorSnapshot
        {
            public string current_state;
            public string previous_state;
            public string transition_reason;
            public int transition_tick;
            public int logical_tick;
            public string current_skill;
            public string current_action;
            public OutcomeSnapshot recent_outcome;
            public OutcomeSnapshot[] recent_outcomes;
        }

        [Serializable]
        private sealed class OutcomeSnapshot
        {
            public string action;
            public string status;
            public bool success;
            public string reason;
            public string timestamp;
            public string viewer_response;
            public string skill_id;
        }

        [Serializable]
        private sealed class SkillsSnapshot
        {
            public bool enabled;
            public RuntimeSnapshot runtime;
        }

        [Serializable]
        private sealed class RuntimeSnapshot
        {
            public ActiveSkillSnapshot active;
            public EvaluationSnapshot[] recent_evaluations;
        }

        [Serializable]
        private sealed class ActiveSkillSnapshot
        {
            public string execution_id;
            public string skill_id;
            public string plan_id;
            public string status;
            public int current_step;
            public int total_steps;
        }

        [Serializable]
        private sealed class EvaluationSnapshot
        {
            public string skill_id;
            public string execution_id;
            public string status;
            public int steps_completed;
            public int steps_failed;
            public string viewer_response;
            public string notes;
        }

        [Serializable]
        private sealed class LearningSkillSnapshot
        {
            public string skill_id;
            public KnowledgeSnapshot[] contexts;
        }

        [Serializable]
        private sealed class KnowledgeSnapshot
        {
            public string skill_id;
            public string context_key;
            public int trials;
            public int successes;
            public int failures;
            public float success_rate;
            public float average_score;
            public float confidence;
        }

        [Serializable]
        private sealed class ViewerMemorySnapshot
        {
            public ViewerSnapshot viewer;
            public InteractionSnapshot[] recent_interactions;
        }

        [Serializable]
        private sealed class ViewerSnapshot
        {
            public string viewer_id;
            public int interaction_count;
            public int successful_interactions;
            public int failed_interactions;
            public float relationship_score;
            public string[] preferences;
            public string[] recent_topics;
        }

        [Serializable]
        private sealed class InteractionSnapshot
        {
            public string interaction_id;
            public string action;
            public bool success;
        }

        [Serializable]
        private sealed class TtsStateSnapshot
        {
            public string provider;
            public string model;
            public string voice;
            public bool enabled;
            public string state;
            public TtsCounters counters;
            public TtsRequestSnapshot last_result;
        }

        [Serializable]
        private sealed class TtsCounters
        {
            public int requested;
            public int synthesized;
            public int reused;
            public int rejected;
            public int failed;
            public int cancelled;
            public int completed;
        }

        [Serializable]
        private sealed class TtsRequestSnapshot
        {
            public string request_id;
            public string state;
            public string source;
            public string provider;
            public string model;
            public string voice;
            public string format;
            public int byte_length;
            public float duration_seconds;
            public string error;
        }
    }
}
