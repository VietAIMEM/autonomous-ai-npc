using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using RoomAgent.Brain;
using RoomAgent.GeneratedMotions;
using RoomAgent.Lacrimosa;
using RoomAgent.Livestream;
using RoomAgent.Smartphone;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Networking;

namespace RoomAgent.Editor
{
    /// <summary>
    /// PHASE 19 acceptance: a returning viewer's stored relationship and learned
    /// preference must change a real decision while everything still executes
    /// through the production chain Viewer -> ViewerMemory -> Relationship ->
    /// Personalization -> Brain -> SkillExecutor -> BrainActionBridge -> Unity.
    ///
    /// Both viewers are real production identities served by the live Brain
    /// server; nothing here is mocked. The validator only observes and asserts.
    /// </summary>
    [InitializeOnLoad]
    public static class Phase19ProductionPersonalizationValidator
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string LibraryPath = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset";
        private const string DanceFbxPath = "Assets/animation/dance/dance.fbx";
        private const string AnimationReference = "Assets/animation/dance";
        // The trailing token is the relationship bucket the Brain derives from
        // viewer A's stored memory (`int(score * 2) / 2`), so the full key is
        // resolved from the live pre-dance snapshot in Finish(). This fallback
        // only covers the case where that snapshot never arrived.
        private const string ExpectedViewerAContextPrefix =
            "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|";
        private const string ExpectedViewerAContextKey = ExpectedViewerAContextPrefix + "0_5";
        private const string PreferredDanceSkillId = "dance_02";

        private const string ViewerAId = "phase19_viewer_01";
        private const string ViewerBId = "phase19_viewer_02";
        private const string GreetingText = "hello lacrimosa";
        private const string DanceText = "Lacrimosa, please dance for me now.";

        private const string BrainBaseUrl = "http://127.0.0.1:8766";
        private const string StateUrl = BrainBaseUrl + "/brain/state";
        private const string PersonalityUrl = BrainBaseUrl + "/personality/state";
        private const string SpeechStateUrl = BrainBaseUrl + "/tts/state";
        private const string ViewerAUrl = BrainBaseUrl + "/memory/viewer/" + ViewerAId;
        private const string ViewerBUrl = BrainBaseUrl + "/memory/viewer/" + ViewerBId;
        private const string DanceSkillUrl = BrainBaseUrl + "/skills/learning/skill/dance_02";
        private const string DanceWarmupSkillUrl = BrainBaseUrl + "/skills/learning/skill/dance_warmup";
        private const string ActiveKey = "Phase19ProductionPersonalizationValidator.Active";

        private const float StatePollSeconds = 0.25f;
        private const float SampleIntervalSeconds = 0.25f;
        private const int MaxPlaybackSamples = 48;
        private const float PlanDeadlineSeconds = 45f;
        private const float SpeechDeadlineSeconds = 150f;
        private const float DanceDeadlineSeconds = 120f;
        private const float CompleteDeadlineSeconds = 150f;
        private const float ScenarioDeadlineSeconds = 600f;
        private const float PlacementDeadlineSeconds = 30f;
        private const float GreetingTtsPollSeconds = 1f;
        // The exchange is only real once the Brain has persisted it, so the
        // validator keeps polling the production viewer memory endpoint after
        // the dance clip ends instead of assuming the clip is the whole story.
        private const float RecordPollSeconds = 1f;
        private const float RecordDeadlineSeconds = 60f;

        // The learning context key includes the semantic placement surface, so
        // the production phone has to sit on the table when the dance request is
        // handled. These are candidate points inside Table/Top's placement area.
        private static readonly Vector3[] TablePlacementCandidates =
        {
            new Vector3(-3.1f, 0.9f, 0f),
            new Vector3(-3.4f, 0.9f, 0.2f),
            new Vector3(-2.8f, 0.9f, -0.2f),
            new Vector3(-3.5f, 0.9f, -0.2f),
            new Vector3(-2.7f, 0.9f, 0.2f)
        };

        private enum ValidationStage
        {
            Idle,
            AwaitGreetingPlan,
            AwaitGreetingComplete,
            AwaitPlacementConfirm,
            AwaitDancePlan,
            AwaitDanceComplete,
            AwaitDanceRecordA,
            AwaitIsolatedPlan,
            AwaitIsolatedComplete,
            AwaitIsolatedRecordB,
            PostFlight,
            Finished
        }

        private static ValidationStage stage = ValidationStage.Idle;
        private static Phase19PersonalizationReport report = new Phase19PersonalizationReport();
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
        private static float stageStartedAt;
        private static float lastPollAt;
        private static float lastSampleAt;
        private static float lastGreetingTtsPollAt;
        private static float lastRecordPollAt;

        private static ValidationStage danceRequestStage = ValidationStage.AwaitDancePlan;
        private static string pendingViewerId;
        private static string pendingViewerLabel;

        static Phase19ProductionPersonalizationValidator()
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

                report = new Phase19PersonalizationReport
                {
                    viewerAId = ViewerAId,
                    viewerBId = ViewerBId,
                    animationReference = AnimationReference,
                    clipPath = DanceFbxPath,
                    expectedViewerAContextKey = ExpectedViewerAContextKey
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
                Fail("Unity exited Play Mode before personalization validation completed.");
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

                var library = AssetDatabase.LoadAssetAtPath<GeneratedMotionLibrary>(LibraryPath);
                expectedClip = library != null ? library.ResolveClip(AnimationReference) : null;
                var danceAssets = AssetDatabase.LoadAllAssetsAtPath(DanceFbxPath);
                report.clipName = expectedClip != null ? expectedClip.name : null;
                report.clipLength = expectedClip != null ? expectedClip.length : 0f;
                report.clipPersistent = expectedClip != null &&
                    AssetDatabase.Contains(expectedClip) &&
                    AssetDatabase.GetAssetPath(expectedClip) == DanceFbxPath &&
                    danceAssets != null &&
                    danceAssets.Any(asset => ReferenceEquals(asset, expectedClip));
                report.libraryResolved = library != null &&
                    expectedClip != null &&
                    ReferenceEquals(library.ResolveClip(AnimationReference), expectedClip);
                if (!report.clipPersistent || !report.libraryResolved)
                    throw new InvalidOperationException("DANCE_MOTION_RESOLUTION_FAILED");

                requestQueue.Clear();
                requestQueue.Add(new KeyValuePair<string, string>("baseline_viewer_a", ViewerAUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_viewer_b", ViewerBUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_personality", PersonalityUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_speech", SpeechStateUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_dance_02", DanceSkillUrl));
                requestQueue.Add(new KeyValuePair<string, string>("baseline_dance_warmup", DanceWarmupSkillUrl));
                stage = ValidationStage.AwaitGreetingPlan;
                scenarioStartedAt = Time.realtimeSinceStartup;
                stageStartedAt = scenarioStartedAt;
                lastPollAt = scenarioStartedAt;
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

            if (stage == ValidationStage.Finished)
                return;

            var now = Time.realtimeSinceStartup;
            // Sampling is decoupled from the HTTP poll cadence so the whole
            // 18.5 s production clip is covered, not just its first second.
            TrackClipPlayback(now);
            if (now - scenarioStartedAt > ScenarioDeadlineSeconds)
            {
                Fail("SCENARIO_TIMEOUT at stage " + stage + " " + CompactState());
                return;
            }

            if (stage == ValidationStage.PostFlight)
                return;

            // The greeting's real synthesized audio is confirmed against the
            // exact request id the Brain queued for this greeting; the generic
            // post-flight snapshot alone is not enough evidence.
            if (stage == ValidationStage.AwaitGreetingComplete &&
                report.greetingSpeechRequested &&
                !report.greetingSynthesisReady &&
                now - lastGreetingTtsPollAt >= GreetingTtsPollSeconds &&
                activeRequest == null &&
                requestQueue.Count == 0)
            {
                lastGreetingTtsPollAt = now;
                requestQueue.Add(new KeyValuePair<string, string>("greeting_tts", SpeechStateUrl));
            }

            // A finished clip is not a finished exchange: the Brain only writes
            // the viewer interaction when the plan's final step is observed, so
            // keep asking the production memory endpoint until it really grew.
            if (stage == ValidationStage.AwaitDanceRecordA ||
                stage == ValidationStage.AwaitIsolatedRecordB)
            {
                if (now - stageStartedAt > RecordDeadlineSeconds)
                {
                    Fail("EXCHANGE_RECORD_TIMEOUT:" + pendingViewerLabel + " " + CompactState());
                    return;
                }

                if (now - lastRecordPollAt >= RecordPollSeconds &&
                    activeRequest == null &&
                    requestQueue.Count == 0)
                {
                    lastRecordPollAt = now;
                    var recordA = stage == ValidationStage.AwaitDanceRecordA;
                    requestQueue.Add(new KeyValuePair<string, string>(
                        recordA ? "record_a" : "record_b",
                        recordA ? ViewerAUrl : ViewerBUrl));
                }
            }

            if (now - lastPollAt >= StatePollSeconds && activeRequest == null && requestQueue.Count == 0)
            {
                lastPollAt = now;
                requestQueue.Add(new KeyValuePair<string, string>("state", StateUrl));
            }
        }

        private static void TrackClipPlayback(float now)
        {
            // Gate 5 only needs viewer A's real playback samples. Viewer B runs
            // afterwards, so leave A's samples intact for the final report.
            if (stage != ValidationStage.AwaitDancePlan && stage != ValidationStage.AwaitDanceComplete)
                return;
            var clipActive = ReferenceEquals(bodyAnimator.CurrentGeneratedClip, expectedClip);
            if (!clipActive)
                return;
            report.clipPlayed = true;
            if (samples.Count < MaxPlaybackSamples && now - lastSampleAt >= SampleIntervalSeconds)
            {
                lastSampleAt = now;
                CaptureSample(now);
            }
        }

        private static void BeginGreeting()
        {
            try
            {
                if (!livestream.IsStreaming && !livestream.StartStream())
                    throw new InvalidOperationException("STREAM_START_FAILED");
                report.streamStarted = livestream.IsStreaming;

                var viewers = livestream.Viewers;
                viewers.ClearViewers();
                report.viewerAAdded = viewers.AddViewer(ViewerAId, "Phase19 Production Viewer A");
                report.viewerBAdded = viewers.AddViewer(ViewerBId, "Phase19 Production Viewer B");
                if (!report.viewerAAdded || !report.viewerBAdded)
                    throw new InvalidOperationException("VIEWER_ADD_FAILED");

                loop.enabled = true;
                loop.StartLoop();
                report.loopRunningAtStart = loop.IsRunning;
                if (!report.loopRunningAtStart)
                    throw new InvalidOperationException("BRAIN_LOOP_NOT_RUNNING");

                loop.SetViewerResponse("positive");
                report.greetingSubmitted = bridge.SubmitChat(ViewerAId, GreetingText);
                if (!report.greetingSubmitted)
                    throw new InvalidOperationException("GREETING_SUBMIT_FAILED");
                AddTimeline("chat:greeting:" + ViewerAId);
                stage = ValidationStage.AwaitGreetingPlan;
                stageStartedAt = Time.realtimeSinceStartup;
            }
            catch (Exception exception)
            {
                Fail(exception.ToString());
            }
        }

        private static string BeginDanceRequest(string viewerId, bool viewerA)
        {
            pendingViewerId = viewerId;
            pendingViewerLabel = viewerA ? "a" : "b";
            danceRequestStage = viewerA ? ValidationStage.AwaitDancePlan : ValidationStage.AwaitIsolatedPlan;
            if (viewerA)
            {
                samples.Clear();
                report.clipPlayed = false;
                lastSampleAt = 0f;
            }
            loop.SetViewerResponse("positive");
            var submitted = bridge.SubmitChat(viewerId, DanceText);
            AddTimeline("chat:dance:" + viewerId);
            stage = danceRequestStage;
            stageStartedAt = Time.realtimeSinceStartup;
            return submitted ? null : "DANCE_SUBMIT_FAILED:" + viewerId;
        }

        private static void HandleState(BrainStateSnapshot snapshot)
        {
            if (snapshot == null)
                return;

            if (stage == ValidationStage.AwaitGreetingPlan)
            {
                if (snapshot.status == "ERROR")
                {
                    Fail("BRAIN_ERROR during personalized greeting: " +
                         DescribeResult(snapshot.last_result));
                    return;
                }
                var personalization = snapshot.personalization;
                var pendingSpeak = snapshot.pending_action != null &&
                    snapshot.pending_action.action == "SPEAK";
                var planSpeaks = snapshot.plan != null && snapshot.plan.steps != null &&
                    snapshot.plan.steps.Any(step => step != null && step.action == "SPEAK");
                if (!planSpeaks && !pendingSpeak)
                {
                    if (Time.realtimeSinceStartup - stageStartedAt > PlanDeadlineSeconds)
                        Fail("GREETING_PLAN_TIMEOUT " + CompactState());
                    return;
                }
                if (personalization == null || !personalization.active ||
                    personalization.viewer_id != ViewerAId)
                    return;

                CapturePersonalization(report, "a", personalization);
                report.personalizedGreeting = true;
                report.greetingReturning = personalization.returning;
                report.greetingSpeechStyle = personalization.speech_style;
                report.greetingSuggested = personalization.greeting;
                report.greetingPlanSummary = DescribePlan(snapshot.plan);
                report.greetingPendingAction = DescribeAction(snapshot.pending_action);
                // Capture the greeting's own speech request immediately; the
                // pending slot may already be reused by the next poll.
                if (snapshot.pending_action != null &&
                    snapshot.pending_action.action == "SPEAK" &&
                    !string.IsNullOrEmpty(snapshot.pending_action.speech_request_id))
                {
                    report.greetingSpeechRequestId = snapshot.pending_action.speech_request_id;
                    report.greetingSpeechRequested = true;
                }
                AddTimeline("greeting-plan:" + report.greetingPlanSummary);
                stage = ValidationStage.AwaitGreetingComplete;
                stageStartedAt = Time.realtimeSinceStartup;
                return;
            }

            if (stage == ValidationStage.AwaitGreetingComplete)
            {
                if (snapshot.status == "ERROR")
                {
                    Fail("BRAIN_ERROR during personalized greeting: " +
                         DescribeResult(snapshot.last_result));
                    return;
                }

                var speechRequestId = snapshot.pending_action != null &&
                    snapshot.pending_action.action == "SPEAK"
                    ? snapshot.pending_action.speech_request_id
                    : null;
                if (!string.IsNullOrEmpty(speechRequestId))
                {
                    report.greetingSpeechRequestId = speechRequestId;
                    report.greetingSpeechRequested = true;
                }

                // The Brain keeps ticking autonomous behaviors after the
                // exchange, so a fully idle snapshot is rarely sampled. The
                // greeting is complete once the SPEAK the Brain queued for it
                // has left the pending slot and the real audio arrived.
                var pendingSpeakId = snapshot.pending_action != null &&
                    snapshot.pending_action.action == "SPEAK"
                    ? snapshot.pending_action.speech_request_id
                    : null;
                var speakDeparted = report.greetingSpeechRequested &&
                    (string.IsNullOrEmpty(pendingSpeakId) ||
                     !string.Equals(
                         pendingSpeakId,
                         report.greetingSpeechRequestId,
                         StringComparison.Ordinal));
                var clipCleared = bodyAnimator == null ||
                    bodyAnimator.CurrentGeneratedClip == null;
                if (speakDeparted && clipCleared)
                {
                    if (!report.greetingSynthesisReady)
                    {
                        if (Time.realtimeSinceStartup - stageStartedAt > SpeechDeadlineSeconds)
                            Fail("GREETING_SYNTHESIS_NOT_CONFIRMED " + CompactState() +
                                 " request_id=" + (report.greetingSpeechRequestId ?? "null"));
                        return;
                    }

                    BeginDanceForViewerA();
                }
                else if (Time.realtimeSinceStartup - stageStartedAt > SpeechDeadlineSeconds)
                {
                    Fail("GREETING_COMPLETION_TIMEOUT " + CompactState() +
                         " speechRequested=" + report.greetingSpeechRequested +
                         " synthesisReady=" + report.greetingSynthesisReady);
                }
                return;
            }

            if (stage == ValidationStage.AwaitPlacementConfirm)
            {
                var surface = snapshot.world != null && snapshot.world.phone != null
                    ? snapshot.world.phone.placement_surface
                    : null;
                report.observedPlacementSurface = surface;
                report.observedPhoneState = snapshot.world != null && snapshot.world.phone != null
                    ? snapshot.world.phone.state
                    : null;
                if (string.Equals(surface, "Table", StringComparison.OrdinalIgnoreCase))
                {
                    AddTimeline("placement-confirmed:Table:" + (report.observedPhoneState ?? "?"));
                    BeginDanceForViewerARequest();
                }
                else if (Time.realtimeSinceStartup - stageStartedAt > PlacementDeadlineSeconds)
                {
                    Fail("PHONE_PLACEMENT_NOT_VISIBLE_TO_BRAIN surface=" +
                         (surface ?? "null") + " " + CompactState());
                }
                return;
            }

            if (stage == ValidationStage.AwaitDancePlan || stage == ValidationStage.AwaitIsolatedPlan)
            {
                var viewerA = stage == ValidationStage.AwaitDancePlan;
                if (snapshot.status == "ERROR")
                {
                    Fail("BRAIN_ERROR during dance for " + pendingViewerId + ": " +
                         DescribeResult(snapshot.last_result));
                    return;
                }
                var personalization = snapshot.personalization;
                if (personalization == null || !personalization.active ||
                    personalization.viewer_id != pendingViewerId)
                    return;

                var firstStep = snapshot.plan != null && snapshot.plan.steps != null &&
                    snapshot.plan.steps.Length > 0
                    ? snapshot.plan.steps[0]
                    : null;
                var isDancePlan = firstStep != null && firstStep.action == "DANCE";
                var active = snapshot.skills != null && snapshot.skills.runtime != null
                    ? snapshot.skills.runtime.active
                    : null;
                // A stale plan from the previous exchange must not be mistaken
                // for this request: only advance with both the dance plan and
                // the real active skill execution observed.
                if (!isDancePlan || active == null)
                {
                    if (Time.realtimeSinceStartup - stageStartedAt > PlanDeadlineSeconds)
                        Fail("DANCE_PLAN_TIMEOUT:" + pendingViewerLabel + " " + CompactState());
                    return;
                }

                CapturePersonalization(report, pendingViewerLabel, personalization);
                if (viewerA)
                {
                    report.dancePlanObserved = true;
                    report.dancePlanSummary = DescribePlan(snapshot.plan);
                    report.danceSkillId = active != null ? active.skill_id : null;
                    report.danceExecutionId = active != null ? active.execution_id : null;
                    report.danceExecutionStatus = active != null ? active.status : null;
                    report.dancePendingAction = DescribeAction(snapshot.pending_action);
                    report.dancePlanTargetMatched = firstStep != null &&
                        firstStep.action == "DANCE" &&
                        firstStep.target == AnimationReference;
                    report.dancePendingDanceObserved = snapshot.pending_action != null &&
                        snapshot.pending_action.action == "DANCE" &&
                        snapshot.pending_action.target == AnimationReference;
                    AddTimeline("dance-plan:" + report.dancePlanSummary +
                                " skill=" + (report.danceSkillId ?? "null"));
                }
                else
                {
                    report.isolatedPlanObserved = true;
                    report.isolatedPlanSummary = DescribePlan(snapshot.plan);
                    report.isolatedSkillId = active != null ? active.skill_id : null;
                    report.isolatedExecutionId = active != null ? active.execution_id : null;
                    report.isolatedPlanTargetMatched = firstStep != null &&
                        firstStep.action == "DANCE" &&
                        firstStep.target == AnimationReference;
                    report.isolatedPendingDanceObserved = snapshot.pending_action != null &&
                        snapshot.pending_action.action == "DANCE" &&
                        snapshot.pending_action.target == AnimationReference;
                    AddTimeline("isolated-plan:" + report.isolatedPlanSummary +
                                " skill=" + (report.isolatedSkillId ?? "null"));
                }

                QueueChecks(active != null ? active.skill_id : null);
                stage = viewerA ? ValidationStage.AwaitDanceComplete : ValidationStage.AwaitIsolatedComplete;
                stageStartedAt = Time.realtimeSinceStartup;
                return;
            }

            if (stage == ValidationStage.AwaitDanceComplete || stage == ValidationStage.AwaitIsolatedComplete)
            {
                var viewerA = stage == ValidationStage.AwaitDanceComplete;
                CaptureOutcome(report, snapshot, viewerA);
                if (viewerA)
                    TrackClipPlayback(Time.realtimeSinceStartup);

                var expectedExecutionId = viewerA
                    ? report.danceExecutionId
                    : report.isolatedExecutionId;
                var active = snapshot.skills != null && snapshot.skills.runtime != null
                    ? snapshot.skills.runtime.active
                    : null;
                var executionFinished = active == null ||
                    active.execution_id != expectedExecutionId ||
                    active.status != "RUNNING";
                var danceDeparted = snapshot.status != "ERROR" &&
                    (snapshot.pending_action == null ||
                     snapshot.pending_action.action != "DANCE");
                var clipCleared = bodyAnimator == null ||
                    bodyAnimator.CurrentGeneratedClip == null;

                if (executionFinished && danceDeparted && clipCleared)
                {
                    if (viewerA)
                    {
                        // Wait for the Brain to persist viewer A's exchange before
                        // viewer B starts, so the two records stay attributable.
                        stage = ValidationStage.AwaitDanceRecordA;
                    }
                    else
                    {
                        stage = ValidationStage.AwaitIsolatedRecordB;
                    }
                    stageStartedAt = Time.realtimeSinceStartup;
                    lastRecordPollAt = 0f;
                }
                else if (Time.realtimeSinceStartup - stageStartedAt >
                         (viewerA ? DanceDeadlineSeconds : CompleteDeadlineSeconds))
                {
                    Fail("DANCE_COMPLETION_TIMEOUT:" + pendingViewerLabel + " " + CompactState() +
                         " executionFinished=" + executionFinished +
                         " danceDeparted=" + danceDeparted +
                         " clipCleared=" + clipCleared);
                }
            }
        }

        private static void BeginDanceForViewerA()
        {
            var error = PlacePhoneOnTable();
            if (error != null)
            {
                Fail(error);
                return;
            }

            // The Brain builds the learning context key from the world snapshot
            // it last received, so wait until that snapshot really reports the
            // table surface before the dance request is submitted.
            stage = ValidationStage.AwaitPlacementConfirm;
            stageStartedAt = Time.realtimeSinceStartup;
        }

        private static void BeginDanceForViewerARequest()
        {
            var error = BeginDanceRequest(ViewerAId, true);
            if (error != null)
                Fail(error);
        }

        private static void BeginDanceForViewerB()
        {
            var error = BeginDanceRequest(ViewerBId, false);
            if (error != null)
                Fail(error);
        }

        private static string PlacePhoneOnTable()
        {
            var phone = UnityEngine.Object.FindFirstObjectByType<SmartphoneController>();
            if (phone == null)
                return "PRODUCTION_PHONE_MISSING";

            var placement = phone.GetComponent<SmartphonePlacement>();
            if (placement == null)
                return "PRODUCTION_PHONE_PLACEMENT_MISSING";

            var anchor = animator != null
                ? animator.GetBoneTransform(HumanBodyBones.RightHand)
                : null;
            if (anchor == null)
                anchor = character.transform;
            phone.PickUp(anchor, character.gameObject);

            var placed = false;
            foreach (var candidate in TablePlacementCandidates)
            {
                if (!placement.TryPlaceAt(candidate))
                    continue;
                placed = true;
                break;
            }

            if (!placed)
                return "PHONE_PLACEMENT_ON_TABLE_FAILED";

            var surface = phone.PlacementSurface;
            report.phonePlacedOnTable = surface != null && phone.CurrentState == SmartphoneState.Placed;
            report.phonePlacementSurface = surface != null
                ? (surface.transform.parent != null ? surface.transform.parent.name : surface.name)
                : null;
            if (!report.phonePlacedOnTable ||
                !string.Equals(report.phonePlacementSurface, "Table", StringComparison.OrdinalIgnoreCase))
                return "PHONE_PLACEMENT_SURFACE_UNEXPECTED surface=" +
                       (report.phonePlacementSurface ?? "null");

            AddTimeline("phone-placed:Table");
            return null;
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

        private static void CapturePersonalization(
            Phase19PersonalizationReport target,
            string label,
            PersonalizationSnapshot personalization)
        {
            if (personalization == null)
                return;
            if (label == "a")
            {
                target.personalizationAViewerId = personalization.viewer_id;
                target.personalizationAReturning = personalization.returning;
                target.personalizationAReason = personalization.reason;
                target.personalizationASpeechStyle = personalization.speech_style;
                target.personalizationARelationshipLevel = personalization.relationship_level;
                target.personalizationAInteractionCount = personalization.interaction_count;
                target.personalizationAPreferredSkills = personalization.preferred_skills ?? new string[0];
                target.personalizationAPreferredTopics = personalization.preferred_topics ?? new string[0];
                target.personalizationAGreeting = personalization.greeting;
                target.personalizationAKnownViewer = personalization.known_viewer;
                target.personalizationADegraded = personalization.degraded;
            }
            else
            {
                target.personalizationBViewerId = personalization.viewer_id;
                target.personalizationBReturning = personalization.returning;
                target.personalizationBReason = personalization.reason;
                target.personalizationBSpeechStyle = personalization.speech_style;
                target.personalizationBRelationshipLevel = personalization.relationship_level;
                target.personalizationBInteractionCount = personalization.interaction_count;
                target.personalizationBPreferredSkills = personalization.preferred_skills ?? new string[0];
                target.personalizationBPreferredTopics = personalization.preferred_topics ?? new string[0];
                target.personalizationBGreeting = personalization.greeting;
                target.personalizationBKnownViewer = personalization.known_viewer;
                target.personalizationBDegraded = personalization.degraded;
            }
        }

        private static void CaptureOutcome(
            Phase19PersonalizationReport target,
            BrainStateSnapshot snapshot,
            bool viewerA)
        {
            var expectedSkill = viewerA ? target.danceSkillId : target.isolatedSkillId;
            if (snapshot.behavior != null && snapshot.behavior.recent_outcomes != null)
            {
                foreach (var outcome in snapshot.behavior.recent_outcomes)
                {
                    if (outcome == null || outcome.action != "DANCE")
                        continue;
                    if (viewerA)
                    {
                        target.danceOutcomeObserved = true;
                        target.danceOutcomeSuccess = outcome.success;
                        target.danceOutcomeSkillId = outcome.skill_id;
                        target.danceOutcomeReason = outcome.reason;
                        target.danceOutcomeViewerResponse = outcome.viewer_response;
                    }
                    else
                    {
                        target.isolatedOutcomeObserved = true;
                        target.isolatedOutcomeSuccess = outcome.success;
                        target.isolatedOutcomeSkillId = outcome.skill_id;
                        target.isolatedOutcomeReason = outcome.reason;
                    }
                }
            }

            var evaluations = snapshot.skills != null && snapshot.skills.runtime != null
                ? snapshot.skills.runtime.recent_evaluations
                : null;
            if (evaluations == null)
                return;
            foreach (var evaluation in evaluations)
            {
                if (evaluation == null)
                    continue;
                if (string.IsNullOrEmpty(expectedSkill))
                    continue;
                if (evaluation.skill_id != expectedSkill)
                    continue;
                if (evaluation.status != "SUCCESS")
                    continue;
                if (viewerA)
                {
                    target.danceEvaluationObserved = true;
                    target.danceEvaluationStatus = evaluation.status;
                    target.danceEvaluationViewerResponse = evaluation.viewer_response;
                }
                else
                {
                    target.isolatedEvaluationObserved = true;
                    target.isolatedEvaluationStatus = evaluation.status;
                }
            }
        }

        private static void QueueChecks(string selectedSkillId)
        {
            // Only the starting count is captured here. Every "after" snapshot is
            // fetched once the production exchange is confirmed persisted, since
            // the Brain writes the viewer interaction when the plan's final step
            // is observed, not when the dance clip ends.
            requestQueue.Add(new KeyValuePair<string, string>(
                "dance_start_" + pendingViewerLabel + "_memory",
                pendingViewerLabel == "a" ? ViewerAUrl : ViewerBUrl));
        }

        private static void QueuePostDanceChecks(bool viewerA)
        {
            var label = viewerA ? "a" : "b";
            requestQueue.Add(new KeyValuePair<string, string>(
                "after_dance_" + label + "_memory",
                viewerA ? ViewerAUrl : ViewerBUrl));
            requestQueue.Add(new KeyValuePair<string, string>(
                "after_dance_" + label + "_speech",
                SpeechStateUrl));
            requestQueue.Add(new KeyValuePair<string, string>(
                "after_dance_" + label + "_personality",
                PersonalityUrl));
            var selectedSkillId = viewerA ? report.danceSkillId : report.isolatedSkillId;
            if (!string.IsNullOrEmpty(selectedSkillId))
            {
                requestQueue.Add(new KeyValuePair<string, string>(
                    "after_dance_" + label + "_skill",
                    BrainBaseUrl + "/skills/learning/skill/" + Uri.EscapeDataString(selectedSkillId)));
                return;
            }

            // Without a selected skill there is no learning snapshot to await.
            if (viewerA)
                BeginDanceForViewerB();
            else
                StartPostFlight();
        }

        private static void HandleRecordPoll(ViewerMetricsSnapshot viewer, bool viewerA)
        {
            if (viewer == null)
                return;
            if (viewerA)
                report.viewerAInteractionsAfter = viewer;
            else
                report.viewerBInteractionsAfter = viewer;

            var baseline = viewerA
                ? report.viewerAInteractionsBeforeDance
                : report.viewerBInteractionsBeforeDance;
            if (baseline == null || viewer.interaction_count <= baseline.interaction_count)
                return;

            if (viewerA)
                report.viewerARecordConfirmed = true;
            else
                report.viewerBRecordConfirmed = true;
            AddTimeline("record-confirmed:" + viewer.viewer_id + ":" + viewer.interaction_count);
            QueuePostDanceChecks(viewerA);
        }

        private static void StartPostFlight()
        {
            stage = ValidationStage.PostFlight;
            requestQueue.Clear();
            requestQueue.Add(new KeyValuePair<string, string>("final_viewer_a", ViewerAUrl));
            requestQueue.Add(new KeyValuePair<string, string>("final_viewer_b", ViewerBUrl));
            requestQueue.Add(new KeyValuePair<string, string>("final_personality", PersonalityUrl));
            requestQueue.Add(new KeyValuePair<string, string>("final_speech", SpeechStateUrl));
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
                case "state":
                    HandleState(Parse<BrainStateSnapshot>(body));
                    break;
                case "greeting_tts":
                    CaptureGreetingTts(body);
                    break;
                case "baseline_viewer_a":
                    report.viewerAInteractionsBefore = ParseViewer(body);
                    break;
                case "baseline_viewer_b":
                    report.viewerBInteractionsBefore = ParseViewer(body);
                    break;
                case "baseline_personality":
                    report.personalityBefore = Parse<PersonalitySnapshot>(body);
                    break;
                case "baseline_speech":
                    var speechBefore = Parse<TtsStateSnapshot>(body);
                    report.speechProvider = speechBefore != null ? speechBefore.provider : null;
                    report.speechModel = speechBefore != null ? speechBefore.model : null;
                    report.speechVoice = speechBefore != null ? speechBefore.voice : null;
                    report.speechFormat = speechBefore != null ? speechBefore.format : null;
                    report.synthesizedBefore = speechBefore != null && speechBefore.counters != null
                        ? speechBefore.counters.synthesized
                        : 0;
                    report.speechFailedBefore = speechBefore != null && speechBefore.counters != null
                        ? speechBefore.counters.failed
                        : 0;
                    break;
                case "baseline_dance_02":
                    report.learnedSkillBefore = Parse<LearningSkillSnapshot>(body);
                    break;
                case "baseline_dance_warmup":
                    report.warmupSkillBefore = Parse<LearningSkillSnapshot>(body);
                    // All six baseline snapshots are consumed; only now does the
                    // real production scenario begin with viewer A's greeting.
                    BeginGreeting();
                    break;
                case "after_dance_a_memory":
                    report.viewerAInteractionsAfter = ParseViewer(body);
                    break;
                case "after_dance_b_memory":
                    report.viewerBInteractionsAfter = ParseViewer(body);
                    break;
                case "dance_start_a_memory":
                    report.viewerAInteractionsBeforeDance = ParseViewer(body);
                    break;
                case "dance_start_b_memory":
                    report.viewerBInteractionsBeforeDance = ParseViewer(body);
                    break;
                case "record_a":
                    HandleRecordPoll(ParseViewer(body), true);
                    break;
                case "record_b":
                    HandleRecordPoll(ParseViewer(body), false);
                    break;
                case "after_dance_a_personality":
                case "after_dance_b_personality":
                    report.personalityAfterDance = Parse<PersonalitySnapshot>(body);
                    break;
                case "after_dance_a_speech":
                    CaptureSpeechAfter(body, true);
                    break;
                case "after_dance_b_speech":
                    CaptureSpeechAfter(body, false);
                    break;
                case "after_dance_a_skill":
                    report.learnedSkillAfter = Parse<LearningSkillSnapshot>(body);
                    BeginDanceForViewerB();
                    break;
                case "after_dance_b_skill":
                    report.isolatedSkillAfter = Parse<LearningSkillSnapshot>(body);
                    StartPostFlight();
                    break;
                case "final_viewer_a":
                    report.finalViewerA = ParseViewer(body);
                    break;
                case "final_viewer_b":
                    report.finalViewerB = ParseViewer(body);
                    break;
                case "final_personality":
                    report.personalityAfter = Parse<PersonalitySnapshot>(body);
                    break;
                case "final_speech":
                    var speechAfter = Parse<TtsStateSnapshot>(body);
                    report.synthesizedAfter = speechAfter != null && speechAfter.counters != null
                        ? speechAfter.counters.synthesized
                        : 0;
                    report.speechFailures = speechAfter != null && speechAfter.counters != null
                        ? speechAfter.counters.failed
                        : 0;
                    report.speechLastState = speechAfter != null && speechAfter.last_result != null
                        ? speechAfter.last_result.state
                        : null;
                    report.speechByteLength = speechAfter != null && speechAfter.last_result != null
                        ? speechAfter.last_result.byte_length
                        : 0;
                    Finish();
                    break;
            }
        }

        private static void CaptureSpeechAfter(string body, bool viewerA)
        {
            var speech = Parse<TtsStateSnapshot>(body);
            if (speech == null)
                return;
            if (speech.counters != null && viewerA)
            {
                report.synthesizedAfter = speech.counters.synthesized;
                report.speechFailures = speech.counters.failed;
            }
            if (!viewerA)
                return;
            // Only the exact greeting request counts; a later SPEAK must never
            // be able to satisfy the personalized-greeting audio evidence.
            if (speech.last_result != null &&
                string.Equals(
                    speech.last_result.request_id,
                    report.greetingSpeechRequestId,
                    StringComparison.Ordinal) &&
                (speech.last_result.state == "READY" || speech.last_result.state == "COMPLETED" ||
                 speech.last_result.state == "PLAYING") &&
                speech.last_result.byte_length > 0)
            {
                report.greetingSynthesisReady = true;
                report.greetingSynthesisByteLength = speech.last_result.byte_length;
                report.greetingSynthesisProvider = speech.last_result.provider;
                report.greetingSynthesisVoice = speech.last_result.voice;
            }
        }

        private static void CaptureGreetingTts(string body)
        {
            var speech = Parse<TtsStateSnapshot>(body);
            var result = speech != null ? speech.last_result : null;
            if (result == null || !report.greetingSpeechRequested)
                return;
            if (!string.Equals(
                    result.request_id,
                    report.greetingSpeechRequestId,
                    StringComparison.Ordinal))
                return;
            if (result.state != "READY" && result.state != "COMPLETED" &&
                result.state != "PLAYING")
                return;
            if (result.byte_length <= 0)
                return;

            report.greetingSynthesisReady = true;
            report.greetingSynthesisByteLength = result.byte_length;
            report.greetingSynthesisProvider = result.provider;
            report.greetingSynthesisVoice = result.voice;
            AddTimeline("greeting-tts:" + result.state + ":" + result.byte_length + "B");
        }

        private static ViewerMetricsSnapshot ParseViewer(string body)
        {
            var snapshot = Parse<ViewerMemorySnapshot>(body);
            return snapshot != null ? snapshot.viewer : null;
        }

        private static T Parse<T>(string body) where T : class
        {
            try
            {
                return JsonUtility.FromJson<T>(body);
            }
            catch (Exception exception)
            {
                Debug.LogWarning("[Phase19ProductionPersonalizationValidator] parse failure: " +
                                 exception.Message);
                return null;
            }
        }

        private static void Finish()
        {
            stage = ValidationStage.Finished;
            report.loopStillRunning = loop != null && loop.IsRunning;
            report.characterIdle = character != null && !character.IsMoving && !character.IsTurning;
            report.generatedClipClearedAtEnd = bodyAnimator != null &&
                bodyAnimator.CurrentGeneratedClip == null;
            report.controllerRestored = animator != null &&
                ReferenceEquals(animator.runtimeAnimatorController, originalController);

            var aBefore = report.viewerAInteractionsBefore;
            var aAfter = report.viewerAInteractionsAfter;
            var bBefore = report.viewerBInteractionsBefore;
            var bAfter = report.viewerBInteractionsAfter;
            var finalA = report.finalViewerA;
            var finalB = report.finalViewerB;

            var personalityStable = SameTraits(report.personalityBefore, report.personalityAfterDance) &&
                SameTraits(report.personalityBefore, report.personalityAfter);
            var voiceStable = report.personalityAfterDance != null &&
                report.personalityAfterDance.speech_style == report.personalityBefore?.speech_style &&
                !string.IsNullOrEmpty(report.personalityBefore?.speech_style);

            var viewerAMemoryGrew = aBefore != null && aAfter != null && finalA != null &&
                finalA.interaction_count > aBefore.interaction_count &&
                aAfter.interaction_count > aBefore.interaction_count;
            // The dance exchange itself has to be persisted, not just observed:
            // both viewers must show a memory record that did not exist before
            // their own request was handled.
            var viewerADancePersisted = report.viewerARecordConfirmed &&
                report.viewerAInteractionsBeforeDance != null &&
                finalA != null &&
                finalA.interaction_count >
                report.viewerAInteractionsBeforeDance.interaction_count;
            var viewerBDancePersisted = report.viewerBRecordConfirmed &&
                report.viewerBInteractionsBeforeDance != null &&
                finalB != null &&
                finalB.interaction_count >
                report.viewerBInteractionsBeforeDance.interaction_count;
            var viewerBIsolated = bBefore != null && bAfter != null && finalB != null &&
                viewerBDancePersisted &&
                finalB.viewer_id == ViewerBId &&
                report.personalizationBViewerId == ViewerBId &&
                report.personalizationBInteractionCount < report.personalizationAInteractionCount &&
                !ContainsIgnoreCase(
                    report.personalizationBPreferredSkills,
                    report.danceSkillId) &&
                !ContainsIgnoreCase(finalB.preferred_skills, report.danceSkillId) &&
                report.personalizationBKnownViewer;

            var expectedContextKey = ResolveExpectedContextKey();
            report.expectedViewerAContextKey = expectedContextKey;
            var learnedSkillContext = FindContext(
                report.learnedSkillAfter,
                report.danceSkillId,
                expectedContextKey);
            var learnedSkillBeforeContext = FindContext(
                report.learnedSkillBefore,
                report.danceSkillId,
                expectedContextKey);
            var learnedSkillTrialsGrew = learnedSkillContext != null &&
                learnedSkillContext.trials >
                (learnedSkillBeforeContext != null ? learnedSkillBeforeContext.trials : 0);
            var warmupContext = FindContext(
                report.learnedSkillAfter,
                "dance_warmup",
                expectedContextKey);
            var warmupContextAfter = FindContext(
                report.warmupSkillBefore,
                "dance_warmup",
                expectedContextKey);
            var warmupEvidenceBefore = warmupContextAfter != null ? warmupContextAfter.trials : 0;
            var warmupEvidenceAfter = warmupContext != null ? warmupContext.trials : 0;

            var personalizedSkillSelected = string.Equals(
                report.danceSkillId,
                PreferredDanceSkillId,
                StringComparison.OrdinalIgnoreCase);
            report.personalizedSkillSelected = personalizedSkillSelected;
            report.personalizedDecisionApplied = personalizedSkillSelected &&
                report.dancePlanTargetMatched &&
                report.dancePendingDanceObserved &&
                report.personalizationAReason?.Contains("prefers dance") == true;
            report.personalizationMatchedViewerA =
                report.personalizationAViewerId == ViewerAId &&
                report.personalizationARelationshipLevel == "REGULAR" &&
                report.personalizationAReturning;
            report.viewerIsolationVerified = viewerBIsolated &&
                report.personalizationBReason != null &&
                !report.personalizationBReason.Contains(ViewerAId) &&
                !ContainsIgnoreCase(report.personalizationBPreferredSkills, report.danceSkillId);
            report.viewerADancePersisted = viewerADancePersisted;
            report.viewerBDancePersisted = viewerBDancePersisted;
            report.relationshipProgressionVerified = report.personalizationARelationshipLevel == "REGULAR" &&
                report.personalizationBRelationshipLevel == "NEW";
            // Learned viewer preference only counts when the skill the learned
            // ranking actually promoted (dance_02) is the one that executed.
            report.learnedPreferenceApplied = personalizedSkillSelected && learnedSkillTrialsGrew;

            var playbackSuccess = report.clipPlayed &&
                samples.Count >= 2 &&
                samples.All(sample => sample.generatedClipActive && sample.danceStateActive) &&
                samples.SelectMany(sample => sample.bonePositions).All(IsFinite) &&
                MaxBoneDisplacement(samples) > 0.01f &&
                report.generatedClipClearedAtEnd &&
                report.controllerRestored;

            report.executionPathVerified = report.dancePlanObserved &&
                report.dancePlanTargetMatched &&
                report.dancePendingDanceObserved &&
                report.danceOutcomeObserved &&
                report.danceOutcomeSuccess &&
                report.danceOutcomeSkillId == report.danceSkillId &&
                report.danceEvaluationObserved &&
                report.danceEvaluationStatus == "SUCCESS" &&
                playbackSuccess;

            report.success =
                report.sceneLoaded &&
                report.clipPersistent &&
                report.libraryResolved &&
                report.streamStarted &&
                report.viewerAAdded &&
                report.viewerBAdded &&
                report.loopRunningAtStart &&
                report.greetingSubmitted &&
                report.personalizedGreeting &&
                report.greetingReturning &&
                report.greetingSpeechRequested &&
                report.phonePlacedOnTable &&
                string.Equals(
                    report.observedPlacementSurface,
                    "Table",
                    StringComparison.OrdinalIgnoreCase) &&
                report.dancePlanObserved &&
                report.dancePlanTargetMatched &&
                report.personalizedDecisionApplied &&
                report.personalizationMatchedViewerA &&
                report.learnedPreferenceApplied &&
                viewerAMemoryGrew &&
                viewerADancePersisted &&
                report.executionPathVerified &&
                report.isolatedPlanObserved &&
                report.isolatedPlanTargetMatched &&
                report.isolatedOutcomeObserved &&
                report.isolatedOutcomeSuccess &&
                viewerBIsolated &&
                report.viewerIsolationVerified &&
                report.relationshipProgressionVerified &&
                personalityStable &&
                voiceStable &&
                report.greetingSynthesisReady &&
                report.synthesizedAfter > report.synthesizedBefore &&
                report.speechFailures == report.speechFailedBefore &&
                report.loopStillRunning &&
                report.characterIdle &&
                report.generatedClipClearedAtEnd &&
                report.controllerRestored;

            report.playbackSamples = samples.Count;
            report.maxBoneDisplacement = MaxBoneDisplacement(samples);
            report.timeline = timeline.ToArray();
            report.samples = samples.ToArray();
            report.warmupEvidenceBefore = warmupEvidenceBefore;
            report.warmupEvidenceAfter = warmupEvidenceAfter;
            report.learnedSkillTrials = learnedSkillContext != null ? learnedSkillContext.trials : 0;
            report.learnedSkillTrialsBefore = learnedSkillBeforeContext != null
                ? learnedSkillBeforeContext.trials
                : 0;

            WriteReport();
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(report.success ? 0 : 1);
                return;
            }

            SessionState.SetBool(ActiveKey, false);
            EditorApplication.isPlaying = false;
        }

        private static bool SameTraits(PersonalitySnapshot first, PersonalitySnapshot second)
        {
            if (first == null || second == null || first.profile == null || second.profile == null)
                return false;
            return first.profile.warmth == second.profile.warmth &&
                   first.profile.playfulness == second.profile.playfulness &&
                   first.profile.curiosity == second.profile.curiosity &&
                   first.profile.energy == second.profile.energy &&
                   first.profile.talkativeness == second.profile.talkativeness &&
                   first.profile.formality == second.profile.formality &&
                   first.profile.calmness == second.profile.calmness &&
                   first.profile.sociability == second.profile.sociability;
        }

        private static bool ContainsIgnoreCase(string[] values, string expected)
        {
            if (values == null || string.IsNullOrEmpty(expected))
                return false;
            var target = expected.Trim();
            return values.Any(value => string.Equals(value, target, StringComparison.OrdinalIgnoreCase));
        }

        private static KnowledgeSnapshot FindContext(
            LearningSkillSnapshot snapshot,
            string skillId,
            string contextKey)
        {
            if (snapshot == null || snapshot.contexts == null || string.IsNullOrEmpty(contextKey))
                return null;
            return snapshot.contexts.FirstOrDefault(context =>
                context != null &&
                (string.IsNullOrEmpty(skillId) ||
                 string.Equals(context.skill_id, skillId, StringComparison.OrdinalIgnoreCase)) &&
                string.Equals(context.context_key, contextKey, StringComparison.OrdinalIgnoreCase));
        }

        private static string ResolveExpectedContextKey()
        {
            var snapshot = report.viewerAInteractionsBeforeDance;
            if (snapshot == null)
                return ExpectedViewerAContextKey;

            // Mirrors backend/learning/context.py: the relationship token is the
            // stored score truncated to the nearest lower half step.
            var bucket = Mathf.Floor(Mathf.Clamp01(snapshot.relationship_score) * 2f) / 2f;
            var token = bucket.ToString("0.0", CultureInfo.InvariantCulture).Replace('.', '_');
            return ExpectedViewerAContextPrefix + token;
        }

        private static void Fail(string error)
        {
            if (stage == ValidationStage.Finished)
                return;
            stage = ValidationStage.Finished;
            report.success = false;
            report.error = error;
            WriteReport();
            Debug.LogError("[Phase19ProductionPersonalizationValidator] " + error);
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
            var path = Path.GetFullPath(Path.Combine(
                Application.dataPath,
                "../../../Temp/Phase19UnityPersonalization.json"));
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase19ProductionPersonalizationValidator] " + JsonUtility.ToJson(report));
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

        private static string DescribeAction(ActionSnapshot action)
        {
            if (action == null)
                return null;
            return action.action + ":" + (action.target ?? "-") + ":" + (action.text ?? "-");
        }

        private static string DescribePlan(PlanSnapshot plan)
        {
            if (plan == null || plan.steps == null)
                return "null";
            return string.Join("|", plan.steps.Select(step =>
                step == null ? "-" : step.action + ":" + (step.target ?? "-")));
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
            if (playbackSamples == null || playbackSamples.Count == 0)
                return maximum;
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

        private static string DescribeResult(LastResultSnapshot result)
        {
            if (result == null)
                return "null";
            return result.action + ":" + (result.status ?? "-") + ":" + (result.reason ?? "-");
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
        private sealed class Phase19PersonalizationReport
        {
            public bool success;
            public string error;
            public string viewerAId;
            public string viewerBId;
            public string animationReference;
            public string clipPath;
            public string clipName;
            public float clipLength;
            public string expectedViewerAContextKey;

            public bool sceneLoaded;
            public bool clipPersistent;
            public bool libraryResolved;
            public bool streamStarted;
            public bool viewerAAdded;
            public bool viewerBAdded;
            public bool loopRunningAtStart;

            public bool greetingSubmitted;
            public bool personalizedGreeting;
            public bool greetingReturning;
            public string greetingSpeechStyle;
            public string greetingSuggested;
            public string greetingPlanSummary;
            public string greetingPendingAction;
            public string greetingSpeechRequestId;
            public bool greetingSpeechRequested;
            public bool greetingSynthesisReady;
            public int greetingSynthesisByteLength;
            public string greetingSynthesisProvider;
            public string greetingSynthesisVoice;

            public bool phonePlacedOnTable;
            public string phonePlacementSurface;
            public string observedPlacementSurface;
            public string observedPhoneState;

            public bool dancePlanObserved;
            public string dancePlanSummary;
            public string danceSkillId;
            public string danceExecutionId;
            public string danceExecutionStatus;
            public string dancePendingAction;
            public bool dancePlanTargetMatched;
            public bool dancePendingDanceObserved;
            public bool clipPlayed;
            public bool danceOutcomeObserved;
            public bool danceOutcomeSuccess;
            public string danceOutcomeSkillId;
            public string danceOutcomeReason;
            public string danceOutcomeViewerResponse;
            public bool danceEvaluationObserved;
            public string danceEvaluationStatus;
            public string danceEvaluationViewerResponse;

            public bool isolatedPlanObserved;
            public string isolatedPlanSummary;
            public string isolatedSkillId;
            public string isolatedExecutionId;
            public bool isolatedPlanTargetMatched;
            public bool isolatedPendingDanceObserved;
            public bool isolatedOutcomeObserved;
            public bool isolatedOutcomeSuccess;
            public string isolatedOutcomeSkillId;
            public string isolatedOutcomeReason;
            public bool isolatedEvaluationObserved;
            public string isolatedEvaluationStatus;
            public LearningSkillSnapshot isolatedSkillAfter;

            public bool personalizedSkillSelected;
            public bool personalizedDecisionApplied;
            public bool personalizationMatchedViewerA;
            public bool viewerIsolationVerified;
            public bool viewerADancePersisted;
            public bool viewerBDancePersisted;
            public bool relationshipProgressionVerified;
            public bool learnedPreferenceApplied;
            public bool executionPathVerified;

            public string personalizationAViewerId;
            public bool personalizationAReturning;
            public string personalizationAReason;
            public string personalizationASpeechStyle;
            public string personalizationARelationshipLevel;
            public int personalizationAInteractionCount;
            public string[] personalizationAPreferredSkills;
            public string[] personalizationAPreferredTopics;
            public string personalizationAGreeting;
            public bool personalizationAKnownViewer;
            public bool personalizationADegraded;

            public string personalizationBViewerId;
            public bool personalizationBReturning;
            public string personalizationBReason;
            public string personalizationBSpeechStyle;
            public string personalizationBRelationshipLevel;
            public int personalizationBInteractionCount;
            public string[] personalizationBPreferredSkills;
            public string[] personalizationBPreferredTopics;
            public string personalizationBGreeting;
            public bool personalizationBKnownViewer;
            public bool personalizationBDegraded;

            public ViewerMetricsSnapshot viewerAInteractionsBefore;
            public ViewerMetricsSnapshot viewerAInteractionsAfter;
            public ViewerMetricsSnapshot viewerBInteractionsBefore;
            public ViewerMetricsSnapshot viewerBInteractionsAfter;
            public ViewerMetricsSnapshot viewerAInteractionsBeforeDance;
            public ViewerMetricsSnapshot viewerBInteractionsBeforeDance;
            public bool viewerARecordConfirmed;
            public bool viewerBRecordConfirmed;
            public ViewerMetricsSnapshot finalViewerA;
            public ViewerMetricsSnapshot finalViewerB;

            public PersonalitySnapshot personalityBefore;
            public PersonalitySnapshot personalityAfterDance;
            public PersonalitySnapshot personalityAfter;

            public LearningSkillSnapshot learnedSkillBefore;
            public LearningSkillSnapshot learnedSkillAfter;
            public LearningSkillSnapshot warmupSkillBefore;
            public int learnedSkillTrialsBefore;
            public int learnedSkillTrials;
            public int warmupEvidenceBefore;
            public int warmupEvidenceAfter;

            public string speechProvider;
            public string speechModel;
            public string speechVoice;
            public string speechFormat;
            public int synthesizedBefore;
            public int synthesizedAfter;
            public int speechFailedBefore;
            public int speechFailures;
            public string speechLastState;
            public int speechByteLength;

            public bool loopStillRunning;
            public bool characterIdle;
            public bool generatedClipClearedAtEnd;
            public bool controllerRestored;
            public int playbackSamples;
            public float maxBoneDisplacement;
            public string[] timeline;
            public PlaybackSample[] samples;
        }

        [Serializable]
        private sealed class BrainStateSnapshot
        {
            public string status;
            public WorldStateSnapshot world;
            public GoalSnapshot goal;
            public PlanSnapshot plan;
            public int step_index;
            public ActionSnapshot pending_action;
            public LastResultSnapshot last_result;
            public PersonalizationSnapshot personalization;
            public BehaviorSnapshot behavior;
            public SkillsSnapshot skills;
        }

        [Serializable]
        private sealed class WorldStateSnapshot
        {
            public PhoneStateSnapshot phone;
        }

        [Serializable]
        private sealed class PhoneStateSnapshot
        {
            public string state;
            public string holder;
            public string placement_surface;
        }

        [Serializable]
        private sealed class LastResultSnapshot
        {
            public string action;
            public string status;
            public string reason;
            public bool reached;
            public string viewer_response;
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
        private sealed class PersonalizationSnapshot
        {
            public bool active;
            public string viewer_id;
            public bool known_viewer;
            public bool returning;
            public string relationship_level;
            public float familiarity;
            public int interaction_count;
            public float engagement_score;
            public string[] preferred_topics;
            public string[] preferred_skills;
            public string recent_topic;
            public string last_interaction;
            public string topic_continuity;
            public string greeting;
            public string speech_style;
            public string reason;
            public string rule;
            public bool degraded;
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
        private sealed class PersonalitySnapshot
        {
            public PersonalityProfileSnapshot profile;
            public string speech_style;
            public string source;
            public string rule;
        }

        [Serializable]
        private sealed class PersonalityProfileSnapshot
        {
            public float warmth;
            public float playfulness;
            public float curiosity;
            public float energy;
            public float talkativeness;
            public float formality;
            public float calmness;
            public float sociability;
        }

        [Serializable]
        private sealed class ViewerMemorySnapshot
        {
            public ViewerMetricsSnapshot viewer;
        }

        [Serializable]
        private sealed class ViewerMetricsSnapshot
        {
            public string viewer_id;
            public int interaction_count;
            public int successful_interactions;
            public int failed_interactions;
            public float relationship_score;
            public string last_interaction;
            public string[] preferences;
            public string[] recent_topics;
            public string relationship_level;
            public float engagement_score;
            public float familiarity;
            public string[] preferred_skills;
            public string[] preferred_topics;
        }

        [Serializable]
        private sealed class TtsStateSnapshot
        {
            public string provider;
            public string model;
            public string voice;
            public string format;
            public bool enabled;
            public bool streaming_supported;
            public string state;
            public int queue_length;
            public int queue_capacity;
            public string active_request_id;
            public string playing_request_id;
            public TtsCounters counters;
            public int consecutive_failures;
            public int cached_audio_requests;
            public TtsRequestSnapshot last_request;
            public TtsRequestSnapshot last_result;
            public TtsRequestSnapshot[] recent;
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
