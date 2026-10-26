using System.Collections.Generic;
using UnityEngine;

namespace RoomAgent.Lacrimosa
{
    /// <summary>
    /// PHASE 28: the single owner of Animator playback for Lacrimosa.
    ///
    /// The existing controller (states, triggers, transitions) is reused as-is;
    /// this component adds observation on top of it so the rest of the runtime can
    /// tell the difference between "an action was requested", "the Animator really
    /// entered the action state", "the action clip finished" and "the action was
    /// interrupted". Every value reported here is read back from the Animator, not
    /// inferred from the request that was sent.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Animator))]
    public sealed class LacrimosaBodyAnimator : MonoBehaviour
    {
        public const string IdleStateName = "Idle";
        public const string WalkStateName = "Walk";
        public const string RunStateName = "Run";
        public const string DanceStateName = "Dance";

        [SerializeField] private string speedParameter = "Speed";
        [SerializeField] private float speedBlendSmoothing = 10f;

        [Tooltip("Cross-fade used when an interrupted action hands control back to locomotion.")]
        [SerializeField] private float locomotionRestoreDuration = 0.3f;

        [Tooltip("How long an action state may fail to appear before the request is reported as never started.")]
        [SerializeField] private float actionStartTimeoutSeconds = 0.75f;

        [Tooltip("Locomotion speed at or above which the restore path re-enters Run instead of Walk.")]
        [SerializeField] private float runRestoreSpeed = 1.8f;

        private static readonly Dictionary<string, string> ActionStateNames =
            new Dictionary<string, string>
            {
                { "Wave", "Waving" },
                { "Dance", "Dance" },
                { "TurnLeft", "LeftTurn" },
                { "TurnRight", "RightTurn" },
                { "Look", "LookingBehind" }
            };

        // The runtime cannot enumerate state names, so the states the controller
        // actually contains are mapped by hash once and anything else is reported
        // by hash rather than by an invented name.
        private static readonly string[] KnownStateNames =
        {
            IdleStateName,
            WalkStateName,
            RunStateName,
            "LeftTurn",
            "RightTurn",
            "LookingBehind",
            "Waving",
            DanceStateName
        };

        /// <summary>What the Animator actually did for one action request.</summary>
        public sealed class ActionObservation
        {
            public string trigger;
            public string stateName;
            public int stateHash;
            public float requestedAt;
            public float startedAt;
            public bool started;
            public bool finished;
            public bool completed;
            public float activeSeconds;
            public float normalizedTime;
            public string reason = "none";

            public ActionObservation Copy() => (ActionObservation)MemberwiseClone();
        }

        private readonly Dictionary<int, string> knownStateHashes = new Dictionary<int, string>();

        private Animator animator;
        private int speedParameterId;
        private int danceStateHash;
        private float currentSpeed;
        private RuntimeAnimatorController originalController;
        private AnimatorOverrideController generatedOverride;
        private AnimationClip originalDanceClip;
        private AnimationClip currentGeneratedClip;

        private string currentStateName = "unknown";
        private string previousStateName = "none";
        private int currentStateHash;
        private int previousStateHash;
        private float currentStateNormalizedTime;
        private float currentStateLength;
        private string lastTransition = "none";
        private bool transitionActive;
        private float transitionProgress;
        private string transitionTargetName = "none";
        private int transitionTargetHash;

        private string requestedAction = "none";
        private string lastCompletedAction = "none";
        private string lastCompletion = "none";
        private string lastCompletionReason = "none";
        private string lastCompletionState = "none";
        private string lastInterruptionReason = "none";
        private string lastRestoreReason = "none";
        private float lastActionActiveSeconds;

        public Animator Animator => animator ? animator : (animator = GetComponent<Animator>());

        public AnimationClip CurrentGeneratedClip => currentGeneratedClip;

        public string GeneratedClipName =>
            currentGeneratedClip != null ? currentGeneratedClip.name : "none";

        /// <summary>Actual Animator state on the base layer, read back this frame.</summary>
        public string CurrentStateName => currentStateName;

        public string PreviousStateName => previousStateName;

        public string LastTransition => lastTransition;

        public bool TransitionActive => transitionActive;

        public float TransitionProgress => transitionProgress;

        public string TransitionTargetName => transitionTargetName;

        public float CurrentStateNormalizedTime => currentStateNormalizedTime;

        public float CurrentStateLengthSeconds => currentStateLength;
        
        public int LayerCount => Animator != null ? Animator.layerCount : 0;

        public float BaseLayerWeight => Animator != null ? Animator.GetLayerWeight(0) : 0f;

        public string RequestedAction => requestedAction;

        public string LastCompletedAction => lastCompletedAction;

        public string LastCompletion => lastCompletion;

        public string LastCompletionReason => lastCompletionReason;

        /// <summary>Animator state the character was in when the action ended.</summary>
        public string LastCompletionState => lastCompletionState;

        public string LastInterruptionReason => lastInterruptionReason;

        public string LastRestoreReason => lastRestoreReason;

        public float LastActionActiveSeconds => lastActionActiveSeconds;

        /// <summary>The most recent action observation; null until an action is requested.</summary>
        public ActionObservation ActiveAction { get; private set; }

        public bool ActionStateActive =>
            ActiveAction != null && ActiveAction.started && !ActiveAction.finished;

        void Awake()
        {
            animator = GetComponent<Animator>();
            speedParameterId = Animator.StringToHash(speedParameter);
            for (var index = 0; index < KnownStateNames.Length; index++)
                knownStateHashes[Animator.StringToHash(KnownStateNames[index])] = KnownStateNames[index];
            danceStateHash = Animator.StringToHash(DanceStateName);
        }

        public void SetLocomotionSpeed(float metersPerSecond)
        {
            currentSpeed = Mathf.Lerp(currentSpeed, metersPerSecond, 1f - Mathf.Exp(-speedBlendSmoothing * Time.deltaTime));
            if (Animator != null && Animator.parameters != null)
                Animator.SetFloat(speedParameterId, currentSpeed);
        }

        /// <summary>
        /// Starts a controller-authored action state. Returns false when the
        /// trigger or its state is not part of the current controller, so callers
        /// never report an action the Animator cannot actually play.
        /// </summary>
        public bool PlayAction(string triggerName)
        {
            if (Animator == null || string.IsNullOrWhiteSpace(triggerName))
                return false;
            if (!ActionStateNames.TryGetValue(triggerName, out var stateName))
                return false;

            SupersedeActiveAction("superseded_by_" + triggerName);
            StopGeneratedClip("superseded_by_action");
            ResetActionTriggers();
            Animator.SetTrigger(triggerName);

            requestedAction = triggerName;
            lastRestoreReason = "none";
            ActiveAction = new ActionObservation
            {
                trigger = triggerName,
                stateName = stateName,
                stateHash = Animator.StringToHash(stateName),
                requestedAt = Time.time
            };
            return true;
        }

        public bool PlayGeneratedClip(AnimationClip clip)
        {
            if (clip == null || Animator == null || Animator.avatar == null || !Animator.avatar.isHuman)
                return false;

            if (originalController == null)
                originalController = Animator.runtimeAnimatorController;
            if (originalDanceClip == null)
                originalDanceClip = FindControllerClip(DanceStateName);
            if (originalController == null || originalDanceClip == null)
                return false;

            if (generatedOverride == null)
                generatedOverride = new AnimatorOverrideController(originalController);

            SupersedeActiveAction("superseded_by_generated_clip");
            var overrides = new List<KeyValuePair<AnimationClip, AnimationClip>>();
            generatedOverride.GetOverrides(overrides);
            for (var index = 0; index < overrides.Count; index++)
            {
                if (string.Equals(overrides[index].Key.name, DanceStateName, System.StringComparison.OrdinalIgnoreCase))
                    overrides[index] = new KeyValuePair<AnimationClip, AnimationClip>(overrides[index].Key, clip);
            }
            foreach (var pair in overrides)
            {
                if (string.Equals(pair.Key.name, DanceStateName, System.StringComparison.OrdinalIgnoreCase))
                    generatedOverride[pair.Key] = clip;
            }

            Animator.runtimeAnimatorController = generatedOverride;
            Animator.ResetTrigger(DanceStateName);
            Animator.Play(DanceStateName, 0, 0f);
            Animator.Update(0f);
            if (!Animator.GetCurrentAnimatorStateInfo(0).IsName(DanceStateName))
            {
                StopGeneratedClip("generated_playback_failed");
                return false;
            }

            currentGeneratedClip = clip;
            requestedAction = DanceStateName;
            lastRestoreReason = "none";
            // The state was verified just above, so this action is observed as
            // started; the completion side is reported by whoever stops it.
            ActiveAction = new ActionObservation
            {
                trigger = DanceStateName,
                stateName = DanceStateName,
                stateHash = danceStateHash,
                requestedAt = Time.time,
                started = true,
                startedAt = Time.time
            };
            return true;
        }

        /// <summary>
        /// True only while the generated clip is the motion the Dance state is
        /// actually playing, verified against the Animator rather than a stored
        /// reference.
        /// </summary>
        public bool IsPlayingGeneratedClip(AnimationClip clip)
        {
            return TryGetGeneratedClipProgress(clip, out _, out var stateActive) && stateActive;
        }

        /// <summary>Playback progress of a generated clip as the Animator reports it.</summary>
        public bool TryGetGeneratedClipProgress(AnimationClip clip, out float normalizedTime, out bool stateActive)
        {
            normalizedTime = 0f;
            stateActive = false;
            if (clip == null || currentGeneratedClip != clip || Animator == null)
                return false;

            var current = Animator.GetCurrentAnimatorStateInfo(0);
            if (current.shortNameHash == danceStateHash)
            {
                normalizedTime = current.normalizedTime;
                stateActive = true;
                return true;
            }

            if (Animator.IsInTransition(0))
            {
                var next = Animator.GetNextAnimatorStateInfo(0);
                if (next.shortNameHash == danceStateHash)
                {
                    normalizedTime = next.normalizedTime;
                    stateActive = true;
                }
            }
            return true;
        }

        /// <summary>Length of the controller clip behind an action state, in seconds.</summary>
        public float ResolveActionStateLength(string triggerName)
        {
            if (!ActionStateNames.TryGetValue(triggerName ?? string.Empty, out var stateName))
                return 0f;
            var clip = FindControllerClip(stateName);
            return clip != null ? clip.length : 0f;
        }

        /// <summary>
        /// PHASE 28: hands animation ownership back to locomotion after an action
        /// finished or was interrupted, with a real cross-fade into the state the
        /// current speed parameter belongs to. Never leaves a stale action state.
        /// </summary>
        public bool RestoreLocomotion(string reason)
        {
            MarkActionEnded(reason, completed: false, interrupted: true);
            return ApplyLocomotionRestore(reason);
        }

        /// <summary>
        /// PHASE 28: closes an action that has no natural end of its own - a
        /// looping action clip - after one full observed pass, recording it as
        /// a real completion rather than an interruption.
        /// </summary>
        public bool CompleteAction(string reason)
        {
            MarkActionEnded(reason, completed: true, interrupted: false);
            return ApplyLocomotionRestore(reason);
        }

        public void StopGeneratedClip()
        {
            StopGeneratedClip("stopped");
        }

        public void StopGeneratedClip(string reason)
        {
            var hadClip = currentGeneratedClip != null;
            currentGeneratedClip = null;

            var observation = ActiveAction;
            if (hadClip && observation != null && !observation.finished)
            {
                observation.finished = true;
                // Only a caller that genuinely observed the clip reach its end
                // may claim completion; every other stop is a real interruption.
                observation.completed = string.Equals(reason, "completed", System.StringComparison.Ordinal);
                observation.reason = reason;
                lastCompletedAction = observation.trigger;
                lastCompletion = $"{observation.trigger}:" +
                    (observation.completed ? "COMPLETED" : "INTERRUPTED");
                lastCompletionReason = reason;
                lastCompletionState = currentStateName;
                if (!observation.completed)
                    lastInterruptionReason = reason;
                lastActionActiveSeconds = observation.activeSeconds;
            }

            if (Animator == null)
                return;

            Animator.ResetTrigger(DanceStateName);
            if (hadClip && originalController != null)
                Animator.runtimeAnimatorController = originalController;
        }

        void Update()
        {
            if (Animator == null || !Animator.isActiveAndEnabled)
                return;

            var deltaTime = Time.deltaTime;
            ObserveState();
            ObserveAction(deltaTime);
        }

        private void ObserveState()
        {
            var info = Animator.GetCurrentAnimatorStateInfo(0);
            if (info.shortNameHash != currentStateHash)
            {
                previousStateHash = currentStateHash;
                previousStateName = currentStateName;
                currentStateHash = info.shortNameHash;
                currentStateName = ResolveStateName(currentStateHash);
                lastTransition = $"{previousStateName} -> {currentStateName}";
            }

            currentStateNormalizedTime = info.normalizedTime;
            currentStateLength = info.length;

            transitionActive = Animator.IsInTransition(0);
            if (transitionActive)
            {
                var transition = Animator.GetAnimatorTransitionInfo(0);
                transitionProgress = Mathf.Clamp01(transition.normalizedTime);
                var next = Animator.GetNextAnimatorStateInfo(0);
                if (next.shortNameHash != transitionTargetHash)
                {
                    transitionTargetHash = next.shortNameHash;
                    transitionTargetName = ResolveStateName(transitionTargetHash);
                    lastTransition = $"{currentStateName} -> {transitionTargetName}";
                }
            }
            else
            {
                transitionProgress = 0f;
                transitionTargetHash = 0;
                transitionTargetName = "none";
            }
        }

        private void ObserveAction(float deltaTime)
        {
            var observation = ActiveAction;
            if (observation == null || observation.finished)
                return;

            var current = Animator.GetCurrentAnimatorStateInfo(0);
            var inActionState = current.shortNameHash == observation.stateHash;
            if (!inActionState && Animator.IsInTransition(0))
                inActionState = Animator.GetNextAnimatorStateInfo(0).shortNameHash == observation.stateHash;

            if (!observation.started)
            {
                if (inActionState)
                {
                    observation.started = true;
                    observation.startedAt = Time.time;
                    return;
                }

                if (Time.time - observation.requestedAt > actionStartTimeoutSeconds)
                {
                    observation.finished = true;
                    observation.reason = "action_state_never_started";
                    RecordCompletion(observation);
                }
                return;
            }

            observation.activeSeconds += deltaTime;
            observation.normalizedTime = current.shortNameHash == observation.stateHash
                ? current.normalizedTime
                : Animator.GetNextAnimatorStateInfo(0).normalizedTime;

            if (!inActionState)
            {
                // The controller returned to locomotion on its own: the action
                // clip reached its exit and the Animator left the state.
                observation.finished = true;
                observation.completed = true;
                observation.reason = "animation_completed";
                RecordCompletion(observation);
            }
        }

        private void RecordCompletion(ActionObservation observation)
        {
            lastCompletedAction = observation.trigger;
            lastCompletion = $"{observation.trigger}:{(observation.completed ? "COMPLETED" : "NOT_STARTED")}";
            lastCompletionReason = observation.reason;
            lastCompletionState = currentStateName;
            lastActionActiveSeconds = observation.activeSeconds;
        }

        private void MarkActionEnded(string reason, bool completed, bool interrupted)
        {
            var observation = ActiveAction;
            if (observation == null || observation.finished)
                return;

            observation.finished = true;
            observation.completed = completed;
            observation.reason = reason;
            lastCompletedAction = observation.trigger;
            lastCompletion = $"{observation.trigger}:{(completed ? "COMPLETED" : "INTERRUPTED")}";
            lastCompletionReason = reason;
            lastCompletionState = currentStateName;
            lastActionActiveSeconds = observation.activeSeconds;
            if (interrupted)
                lastInterruptionReason = reason;
        }

        private bool ApplyLocomotionRestore(string reason)
        {
            if (Animator == null)
                return false;

            StopGeneratedClip(reason);
            ResetActionTriggers();

            var target = currentSpeed >= runRestoreSpeed
                ? RunStateName
                : currentSpeed > 0.05f
                    ? WalkStateName
                    : IdleStateName;
            if (!Animator.GetCurrentAnimatorStateInfo(0).IsName(target))
                Animator.CrossFadeInFixedTime(target, locomotionRestoreDuration, 0, 0f);
            lastRestoreReason = reason;
            return true;
        }

        /// <summary>
        /// A newer action request takes ownership from an unfinished one, and
        /// the older request is recorded as interrupted rather than as a
        /// completion it never observed.
        /// </summary>
        private void SupersedeActiveAction(string reason)
        {
            var observation = ActiveAction;
            if (observation == null || observation.finished)
                return;

            observation.finished = true;
            observation.completed = false;
            observation.reason = reason;
            lastCompletedAction = observation.trigger;
            lastCompletion = $"{observation.trigger}:INTERRUPTED";
            lastCompletionReason = reason;
            lastInterruptionReason = reason;
            lastCompletionState = currentStateName;
            lastActionActiveSeconds = observation.activeSeconds;
        }

        private void ResetActionTriggers()
        {
            foreach (var pair in ActionStateNames)
                Animator.ResetTrigger(pair.Key);
        }

        private string ResolveStateName(int stateHash)
        {
            if (knownStateHashes.TryGetValue(stateHash, out var name))
                return name;
            return stateHash == 0 ? "none" : $"hash:{stateHash}";
        }

        private AnimationClip FindControllerClip(string clipName)
        {
            var controller = originalController != null ? originalController : Animator?.runtimeAnimatorController;
            if (controller == null)
                return null;

            var overrides = new List<KeyValuePair<AnimationClip, AnimationClip>>();
            var probe = new AnimatorOverrideController(controller);
            probe.GetOverrides(overrides);
            foreach (var pair in overrides)
            {
                if (string.Equals(pair.Key.name, clipName, System.StringComparison.OrdinalIgnoreCase))
                    return pair.Key;
            }
            return null;
        }
    }
}
