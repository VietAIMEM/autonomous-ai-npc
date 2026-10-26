using System.Collections;
using UnityEngine;

namespace RoomAgent.Lacrimosa
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(LacrimosaMovement))]
    public sealed class LacrimosaController : MonoBehaviour
    {
        [SerializeField] private float turnSpeed = 240f;
        [SerializeField] private float turnTolerance = 1.5f;
        [SerializeField] private float turnTimeout = 2f;

        private LacrimosaMovement movement;
        private LacrimosaLookAt lookAt;
        private LacrimosaBodyAnimator bodyAnimator;
        private bool turning;
        private float turnTimer;
        private Vector3 turnTarget;
        private bool agentRotationSuppressed;

        private RoomAgent.Smartphone.SmartphoneController heldPhone;

        public bool IsTurning => turning;

        public bool IsMoving => movement != null && movement.IsMoving;

        // PHASE 28 diagnostics: every value below is read from the live
        // transform / agent / Animator, never from the request that was sent.
        public Vector3 OrientationTarget => turnTarget;

        public float OrientationAngleRemaining
        {
            get
            {
                if (!turning)
                    return 0f;
                var direction = turnTarget - transform.position;
                direction.y = 0f;
                if (direction.sqrMagnitude < 0.001f)
                    return 0f;
                return Quaternion.Angle(
                    transform.rotation,
                    Quaternion.LookRotation(direction.normalized, Vector3.up));
            }
        }

        public string OrientationState =>
            turning ? "turning" : IsMoving ? "moving" : "settled";

        /// <summary>
        /// Who currently owns the character's yaw. The NavMeshAgent rotates the
        /// body while walking; a scripted turn takes that ownership away for
        /// exactly as long as the turn runs, so the two never fight.
        /// </summary>
        public string RotationOwner =>
            turning || agentRotationSuppressed ? "script_turn" : "navmesh_agent";

        public LacrimosaBodyAnimator BodyAnimator =>
            bodyAnimator != null ? bodyAnimator : (bodyAnimator = GetComponent<LacrimosaBodyAnimator>());

        void Awake()
        {
            movement = GetComponent<LacrimosaMovement>();
            lookAt = GetComponent<LacrimosaLookAt>();
            bodyAnimator = GetComponent<LacrimosaBodyAnimator>();
        }

        public bool MoveTo(Vector3 worldTarget, bool run = false)
        {
            turning = false;
            CancelActiveAction("superseded_by_move");
            RestoreAgentRotation();
            return movement.MoveTo(worldTarget, run);
        }

        public void Stop()
        {
            Stop("stop_requested");
        }

        /// <summary>
        /// Stops navigation and settles the body back into locomotion, recording
        /// the given reason on the animation observation.
        /// </summary>
        public void Stop(string reason)
        {
            turning = false;
            RestoreAgentRotation();
            if (bodyAnimator != null)
            {
                bodyAnimator.StopGeneratedClip(reason);
                bodyAnimator.RestoreLocomotion(reason);
            }
            movement.Stop();
        }

        public bool TurnTo(Vector3 worldPosition)
        {
            movement.Stop();
            CancelActiveAction("superseded_by_turn");
            var direction = worldPosition - transform.position;
            direction.y = 0f;
            if (direction.sqrMagnitude < 0.001f)
                return false;

            SuppressAgentRotation();
            turnTarget = worldPosition;
            turning = true;
            turnTimer = 0f;
            return true;
        }

        public void LookAt(Vector3 worldPosition)
        {
            if (lookAt != null)
                lookAt.LookAt(worldPosition);
        }

        public void ClearLookAt()
        {
            if (lookAt != null)
                lookAt.ClearLookAt();
        }

        public bool PlayWave()
        {
            return PlayAction("Wave");
        }

        public bool PlayDance()
        {
            return PlayAction("Dance");
        }

        /// <summary>
        /// Returns false when the trigger is not part of the current controller,
        /// so callers never report an action the Animator cannot play.
        /// </summary>
        public bool PlayAction(string triggerName)
        {
            movement.Stop();
            turning = false;
            RestoreAgentRotation();
            return bodyAnimator != null && bodyAnimator.PlayAction(triggerName);
        }

        public bool PlayGeneratedClip(AnimationClip clip)
        {
            movement.Stop();
            turning = false;
            RestoreAgentRotation();
            return bodyAnimator != null && bodyAnimator.PlayGeneratedClip(clip);
        }

        public bool IsPlayingGeneratedClip(AnimationClip clip)
        {
            return bodyAnimator != null && bodyAnimator.IsPlayingGeneratedClip(clip);
        }

        /// <summary>Playback progress of a generated clip as the Animator reports it.</summary>
        public bool TryGetGeneratedClipProgress(
            AnimationClip clip,
            out float normalizedTime,
            out bool stateActive)
        {
            normalizedTime = 0f;
            stateActive = false;
            return bodyAnimator != null &&
                   bodyAnimator.TryGetGeneratedClipProgress(clip, out normalizedTime, out stateActive);
        }

        public float ResolveActionStateLength(string triggerName)
        {
            return bodyAnimator != null ? bodyAnimator.ResolveActionStateLength(triggerName) : 0f;
        }

        public LacrimosaBodyAnimator.ActionObservation ActiveAction =>
            bodyAnimator != null ? bodyAnimator.ActiveAction : null;

        public void StopGeneratedClip()
        {
            if (bodyAnimator != null)
                bodyAnimator.StopGeneratedClip();
        }

        public void StopGeneratedClip(string reason)
        {
            if (bodyAnimator != null)
                bodyAnimator.StopGeneratedClip(reason);
        }

        /// <summary>
        /// PHASE 28: hands animation ownership back to locomotion. Used by the
        /// interruption path so a cancelled action can never leave the Animator
        /// stranded in an action state.
        /// </summary>
        public bool RestoreLocomotion(string reason)
        {
            return bodyAnimator != null && bodyAnimator.RestoreLocomotion(reason);
        }

        /// <summary>
        /// Ends a looping action clip after one full observed pass and hands
        /// animation back to locomotion.
        /// </summary>
        public bool CompleteAction(string reason)
        {
            return bodyAnimator != null && bodyAnimator.CompleteAction(reason);
        }

        /// <summary>
        /// Interrupts whatever action animation is playing, if any, and returns
        /// to locomotion. Returns false when there was nothing to interrupt.
        /// </summary>
        public bool CancelActiveAction(string reason)
        {
            if (bodyAnimator == null)
                return false;
            if (!bodyAnimator.ActionStateActive && bodyAnimator.CurrentGeneratedClip == null)
                return false;

            bodyAnimator.StopGeneratedClip(reason);
            return bodyAnimator.RestoreLocomotion(reason);
        }

        public RoomAgent.Smartphone.SmartphoneController HeldPhone => heldPhone;

        public bool PickUp(RoomAgent.Smartphone.SmartphoneController phone)
        {
            if (phone == null)
                return false;

            var interaction = phone.GetComponent<RoomAgent.Smartphone.SmartphoneInteraction>();
            if (interaction == null || !interaction.Interact(gameObject))
                return false;

            heldPhone = phone;
            return true;
        }

        public bool CanInteract(RoomAgent.Smartphone.SmartphoneController phone)
        {
            if (phone == null)
                return false;

            var interaction = phone.GetComponent<RoomAgent.Smartphone.SmartphoneInteraction>();
            return interaction != null && interaction.CanInteract(gameObject);
        }

        public bool PlaceAt(Vector3 worldPosition)
        {
            return PlaceAt(worldPosition, null);
        }

        public bool PlaceAt(Vector3 worldPosition, float placedYaw)
        {
            return PlaceAt(worldPosition, (float?)placedYaw);
        }

        private bool PlaceAt(Vector3 worldPosition, float? placedYaw)
        {
            if (heldPhone == null)
                return false;

            var placement = heldPhone.GetComponent<RoomAgent.Smartphone.SmartphonePlacement>();
            var placed = placedYaw.HasValue
                ? placement != null && placement.TryPlaceAt(worldPosition, placedYaw.Value)
                : placement != null && placement.TryPlaceAt(worldPosition);
            if (!placed)
                return false;

            heldPhone = null;
            return true;
        }

        /// <summary>
        /// The scripted turn owns the body's yaw, so the agent's own rotation is
        /// switched off for the duration and switched back afterwards.
        /// </summary>
        private void SuppressAgentRotation()
        {
            if (movement == null || agentRotationSuppressed)
                return;

            var agent = movement.Agent;
            if (agent == null)
                return;

            agent.updateRotation = false;
            agentRotationSuppressed = true;
        }

        private void RestoreAgentRotation()
        {
            if (!agentRotationSuppressed)
                return;

            agentRotationSuppressed = false;
            var agent = movement != null ? movement.Agent : null;
            if (agent != null)
                agent.updateRotation = true;
        }

        void Update()
        {
            if (!turning)
                return;

            turnTimer += Time.deltaTime;
            var direction = turnTarget - transform.position;
            direction.y = 0f;
            var targetRotation = direction.sqrMagnitude < 0.001f
                ? transform.rotation
                : Quaternion.LookRotation(direction.normalized, Vector3.up);

            transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRotation, turnSpeed * Time.deltaTime);
            if (Quaternion.Angle(transform.rotation, targetRotation) <= turnTolerance || turnTimer > turnTimeout)
            {
                turning = false;
                RestoreAgentRotation();
            }
        }

        void OnDisable()
        {
            turning = false;
            RestoreAgentRotation();
        }
    }
}
