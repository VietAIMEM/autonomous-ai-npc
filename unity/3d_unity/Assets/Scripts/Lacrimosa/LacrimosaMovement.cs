using UnityEngine;
using RoomAgent.Navigation;

namespace RoomAgent.Lacrimosa
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(UnityEngine.AI.NavMeshAgent))]
    public sealed class LacrimosaMovement : MonoBehaviour
    {
        [SerializeField] private float walkSpeed = 1.4f;
        [SerializeField] private float runSpeed = 3.0f;
        [SerializeField] private float rotationSpeed = 240f;
        [SerializeField] private float stoppingDistance = 0.3f;

        private UnityEngine.AI.NavMeshAgent agent;
        private NavMeshWalker walker;
        private LacrimosaBodyAnimator bodyAnimator;

        public UnityEngine.AI.NavMeshAgent Agent => agent ? agent : (agent = GetComponent<UnityEngine.AI.NavMeshAgent>());
        public bool IsNavigationPending => walker != null && walker.IsNavigationPending;
        public bool IsMoving => walker != null && walker.IsMoving;

        void Awake()
        {
            agent = GetComponent<UnityEngine.AI.NavMeshAgent>();
            walker = GetComponent<NavMeshWalker>();
            bodyAnimator = GetComponent<LacrimosaBodyAnimator>();
            ApplyDefaultMovement();
        }

        public bool MoveTo(Vector3 worldTarget, bool run = false)
        {
            if (walker == null)
                walker = GetComponent<NavMeshWalker>();

            Agent.speed = run ? runSpeed : walkSpeed;
            Agent.angularSpeed = rotationSpeed;
            return walker.MoveTo(worldTarget, stoppingDistance);
        }

        public void Stop()
        {
            if (walker != null)
                walker.Stop();
        }

        private void ApplyDefaultMovement()
        {
            if (Agent != null)
            {
                Agent.speed = walkSpeed;
                Agent.angularSpeed = rotationSpeed;
            }
        }

        void Update()
        {
            if (bodyAnimator != null && Agent != null)
                bodyAnimator.SetLocomotionSpeed(Agent.velocity.magnitude);
        }
    }
}
