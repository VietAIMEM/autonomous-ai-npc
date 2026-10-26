using UnityEngine;
using UnityEngine.AI;

namespace RoomAgent.Navigation
{
    /// <summary>
    /// Thin movement seam over a NavMeshAgent.
    /// Phase 01 only: it turns "go to this place" into agent commands and reports
    /// whether the walk finished. Higher level decision making lives elsewhere.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(NavMeshAgent))]
    public sealed class NavMeshWalker : MonoBehaviour
    {
        [Tooltip("How far away from the requested point is close enough.")]
        [SerializeField] float arrivalTolerance = 0.3f;

        [Tooltip("NavMesh points are snapped to the nearest walkable surface inside this radius.")]
        [SerializeField] float projectionRadius = 1.5f;

        NavMeshAgent agent;
        bool requestActive;

        public NavMeshAgent Agent => agent ? agent : (agent = GetComponent<NavMeshAgent>());
        public bool IsReady => Agent != null && Agent.enabled && Agent.isOnNavMesh;
        public bool IsNavigationPending => IsReady && requestActive && Agent.pathPending;
        public Vector3 Position => transform.position;
        public Vector3 Velocity => IsReady ? Agent.velocity : Vector3.zero;
        public bool HasRequestedDestination => requestActive;

        public bool IsMoving => IsNavigationPending || (IsReady && requestActive && Agent.hasPath && !Agent.isStopped
                                && Agent.remainingDistance < float.MaxValue * 0.5f
                                && Agent.remainingDistance > Agent.stoppingDistance);

        public bool HasArrived => IsReady && requestActive && !IsMoving;

        void Awake()
        {
            agent = GetComponent<NavMeshAgent>();
        }

        /// <summary>Path to a world position using the walker's default tolerance.</summary>
        public bool MoveTo(Vector3 worldTarget)
        {
            return MoveTo(worldTarget, arrivalTolerance);
        }

        /// <summary>Path to a world position, snapped onto the NavMesh. Returns false when unreachable.</summary>
        public bool MoveTo(Vector3 worldTarget, float tolerance)
        {
            if (!IsReady && !ReattachToNavMesh())
                return false;

            Agent.stoppingDistance = Mathf.Max(0f, tolerance);

            if (!NavMesh.SamplePosition(worldTarget, out var hit, projectionRadius, NavMesh.AllAreas))
                return false;

            Agent.isStopped = false;
            requestActive = Agent.SetDestination(hit.position);
            return requestActive;
        }

        public void Stop()
        {
            requestActive = false;
            if (IsReady)
                Agent.ResetPath();
        }

        bool ReattachToNavMesh()
        {
            if (IsReady)
                return true;
            if (Agent == null || !NavMesh.SamplePosition(transform.position, out var hit, projectionRadius * 4f, NavMesh.AllAreas))
                return false;
            return Agent.Warp(hit.position);
        }
    }
}
