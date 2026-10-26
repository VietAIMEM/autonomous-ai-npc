using RoomAgent.Interaction;
using UnityEngine;

namespace RoomAgent.Smartphone
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(SmartphoneController))]
    public sealed class SmartphonePlacement : MonoBehaviour
    {
        private const float PlacementBoundaryEpsilon = 0.001f;

        [SerializeField] private float edgeMargin = 0.035f;
        [SerializeField] private float surfaceCosineThreshold = 0.7f;
        [SerializeField] private float maxRaycastDistance = 4f;

        public bool TryPlaceAt(Vector3 desiredPosition)
        {
            return TryPlaceAt(desiredPosition, transform.eulerAngles.y);
        }

        public bool TryPlaceAt(Vector3 desiredPosition, float placedYaw)
        {
            return TryPlaceAt(desiredPosition, placedYaw, out _, out _, out _);
        }

        /// <summary>
        /// Chooses the furthest valid placement point from a desired surface
        /// point along a requested horizontal direction. This keeps a virtual
        /// camera near the table edge instead of looking across the whole top.
        /// </summary>
        public bool TryResolveEdgePlacementPoint(
            Vector3 desiredPosition,
            Vector3 desiredDirection,
            out Vector3 edgePosition)
        {
            edgePosition = desiredPosition;

            var phoneCollider = GetComponent<Collider>();
            if (phoneCollider == null ||
                !TryResolvePlacementSurface(desiredPosition, out var hit, out _))
                return false;

            var direction = Vector3.ProjectOnPlane(desiredDirection, hit.normal);
            direction.y = 0f;
            if (direction.sqrMagnitude < 0.0001f)
                return true;
            direction.Normalize();

            var bounds = hit.collider.bounds;
            var horizontalExtent = Mathf.Max(
                phoneCollider.bounds.extents.x,
                phoneCollider.bounds.extents.z);
            var minimumInside = horizontalExtent + edgeMargin;
            var minimumX = bounds.min.x + minimumInside;
            var maximumX = bounds.max.x - minimumInside;
            var minimumZ = bounds.min.z + minimumInside;
            var maximumZ = bounds.max.z - minimumInside;
            if (minimumX > maximumX || minimumZ > maximumZ)
                return false;

            var distance = float.PositiveInfinity;
            if (direction.x > 0.0001f)
                distance = Mathf.Min(distance, (maximumX - desiredPosition.x) / direction.x);
            else if (direction.x < -0.0001f)
                distance = Mathf.Min(distance, (minimumX - desiredPosition.x) / direction.x);

            if (direction.z > 0.0001f)
                distance = Mathf.Min(distance, (maximumZ - desiredPosition.z) / direction.z);
            else if (direction.z < -0.0001f)
                distance = Mathf.Min(distance, (minimumZ - desiredPosition.z) / direction.z);

            if (distance < 0f || distance == float.PositiveInfinity)
                return false;

            distance = Mathf.Max(0f, distance - PlacementBoundaryEpsilon);
            edgePosition = desiredPosition + direction * distance;
            edgePosition.x = Mathf.Clamp(edgePosition.x, minimumX, maximumX);
            edgePosition.z = Mathf.Clamp(edgePosition.z, minimumZ, maximumZ);
            edgePosition.y = desiredPosition.y;
            return true;
        }

        public bool TryPlaceAt(
            Vector3 desiredPosition,
            out Vector3 placedPosition,
            out Quaternion placedRotation,
            out Collider surfaceCollider)
        {
            return TryPlaceAt(
                desiredPosition,
                transform.eulerAngles.y,
                out placedPosition,
                out placedRotation,
                out surfaceCollider);
        }

        public bool TryPlaceAt(
            Vector3 desiredPosition,
            float placedYaw,
            out Vector3 placedPosition,
            out Quaternion placedRotation,
            out Collider surfaceCollider)
        {
            placedPosition = default;
            placedRotation = default;
            surfaceCollider = null;

            var phoneCollider = GetComponent<Collider>();
            if (phoneCollider == null
                || !TryResolvePlacementSurface(desiredPosition, out var hit, out _))
                return false;

            var bounds = hit.collider.bounds;
            var horizontalExtent = Mathf.Max(
                phoneCollider.bounds.extents.x,
                phoneCollider.bounds.extents.z);
            var minimumInside = horizontalExtent + edgeMargin;
            if (desiredPosition.x < bounds.min.x + minimumInside
                || desiredPosition.x > bounds.max.x - minimumInside
                || desiredPosition.z < bounds.min.z + minimumInside
                || desiredPosition.z > bounds.max.z - minimumInside)
                return false;

            placedPosition = hit.point + hit.normal * (phoneCollider.bounds.extents.y + 0.001f);
            placedRotation = Quaternion.Euler(0f, placedYaw, 0f);
            surfaceCollider = hit.collider;
            return GetComponent<SmartphoneController>().Release(placedPosition, placedRotation, hit.collider);
        }

        private bool TryResolvePlacementSurface(
            Vector3 desiredPosition,
            out RaycastHit hit,
            out InteractionSurface surface)
        {
            hit = default;
            surface = null;

            var origin = new Vector3(desiredPosition.x, 2.5f, desiredPosition.z);
            var hits = Physics.RaycastAll(
                origin,
                Vector3.down,
                maxRaycastDistance,
                ~0,
                QueryTriggerInteraction.Ignore);
            var nearestDistance = float.PositiveInfinity;
            foreach (var candidate in hits)
            {
                var candidateSurface = candidate.collider.GetComponent<InteractionSurface>();
                if (candidateSurface == null
                    || !candidateSurface.AcceptsPlacement
                    || Vector3.Dot(candidate.normal, Vector3.up) < surfaceCosineThreshold
                    || candidate.distance >= nearestDistance)
                    continue;

                nearestDistance = candidate.distance;
                hit = candidate;
                surface = candidateSurface;
            }

            return surface != null;
        }
    }
}
