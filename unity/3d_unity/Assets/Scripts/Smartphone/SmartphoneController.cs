using UnityEngine;

namespace RoomAgent.Smartphone
{
    public enum SmartphoneState
    {
        World,
        Held,
        Placed
    }

    [DisallowMultipleComponent]
    public sealed class SmartphoneController : MonoBehaviour
    {
        [SerializeField] private Transform handAnchor;
        [SerializeField] private Vector3 heldLocalPosition = new Vector3(-0.015f, 0.005f, 0.05f);
        [SerializeField] private Vector3 heldLocalEuler = Vector3.zero;

        private SmartphoneState currentState = SmartphoneState.World;
        private GameObject holder;
        private Collider placementSurface;
        private Rigidbody body;
        private Collider bodyCollider;

        void Awake()
        {
            body = GetComponent<Rigidbody>();
            bodyCollider = GetComponent<Collider>();
        }

        public SmartphoneState CurrentState => currentState;
        public GameObject Holder => holder;
        public bool IsHeld => currentState == SmartphoneState.Held;
        public Collider PlacementSurface => placementSurface;

        public bool PickUp(Transform anchor, GameObject newHolder = null)
        {
            if (anchor == null || currentState == SmartphoneState.Held)
                return false;

            SetSimulationEnabled(false);
            transform.SetParent(anchor, false);
            transform.localPosition = heldLocalPosition;
            transform.localRotation = Quaternion.Euler(heldLocalEuler);
            transform.localScale = Vector3.one;

            currentState = SmartphoneState.Held;
            holder = newHolder;
            placementSurface = null;
            return true;
        }

        public bool Release(Vector3 worldPosition, Quaternion worldRotation, Collider surface)
        {
            transform.SetParent(null, true);
            transform.SetPositionAndRotation(worldPosition, worldRotation);
            transform.localScale = Vector3.one;
            SetSimulationEnabled(true);

            currentState = surface == null ? SmartphoneState.World : SmartphoneState.Placed;
            holder = null;
            placementSurface = surface;
            return true;
        }

        private void SetSimulationEnabled(bool enabled)
        {
            if (body != null)
            {
                body.isKinematic = !enabled;
                body.useGravity = enabled;
            }

            if (bodyCollider != null)
                bodyCollider.enabled = enabled;
        }
    }
}
