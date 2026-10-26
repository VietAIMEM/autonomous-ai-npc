using UnityEngine;

namespace RoomAgent.Lacrimosa
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Animator))]
    public sealed class LacrimosaLookAt : MonoBehaviour
    {
        [SerializeField] private float blendSpeed = 6f;
        [SerializeField] private float maxWeight = 1f;
        [SerializeField] private float bodyWeight = 0.35f;
        [SerializeField] private float headWeight = 1f;
        [SerializeField] private float eyesWeight = 0.2f;
        [SerializeField] private float clampWeight = 0.75f;

        private Transform target;
        private Vector3 targetPosition;
        private bool active;
        private float weight;

        public bool HasTarget => target != null;

        // PHASE 28 diagnostics: the live look-at state, read directly.
        public bool IsActive => active;

        public float CurrentWeight => weight;

        public Vector3 TargetPosition => targetPosition;

        public Transform Target => target;

        public void LookAt(Transform newTarget)
        {
            active = true;
            target = newTarget;
            targetPosition = newTarget != null ? newTarget.position : targetPosition;
        }

        public void LookAt(Vector3 worldPosition)
        {
            active = true;
            target = null;
            targetPosition = worldPosition;
        }

        public void ClearLookAt()
        {
            target = null;
            active = false;
        }

        void OnAnimatorIK(int layerIndex)
        {
            var animator = GetComponent<Animator>();
            if (animator == null)
                return;

            if (target != null)
                targetPosition = target.position;

            var desiredWeight = active ? maxWeight : 0f;
            weight = Mathf.MoveTowards(weight, desiredWeight, blendSpeed * Time.deltaTime);
            animator.SetLookAtWeight(weight, bodyWeight, headWeight, eyesWeight, clampWeight);

            if (weight > 0.01f)
                animator.SetLookAtPosition(targetPosition);
        }
    }
}
