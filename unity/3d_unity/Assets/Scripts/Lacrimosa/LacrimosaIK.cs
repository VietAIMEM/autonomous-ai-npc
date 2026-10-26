using UnityEngine;

namespace RoomAgent.Lacrimosa
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Animator))]
    public sealed class LacrimosaIK : MonoBehaviour
    {
        [SerializeField] private Transform headTarget;
        [SerializeField] private Transform leftHandTarget;
        [SerializeField] private Transform rightHandTarget;
        [SerializeField] private Transform rightHandRotationTarget;
        [SerializeField] private float headWeight = 0f;
        [SerializeField] private float leftHandWeight = 0f;
        [SerializeField] private float rightHandWeight = 0f;
        [SerializeField] private float rightHandRotationWeight = 0f;
        private LacrimosaLookAt lookAt;
        private LacrimosaBodyAnimator bodyAnimator;

        public Transform HeadTarget => headTarget;
        public Transform LeftHandTarget => leftHandTarget;
        public Transform RightHandTarget => rightHandTarget;
        public Transform RightHandRotationTarget => rightHandRotationTarget;

        // PHASE 28 diagnostics: the weights actually handed to the Mecanim IK
        // pass, so a caller can verify that an interaction restored the pose.
        public float HeadWeight => headWeight;

        public float LeftHandWeight => leftHandWeight;

        public float RightHandWeight => rightHandWeight;
        public float RightHandRotationWeight => rightHandRotationWeight;

        public bool IsActive => leftHandTarget != null || rightHandTarget != null;

        public void SetHeadTarget(Transform target, float weight = 1f)
        {
            headTarget = target;
            headWeight = weight;
            if (lookAt != null && target != null)
                lookAt.LookAt(target);
        }

        public void SetLeftHandTarget(Transform target, float weight = 1f)
        {
            leftHandTarget = target;
            leftHandWeight = weight;
        }

        public void SetRightHandTarget(Transform target, float weight = 1f)
        {
            rightHandTarget = target;
            rightHandWeight = weight;
        }

        public void SetRightHandRotationTarget(Transform target, float weight = 1f)
        {
            rightHandRotationTarget = target;
            rightHandRotationWeight = weight;
        }

        public void SetRightHandPoseWeight(float positionWeight, float rotationWeight)
        {
            rightHandWeight = Mathf.Clamp01(positionWeight);
            rightHandRotationWeight = Mathf.Clamp01(rotationWeight);
        }

        public void ClearTargets()
        {
            headTarget = null;
            leftHandTarget = null;
            rightHandTarget = null;
            rightHandRotationTarget = null;
            headWeight = 0f;
            leftHandWeight = 0f;
            rightHandWeight = 0f;
            rightHandRotationWeight = 0f;
            if (lookAt != null)
                lookAt.ClearLookAt();
        }

        void Awake()
        {
            lookAt = GetComponent<LacrimosaLookAt>();
            bodyAnimator = GetComponent<LacrimosaBodyAnimator>();
        }

        void OnAnimatorIK(int layerIndex)
        {
            var animator = GetComponent<Animator>();
            if (animator == null)
                return;

            var generatedClipActive = bodyAnimator != null && bodyAnimator.CurrentGeneratedClip != null;
            if (leftHandTarget != null)
            {
                animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, leftHandWeight);
                animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandTarget.position);
            }
            else if (!generatedClipActive)
            {
                animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 0f);
            }

            if (rightHandTarget != null)
            {
                animator.SetIKPositionWeight(AvatarIKGoal.RightHand, rightHandWeight);
                animator.SetIKPosition(AvatarIKGoal.RightHand, rightHandTarget.position);
            }
            else if (!generatedClipActive)
            {
                animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 0f);
            }

            if (rightHandRotationTarget != null)
            {
                animator.SetIKRotationWeight(AvatarIKGoal.RightHand, rightHandRotationWeight);
                animator.SetIKRotation(AvatarIKGoal.RightHand, rightHandRotationTarget.rotation);
            }
            else if (!generatedClipActive)
            {
                animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 0f);
            }
        }
    }
}
