using System;
using RoomAgent.Lacrimosa;
using RoomAgent.Smartphone;
using UnityEngine;

namespace RoomAgent.Livestream
{
    /// <summary>
    /// Owns the held-phone virtual livestream viewpoint.
    ///
    /// This is deliberately a virtual camera, not the physical smartphone lens.
    /// The phone stays attached to the right hand and the physical camera stays
    /// parented under CameraAnchor; this component only decides which camera the
    /// livestream should render from and drives a separate runtime-only camera
    /// that follows the character's head and upper torso with smoothing.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class HeldPhoneSelfieCamera : MonoBehaviour
    {
        [Tooltip("Horizontal pull-back distance from the character's face.")]
        [SerializeField] private float distance = 0.85f;

        [Tooltip("Vertical offset above the character's face.")]
        [SerializeField] private float heightOffset = 0.18f;

        [Tooltip("Lateral offset in the character's local right direction.")]
        [SerializeField] private float lateralOffset = 0f;

        [Tooltip("Field of view used by the virtual selfie camera.")]
        [SerializeField] private float fieldOfView = 55f;

        [Tooltip("Seconds for the virtual camera to converge on its follow pose.")]
        [SerializeField] private float followSmoothingSeconds = 0.16f;

        private const float MinimumDistance = 0.6f;
        private const float MaximumDistance = 1.6f;
        private const float MinimumHeightOffset = -0.05f;
        private const float MaximumHeightOffset = 0.5f;

        private LivestreamSystem livestream;
        private SmartphoneController phone;
        private Animator animator;
        private Camera virtualCamera;
        private SmartphoneState lastState = SmartphoneState.World;
        private bool initialized;
        private bool usingVirtualCamera;
        private bool hasPose;
        private Vector3 posePosition;
        private Quaternion poseRotation;
        private static int instanceCount;

        public bool UsesVirtualCamera => usingVirtualCamera;
        public Camera VirtualCamera => virtualCamera;
        public Camera PhysicalCamera =>
            livestream != null ? livestream.PhysicalCamera : null;
        public bool HasPose => hasPose;
        public Vector3 PosePosition => posePosition;
        public Quaternion PoseRotation => poseRotation;
        /// <summary>Live controller count, used to prove no duplicates appear.</summary>
        public static int InstanceCount => instanceCount;

        private void Awake()
        {
            instanceCount++;
            if (livestream == null)
                livestream = GetComponent<LivestreamSystem>();
            ResolveReferences();
        }

        /// <summary>
        /// Wires the mode controller when it is installed by the livestream
        /// system at runtime, so the feature needs no saved scene reference.
        /// </summary>
        public void ConfigureReferences(
            LivestreamSystem system,
            SmartphoneController smartphone,
            LacrimosaBodyAnimator body)
        {
            livestream = system != null ? system : livestream;
            phone = smartphone != null ? smartphone : phone;
            if (body != null)
                animator = body.Animator;
            ResolveReferences();
        }

        private void ResolveReferences()
        {
            if (livestream == null)
                livestream = GetComponent<LivestreamSystem>();
            if (phone == null)
                phone = FindFirstObjectByType<SmartphoneController>();
            if (animator == null)
            {
                var body = FindFirstObjectByType<LacrimosaBodyAnimator>();
                animator = body != null ? body.Animator : null;
            }

            EnsureVirtualCamera();
            initialized = livestream != null && phone != null && animator != null;
            if (initialized)
                lastState = phone.CurrentState;
        }

        private void OnDestroy()
        {
            instanceCount = Mathf.Max(0, instanceCount - 1);
            if (virtualCamera != null)
            {
                virtualCamera.gameObject.SetActive(false);
                DestroyImmediate(virtualCamera.gameObject);
            }
        }

        private void Update()
        {
            if (!initialized)
                return;

            var current = phone.CurrentState;
            if (current != lastState)
            {
                lastState = current;
                ApplyState(current);
            }
            else if (usingVirtualCamera)
            {
                EnsureActiveCamera();
            }

            if (usingVirtualCamera)
                UpdateSelfiePose();
        }

        /// <summary>
        /// Applies a phone state to the camera mode. Called for every observed
        /// transition and safe to call repeatedly with the same state.
        /// </summary>
        public bool ApplyState(SmartphoneState state)
        {
            if (livestream == null)
                return false;

            switch (state)
            {
                case SmartphoneState.Held:
                    EnsureVirtualCamera();
                    if (virtualCamera == null ||
                        !livestream.SetCameraMode(
                            virtualCamera,
                            LivestreamCameraMode.VirtualSelfie))
                        return false;
                    usingVirtualCamera = true;
                    hasPose = false;
                    UpdateSelfiePose();
                    EnsureActiveCamera();
                    return true;

                case SmartphoneState.Placed:
                case SmartphoneState.World:
                    var restored = livestream.RestorePhysicalCameraMode();
                    usingVirtualCamera = false;
                    hasPose = false;
                    return restored;

                default:
                    return false;
            }
        }

        /// <summary>
        /// Used by callers that have no phone component, such as focused editor
        /// tests, to switch modes explicitly and deterministically.
        /// </summary>
        public bool ApplyMode(bool held)
        {
            return ApplyState(held ? SmartphoneState.Held : SmartphoneState.Placed);
        }

        private void EnsureActiveCamera()
        {
            if (livestream.ActiveCamera != virtualCamera)
                livestream.SetCameraMode(virtualCamera, LivestreamCameraMode.VirtualSelfie);
        }

        private void UpdateSelfiePose()
        {
            if (animator == null || virtualCamera == null)
                return;

            var head = animator.GetBoneTransform(HumanBodyBones.Head);
            var chest = animator.GetBoneTransform(HumanBodyBones.Chest);
            if (head == null || chest == null)
                return;

            var lookPoint = (head.position + chest.position) * 0.5f;
            var root = animator.transform;
            var forward = root.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 0.0001f)
                forward = Vector3.forward;
            forward.Normalize();
            var right = root.right;
            right.y = 0f;
            if (right.sqrMagnitude < 0.0001f)
                right = Vector3.right;
            right.Normalize();

            var clampedDistance = Mathf.Clamp(distance, MinimumDistance, MaximumDistance);
            var clampedHeight = Mathf.Clamp(
                heightOffset,
                MinimumHeightOffset,
                MaximumHeightOffset);
            var targetPosition =
                lookPoint +
                forward * clampedDistance +
                right * lateralOffset +
                Vector3.up * clampedHeight;
            var targetRotation = Quaternion.LookRotation(
                lookPoint - targetPosition,
                Vector3.up);

            if (!hasPose)
            {
                posePosition = targetPosition;
                poseRotation = targetRotation;
                hasPose = true;
            }
            else
            {
                var follow = SmoothingAlpha(followSmoothingSeconds);
                posePosition = Vector3.Lerp(posePosition, targetPosition, follow);
                poseRotation = Quaternion.Slerp(poseRotation, targetRotation, follow);
            }

            virtualCamera.transform.SetPositionAndRotation(posePosition, poseRotation);
        }

        private float SmoothingAlpha(float seconds)
        {
            var resolved = Mathf.Max(0.01f, seconds);
            return 1f - Mathf.Exp(-Time.deltaTime / resolved);
        }

        private void EnsureVirtualCamera()
        {
            if (virtualCamera != null)
                return;

            var physical = livestream != null ? livestream.PhysicalCamera : null;
            if (physical == null)
                return;

            var cameraObject = new GameObject("VirtualSelfieCamera")
            {
                hideFlags = HideFlags.DontSave
            };
            cameraObject.transform.SetParent(transform, false);
            var camera = cameraObject.AddComponent<Camera>();
            camera.enabled = false;
            camera.fieldOfView = Mathf.Clamp(fieldOfView, 25f, 90f);
            camera.nearClipPlane = physical.nearClipPlane;
            camera.farClipPlane = physical.farClipPlane;
            camera.clearFlags = physical.clearFlags;
            camera.backgroundColor = physical.backgroundColor;
            camera.cullingMask = physical.cullingMask;
            camera.depth = physical.depth;
            camera.allowHDR = physical.allowHDR;
            camera.allowMSAA = physical.allowMSAA;
            camera.orthographic = false;
            camera.targetTexture = null;
            camera.tag = physical.tag;

            var physicalData = physical.GetComponent<UnityEngine.Rendering.Universal.UniversalAdditionalCameraData>();
            if (physicalData != null)
            {
                var data = cameraObject.AddComponent<
                    UnityEngine.Rendering.Universal.UniversalAdditionalCameraData>();
                data.renderPostProcessing = physicalData.renderPostProcessing;
                data.antialiasing = physicalData.antialiasing;
                data.stopNaN = physicalData.stopNaN;
                data.dithering = physicalData.dithering;
                data.renderShadows = physicalData.renderShadows;
                data.volumeLayerMask = physicalData.volumeLayerMask;
                data.volumeTrigger = physicalData.volumeTrigger;
            }

            virtualCamera = camera;
        }
    }
}
