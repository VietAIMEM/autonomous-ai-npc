using System;
using RoomAgent.Smartphone;
using UnityEngine;

namespace RoomAgent.Livestream
{
    /// <summary>
    /// Which viewpoint the livestream is rendered from. Placed uses the physical
    /// smartphone camera; selfie uses a virtual viewpoint near the character so
    /// the held composition never depends on an unreachable physical lens.
    /// </summary>
    public enum LivestreamCameraMode
    {
        Uninitialized,
        PhysicalPlaced,
        VirtualSelfie
    }

    [DisallowMultipleComponent]
    public sealed class LivestreamSystem : MonoBehaviour
    {
        [SerializeField] private Camera streamCamera;
        [SerializeField] private StreamViewerManager viewerManager;
        [SerializeField] private Transform currentTarget;
        [SerializeField] private float defaultFieldOfView = 60f;

        private bool isStreaming;
        private float completedDuration;
        private float startedRealtime;
        private DateTime startedUtc;
        private Camera physicalCamera;
        private LivestreamCameraMode cameraMode = LivestreamCameraMode.Uninitialized;

        public bool IsStreaming => isStreaming;
        public bool HasCamera => streamCamera != null;
        public bool IsCameraActive => streamCamera != null
            && streamCamera.enabled
            && streamCamera.gameObject.activeInHierarchy;
        public Camera ActiveCamera => streamCamera;
        public Camera PhysicalCamera =>
            physicalCamera != null ? physicalCamera : streamCamera;
        public LivestreamCameraMode CameraMode => cameraMode;
        public bool IsVirtualSelfieCamera =>
            cameraMode == LivestreamCameraMode.VirtualSelfie;
        public Transform Target => currentTarget;
        public StreamViewerManager Viewers => EnsureViewerManager();
        public float DurationSeconds => isStreaming
            ? Time.realtimeSinceStartup - startedRealtime
            : completedDuration;

        private void Awake()
        {
            EnsureViewerManager();
            physicalCamera = streamCamera;
            ConfigureCamera(streamCamera);
            DeactivateCamera();
            if (streamCamera != null)
                cameraMode = LivestreamCameraMode.PhysicalPlaced;
            InstallPhoneCameraMode();
        }

        /// <summary>
        /// Ensures the held/placed camera mode controller exists without editing
        /// the saved scene. The controller is added at runtime and owns a
        /// runtime-only virtual selfie camera, so nothing is serialized.
        /// </summary>
        private void InstallPhoneCameraMode()
        {
            if (GetComponent<HeldPhoneSelfieCamera>() != null)
                return;

            var controller = gameObject.AddComponent<HeldPhoneSelfieCamera>();
            controller.ConfigureReferences(
                this,
                FindFirstObjectByType<RoomAgent.Smartphone.SmartphoneController>(),
                FindFirstObjectByType<RoomAgent.Lacrimosa.LacrimosaBodyAnimator>());
        }

        public bool StartStream(Camera requestedCamera = null)
        {
            if (isStreaming)
                return false;

            var camera = requestedCamera != null ? requestedCamera : streamCamera;
            if (!SetCamera(camera))
                return false;

            ConfigureCamera(camera);
            ActivateCamera();
            if (camera != physicalCamera)
                cameraMode = LivestreamCameraMode.VirtualSelfie;
            else
                cameraMode = LivestreamCameraMode.PhysicalPlaced;
            isStreaming = true;
            completedDuration = 0f;
            startedRealtime = Time.realtimeSinceStartup;
            startedUtc = DateTime.UtcNow;
            return true;
        }

        public bool StopStream()
        {
            if (!isStreaming)
                return false;

            completedDuration = DurationSeconds;
            isStreaming = false;
            DeactivateCamera();
            return true;
        }

        /// <summary>
        /// Reconciles the logical flag with the real transport after an
        /// out-of-band event such as a backend restart. Only the "transport is
        /// not live" direction is applied: a live transport with a stopped local
        /// flag is left to the Brain's stream-maintenance goal, so Unity never
        /// claims a stream it is not actually feeding.
        /// </summary>
        public bool ApplyTransportState(bool backendLive)
        {
            if (backendLive)
                return false;

            return StopStream();
        }

        public bool SetCamera(Camera camera)
        {
            if (camera == null)
                return false;

            var previousCamera = streamCamera;
            streamCamera = camera;
            if (physicalCamera == null && cameraMode == LivestreamCameraMode.Uninitialized)
            {
                physicalCamera = camera;
                cameraMode = LivestreamCameraMode.PhysicalPlaced;
            }

            if (isStreaming && previousCamera != null && previousCamera != camera)
            {
                previousCamera.enabled = false;
                ConfigureCamera(camera);
                camera.enabled = true;
            }

            RebindRunningPump(camera);
            return true;
        }

        /// <summary>
        /// Switches the active camera between the physical placed camera and a
        /// virtual selfie camera. The physical camera's transform is never
        /// touched; only its enabled state changes while the stream is stopped.
        /// </summary>
        public bool SetCameraMode(Camera targetCamera, LivestreamCameraMode mode)
        {
            if (targetCamera == null)
                return false;

            if (physicalCamera == null)
                physicalCamera = streamCamera;

            var previousCamera = streamCamera;
            streamCamera = targetCamera;
            cameraMode = mode;

            if (!isStreaming)
            {
                if (previousCamera != null && previousCamera != targetCamera)
                    previousCamera.enabled = false;
                targetCamera.enabled = false;
                RebindRunningPump(targetCamera);
                return true;
            }

            if (previousCamera != targetCamera)
            {
                if (previousCamera != null)
                    previousCamera.enabled = false;
                ConfigureCamera(targetCamera);
                targetCamera.enabled = true;
            }

            RebindRunningPump(targetCamera);
            return true;
        }

        /// <summary>
        /// Keeps an already-running capture pump pointed at the selected camera.
        /// The swap is a single field update on the main thread and does not
        /// restart the pump, reset its counters, or create a second encoder.
        /// </summary>
        private void RebindRunningPump(Camera camera)
        {
            var ingest = StreamIngestClient.Instance;
            if (ingest == null || !ingest.IsPumping)
                return;
            ingest.SetSourceCamera(camera);
        }

        /// <summary>
        /// Restores the physical placed camera without reconfiguring it, so its
        /// saved FOV, clipping planes, and transform remain byte-for-byte intact.
        /// </summary>
        public bool RestorePhysicalCameraMode()
        {
            if (physicalCamera == null)
                return false;

            return SetCameraMode(physicalCamera, LivestreamCameraMode.PhysicalPlaced);
        }

        public void SetTarget(Transform target)
        {
            currentTarget = target;
        }

        public void ClearTarget()
        {
            currentTarget = null;
        }

        public StreamStateSnapshot GetStreamState()
        {
            var snapshot = new StreamStateSnapshot
            {
                IsStreaming = isStreaming,
                HasCamera = streamCamera != null,
                IsCameraActive = IsCameraActive,
                CameraName = streamCamera != null ? streamCamera.name : string.Empty,
                DurationSeconds = DurationSeconds,
                StartedUtc = isStreaming ? startedUtc : (DateTime?)null,
                ViewerCount = EnsureViewerManager().ViewerCount,
                HasTarget = currentTarget != null,
                TargetName = currentTarget != null ? currentTarget.name : string.Empty
            };

            if (streamCamera != null)
            {
                snapshot.CameraPosition = streamCamera.transform.position;
                snapshot.CameraRotationEuler = streamCamera.transform.eulerAngles;
                snapshot.FieldOfView = streamCamera.fieldOfView;
            }

            return snapshot;
        }

        private void ConfigureCamera(Camera camera)
        {
            if (camera != null && defaultFieldOfView > 0f)
                camera.fieldOfView = defaultFieldOfView;
        }

        private void ActivateCamera()
        {
            if (streamCamera != null)
                streamCamera.enabled = true;
        }

        private void DeactivateCamera()
        {
            if (streamCamera != null)
                streamCamera.enabled = false;
        }

        private StreamViewerManager EnsureViewerManager()
        {
            if (viewerManager == null)
                viewerManager = GetComponent<StreamViewerManager>();

            return viewerManager;
        }
    }
}
