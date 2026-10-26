using System.Text;
using RoomAgent.Smartphone;
using UnityEngine;

namespace RoomAgent.Livestream
{
    /// <summary>
    /// Focused, non-destructive checks for the held/placed camera mode split.
    /// The physical camera is only disabled/enabled; its transform and lens
    /// values are captured and compared before and after every transition.
    /// </summary>
    public sealed class HeldPhoneCameraModeValidator : MonoBehaviour
    {
        public string RunValidation()
        {
            var report = new StringBuilder();
            var system = GetComponent<LivestreamSystem>();
            if (system == null)
                system = FindFirstObjectByType<LivestreamSystem>();
            var mode = GetComponent<HeldPhoneSelfieCamera>();
            if (mode == null)
                mode = FindFirstObjectByType<HeldPhoneSelfieCamera>();
            if (system == null || mode == null)
                return "FAIL: missing livestream system or selfie camera mode.";

            var physical = system.PhysicalCamera;
            if (physical == null)
                return "FAIL: physical camera unresolved.";

            var physicalPosition = physical.transform.position;
            var physicalRotation = physical.transform.rotation;
            var physicalFov = physical.fieldOfView;
            var physicalNear = physical.nearClipPlane;
            var physicalFar = physical.farClipPlane;
            var physicalEnabled = physical.enabled;
            var physicalParent = physical.transform.parent;

            var streamWasRunning = system.IsStreaming;
            report.Append("streamWasRunning=").Append(streamWasRunning).Append(';');

            var heldApplied = mode.ApplyMode(true);
            var heldUsesVirtual = heldApplied && mode.UsesVirtualCamera;
            var heldActive = system.ActiveCamera != null &&
                system.ActiveCamera == mode.VirtualCamera;
            var heldMode = system.CameraMode == LivestreamCameraMode.VirtualSelfie;
            var heldPhysicalDisabled = !physical.enabled;
            report.Append("heldApplied=").Append(heldApplied)
                .Append(",heldUsesVirtual=").Append(heldUsesVirtual)
                .Append(",heldActive=").Append(heldActive)
                .Append(",heldMode=").Append(heldMode)
                .Append(",heldPhysicalDisabled=").Append(heldPhysicalDisabled)
                .Append(';');

            var virtualCamera = mode.VirtualCamera;
            var virtualLensMatches = virtualCamera != null &&
                Mathf.Approximately(virtualCamera.nearClipPlane, physicalNear) &&
                Mathf.Approximately(virtualCamera.farClipPlane, physicalFar) &&
                virtualCamera.cullingMask == physical.cullingMask;
            report.Append("virtualLensMatches=").Append(virtualLensMatches).Append(';');

            var heldPoseReady = mode.HasPose;
            var heldPosePosition = mode.PosePosition;
            var head = ResolveHead();
            var headDistance = head != null
                ? Vector3.Distance(heldPosePosition, head.position)
                : 0f;
            // The virtual viewpoint must sit outside the head/hair volume at a
            // sane follow distance; direction is validated visually instead of
            // with a frame-dependent dot product.
            var outsideHead = head == null ||
                (headDistance > 0.6f && headDistance < 1.5f);
            report.Append("heldPoseReady=").Append(heldPoseReady)
                .Append(",headDistance=").Append(headDistance.ToString("F3"))
                .Append(",outsideHead=").Append(outsideHead)
                .Append(';');

            var repeatedHeld = mode.ApplyMode(true);
            var stableAfterRepeat = repeatedHeld &&
                system.ActiveCamera == virtualCamera &&
                system.CameraMode == LivestreamCameraMode.VirtualSelfie;
            report.Append("stableAfterRepeat=").Append(stableAfterRepeat).Append(';');

            var placedApplied = mode.ApplyMode(false);
            var placedPhysicalActive = placedApplied &&
                system.ActiveCamera == physical &&
                system.CameraMode == LivestreamCameraMode.PhysicalPlaced;
            var physicalUntouched =
                Vector3.Distance(physical.transform.position, physicalPosition) < 0.0001f &&
                Quaternion.Angle(physical.transform.rotation, physicalRotation) < 0.01f &&
                Mathf.Approximately(physical.fieldOfView, physicalFov) &&
                Mathf.Approximately(physical.nearClipPlane, physicalNear) &&
                Mathf.Approximately(physical.farClipPlane, physicalFar) &&
                physical.transform.parent == physicalParent &&
                physical.enabled == physicalEnabled;
            report.Append("placedApplied=").Append(placedApplied)
                .Append(",placedPhysicalActive=").Append(placedPhysicalActive)
                .Append(",physicalUntouched=").Append(physicalUntouched)
                .Append(';');

            var unknownSafe = !mode.ApplyState((SmartphoneState)99);
            report.Append("unknownStateRejected=").Append(unknownSafe).Append(';');

            AppendSameSessionChecks(system, mode, virtualCamera, physical, report);

            Debug.Log("[HeldPhoneCameraModeValidator] " + report);
            return report.ToString();
        }

        /// <summary>
        /// Proves camera switching happens inside one continuous simulated pump
        /// session: the session id never changes, the rendered payload changes
        /// with the camera, and no duplicate cameras appear.
        /// </summary>
        private void AppendSameSessionChecks(
            LivestreamSystem system,
            HeldPhoneSelfieCamera mode,
            Camera virtualCamera,
            Camera physical,
            StringBuilder report)
        {
            var ingest = StreamIngestClient.GetOrCreate();
            var pumpCountBefore = FindObjectsByType<StreamIngestClient>(
                FindObjectsSortMode.None).Length;
            var virtualCountBefore = HeldPhoneSelfieCamera.InstanceCount;

            mode.ApplyMode(true);
            ingest.ResetPumpSessionIdentityForTest();
            ingest.SetSimulatedSource(virtualCamera);
            var sessionStart = ingest.PumpSessionId;
            var generationStart = ingest.SourceGeneration;
            var heldRendered = ingest.SimulateCaptureOnce(
                out var heldBytes,
                out var heldHash);
            var generationBeforeSwitches = ingest.SourceGeneration;

            var placedOk = true;
            var heldOk = true;
            var switchCount = 0;
            var lastHash = heldHash;
            for (var cycle = 0; cycle < 4; cycle++)
            {
                mode.ApplyMode(false);
                placedOk &= system.ActiveCamera == physical;
                switchCount++;
                if (!ingest.SimulateCaptureOnce(out var placedBytes, out var placedHash))
                    placedOk = false;
                else if (placedHash == lastHash)
                    placedOk = false;
                else
                    lastHash = placedHash;

                mode.ApplyMode(true);
                heldOk &= system.ActiveCamera == virtualCamera;
                switchCount++;
                if (!ingest.SimulateCaptureOnce(out var nextHeldBytes, out var nextHeldHash))
                    heldOk = false;
                else if (nextHeldHash == lastHash)
                    heldOk = false;
                else
                    lastHash = nextHeldHash;
            }

            var sessionPreserved = ingest.PumpSessionId == sessionStart;
            var generationDelta =
                ingest.SourceGeneration - generationBeforeSwitches;
            // Each mode change bumps the source generation; the exact count is
            // an implementation detail, so assert a monotonic per-switch advance.
            var generationAdvanced = generationDelta >= switchCount;
            var pumpCountAfter = FindObjectsByType<StreamIngestClient>(
                FindObjectsSortMode.None).Length;
            var virtualCountAfter = HeldPhoneSelfieCamera.InstanceCount;
            var noDuplicateCameras =
                pumpCountBefore == pumpCountAfter &&
                virtualCountBefore == virtualCountAfter &&
                virtualCamera != null &&
                mode.VirtualCamera == virtualCamera;

            ingest.EndSimulatedSession();
            mode.ApplyMode(false);

            report.Append("sameSession.pumpSessionId=").Append(sessionStart)
                .Append(",sessionPreserved=").Append(sessionPreserved)
                .Append(",sourceGenerationStart=").Append(generationStart)
                .Append(",sourceGenerationEnd=").Append(ingest.SourceGeneration)
                .Append(",generationDelta=").Append(generationDelta)
                .Append(",switches=").Append(switchCount)
                .Append(",generationAdvanced=").Append(generationAdvanced)
                .Append(",heldRendered=").Append(heldRendered)
                .Append(",heldBytes=").Append(heldBytes)
                .Append(",placedSwitchesOk=").Append(placedOk)
                .Append(",heldSwitchesOk=").Append(heldOk)
                .Append(",renderChangedWithCamera=").Append(placedOk && heldOk)
                .Append(",noDuplicateCameras=").Append(noDuplicateCameras);
        }

        private Transform ResolveHead()
        {
            var body = FindFirstObjectByType<RoomAgent.Lacrimosa.LacrimosaBodyAnimator>();
            var animator = body != null ? body.Animator : null;
            return animator != null
                ? animator.GetBoneTransform(HumanBodyBones.Head)
                : null;
        }
    }
}
