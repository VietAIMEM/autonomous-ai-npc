using System.Collections.Generic;
using System.Text;
using RoomAgent.Smartphone;
using UnityEngine;
using UnityEngine.AI;

namespace RoomAgent.Livestream
{
    public sealed class Phase04Validator : MonoBehaviour
    {
        public string RunValidation(Transform character, SmartphoneController phone)
        {
            var system = GetComponent<LivestreamSystem>();
            var viewers = GetComponent<StreamViewerManager>();
            var report = new StringBuilder();

            if (system == null || viewers == null || character == null || phone == null)
                return "FAIL: missing livestream system, viewer manager, character, or smartphone.";

            viewers.ClearViewers();
            system.StopStream();
            report.Append("initial=").Append(system.GetStreamState().IsStreaming ? "unexpected-active" : "stopped");
            report.Append(";");

            var phoneStartParent = phone.transform.parent;
            var phoneStartPosition = phone.transform.position;
            var phoneStartRotation = phone.transform.rotation;
            var characterStartPosition = character.position;
            var characterStartRotation = character.rotation;
            var handAnchor = ResolveHandAnchor(character);

            var heldPickup = phone.PickUp(handAnchor, character.gameObject);
            var heldStartCamera = system.ActiveCamera != null
                ? system.ActiveCamera.transform.position
                : Vector3.zero;
            character.position += Vector3.forward * 0.75f;
            character.rotation = Quaternion.Euler(0f, 45f, 0f);
            var heldMovedCamera = system.ActiveCamera != null
                ? system.ActiveCamera.transform.position
                : Vector3.zero;
            var heldCameraMoved = Vector3.Distance(heldStartCamera, heldMovedCamera) > 0.2f;
            var heldCameraTracksAnchor = system.ActiveCamera != null
                && Vector3.Distance(
                    system.ActiveCamera.transform.position,
                    phone.transform.Find("CameraAnchor").position) < 0.001f;
            phone.Release(phoneStartPosition, phoneStartRotation, null);
            character.SetPositionAndRotation(characterStartPosition, characterStartRotation);
            report.Append("heldPickup=").Append(heldPickup);
            report.Append(",heldCameraFollows=").Append(heldCameraMoved && heldCameraTracksAnchor);
            report.Append(";");

            var placedCameraStart = system.ActiveCamera != null
                ? system.ActiveCamera.transform.position
                : Vector3.zero;
            character.position += Vector3.right * 1.5f;
            var placedCameraEnd = system.ActiveCamera != null
                ? system.ActiveCamera.transform.position
                : Vector3.zero;
            var placedPhoneEnd = phone.transform.position;
            var placedCameraStatic = Vector3.Distance(placedCameraStart, placedCameraEnd) < 0.001f
                && Vector3.Distance(placedPhoneEnd, phoneStartPosition) < 0.001f;
            character.SetPositionAndRotation(characterStartPosition, characterStartRotation);
            report.Append("placedCameraStatic=").Append(placedCameraStatic);
            report.Append(";");

            var stateStart = system.StartStream();
            var durationBefore = system.DurationSeconds;
            System.Threading.Thread.Sleep(25);
            var durationAfter = system.DurationSeconds;
            var durationIncreased = stateStart && durationAfter > durationBefore;
            var stop1 = system.StopStream();
            var durationStopped = !system.IsStreaming && Mathf.Abs(system.DurationSeconds - durationAfter) < 0.001f;
            report.Append("streamStart=").Append(stateStart)
                .Append(",durationIncrease=").Append(durationIncreased)
                .Append(",streamStop=").Append(stop1)
                .Append(",durationPreserved=").Append(durationStopped);
            report.Append(";");

            var viewerJoinA = viewers.AddViewer("viewer_01", "Viewer01");
            var viewerJoinB = viewers.AddViewer("viewer_02", "Viewer02");
            var countTwo = viewers.ViewerCount == 2;
            var viewerLeaveA = viewers.RemoveViewer("viewer_01");
            var countOne = viewers.ViewerCount == 1;
            var viewerLeaveB = viewers.RemoveViewer("viewer_02");
            var countZero = viewers.ViewerCount == 0;
            report.Append("viewerJoin=").Append(viewerJoinA && viewerJoinB && countTwo)
                .Append(",viewerLeave=").Append(viewerLeaveA && viewerLeaveB && countOne && countZero);
            report.Append(";");

            var restart1 = system.StartStream() && system.ActiveCamera != null;
            var restartStop1 = system.StopStream();
            var restart2 = system.StartStream() && system.ActiveCamera != null;
            var restartStop2 = system.StopStream();
            var cameraCount = Object.FindObjectsOfType<Camera>().Length;
            report.Append("restart=").Append(restart1 && restartStop1 && restart2 && restartStop2)
                .Append(",singlePhone=").Append(Object.FindObjectsOfType<SmartphoneController>().Length == 1)
                .Append(",streamCameras=").Append(cameraCount);
            report.Append(";");

            phone.transform.SetParent(phoneStartParent, true);
            phone.transform.SetPositionAndRotation(phoneStartPosition, phoneStartRotation);
            character.SetPositionAndRotation(characterStartPosition, characterStartRotation);
            var finalState = system.GetStreamState();
            report.Append("finalStream=").Append(finalState.IsStreaming ? "active" : "stopped")
                .Append(",finalCameraActive=").Append(finalState.IsCameraActive);

            Debug.Log("[Phase04Validator] " + report);
            return report.ToString();
        }

        private static Transform ResolveHandAnchor(Transform character)
        {
            var animator = character.GetComponentInChildren<Animator>();
            var hand = animator != null ? animator.GetBoneTransform(HumanBodyBones.RightHand) : null;
            return hand != null ? hand : character;
        }
    }
}
