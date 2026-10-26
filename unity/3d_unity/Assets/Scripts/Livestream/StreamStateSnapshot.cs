using System;
using UnityEngine;

namespace RoomAgent.Livestream
{
    [Serializable]
    public sealed class StreamStateSnapshot
    {
        public bool IsStreaming;
        public bool HasCamera;
        public bool IsCameraActive;
        public string CameraName;
        public float DurationSeconds;
        public DateTime? StartedUtc;
        public int ViewerCount;
        public bool HasTarget;
        public string TargetName;
        public Vector3 CameraPosition;
        public Vector3 CameraRotationEuler;
        public float FieldOfView;
    }
}
