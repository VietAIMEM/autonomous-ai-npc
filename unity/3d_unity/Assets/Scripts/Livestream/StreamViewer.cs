using System;

namespace RoomAgent.Livestream
{
    [Serializable]
    public sealed class StreamViewer
    {
        public string ViewerId;
        public string DisplayName;
        public DateTime JoinedAtUtc;
        public bool IsPresent;

        public StreamViewer(string viewerId, string displayName = null)
        {
            ViewerId = viewerId;
            DisplayName = string.IsNullOrWhiteSpace(displayName) ? viewerId : displayName;
            JoinedAtUtc = DateTime.UtcNow;
            IsPresent = true;
        }
    }
}
