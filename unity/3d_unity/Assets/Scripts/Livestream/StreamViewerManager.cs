using System;
using System.Collections.Generic;
using UnityEngine;

namespace RoomAgent.Livestream
{
    [DisallowMultipleComponent]
    public sealed class StreamViewerManager : MonoBehaviour
    {
        [SerializeField] private List<StreamViewer> viewers = new List<StreamViewer>();

        public IReadOnlyList<StreamViewer> Viewers => viewers;
        public int ViewerCount => viewers.Count;

        public bool AddViewer(string viewerId, string displayName = null)
        {
            if (string.IsNullOrWhiteSpace(viewerId) || FindViewer(viewerId) != null)
                return false;

            viewers.Add(new StreamViewer(viewerId, displayName));
            return true;
        }

        public bool RemoveViewer(string viewerId)
        {
            var viewer = FindViewer(viewerId);
            return viewer != null && viewers.Remove(viewer);
        }

        public StreamViewer GetViewer(string viewerId)
        {
            return FindViewer(viewerId);
        }

        public void ClearViewers()
        {
            viewers.Clear();
        }

        private StreamViewer FindViewer(string viewerId)
        {
            if (string.IsNullOrWhiteSpace(viewerId))
                return null;

            return viewers.Find(candidate => string.Equals(
                candidate.ViewerId,
                viewerId,
                StringComparison.OrdinalIgnoreCase));
        }
    }
}
