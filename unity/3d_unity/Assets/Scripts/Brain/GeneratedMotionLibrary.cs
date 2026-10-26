using System;
using System.Collections.Generic;
using UnityEngine;

namespace RoomAgent.GeneratedMotions
{
    [CreateAssetMenu(menuName = "Room Agent/Generated Motion Library")]
    public sealed class GeneratedMotionLibrary : ScriptableObject
    {
        [Serializable]
        public sealed class Motion
        {
            [SerializeField] private string sourceReference;
            [SerializeField] private AnimationClip generatedClip;

            public string SourceReference
            {
                get => sourceReference;
                set => sourceReference = value;
            }

            public AnimationClip GeneratedClip
            {
                get => generatedClip;
                set => generatedClip = value;
            }
        }

        [SerializeField] private List<Motion> motions = new List<Motion>();

        public IReadOnlyList<Motion> Motions => motions;

        public AnimationClip ResolveClip(string sourceReference)
        {
            var normalized = Normalize(sourceReference);
            if (normalized == null)
                return null;

            foreach (var motion in motions)
            {
                if (Normalize(motion.SourceReference) == normalized)
                    return motion.GeneratedClip;
            }
            return null;
        }

        public void Configure(IEnumerable<Motion> configuredMotions)
        {
            motions.Clear();
            foreach (var motion in configuredMotions)
                motions.Add(motion);
        }

        public void Upsert(Motion motion)
        {
            var normalized = Normalize(motion.SourceReference);
            for (var index = 0; index < motions.Count; index++)
            {
                if (Normalize(motions[index].SourceReference) != normalized)
                    continue;

                motions[index] = motion;
                return;
            }

            motions.Add(motion);
        }

        private static string Normalize(string value)
        {
            var normalized = (value ?? string.Empty).Trim().Replace('\\', '/');
            while (normalized.EndsWith("/", StringComparison.Ordinal))
                normalized = normalized.Substring(0, normalized.Length - 1);
            return normalized.Length == 0 ? null : normalized;
        }
    }
}
