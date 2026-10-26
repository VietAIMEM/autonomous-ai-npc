using UnityEngine;

namespace RoomAgent.Speech
{
    /// <summary>
    /// Owns the single voice AudioSource for Lacrimosa. The host and source are
    /// created at runtime so the saved scene never changes.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class SpeechAudioPlayer : MonoBehaviour
    {
        private AudioSource source;

        public static SpeechAudioPlayer GetOrCreate(Transform parent)
        {
            var anchor = parent != null ? parent : GameObject.Find("Lacrimosa")?.transform;
            if (anchor == null)
            {
                Debug.LogWarning("[SpeechAudioPlayer] no Lacrimosa transform to attach to");
                return null;
            }

            var existing = anchor.GetComponentInChildren<SpeechAudioPlayer>(true);
            if (existing != null)
                return existing;

            var host = new GameObject("SpeechAudioPlayer");
            host.transform.SetParent(anchor, false);
            var player = host.AddComponent<SpeechAudioPlayer>();
            player.EnsureSource();
            return player;
        }

        private void Awake() => EnsureSource();

        private void EnsureSource()
        {
            if (source != null)
                return;

            source = GetComponent<AudioSource>();
            if (source == null)
                source = gameObject.AddComponent<AudioSource>();

            source.playOnAwake = false;
            source.loop = false;
            source.spatialBlend = 0f;
            source.mute = false;
            source.volume = 1f;
        }

        public bool IsPlaying => source != null && source.isPlaying;

        public float PlayedSeconds => source != null ? source.time : 0f;

        public AudioClip CurrentClip => source != null ? source.clip : null;

        /// <summary>False when the clip is missing or decoded to nothing.</summary>
        public bool Play(AudioClip clip)
        {
            EnsureSource();
            if (!IsPlayable(clip))
            {
                Debug.LogWarning("[SpeechAudioPlayer] refused to play an invalid audio clip");
                return false;
            }

            source.Stop();
            source.clip = clip;
            source.time = 0f;
            source.Play();
            Debug.Log($"[SpeechAudioPlayer] playback started {Describe(clip)}");
            return source.isPlaying || source.time > 0f;
        }

        public void Stop()
        {
            EnsureSource();
            source.Stop();
        }

        // A UnityWebRequestAudio decode that yields no samples still hands back a
        // non-null clip, so duration and rate are the honest validity signal.
        public static bool IsPlayable(AudioClip clip) =>
            clip != null && !ReferenceEquals(clip, null) && clip.length > 0.01f && clip.frequency > 0;

        public static string Describe(AudioClip clip) =>
            clip == null || ReferenceEquals(clip, null)
                ? "clip:<none>"
                : $"clip:{clip.name} length:{clip.length:F2}s rate:{clip.frequency} " +
                  $"channels:{clip.channels}";
    }
}
