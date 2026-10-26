using UnityEngine;

namespace RoomAgent.Livestream
{
    /// <summary>
    /// Forwards whatever Unity is actually about to output for this GameObject's
    /// AudioSource into the livestream PCM ring. Attached at runtime to the
    /// SpeechAudioPlayer host, never in the editor, so the scene stays clean.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class StreamAudioTap : MonoBehaviour
    {
        private void OnAudioFilterRead(float[] data, int channels)
        {
            var client = StreamIngestClient.Instance;
            if (client != null && client.IsCapturing)
                client.PushAudio(data, channels);
        }
    }
}
