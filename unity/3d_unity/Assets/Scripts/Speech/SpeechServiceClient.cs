using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace RoomAgent.Speech
{
    /// <summary>
    /// Talks to the backend speech service. Unity never reaches the TTS
    /// provider directly: the backend owns credentials and synthesis, Unity
    /// only asks for the audio the Brain already requested.
    /// </summary>
    public static class SpeechServiceClient
    {
        public const string DefaultBaseUrl = "http://127.0.0.1:8766";

        public static IEnumerator FetchState(
            string baseUrl,
            string requestId,
            int timeoutSeconds,
            Action<SpeechStatePayload> onDone)
        {
            SpeechStatePayload parsed = null;
            var url = $"{Normalize(baseUrl)}/speech/state/{Uri.EscapeDataString(requestId ?? string.Empty)}";
            using var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET)
            {
                downloadHandler = new DownloadHandlerBuffer(),
                timeout = Mathf.Max(1, timeoutSeconds)
            };
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    parsed = JsonUtility.FromJson<SpeechStatePayload>(request.downloadHandler.text);
                }
                catch (Exception exception)
                {
                    Debug.LogWarning($"[SpeechServiceClient] invalid state payload: {exception.Message}");
                }
            }
            else
            {
                Debug.LogWarning($"[SpeechServiceClient] state request failed: {request.error}");
            }

            onDone?.Invoke(parsed);
        }

        public static IEnumerator PostFlag(
            string baseUrl,
            string operation,
            string requestId,
            int timeoutSeconds,
            Action<bool> onDone)
        {
            var url = $"{Normalize(baseUrl)}/speech/{operation}/{Uri.EscapeDataString(requestId ?? string.Empty)}";
            using var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST)
            {
                downloadHandler = new DownloadHandlerBuffer(),
                uploadHandler = new UploadHandlerRaw(Array.Empty<byte>()),
                timeout = Mathf.Max(1, timeoutSeconds)
            };
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();

            var accepted = request.result == UnityWebRequest.Result.Success;
            if (!accepted)
                Debug.LogWarning($"[SpeechServiceClient] {operation} failed: {request.error}");
            onDone?.Invoke(accepted);
        }

        public static IEnumerator DownloadClip(
            string baseUrl,
            string requestId,
            AudioType audioType,
            int timeoutSeconds,
            Action<AudioClip> onDone)
        {
            AudioClip clip = null;
            var url = $"{Normalize(baseUrl)}/speech/audio/{Uri.EscapeDataString(requestId ?? string.Empty)}";
            using var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET)
            {
                downloadHandler = new DownloadHandlerAudioClip(url, audioType),
                timeout = Mathf.Max(1, timeoutSeconds)
            };
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
                clip = DownloadHandlerAudioClip.GetContent(request);

            if (clip == null)
                Debug.LogWarning($"[SpeechServiceClient] clip download failed: {request.error}");

            onDone?.Invoke(clip);
        }

        public static AudioType ResolveAudioType(string format)
        {
            switch ((format ?? string.Empty).Trim().ToLowerInvariant())
            {
                case "wav":
                    return AudioType.WAV;
                case "ogg":
                    return AudioType.OGGVORBIS;
                default:
                    return AudioType.MPEG;
            }
        }

        private static string Normalize(string baseUrl) =>
            string.IsNullOrWhiteSpace(baseUrl) ? DefaultBaseUrl : baseUrl.TrimEnd('/');
    }

    [Serializable]
    public sealed class SpeechStatePayload
    {
        public string request_id;
        public string state;
        public string source;
        public string text;
        public string provider;
        public string model;
        public string voice;
        public string format;
        public string error;
        public int attempt;
        public int byte_length;
        public float duration_seconds;
        public SpeechMetricsPayload metrics;
    }

    [Serializable]
    public sealed class SpeechMetricsPayload
    {
        public float time_to_first_audio_ms;
        public float synthesis_ms;
        public float playback_ms;
        public float total_ms;
    }
}
