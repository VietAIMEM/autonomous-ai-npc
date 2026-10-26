using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace RoomAgent.Brain
{
    public static class BrainChatSubmitter
    {
        public static bool Submit(string viewerId, string text)
        {
            return RunCoroutine(SubmitAsync(viewerId, text));
        }

        public static IEnumerator SubmitAsync(string viewerId, string text)
        {
            if (string.IsNullOrWhiteSpace(viewerId) || string.IsNullOrWhiteSpace(text))
                yield break;

            var payload = $"{{\"viewer_id\":{JsonUtilityQuote(viewerId)},\"text\":{JsonUtilityQuote(text)}}}";
            using var request = BuildRequest("http://127.0.0.1:8765/chat/message", payload);
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[BrainChatSubmitter] chat submit failed: {request.error}");
                yield break;
            }

            Debug.Log("[BrainChatSubmitter] chat message accepted");
        }

        private static UnityWebRequest BuildRequest(string url, string payload)
        {
            var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST)
            {
                downloadHandler = new DownloadHandlerBuffer(),
                timeout = 5
            };
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(payload));
            request.SetRequestHeader("Content-Type", "application/json");
            return request;
        }

        private static string JsonUtilityQuote(string value)
        {
            var safeValue = value ?? string.Empty;
            return "\"" + safeValue
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n")
                .Replace("\t", "\\t") + "\"";
        }

        private static bool RunCoroutine(IEnumerator coroutine)
        {
            var host = new GameObject("BrainChatSubmitterHost");
            UnityEngine.Object.DontDestroyOnLoad(host);
            host.AddComponent<BrainChatSubmitterHost>().Run(coroutine);
            return true;
        }

        private sealed class BrainChatSubmitterHost : MonoBehaviour
        {
            public void Run(IEnumerator coroutine)
            {
                StartCoroutine(RunAndDestroy(coroutine));
            }

            private IEnumerator RunAndDestroy(IEnumerator coroutine)
            {
                yield return coroutine;
                Destroy(gameObject);
            }

            private void OnDestroy()
            {
                StopAllCoroutines();
            }
        }
    }
}
