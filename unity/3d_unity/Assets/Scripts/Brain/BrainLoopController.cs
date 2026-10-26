using System;
using System.Collections;
using System.Text;
using RoomAgent.Brain;
using UnityEngine;
using UnityEngine.Networking;

namespace RoomAgent.Brain
{
    [DisallowMultipleComponent]
    public sealed class BrainLoopController : MonoBehaviour
    {
        [SerializeField] private string tickUrl = "http://127.0.0.1:8766/brain/tick";
        [SerializeField] private float decisionIntervalSeconds = 5f;
        [SerializeField] private float requestTimeoutSeconds = 5f;
        [SerializeField] private bool startAutomatically = true;

        // Every bridge action path is bounded in time (the longest, speech
        // synthesis plus playback, is under a minute), so a tick that outlives
        // this bound no longer has a coroutine behind it. A script recompile
        // during Play Mode reloads the domain and kills in-flight coroutines
        // without ever invoking their callbacks; without this bound the flag
        // they set would stay raised and the Brain loop would wedge forever.
        [SerializeField] private float abandonedTickSeconds = 120f;

        private BrainActionBridge bridge;
        private bool requestInFlight;
        private float requestStartedAt;
        private float nextTickTime;
        private ActionResultSnapshot lastResult;
        private string pendingViewerResponse = "none";

        //: PHASE 33: longest failure-reason token forwarded to the backend.
        private const int MaxFailureReasonLength = 64;

        //: PHASE 35: the smallest HTTP timeout a Brain tick may use, in
        //: seconds. The backend plans synchronously and its provider transport
        //: is allowed to block for up to 120 s before it gives up, so a tick
        //: abandoned earlier discards a plan the backend may still return.
        //: 135 s keeps a 15 s margin over that provider budget; this is a floor
        //: only, so a larger configured value is preserved untouched.
        public const int MinimumTickTimeoutSeconds = 135;

        public bool IsRunning { get; private set; }

        public event Action<string> Log;

        public bool SetViewerResponse(string response)
        {
            var normalized = (response ?? string.Empty).Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "":
                case "none":
                case "positive":
                case "neutral":
                case "negative":
                    pendingViewerResponse = normalized == "" ? "none" : normalized;
                    return true;
                default:
                    EmitLog("viewer response ignored");
                    return false;
            }
        }

        public void StartLoop()
        {
            if (IsRunning)
                return;

            bridge = bridge != null ? bridge : GetComponent<BrainActionBridge>();
            bridge = bridge != null ? bridge : FindFirstObjectByType<BrainActionBridge>();
            if (bridge == null)
            {
                EmitLog("BrainActionBridge missing");
                return;
            }

            IsRunning = true;
            nextTickTime = 0f;
            EmitLog("brain loop started");
        }

        public void StopLoop()
        {
            IsRunning = false;
            EmitLog("brain loop stopped");
        }

        private void Start()
        {
            if (startAutomatically)
                StartLoop();
        }

        private void Update()
        {
            if (!IsRunning || bridge == null)
                return;

            if (requestInFlight)
            {
                if (Time.time - requestStartedAt < Mathf.Max(1f, abandonedTickSeconds))
                    return;

                // Releasing the flag lets the next tick re-request the pending
                // action; the backend re-dispatches it idempotently under the
                // same action id while it awaits a result.
                EmitLog($"tick abandoned after {abandonedTickSeconds:0}s without completion");
                requestInFlight = false;
            }

            if (bridge.IsBusy)
                return;
            if (Time.time < nextTickTime)
                return;

            StartCoroutine(Tick());
       }

        private IEnumerator Tick()
        {
            requestInFlight = true;
            requestStartedAt = Time.time;
            var payload = BuildTickPayload();
            var request = BuildRequest(payload);
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                var responseText = request.downloadHandler != null ? request.downloadHandler.text : string.Empty;
                EmitLog(string.IsNullOrEmpty(responseText)
                    ? $"tick failed: {request.error}"
                    : $"tick failed: {request.error}: {responseText}");
                requestInFlight = false;
                nextTickTime = Time.time + decisionIntervalSeconds;
                yield break;
            }

            // The backend accepted this result, so do not replay it on a later tick.
            lastResult = null;
            var action = ParseAction(request.downloadHandler.text);
            if (action == null)
            {
                EmitLog("tick returned no action");
                requestInFlight = false;
                nextTickTime = Time.time + decisionIntervalSeconds;
                yield break;
            }

            EmitLog($"action {action.action}:{action.target ?? "-"}");
            var resultViewerResponse = pendingViewerResponse;
            var completed = false;
            var success = false;
            var accepted = bridge.Execute(action.action, action.target, action.text, action.speech_request_id, reportedSuccess =>
            {
                success = reportedSuccess;
                completed = true;
            });
            if (!accepted)
            {
                // The bridge refused to start (it reported itself busy), so no
                // callback will arrive. Leave the tick result unset and retry on
                // the normal interval instead of waiting forever.
                EmitLog($"action {action.action} refused: bridge busy");
                requestInFlight = false;
                nextTickTime = Time.time + decisionIntervalSeconds;
                yield break;
            }

            yield return new WaitUntil(() => completed);
            lastResult = new ActionResultSnapshot
            {
                action = action.action,
                status = success ? "SUCCESS" : "FAILED",
                // PHASE 33: a failure carries the bounded reason the bridge
                // already recorded for it (target_unresolved, move_unreachable,
                // turn_unreachable, an interruption token, ...) instead of the
                // constant "bridge_failed". Success is unchanged.
                reason = success ? "bridge_success" : ResolveFailureReason(bridge),
                reached = success,
                viewer_response = resultViewerResponse,
                // PHASE 34: echo the id of the action that produced this result
                // so the backend can confirm it completes exactly the dispatch
                // that requested it. Null for a legacy request that had no id.
                dispatch_id = action.dispatch_id
            };
            pendingViewerResponse = "none";
            requestInFlight = false;
            nextTickTime = Time.time + 0.1f;
        }

        /// <summary>
        /// PHASE 33: the bounded, machine-readable reason for a failed action.
        ///
        /// <see cref="BrainActionBridge.LastObservedReason"/> is written by
        /// <c>Complete(...)</c> immediately *before* it invokes the callback this
        /// method reads it for, so it always describes the action that just
        /// finished. Only an identifier-shaped token is forwarded (the bridge's
        /// own labels are lowercase tokens such as <c>target_unresolved</c>);
        /// anything else -- an over-long string, punctuation, embedded spacing,
        /// or a caller-supplied interruption reason -- falls back to the
        /// historical constant, so arbitrary text can never reach the backend
        /// or its persisted failure details.
        /// </summary>
        private static string ResolveFailureReason(BrainActionBridge source)
        {
            var reason = source != null ? source.LastObservedReason : null;
            if (string.IsNullOrWhiteSpace(reason))
                return "bridge_failed";

            var trimmed = reason.Trim();
            if (trimmed.Length > MaxFailureReasonLength)
                return "bridge_failed";

            for (var index = 0; index < trimmed.Length; index++)
            {
                var character = trimmed[index];
                var allowed =
                    (character >= 'a' && character <= 'z') ||
                    (character >= 'A' && character <= 'Z') ||
                    (character >= '0' && character <= '9') ||
                    character == '_' ||
                    character == '-' ||
                    character == '.';
                if (!allowed)
                    return "bridge_failed";
            }

            return trimmed;
        }

        private string BuildTickPayload()
        {
            var snapshot = bridge.BuildSnapshotJson();
            var hasUsableResult = lastResult != null && !string.IsNullOrWhiteSpace(lastResult.action);
            var result = !hasUsableResult
                ? "null"
                : BuildResultJson(lastResult);
            var viewerResponse = hasUsableResult
                ? BrainActionBridge.JsonUtilityQuote(lastResult.viewer_response)
                : "\"none\"";
            return $"{{\"world\":{snapshot},\"result\":{result},\"viewer_response\":{viewerResponse}}}";
        }

        /// <summary>
        /// PHASE 34: the action result exactly as Unity already sent it, with the
        /// echoed <c>dispatch_id</c> appended only when the request actually
        /// carried one. A legacy request (no id) keeps the historical result
        /// shape, and an id is never fabricated for a result whose originating
        /// dispatch had none.
        /// </summary>
        private static string BuildResultJson(ActionResultSnapshot snapshot)
        {
            var builder = new StringBuilder();
            builder.Append("{\"action\":").Append(BrainActionBridge.JsonUtilityQuote(snapshot.action));
            builder.Append(",\"status\":").Append(BrainActionBridge.JsonUtilityQuote(snapshot.status));
            builder.Append(",\"reason\":").Append(BrainActionBridge.JsonUtilityQuote(snapshot.reason));
            builder.Append(",\"reached\":").Append(snapshot.reached ? "true" : "false");
            builder.Append(",\"viewer_response\":").Append(BrainActionBridge.JsonUtilityQuote(snapshot.viewer_response));
            if (!string.IsNullOrEmpty(snapshot.dispatch_id))
                builder.Append(",\"dispatch_id\":").Append(BrainActionBridge.JsonUtilityQuote(snapshot.dispatch_id));
            return builder.Append('}').ToString();
        }

        private static BrainActionRequest ParseAction(string json)
        {
            try
            {
                var response = JsonUtility.FromJson<BrainTickResponse>(json);
                var action = response?.tick?.action;
                return string.IsNullOrWhiteSpace(action?.action) ? null : action;
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[BrainLoopController] invalid tick response: {exception.Message}");
                return null;
            }
        }

        private UnityWebRequest BuildRequest(string payload)
        {
            var request = new UnityWebRequest(tickUrl, UnityWebRequest.kHttpVerbPOST)
            {
                downloadHandler = new DownloadHandlerBuffer(),
                timeout = EffectiveTickTimeoutSeconds(requestTimeoutSeconds)
            };
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(payload));
            request.SetRequestHeader("Content-Type", "application/json");
            return request;
        }

        /// <summary>
        /// PHASE 35: the effective Brain tick HTTP timeout, in whole seconds.
        ///
        /// <see cref="requestTimeoutSeconds"/> is serialized in the scene and
        /// may hold a value below the backend's own provider budget, so the
        /// floor is applied here instead of trusting the asset. A larger
        /// configured value is returned unchanged, and
        /// <see cref="UnityWebRequest.timeout"/> takes whole seconds.
        /// </summary>
        public static int EffectiveTickTimeoutSeconds(float configuredSeconds)
        {
            // Mathf.RoundToInt throws on NaN, and this runs on every tick, so a
            // hand-edited serialized value must never reach it.
            if (float.IsNaN(configuredSeconds) || float.IsInfinity(configuredSeconds))
                return MinimumTickTimeoutSeconds;
            return Mathf.Max(Mathf.RoundToInt(configuredSeconds), MinimumTickTimeoutSeconds);
        }

        private void EmitLog(string message)
        {
            Log?.Invoke(message);
            Debug.Log($"[BrainLoopController] {message}");
        }
    }

    [Serializable]
    public sealed class BrainActionRequest
    {
        public string action;
        public string target;
        public string text;
        public string position;
        public string speech_request_id;
        // PHASE 34: the correlation id for this exact dispatch, echoed back with
        // the action result. Null for a legacy backend that sends no id.
        public string dispatch_id;
    }

    [Serializable]
    public sealed class BrainActionResult
    {
        public string action;
        public string status;
        public string reason;
        public bool reached;
    }

    [Serializable]
    public sealed class BrainTickResponse
    {
        public BrainTickEnvelope tick;
    }

    [Serializable]
    public sealed class BrainTickEnvelope
    {
        public string status;
        public BrainActionRequest action;
        public string message;
    }

    [Serializable]
    internal sealed class ActionResultSnapshot
    {
        public string action;
        public string status;
        public string reason;
        public bool reached;
        public string viewer_response;
        // PHASE 34: the id of the action this result completes.
        public string dispatch_id;
    }
}
