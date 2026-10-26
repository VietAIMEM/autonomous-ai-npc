using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;

namespace RoomAgent.Editor
{
    /// <summary>
    /// Opt-in request watcher used by the Phase 23 admin runtime when
    /// MOTION_UNITY_MODE=editor. It only dispatches to the existing Phase 16/17
    /// editor entry points and signals completion; the backend parses the
    /// reports with the same adapters used by batch mode.
    /// </summary>
    [InitializeOnLoad]
    public static class Phase23AdminMotionBridge
    {
        private const string RequestDirectory = "Temp/Phase23AdminMotionBridge";
        private const string RequestSuffix = ".request.json";
        private const string ResponseSuffix = ".response.json";
        private const string ImportReport = "Temp/Phase16MotionImport.json";
        private const string ProcessingReport = "Temp/Phase17MotionProcessing.json";
        private const string PlaybackReport = "Temp/Phase17ProductionPlayback.json";
        private const double PollIntervalSeconds = 0.25;
        private const double PlaybackTimeoutSeconds = 420.0;
        private static readonly Regex RequestNamePattern = new Regex(
            @"^[A-Za-z0-9_]+\.request\.json$", RegexOptions.CultureInvariant);

        [Serializable]
        private sealed class BridgeRequest
        {
            public string id;
            public string operation;
            public string responsePath;
            public string sourceFile;
            public string clipName;
            public string motionId;
            public int version;
            public string sourceMotionId;
            public int sourceVersion;
            public string sourceReference;
            public string processedMotionId;
            public int processingVersion;
            public string processedClipPath;
        }

        [Serializable]
        private sealed class BridgeResponse
        {
            public string id;
            public string status;
            public bool success;
            public string error;
            public string message;
            public string reportPath;
        }

        private sealed class PendingPlaybackRequest
        {
            public BridgeRequest request;
            public double deadline;
        }

        private static readonly List<PendingPlaybackRequest> pendingPlayback =
            new List<PendingPlaybackRequest>();
        private static double nextPollAt;

        static Phase23AdminMotionBridge()
        {
            EditorApplication.update += OnEditorUpdate;
        }

        private static void OnEditorUpdate()
        {
            var now = EditorApplication.timeSinceStartup;
            if (now >= nextPollAt)
            {
                nextPollAt = now + PollIntervalSeconds;
                PollRequests();
            }
            ProcessPendingPlayback();
        }

        private static void PollRequests()
        {
            var projectRoot = Directory.GetCurrentDirectory();
            var directory = Path.Combine(projectRoot, RequestDirectory);
            if (!Directory.Exists(directory))
                return;

            string[] files;
            try
            {
                files = Directory.GetFiles(directory, "*" + RequestSuffix);
            }
            catch (Exception)
            {
                return;
            }

            foreach (var file in files)
            {
                var name = Path.GetFileName(file);
                if (!RequestNamePattern.IsMatch(name))
                    continue;
                try
                {
                    var text = File.ReadAllText(file);
                    File.Delete(file);
                    var request = JsonUtility.FromJson<BridgeRequest>(text);
                    if (request == null || string.IsNullOrWhiteSpace(request.id))
                    {
                        TryWriteResponse(
                            ResponsePathFor(projectRoot, name),
                            Error("", "UNITY_BRIDGE_REQUEST_INVALID", "missing_request_id"));
                        continue;
                    }
                    Dispatch(request);
                }
                catch (Exception exception)
                {
                    TryWriteResponse(
                        ResponsePathFor(projectRoot, name),
                        Error("", "UNITY_BRIDGE_REQUEST_INVALID", exception.Message));
                }
            }
        }

        private static void Dispatch(BridgeRequest request)
        {
            // Phase 16/17 open scenes and drive Play mode themselves, so they
            // cannot run while the editor is already playing. Refuse the
            // operation instead of letting the importer throw mid-flight.
            if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
            {
                WriteError(
                    request,
                    "UNITY_EDITOR_PLAY_MODE_ACTIVE",
                    "Unity editor is in Play mode; stop Play mode before running motion " +
                    request.operation + ".");
                return;
            }
            try
            {
                switch (request.operation)
                {
                    case "import":
                        Phase16MotionFileImporter.Run(
                            request.sourceFile, request.clipName, request.motionId, request.version);
                        Complete(request, ImportReport);
                        break;
                    case "process":
                        Phase17MotionProcessor.Run(
                            request.sourceMotionId,
                            request.sourceVersion,
                            request.sourceReference,
                            request.processedMotionId,
                            request.processingVersion,
                            false);
                        Complete(request, ProcessingReport);
                        break;
                    case "playback":
                        Phase17ProductionPlaybackValidator.Run(
                            request.sourceMotionId,
                            request.sourceVersion,
                            request.processedMotionId,
                            request.processingVersion,
                            request.sourceReference,
                            request.processedClipPath);
                        pendingPlayback.Add(new PendingPlaybackRequest
                        {
                            request = request,
                            deadline = EditorApplication.timeSinceStartup + PlaybackTimeoutSeconds
                        });
                        break;
                    default:
                        WriteError(
                            request, "UNITY_BRIDGE_OPERATION_UNKNOWN", request.operation ?? "");
                        break;
                }
            }
            catch (Exception exception)
            {
                WriteError(request, "UNITY_BRIDGE_DISPATCH_FAILED", exception.Message);
            }
        }

        private static void ProcessPendingPlayback()
        {
            if (pendingPlayback.Count == 0)
                return;
            for (var index = pendingPlayback.Count - 1; index >= 0; index--)
            {
                var pending = pendingPlayback[index];
                if (ReadReport(PlaybackReport) != null)
                {
                    Complete(pending.request, PlaybackReport);
                    pendingPlayback.RemoveAt(index);
                    continue;
                }
                if (EditorApplication.timeSinceStartup >= pending.deadline)
                {
                    WriteError(pending.request, "UNITY_PLAYBACK_TIMEOUT", PlaybackReport);
                    pendingPlayback.RemoveAt(index);
                }
            }
        }

        private static void Complete(BridgeRequest request, string relativeReportPath)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), relativeReportPath);
            if (!File.Exists(path))
            {
                WriteError(request, "UNITY_BRIDGE_REPORT_MISSING", relativeReportPath);
                return;
            }
            WriteResponse(request, new BridgeResponse
            {
                id = request.id,
                status = "OK",
                success = true,
                reportPath = relativeReportPath
            });
        }

        private static BridgeResponse ReadReport(string relativeReportPath)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), relativeReportPath);
            if (!File.Exists(path))
                return null;
            try
            {
                // Only a completion probe: the backend re-parses the full file
                // with the canonical Phase 16/17 report adapters.
                return JsonUtility.FromJson<BridgeResponse>(File.ReadAllText(path));
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static void WriteError(BridgeRequest request, string error, string message)
        {
            WriteResponse(request, Error(request == null ? "" : request.id, error, message));
        }

        private static BridgeResponse Error(string id, string error, string message)
        {
            return new BridgeResponse
            {
                id = id ?? "",
                status = "ERROR",
                success = false,
                error = error ?? "",
                message = message ?? ""
            };
        }

        private static void WriteResponse(BridgeRequest request, BridgeResponse response)
        {
            var responsePath = request != null ? request.responsePath : null;
            if (string.IsNullOrWhiteSpace(responsePath))
            {
                Debug.LogError("[Phase23AdminMotionBridge] response path missing for " +
                               (request == null ? "unknown" : request.id));
                return;
            }
            TryWriteResponse(responsePath, response);
        }

        private static void TryWriteResponse(string path, BridgeResponse response)
        {
            try
            {
                var directory = Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                var pending = path + ".pending";
                File.WriteAllText(pending, JsonUtility.ToJson(response, true));
                if (File.Exists(path))
                    File.Delete(path);
                File.Move(pending, path);
            }
            catch (Exception exception)
            {
                Debug.LogError("[Phase23AdminMotionBridge] response write failed: " +
                               exception.Message);
            }
        }

        private static string ResponsePathFor(string projectRoot, string requestName)
        {
            var safeName = string.IsNullOrEmpty(requestName) ? "unknown" : requestName;
            var baseName = safeName.EndsWith(RequestSuffix, StringComparison.Ordinal)
                ? safeName.Substring(0, safeName.Length - RequestSuffix.Length)
                : Path.GetFileNameWithoutExtension(safeName);
            return Path.Combine(projectRoot, RequestDirectory, baseName + ResponseSuffix);
        }
    }
}
