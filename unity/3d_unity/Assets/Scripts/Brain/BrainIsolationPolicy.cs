using System;
using RoomAgent.Livestream;

namespace RoomAgent.Brain
{
    /// <summary>
    /// Fail-closed authorization for an E2E test fixture. Production keeps this
    /// disabled; a test must both enable it and provide a non-production stream
    /// base URL before any action may run. Stream-control actions are denied in
    /// test mode because there is no safe fake stream dependency yet.
    /// </summary>
    public static class BrainIsolationPolicy
    {
        private static readonly Uri ProductionStreamBaseUri =
            new(StreamIngestClient.DefaultBaseUrl, UriKind.Absolute);

        public const string MissingConfigurationReason = "e2e_isolation_config_missing";
        public const string InvalidConfigurationReason = "e2e_isolation_config_invalid";
        public const string ProductionConfigurationReason = "e2e_isolation_config_production_equivalent";
        public const string StreamActionDeniedReason = "e2e_stream_action_denied";

        public static bool TryAuthorize(
            bool e2eIsolationEnabled,
            string streamBaseUrl,
            string action,
            out string reason)
        {
            reason = null;
            if (!e2eIsolationEnabled)
                return true;

            if (!TryValidateE2EStreamConfiguration(streamBaseUrl, out reason))
                return false;

            if (IsStreamControlAction(action))
            {
                reason = StreamActionDeniedReason;
                return false;
            }

            return true;
        }

        public static bool TryValidateE2EStreamConfiguration(
            string streamBaseUrl,
            out string reason)
        {
            reason = null;
            if (string.IsNullOrWhiteSpace(streamBaseUrl))
            {
                reason = MissingConfigurationReason;
                return false;
            }

            if (!Uri.TryCreate(streamBaseUrl.Trim(), UriKind.Absolute, out var uri) ||
                (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps) ||
                string.IsNullOrWhiteSpace(uri.Host) ||
                uri.Port == -1)
            {
                reason = InvalidConfigurationReason;
                return false;
            }

            if (IsProductionEquivalent(uri))
            {
                reason = ProductionConfigurationReason;
                return false;
            }

            return true;
        }

        public static bool IsStreamControlAction(string action)
        {
            return action is not null && action.Trim().ToUpperInvariant() is "START_STREAM" or "STOP_STREAM";
        }

        private static bool IsProductionEquivalent(Uri uri)
        {
            return uri.Port == ProductionStreamBaseUri.Port &&
                IsLoopbackHost(uri.Host);
        }

        private static bool IsLoopbackHost(string host)
        {
            var normalized = (host ?? string.Empty).Trim().Trim('[', ']').ToLowerInvariant();
            return normalized is "127.0.0.1" or "localhost" or "::1";
        }
    }
}
