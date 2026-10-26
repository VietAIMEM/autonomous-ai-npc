"""Fail-closed E2E isolation contract for the Unity action bridge.

The production bridge must keep its existing behavior when the E2E flag is
off. When the flag is on, the bridge must reject missing, invalid, or
production-equivalent stream configuration before resolving references or
entering the action switch. Stream-control actions are denied in test mode
because no safe fake stream dependency exists yet.

These tests mirror the C# policy and pin the exact dispatch-boundary order,
following the existing repo pattern for Unity source-contract tests.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SOURCE = REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainActionBridge.cs"
POLICY_SOURCE = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainIsolationPolicy.cs"
)
STREAM_CLIENT_SOURCE = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Livestream/StreamIngestClient.cs"
)

PRODUCTION_STREAM_BASE_URL = "http://127.0.0.1:8766"


class _IsolationDenial(Exception):
    """Python mirror of the C# ``TryAuthorize`` failure path."""


def _normalized_host(host: str) -> str:
    return host.removeprefix("[").removesuffix("]").lower()


def _production_equivalent(parts) -> bool:
    production = urlsplit(PRODUCTION_STREAM_BASE_URL)
    host = _normalized_host(parts.hostname or "")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    return (
        parts.port == production.port
    )


def try_authorize(
    e2e_isolation_enabled: bool,
    stream_base_url: str | None,
    action: str,
) -> None:
    """Raise on any test-mode denial; return normally when authorized."""
    if not e2e_isolation_enabled:
        return

    if not stream_base_url or not stream_base_url.strip():
        raise _IsolationDenial("e2e_isolation_config_missing")

    parts = urlsplit(stream_base_url.strip())
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.port is None
    ):
        raise _IsolationDenial("e2e_isolation_config_invalid")
    if _production_equivalent(parts):
        raise _IsolationDenial("e2e_isolation_config_production_equivalent")

    normalized_action = action.strip().upper()
    if normalized_action in {"START_STREAM", "STOP_STREAM"}:
        raise _IsolationDenial("e2e_stream_action_denied")


class IsolationPolicyMirrorTests(unittest.TestCase):
    """The Python mirror used to pin the C# fail-closed contract."""

    def test_production_mode_keeps_the_existing_path(self) -> None:
        for action in ("START_STREAM", "STOP_STREAM", "LOOK_AT"):
            with self.subTest(action=action):
                self.assertIsNone(
                    try_authorize(False, PRODUCTION_STREAM_BASE_URL, action)
                )

    def test_test_mode_without_configuration_fails_closed(self) -> None:
        for stream_base_url in (None, "", "   "):
            with self.subTest(stream_base_url=stream_base_url):
                with self.assertRaises(_IsolationDenial) as caught:
                    try_authorize(True, stream_base_url, "LOOK_AT")
                self.assertEqual(
                    str(caught.exception), "e2e_isolation_config_missing"
                )

    def test_invalid_test_configuration_fails_closed(self) -> None:
        for stream_base_url in (
            "not-a-url",
            "ftp://127.0.0.1:18766",
            "http://127.0.0.1",
        ):
            with self.subTest(stream_base_url=stream_base_url):
                with self.assertRaises(_IsolationDenial) as caught:
                    try_authorize(True, stream_base_url, "LOOK_AT")
                self.assertEqual(
                    str(caught.exception), "e2e_isolation_config_invalid"
                )

    def test_production_equivalent_test_configuration_fails_closed(self) -> None:
        for stream_base_url in (
            "http://127.0.0.1:8766",
            "http://127.0.0.1:8766/",
            "http://localhost:8766",
            "http://LOCALHOST:8766",
        ):
            with self.subTest(stream_base_url=stream_base_url):
                with self.assertRaises(_IsolationDenial) as caught:
                    try_authorize(True, stream_base_url, "LOOK_AT")
                self.assertEqual(
                    str(caught.exception),
                    "e2e_isolation_config_production_equivalent",
                )

    def test_stream_control_actions_are_denied_in_test_mode(self) -> None:
        for action in ("START_STREAM", "start_stream", "STOP_STREAM", "stop_stream"):
            with self.subTest(action=action):
                with self.assertRaises(_IsolationDenial) as caught:
                    try_authorize(True, "http://127.0.0.1:18766", action)
                self.assertEqual(str(caught.exception), "e2e_stream_action_denied")

    def test_safe_non_streaming_actions_use_the_intended_test_path(self) -> None:
        for action in (
            "MOVE_TO",
            "TURN_TO",
            "LOOK_AT",
            "STOP",
            "SPEAK",
            "WAVE",
            "DANCE",
            "RUN",
            "PICK_UP_PHONE",
            "PLACE_PHONE",
        ):
            with self.subTest(action=action):
                self.assertIsNone(
                    try_authorize(True, "http://127.0.0.1:18766", action)
                )

    def test_maintain_stream_start_cannot_reach_production(self) -> None:
        with self.assertRaises(_IsolationDenial) as caught:
            try_authorize(
                True,
                "http://127.0.0.1:18766",
                "START_STREAM",
            )
        self.assertEqual(str(caught.exception), "e2e_stream_action_denied")


class UnityExecutionBoundaryContractTests(unittest.TestCase):
    """Pin the C# policy and the exact place the bridge enforces it."""

    def setUp(self) -> None:
        self.bridge = BRIDGE_SOURCE.read_text(encoding="utf-8")
        self.policy = POLICY_SOURCE.read_text(encoding="utf-8")
        self.stream_client = STREAM_CLIENT_SOURCE.read_text(encoding="utf-8")

    def test_policy_exists_and_uses_the_production_constant(self) -> None:
        self.assertIn("public static class BrainIsolationPolicy", self.policy)
        self.assertIn(
            "new(StreamIngestClient.DefaultBaseUrl, UriKind.Absolute)",
            self.policy,
        )

    def test_policy_has_a_fail_closed_test_mode_gate(self) -> None:
        self.assertIn(
            "public static bool TryAuthorize(", self.policy
        )
        self.assertIn(
            "public static bool TryValidateE2EStreamConfiguration(", self.policy
        )
        self.assertIn("if (!e2eIsolationEnabled)", self.policy)
        self.assertIn(
            'if (!TryValidateE2EStreamConfiguration(streamBaseUrl, out reason))',
            self.policy,
        )
        self.assertIn("IsStreamControlAction(action)", self.policy)

    def test_production_base_url_matches_the_stream_client_default(self) -> None:
        match = re.search(
            r'public const string DefaultBaseUrl = "([^"]+)";',
            self.stream_client,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), PRODUCTION_STREAM_BASE_URL)

    def test_bridge_has_an_explicit_test_mode_flag(self) -> None:
        self.assertIn("[SerializeField] private bool e2eIsolationEnabled;", self.bridge)

    def test_bridge_authorizes_before_reference_resolution_or_dispatch(self) -> None:
        gate_position = self.bridge.index("BrainIsolationPolicy.TryAuthorize(")
        reference_position = self.bridge.index("ResolveReferences();", gate_position)
        switch_position = self.bridge.index("switch (action.ToUpperInvariant())", gate_position)
        stream_position = self.bridge.index('case "START_STREAM":', switch_position)
        self.assertLess(gate_position, reference_position)
        self.assertLess(gate_position, switch_position)
        self.assertLess(gate_position, stream_position)

    def test_bridge_rejects_the_action_instead_of_dispatching_it(self) -> None:
        gate_text = self.bridge[
            self.bridge.index("if (!BrainIsolationPolicy.TryAuthorize(") :
            self.bridge.index("ResolveReferences();")
        ]
        self.assertIn("Complete(callback, false, isolationReason);", gate_text)
        self.assertIn("return true;", gate_text)
        self.assertNotIn("IsBusy = true", gate_text)
        self.assertNotIn("ResolveReferences()", gate_text)

    def test_rejection_occurs_before_the_stream_switch_or_busy_flag(self) -> None:
        gate_position = self.bridge.index("BrainIsolationPolicy.TryAuthorize(")
        busy_position = self.bridge.index("IsBusy = true;", gate_position)
        stream_position = self.bridge.index('case "START_STREAM":', gate_position)
        self.assertLess(gate_position, busy_position)
        self.assertLess(gate_position, stream_position)

    def test_the_policy_does_not_call_the_production_stream_endpoint(self) -> None:
        self.assertNotIn("/stream/start", self.policy)
        self.assertNotIn("/stream/stop", self.policy)
        self.assertNotIn("UnityWebRequest", self.policy)

    def test_stream_cases_remain_after_the_gate(self) -> None:
        gate_position = self.bridge.index("BrainIsolationPolicy.TryAuthorize(")
        for action in ("START_STREAM", "STOP_STREAM"):
            position = self.bridge.index(f'case "{action}":', gate_position)
            self.assertGreater(position, gate_position)


if __name__ == "__main__":
    unittest.main()
