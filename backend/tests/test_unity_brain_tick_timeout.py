"""Focused tests for the PHASE 35 Unity Brain tick HTTP timeout floor.

Reference evidence (``LLM_Transport_Timeout_Runtime_Verification_20261019T123633Z``):
the backend plans synchronously and its provider transport is allowed to block
for the full 120 s socket timeout (four natural failures were classified
``TimeoutError`` / ``transport_phase=read`` at ~120.2 s). Unity's Brain tick
``UnityWebRequest`` timeout is serialized in ``SampleScene.unity`` as 90 s, so
Unity abandoned requests the backend might still have answered.

``BrainLoopController.EffectiveTickTimeoutSeconds`` now floors the tick request
at 135 s (15 s of margin over the provider budget) while preserving any larger
configured value. These tests mirror that pure calculation and pin the shipped
C# and the serialized scene value that it has to out-rank.
"""

from __future__ import annotations

import math
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_CONTROLLER = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainLoopController.cs"
)
SAMPLE_SCENE = REPO_ROOT / "unity/3d_unity/Assets/Scenes/SampleScene.unity"

#: Mirrors ``BrainLoopController.MinimumTickTimeoutSeconds``.
MINIMUM_TICK_TIMEOUT_SECONDS = 135
#: The provider transport budget the floor has to clear, with margin.
PROVIDER_TIMEOUT_SECONDS = 120


def effective_tick_timeout(configured_seconds: float) -> int:
    """Python mirror of ``BrainLoopController.EffectiveTickTimeoutSeconds``.

    C#'s ``Mathf.RoundToInt`` rounds half to even, which is Python's ``round``,
    and its NaN/Infinity guard is reproduced here so the mirror stays honest.
    """
    value = float(configured_seconds)
    if math.isnan(value) or math.isinf(value):
        return MINIMUM_TICK_TIMEOUT_SECONDS
    return max(round(value), MINIMUM_TICK_TIMEOUT_SECONDS)


class EffectiveTickTimeoutTests(unittest.TestCase):
    """The pure timeout calculation."""

    def test_configured_90_seconds_is_raised_to_at_least_135(self):
        self.assertEqual(effective_tick_timeout(90.0), 135)
        self.assertGreaterEqual(
            effective_tick_timeout(90.0), MINIMUM_TICK_TIMEOUT_SECONDS
        )

    def test_configured_120_seconds_is_raised_to_at_least_135(self):
        self.assertEqual(effective_tick_timeout(120.0), 135)
        self.assertGreaterEqual(
            effective_tick_timeout(120.0), MINIMUM_TICK_TIMEOUT_SECONDS
        )
        # The floor must also clear the provider budget itself.
        self.assertGreater(
            effective_tick_timeout(120.0), PROVIDER_TIMEOUT_SECONDS
        )

    def test_a_larger_configured_timeout_is_preserved(self):
        for value in (136.0, 150.0, 200.0, 3600.0):
            with self.subTest(value=value):
                self.assertEqual(effective_tick_timeout(value), int(round(value)))
                self.assertGreater(
                    effective_tick_timeout(value), MINIMUM_TICK_TIMEOUT_SECONDS
                )

    def test_exactly_the_minimum_is_kept(self):
        self.assertEqual(effective_tick_timeout(135.0), 135)

    def test_degenerate_values_fall_back_to_the_minimum(self):
        for value in (0.0, -5.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertEqual(effective_tick_timeout(value), 135)

    def test_fractional_seconds_round_as_unity_does(self):
        self.assertEqual(effective_tick_timeout(134.6), 135)
        self.assertEqual(effective_tick_timeout(200.4), 200)


class UnitySourceContractTests(unittest.TestCase):
    """The shipped C# and the asset value it must not be defeated by."""

    def setUp(self) -> None:
        self.controller = LOOP_CONTROLLER.read_text(encoding="utf-8")
        self.scene = SAMPLE_SCENE.read_text(encoding="utf-8")

    def test_minimum_constant_is_135_seconds(self):
        self.assertIn(
            "public const int MinimumTickTimeoutSeconds = 135;", self.controller
        )

    def test_the_floor_is_applied_to_the_brain_tick_request(self):
        self.assertIn(
            "timeout = EffectiveTickTimeoutSeconds(requestTimeoutSeconds)",
            self.controller,
        )
        # The raw serialized field must no longer reach UnityWebRequest.timeout.
        self.assertNotIn(
            "timeout = Mathf.RoundToInt(requestTimeoutSeconds)", self.controller
        )

    def test_only_the_brain_tick_request_is_affected(self):
        # The controller owns exactly one HTTP client -- the Brain tick POST --
        # and exactly one timeout assignment.
        self.assertEqual(self.controller.count("new UnityWebRequest("), 1)
        self.assertEqual(self.controller.count("timeout ="), 1)

    def test_helper_is_a_max_of_configured_and_minimum(self):
        self.assertIn(
            "public static int EffectiveTickTimeoutSeconds(float configuredSeconds)",
            self.controller,
        )
        self.assertIn(
            "Mathf.Max(Mathf.RoundToInt(configuredSeconds), MinimumTickTimeoutSeconds)",
            self.controller,
        )
        # Mathf.RoundToInt throws on NaN, so the helper must guard first.
        self.assertIn("float.IsNaN(configuredSeconds)", self.controller)
        self.assertIn("float.IsInfinity(configuredSeconds)", self.controller)

    def test_serialized_scene_value_cannot_defeat_the_floor(self):
        match = re.search(
            r"^\s*requestTimeoutSeconds:\s*([0-9.]+)\s*$",
            self.scene,
            re.MULTILINE,
        )
        self.assertIsNotNone(
            match, "SampleScene must serialize requestTimeoutSeconds"
        )
        serialized = float(match.group(1))
        self.assertEqual(serialized, 90.0)
        self.assertGreaterEqual(
            effective_tick_timeout(serialized), MINIMUM_TICK_TIMEOUT_SECONDS
        )

    def test_existing_tick_semantics_are_untouched(self):
        # Dispatch-ID correlation and the bounded action-result reason pipeline
        # must survive this change unchanged.
        self.assertIn("dispatch_id = action.dispatch_id", self.controller)
        self.assertEqual(
            self.controller.count("public string dispatch_id;"),
            2,
        )
        self.assertIn(
            'reason = success ? "bridge_success" : ResolveFailureReason(bridge)',
            self.controller,
        )
        self.assertIn("BuildResultJson(lastResult)", self.controller)


if __name__ == "__main__":
    unittest.main()
