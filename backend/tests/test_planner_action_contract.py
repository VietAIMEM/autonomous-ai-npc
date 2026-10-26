"""Focused tests for the action-name contract in the transmitted planner prompt.

Background (see ``Nemotron_Parse_Rejection_Audit_20261019T201623Z``): the
planner prompt presents canonical uppercase ``available_actions`` alongside
lowercase ``available_skills``, and 11 of the 14 production skill ids are the
lowercase form of an action name. Four production rejections were valid JSON
that used that lowercase spelling in the ``action`` field. The parser now
normalises case; this file pins the independent, prompt-side hardening that
states the distinction explicitly.

All tests are deterministic and offline; no provider call is made.
"""

from __future__ import annotations

import json
import re
import types
import unittest
from pathlib import Path
from unittest import mock

from backend.brain.models import (
    ActionType,
    Goal,
    GoalStatus,
    GoalType,
    PersonalityProfile,
    PlanStep,
)
from backend.brain.planner import MAX_PLANNER_PROMPT_CHARS, BrainPlanner
from backend.brain.reasoning import BrainReasoner
from backend.brain.world_state import build_world_state
from backend.llm.base import LLMProvider, LLMProviderError, LLMRequest


REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER_SOURCE = REPO_ROOT / "backend/brain/planner.py"


def target(navigable: bool, lookable: bool) -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "navigation_target": navigable,
        "look_at_target": lookable,
    }


#: Production-shaped flags: standing points navigate, device anchors look.
PRODUCTION_TARGETS = {
    "CENTER": target(True, True),
    "TABLE": target(True, True),
    "BENCH": target(True, True),
    "FLOOR": target(True, True),
    "PHONE": target(False, True),
    "CAMERA": target(False, True),
    "CAMERA_STAND": target(True, False),
    "PHONE_STAND": target(True, False),
}

#: The production skill vocabulary shape: lowercase ids that mostly collide with
#: the lowercase form of an action name.
PRODUCTION_SKILLS = [
    {"skill_id": skill_id, "description": f"{skill_id} skill"}
    for skill_id in (
        "walk_to", "stop", "turn_to", "look_at", "wave", "dance", "run",
        "pick_up_phone", "place_phone", "start_stream", "stop_stream", "speak",
    )
]


def world():
    return build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": PRODUCTION_TARGETS,
        }
    )


def goal() -> Goal:
    return Goal("goal", GoalType.EXPLORE_ROOM, 0.5, "test", status=GoalStatus.ACTIVE)


def prompt_payload(*, skills=None, memories=None) -> dict:
    planner = BrainPlanner()
    prompt = planner.build_prompt(
        world(),
        goal(),
        PersonalityProfile(),
        None,
        memories,
        skills if skills is not None else PRODUCTION_SKILLS,
        None,
        None,
        None,
    )
    return json.loads(prompt)


def prompt_text(**kwargs) -> str:
    return BrainPlanner().build_prompt(
        world(), goal(), PersonalityProfile(), None,
        kwargs.get("memories"), kwargs.get("skills", PRODUCTION_SKILLS),
        None, None, None,
    )


class _CapturingProvider(LLMProvider):
    """Records the exact ``request.prompt`` the reasoner transmits."""

    name = "capturing"

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(model="test/model-1")
        self.prompts: list[str] = []

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - stub
        self.prompts.append(request.prompt)
        return json.dumps(
            {"reason": "ok", "steps": [{"action": "LOOK_AT", "target": "CAMERA"}]}
        )


class TransmittedPromptTests(unittest.TestCase):
    """The rules must be in the string the provider actually receives."""

    def test_request_prompt_carries_the_action_name_contract(self):
        provider = _CapturingProvider()
        reasoner = BrainReasoner(provider, BrainPlanner(), None)
        plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNotNone(plan, reason)
        self.assertEqual(len(provider.prompts), 1)
        sent = provider.prompts[0]

        self.assertIn("one uppercase name from available_actions", sent)
        self.assertIn("available_actions", sent)
        self.assertIn("(the action vocabulary)", sent)
        self.assertIn("never copy one into action", sent)
        self.assertIn("skill_output_schema", sent)
        self.assertIn("do not invent one", sent)
        self.assertIn("no aliases, suffixes, target names", sent)

    def test_contract_lines_are_in_the_truncation_protected_constraints(self):
        payload = prompt_payload()
        constraints = payload["constraints"]
        self.assertIsInstance(constraints, list)
        joined = " | ".join(constraints)
        self.assertIn("one uppercase name from available_actions", joined)
        self.assertIn("never copy one into action", joined)
        # The constraint block sits before every variable-length context block,
        # so the JSON key order is the truncation order.
        order = list(payload.keys())
        for late in ("available_skills", "learned_skill_ranking", "relevant_memories", "personalization"):
            self.assertLess(order.index("constraints"), order.index(late))

    def test_contract_survives_truncation_with_oversized_context(self):
        huge_memories = [{"summary": "m" * 400} for _ in range(80)]
        huge_skills = [
            {"skill_id": f"skill_{i}", "description": "d" * 400} for i in range(80)
        ]
        sent = prompt_text(memories=huge_memories, skills=huge_skills)
        self.assertLessEqual(len(sent), MAX_PLANNER_PROMPT_CHARS)
        # The guard really did clip: the string is exactly at the cap and the
        # oversized trailing context was dropped.
        self.assertEqual(len(sent), MAX_PLANNER_PROMPT_CHARS)
        self.assertNotIn("m" * 50, sent)
        for marker in (
            "one uppercase name from available_actions",
            "(the action vocabulary)",
            "never copy one into action",
            "do not invent one",
            "action_target_rules",
            "valid_look_targets",
            "valid_navigation_targets",
        ):
            self.assertIn(marker, sent)

    def test_production_fixture_prompt_stays_under_the_cap(self):
        sent = prompt_text()
        self.assertLess(len(sent), MAX_PLANNER_PROMPT_CHARS)
        # The hardening is a small addition, not a prompt rewrite.
        self.assertGreater(len(sent), 3000)


class VocabularyTests(unittest.TestCase):
    def test_available_actions_comes_from_the_enum_in_canonical_case(self):
        payload = prompt_payload()
        self.assertEqual(
            payload["available_actions"], [member.value for member in ActionType]
        )
        for name in payload["available_actions"]:
            self.assertEqual(name, name.upper())
            self.assertIn(name, {member.value for member in ActionType})

    def test_every_canonical_action_is_still_advertised(self):
        payload = prompt_payload()
        self.assertEqual(set(payload["available_actions"]), {m.value for m in ActionType})
        self.assertEqual(len(payload["available_actions"]), 13)

    def test_available_skills_stays_a_separate_lowercase_vocabulary(self):
        payload = prompt_payload()
        skill_ids = [item["skill_id"] for item in payload["available_skills"]]
        self.assertIn("look_at", skill_ids)
        self.assertIn("speak", skill_ids)
        # The skill block is its own key and never merged into the action list.
        actions = set(payload["available_actions"])
        self.assertFalse(actions & set(skill_ids))
        self.assertNotIn("look_at", actions)

    def test_no_second_hard_coded_action_vocabulary_exists(self):
        source = PLANNER_SOURCE.read_text(encoding="utf-8")
        # The action list is still derived from the enum...
        self.assertIn('"available_actions": [action.value for action in ActionType]', source)
        # ...and the hardening text never enumerates action names itself: it
        # always defers to the ``available_actions`` list above.
        hardening_lines = (
            "action must be one uppercase name from available_actions ",
            "(the action vocabulary)",
            "available_skills ids are skill ids, not actions; never copy ",
            "one into action",
            "invoke a skill only via skill_output_schema, not as an ",
            "action step",
            "action names take no aliases, suffixes, target names or ",
            "punctuation",
            "if a request is not expressible with these actions, do not ",
            "invent one",
        )
        for line in hardening_lines:
            for member in ActionType:
                self.assertNotIn(member.value, line)
        # No literal duplicate list of the whole enum was introduced.
        self.assertNotIn("HARD_CODED_ACTIONS", source)
        self.assertNotIn("ACTION_VOCABULARY = [", source)


class TargetGuaranteeTests(unittest.TestCase):
    def test_dynamic_target_lists_and_rules_are_unchanged(self):
        payload = prompt_payload()
        world_block = payload["world"]
        self.assertEqual(
            world_block["valid_look_targets"],
            sorted(n for n, t in PRODUCTION_TARGETS.items() if t["look_at_target"]),
        )
        self.assertEqual(
            world_block["valid_navigation_targets"],
            sorted(n for n, t in PRODUCTION_TARGETS.items() if t["navigation_target"]),
        )
        self.assertIn("CAMERA_STAND", world_block["navigation_only_targets"])
        self.assertIn("PHONE_STAND", world_block["navigation_only_targets"])
        self.assertNotIn("CAMERA_STAND", world_block["valid_look_targets"])
        rules = " | ".join(payload["action_target_rules"])
        self.assertIn("LOOK_AT and TURN_TO accept only targets", rules)
        self.assertIn("MOVE_TO accepts only targets", rules)


class ParserAndValidationUnchangedTests(unittest.TestCase):
    def _parse(self, steps):
        planner = BrainPlanner()
        raw = json.dumps({"reason": "test", "steps": steps})
        return planner.parse_plan(raw, goal(), world())

    def test_lowercase_normalization_still_accepts_supported_variants(self):
        for spelling in ("look_at", "Look_At", "look_AT", "LOOK_AT"):
            with self.subTest(action=spelling):
                plan, reason = self._parse([{"action": spelling, "target": "CAMERA"}])
                self.assertIsNotNone(plan, reason)
                self.assertEqual(plan.steps[0].action, ActionType.LOOK_AT)

    def test_unsupported_actions_still_fail_unchanged(self):
        for bad in ("fly_to", "look", "walk_to", "dance!", ""):
            with self.subTest(action=bad):
                plan, reason = self._parse([{"action": bad, "target": "CENTER"}])
                self.assertIsNone(plan)
                self.assertEqual(reason, "UNKNOWN_ACTION")

    def test_validate_step_and_codes_are_unchanged(self):
        planner = BrainPlanner()
        w = world()
        for action, name, expected in (
            (ActionType.LOOK_AT, "CAMERA_STAND", "TARGET_NOT_LOOKABLE"),
            (ActionType.TURN_TO, "PHONE_STAND", "TARGET_NOT_LOOKABLE"),
            (ActionType.MOVE_TO, "CAMERA", "TARGET_NOT_NAVIGABLE"),
        ):
            with self.subTest(action=action):
                validation = planner.validate_step(PlanStep(action, name), w)
                self.assertFalse(validation.valid)
                self.assertEqual(validation.reason, expected)
        # The parser still routes a case variant through the same validator.
        plan, reason = self._parse([{"action": "look_at", "target": "CAMERA_STAND"}])
        self.assertIsNone(plan)
        self.assertEqual(reason, "TARGET_NOT_LOOKABLE")

    def test_neighbouring_contracts_are_untouched(self):
        planner_source = PLANNER_SOURCE.read_text(encoding="utf-8")
        loop_source = (REPO_ROOT / "backend/brain/loop.py").read_text(encoding="utf-8")
        self.assertIn("ActionType(str(action_name).strip().upper())", planner_source)
        self.assertIn("dispatch_id=_new_dispatch_id()", loop_source)
        self.assertIn("def _is_awaited_result", loop_source)

    def test_provider_diagnostics_are_unchanged(self):
        class _Raising(LLMProvider):
            name = "test-provider"

            def __init__(self) -> None:
                self.config = types.SimpleNamespace(model="vendor/model-1")

            def generate(self, request: LLMRequest) -> str:  # pragma: no cover - mock
                raise LLMProviderError("boom")

        reasoner = BrainReasoner(_Raising(), BrainPlanner(), None)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "LLMProviderError")
        self.assertEqual(spy.call_args.args[1], "brain.llm_request_failed")


if __name__ == "__main__":
    unittest.main()
