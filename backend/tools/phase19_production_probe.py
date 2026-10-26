"""PHASE 19 production probe: returning-viewer personalization end to end.

This is not a unit test. It drives the real production components
(``AutonomousBrain``, ``SocialAutonomyService``, ``SkillLearningService`` on
SQLite, ``SpeechService`` and ``ChatService``) against the Phase 19 probe
databases and reports the five Phase 19 completion gates:

1. relationship/history/preferences/personality retained across a restart,
2. learned viewer preferences change a later skill decision,
3. a new viewer stays isolated from that personalized context,
4. the personality profile stays stable across positive/negative events,
5. the personalized decision is produced by the normal Brain loop.

The planning and chat LLMs stay the deterministic local providers that the
API development mode already uses; every other layer is the production
implementation. Databases live under ``Logs/`` so the production stores in
``backend/data`` are never touched.

Usage:
    python -m backend.tools.phase19_production_probe
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.personality import personality_traits
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.service import BrainService
from backend.chat.service import ChatService, ChatServiceConfig
from backend.learning.models import LearningConfig
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.llm.base import LLMProvider, LLMRequest
from backend.memory.models import MemoryConfig
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.skills.library import create_default_skill_library
from backend.skills.models import SkillCategory
from backend.social.service import SocialAutonomyService
from backend.speech.service import SpeechService
from backend.tts.config import PROVIDER_MOCK, TTSConfig


BASE_TIME = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DANCE_SOURCE = PROJECT_ROOT / "unity/3d_unity/Assets/animation/dance/dance.fbx"
DANCE_REFERENCE = "Assets/animation/dance"
DANCE_DURATION = 18.466667
DANCE_SAMPLE_RATE = 30.0

MEMORY_DB = PROJECT_ROOT / "Logs/phase19_memory.db"
LEARNING_DB = PROJECT_ROOT / "Logs/phase19_learning.db"
REPORT_PATH = PROJECT_ROOT / "Temp/Phase19Production.json"

VIEWER_A = "phase19_viewer_01"
VIEWER_B = "phase19_viewer_02"

#: ViewerMemory starts at relationship_score 0.20 and gains 0.05 per success,
#: so exchanges 1-5 land in this bucket (score 0.20 -> 0.45).
CONTEXT_KEY_00 = (
    "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|0_0"
)
#: Exchanges 6-7 land here (score 0.50 -> 0.55). It is *not* a fresh bucket:
#: the first seven exchanges already left global SkillKnowledge in it.
CONTEXT_KEY_05 = (
    "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|0_5"
)
#: Fresh, decision-relevant buckets: the phone sits on the floor, so neither
#: dance_02 nor dance_warmup has any recorded evidence in them.
FLOOR_CONTEXT_KEY_00 = (
    "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|FLOOR|0_0"
)
FLOOR_CONTEXT_KEY_05 = (
    "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|FLOOR|0_5"
)

DANCE_TEXTS = (
    "Lacrimosa, please dance again like before.",
    "Lacrimosa, can you dance for us now?",
    "Lacrimosa, would you dance once more please?",
    "Lacrimosa, I would love to see you dance now.",
    "Lacrimosa, can you dance one more time?",
    "Lacrimosa, please dance for the stream again.",
    "Lacrimosa, could you dance for everyone watching?",
)
DECISIVE_DANCE_TEXT = "Lacrimosa, please dance again for me."


# --------------------------------------------------------------------------- #
# deterministic environment
# --------------------------------------------------------------------------- #


class FixedClock:
    """Deterministic clock so cooldowns and rate limits are exercised for real."""

    def __init__(self, start: datetime = BASE_TIME) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class CapturingChatLLM(LLMProvider):
    """Chat provider that records the exact prompt the CharacterContext built."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request: LLMRequest) -> str:
        self.prompts.append(request.prompt)
        intent_value = str(request.context.get("intent"))
        target = request.context.get("target")
        requested = (
            target
            if intent_value in {"REQUEST_ACTION", "REQUEST_MOVEMENT"} and target
            else None
        )
        return json.dumps(
            {
                "speech": "Of course, I can do that.",
                "intent": intent_value,
                "requested_action": requested,
                "priority": float(request.context.get("priority", 0.0)),
            }
        )


class CapturingBrainProvider(LLMProvider):
    """Production mock planner plus a record of what the planner was given."""

    def __init__(self) -> None:
        self.inner = MockBrainLLMProvider()
        self.requests: list[dict[str, Any]] = []

    def generate(self, request: LLMRequest) -> str:
        self.requests.append(
            {"prompt": request.prompt, "context": dict(request.context)}
        )
        return self.inner.generate(request)


class FailingMemoryRepository:
    """Every repository call fails; personalization must degrade, not raise."""

    def __getattr__(self, name: str):
        def raise_memory_error(*args: Any, **kwargs: Any):
            raise RuntimeError("database unavailable")

        return raise_memory_error


def _snapshot(placement_surface: str = "Table") -> dict[str, Any]:
    """Live production shape: streaming, phone placed on the floor/table."""
    return {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 209,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {
            "state": "Placed",
            "position": [-3.1, 0.763, 0],
            "holder": None,
            "placement_surface": placement_surface,
        },
        "stream": {
            "is_streaming": True,
            "camera_available": True,
            "camera_active": True,
            "duration_seconds": 1720,
            "viewer_count": 1,
        },
        "targets": {
            "CENTER": {
                "position": [0, 0.05, 0],
                "navigation_target": True,
                "look_at_target": True,
            },
            "TABLE": {
                "position": [-3.1, 0.735, 0],
                "navigation_target": True,
                "look_at_target": True,
            },
            "CAMERA": {
                "position": [-3.1, 0.766, 0],
                "navigation_target": False,
                "look_at_target": True,
            },
        },
    }


def _result(action: str, viewer_response: str = "positive") -> dict[str, Any]:
    return {
        "action": action,
        "status": "SUCCESS",
        "reason": "probe_success",
        "reached": True,
        "viewer_response": viewer_response,
    }


# --------------------------------------------------------------------------- #
# report plumbing
# --------------------------------------------------------------------------- #


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {}
        self.gates: dict[str, str] = {}

    def check(
        self,
        name: str,
        ok: bool,
        detail: Any = None,
        *,
        gate: str | None = None,
    ) -> bool:
        entry: dict[str, Any] = {"check": name, "ok": bool(ok)}
        if gate:
            entry["gate"] = gate
            self.gates[gate] = "PASS" if ok else "BLOCKED"
        if detail is not None:
            entry["detail"] = detail
        self.checks.append(entry)
        return bool(ok)

    def record(self, key: str, value: Any) -> None:
        self.evidence[key] = value

    def failed(self) -> list[str]:
        return [item["check"] for item in self.checks if not item["ok"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "19",
            "probe": "backend.tools.phase19_production_probe",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "databases": {"memory": str(MEMORY_DB), "learning": str(LEARNING_DB)},
            "gates": self.gates,
            "summary": {
                "checks": len(self.checks),
                "passed": len(self.checks) - len(self.failed()),
                "failed": len(self.failed()),
                "failed_checks": self.failed(),
            },
            "checks": self.checks,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------- #
# production stack
# --------------------------------------------------------------------------- #


@dataclass
class Stack:
    """One brain generation, wired exactly like ``create_default_service``."""

    service: BrainService
    memory: MemoryService
    speech: SpeechService
    planner: CapturingBrainProvider

    def close(self) -> None:
        for step in (
            self.service.brain.learning.close,
            self.memory.close,
            self.speech.shutdown,
        ):
            try:
                step()
            except Exception:
                pass


def build_stack(clock: FixedClock, chat: ChatService) -> Stack:
    """Mirror ``backend.api.brain.create_default_service`` with a fixed clock.

    Same classes, same wiring and the same one-shot personality persistence;
    only the clock is injected so cooldowns advance deterministically.
    """
    memory = MemoryService(
        SQLiteMemoryRepository(MEMORY_DB),
        MemoryConfig(database_path=MEMORY_DB),
    )
    social = SocialAutonomyService(clock=clock)
    library = create_default_skill_library()
    learning_config = LearningConfig(database_path=LEARNING_DB)
    learning = SkillLearningService(
        library,
        learning_config,
        SQLiteSkillLearningRepository(learning_config.database_path),
        clock=clock,
    )
    speech = SpeechService(
        tts_config=TTSConfig(provider=PROVIDER_MOCK),
        clock=clock,
        start_worker=False,
    )
    planner = CapturingBrainProvider()
    brain = AutonomousBrain(
        planner,
        chat,
        BrainConfig(decision_interval_seconds=0.0),
        clock=clock,
        memory_service=memory,
        social_service=social,
        skill_library=library,
        learning_service=learning,
        speech_service=speech,
    )
    service = BrainService(brain, chat)
    # The factory persists the stable profile once; nothing drifts it later.
    memory.set_personality(brain.personality)
    return Stack(service=service, memory=memory, speech=speech, planner=planner)


def _register_dance(learning: SkillLearningService, motion_id: str) -> None:
    learning.register_motion(
        motion_id,
        motion_id.replace("_", " ").title(),
        f"Phase 19 production probe motion {motion_id}",
        DANCE_REFERENCE,
        DANCE_DURATION,
        tags=("dance",),
        source_path=DANCE_SOURCE,
        source_format="FBX",
        clip_name="Dance",
        sample_rate=DANCE_SAMPLE_RATE,
        skill_category=SkillCategory.MOVEMENT,
    )


# --------------------------------------------------------------------------- #
# production interactions
# --------------------------------------------------------------------------- #


def _complete(
    stack: Stack,
    response: str = "positive",
    *,
    snapshot: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Feed SUCCESS results for each pending action until the plan finishes."""
    world = snapshot if snapshot is not None else _snapshot()
    steps: list[dict[str, str]] = []
    for _ in range(8):
        pending = stack.service.get_state().get("pending_action")
        if not pending:
            break
        action = str(pending["action"])
        tick = stack.service.tick(world, _result(action, response))
        steps.append(
            {"action": action, "status": tick.status.value, "message": tick.message}
        )
        if tick.status is BrainStatus.IDLE:
            break
    return steps


def _request(
    stack: Stack,
    clock: FixedClock,
    viewer_id: str,
    text: str,
    *,
    message_id: str,
    gap: float = 35.0,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One real chat request: submit, then the Brain's first decision tick."""
    clock.advance(gap)
    submission = stack.service.chat_service.submit(
        viewer_id, text, message_id=message_id
    )
    tick = stack.service.tick(
        snapshot if snapshot is not None else _snapshot(), None
    )
    return {
        "accepted": submission.accepted,
        "tick": tick,
        "skill_id": stack.service.brain.current_skill_id(),
        "personalization": stack.service.brain.last_personalization,
        "context_key": stack.service.brain._active_context_key,  # noqa: SLF001
    }


def _dance_exchange(
    stack: Stack,
    clock: FixedClock,
    text: str,
    *,
    message_id: str,
    viewer_id: str = VIEWER_A,
    response: str = "positive",
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = _request(
        stack,
        clock,
        viewer_id,
        text,
        message_id=message_id,
        snapshot=snapshot,
    )
    completion = _complete(stack, response, snapshot=snapshot)
    return {"request": request, "completion": completion}


def _viewer(stack: Stack, viewer_id: str) -> dict[str, Any]:
    return stack.memory.get_viewer(viewer_id)


def _sqlite_evidence() -> dict[str, Any]:
    connection = sqlite3.connect(MEMORY_DB)
    connection.row_factory = sqlite3.Row
    try:
        viewers = {
            row["viewer_id"]: {
                key: row[key]
                for key in row.keys()
                if key not in {"preferences_json", "preferred_topics_json"}
            }
            for row in connection.execute("SELECT * FROM viewer_memories")
        }
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM memory_metadata")
        }
        counts = {
            "episodes": connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "interactions": connection.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0],
            "viewer_memories": len(viewers),
        }
    finally:
        connection.close()
    return {"viewer_memories": viewers, "metadata": metadata, "counts": counts}


def _skill_view(stack: Stack, context_key: str, skill_ids: tuple[str, ...]) -> dict[str, Any]:
    ranking = stack.service.brain.learning.ranking_prompt_context(context_key)
    return {
        item["skill_id"]: item
        for item in ranking
        if item["skill_id"] in skill_ids
    }


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #


def run() -> Report:
    report = Report()
    for path in (MEMORY_DB, LEARNING_DB):
        if path.exists():
            path.unlink()

    clock = FixedClock()
    chat_llm = CapturingChatLLM()
    chat = ChatService(
        chat_llm,
        ChatServiceConfig(max_messages_per_rate_window=50),
        clock=clock,
    )

    # ------------------------------------------------------------------ #
    # session 1: seven successful dance exchanges for viewer A
    # ------------------------------------------------------------------ #
    first = build_stack(clock, chat)
    personality_at_start = personality_traits(first.service.brain.personality)
    report.record("personality_at_start", personality_at_start)
    report.record(
        "personality_diagnostics_at_start",
        first.memory.personality_diagnostics(),
    )
    _register_dance(first.service.brain.learning, "dance_02")

    exchanges: list[dict[str, Any]] = []
    progression: list[dict[str, Any]] = []
    for index, text in enumerate(DANCE_TEXTS, start=1):
        exchange = _dance_exchange(
            first,
            clock,
            text,
            message_id=f"phase19_a_{index:02d}",
        )
        exchanges.append(
            {
                "message_id": f"phase19_a_{index:02d}",
                "text": text,
                "action": exchange["request"]["tick"].action.action.value
                if exchange["request"]["tick"].action
                else None,
                "skill_id": exchange["request"]["skill_id"],
                "relationship_level": _viewer(first, VIEWER_A)["viewer"][
                    "relationship_level"
                ],
                "completion": exchange["completion"],
            }
        )
        snapshot_row = _viewer(first, VIEWER_A)["viewer"]
        progression.append(
            {
                "exchange": index,
                "relationship_level": snapshot_row["relationship_level"],
                "relationship_score": snapshot_row["relationship_score"],
                "familiarity": snapshot_row["familiarity"],
                "engagement_score": snapshot_row["engagement_score"],
            }
        )
    report.record("viewer_a_exchanges", exchanges)
    report.record("viewer_a_progression", progression)

    viewer_a_after_seven = _viewer(first, VIEWER_A)
    report.record("viewer_a_after_seven", viewer_a_after_seven)
    payload_a7 = viewer_a_after_seven["viewer"]
    report.check(
        "viewer_a seven successful exchanges recorded",
        payload_a7 is not None
        and payload_a7["interaction_count"] == 7
        and payload_a7["successful_interactions"] == 7,
        payload_a7,
        gate="returning_viewer_retention",
    )
    report.check(
        "viewer_a reached FAMILIAR gradually",
        payload_a7["relationship_level"] == "FAMILIAR",
        {
            "relationship_level": payload_a7["relationship_level"],
            "familiarity": payload_a7["familiarity"],
            "relationship_score": payload_a7["relationship_score"],
            "engagement_score": payload_a7["engagement_score"],
        },
        gate="relationship_progression",
    )
    ranks = {"NEW": 0, "FAMILIAR": 1, "REGULAR": 2, "TRUSTED": 3}
    rank_steps = [ranks[row["relationship_level"]] for row in progression]
    score_steps = [
        round(progression[index + 1]["relationship_score"] - progression[index]["relationship_score"], 6)
        for index in range(len(progression) - 1)
    ]
    report.check(
        "relationship accumulation is gradual (at most one level per exchange)",
        all(second - first_rank in (0, 1) for first_rank, second in zip(rank_steps, rank_steps[1:]))
        and rank_steps[0] == 0
        and rank_steps[-1] == 1
        and rank_steps.index(1) >= 2
        and set(score_steps) == {0.05},
        progression,
        gate="relationship_progression",
    )
    report.check(
        "every viewer A exchange executed through the learned dance skill",
        all(exchange["skill_id"] == "dance_02" for exchange in exchanges[1:]),
        [exchange["skill_id"] for exchange in exchanges],
    )
    report.check(
        "repeated dance evidence produced bounded preferences",
        "dance" in payload_a7["preferred_topics"]
        and {"dance", "dance_02", "speak"} <= set(payload_a7["preferred_skills"]),
        {
            "preferred_topics": payload_a7["preferred_topics"],
            "preferred_skills": payload_a7["preferred_skills"],
        },
        gate="preference_learning",
    )

    sqlite_before_restart = _sqlite_evidence()
    report.record("sqlite_before_restart", sqlite_before_restart)

    first.close()

    # ------------------------------------------------------------------ #
    # session 2: restart on the same stores
    # ------------------------------------------------------------------ #
    second = build_stack(clock, chat)
    personality_after_restart = personality_traits(second.service.brain.personality)
    report.check(
        "personality profile retained across restart",
        personality_after_restart == personality_at_start
        and second.memory.personality_diagnostics()["source"] == "persisted",
        {
            "traits": personality_after_restart,
            "source": second.memory.personality_diagnostics()["source"],
        },
        gate="personality_stability",
    )

    viewer_a_after_restart = _viewer(second, VIEWER_A)
    report.record("viewer_a_after_restart", viewer_a_after_restart)
    payload_a_restart = viewer_a_after_restart["viewer"]
    report.check(
        "viewer A relationship/preferences/interactions survived restart",
        payload_a_restart is not None
        and payload_a_restart["interaction_count"] == payload_a7["interaction_count"]
        and payload_a_restart["relationship_level"] == payload_a7["relationship_level"]
        and payload_a_restart["preferred_topics"] == payload_a7["preferred_topics"]
        and payload_a_restart["preferred_skills"] == payload_a7["preferred_skills"]
        and abs(payload_a_restart["familiarity"] - payload_a7["familiarity"]) < 1e-6
        and payload_a_restart["last_interaction"] == payload_a7["last_interaction"],
        payload_a_restart,
        gate="returning_viewer_retention",
    )
    reloaded_dance_02 = second.service.get_learning_skill("dance_02")
    reloaded_contexts = list(reloaded_dance_02["contexts"])
    reloaded_dance_contexts = {
        item["context_key"]: item
        for item in reloaded_contexts
        if str(item["context_key"]).startswith("REQUEST_ACTION|DANCE|")
    }
    report.record("reloaded_dance_02_knowledge", reloaded_dance_02)
    report.check(
        "SkillKnowledge history reloaded by the learning service",
        sum(item["trials"] for item in reloaded_contexts) == 7
        and sum(item["successes"] for item in reloaded_contexts) == 7
        and reloaded_dance_contexts[CONTEXT_KEY_00]["trials"] == 5
        and reloaded_dance_contexts[CONTEXT_KEY_05]["trials"] >= 1,
        reloaded_dance_02,
        gate="skill_learning",
    )

    # ------------------------------------------------------------------ #
    # ranking: learned history + viewer preference + personality
    # ------------------------------------------------------------------ #
    # The floor buckets are untouched by the probe so far: both dance skills
    # have zero learned evidence there, which is what makes the counterfactual
    # below a clean one-variable comparison.
    _register_dance(second.service.brain.learning, "dance_warmup")
    baseline_ranking = _skill_view(
        second, FLOOR_CONTEXT_KEY_05, ("dance_02", "dance_warmup")
    )
    baseline_ranking_00 = _skill_view(
        second, FLOOR_CONTEXT_KEY_00, ("dance_02", "dance_warmup")
    )
    report.record("ranking_floor_0_5_unpersonalized", baseline_ranking)
    report.record("ranking_floor_0_0_unpersonalized", baseline_ranking_00)
    report.check(
        "fresh context starts with equal learned evidence",
        all(
            view["dance_02"]["advisory_score"]
            == view["dance_warmup"]["advisory_score"]
            == 0.45
            for view in (baseline_ranking, baseline_ranking_00)
        ),
        {"floor_0_5": baseline_ranking, "floor_0_0": baseline_ranking_00},
    )
    baseline_tie_break = max(
        (item["advisory_score"], skill_id)
        for skill_id, item in baseline_ranking.items()
    )[1]
    report.check(
        "unpersonalized tie-break favours dance_warmup",
        baseline_tie_break == "dance_warmup",
        baseline_tie_break,
    )

    personalized = second.memory.personalize(
        VIEWER_A,
        message_text=DECISIVE_DANCE_TEXT,
        intent="REQUEST_ACTION",
        personality=second.service.brain.personality,
    )
    personalized_ranking = second.service.brain.learning.ranking_prompt_context(
        FLOOR_CONTEXT_KEY_05,
        personalization=personalized,
        personality=second.service.brain.personality,
    )
    personalized_view = {
        item["skill_id"]: item
        for item in personalized_ranking
        if item["skill_id"] in {"dance_02", "dance_warmup"}
    }
    report.record("personalized_context_for_8th_request", personalized.to_dict())
    report.record("ranking_floor_0_5_personalized", personalized_view)
    report.check(
        "viewer preference + personality bias the advisory ranking only",
        personalized_view["dance_02"]["advisory_score"]
        > personalized_view["dance_warmup"]["advisory_score"]
        and personalized_view["dance_02"]["personalization_bias"] > 0
        and personalized_view["dance_02"]["learned_success_rate"]
        == baseline_ranking["dance_02"]["learned_success_rate"],
        personalized_view,
        gate="preference_affects_decision",
    )

    # ------------------------------------------------------------------ #
    # 8th request through the production loop
    # ------------------------------------------------------------------ #
    eighth = _dance_exchange(
        second,
        clock,
        DECISIVE_DANCE_TEXT,
        message_id="phase19_a_08",
        snapshot=_snapshot("Floor"),
    )
    eighth_tick = eighth["request"]["tick"]
    planner_context = (
        second.planner.requests[-1]["context"] if second.planner.requests else {}
    )
    planner_ranking = {
        item.get("skill_id"): item
        for item in planner_context.get("learned_ranking", [])
        if isinstance(item, dict)
    }
    report.record(
        "eighth_request",
        {
            "text": DECISIVE_DANCE_TEXT,
            "action": eighth_tick.action.action.value if eighth_tick.action else None,
            "plan_steps": [
                step.action.value for step in eighth_tick.plan.steps
            ]
            if eighth_tick.plan
            else [],
            "executing_skill_id": eighth["request"]["skill_id"],
            "context_key": eighth["request"]["context_key"],
            "ranking_passed_to_planner": planner_ranking,
            "completion": eighth["completion"],
        },
    )
    report.check(
        "the 8th request ran in the fresh floor bucket",
        eighth["request"]["context_key"] == FLOOR_CONTEXT_KEY_05,
        eighth["request"]["context_key"],
    )
    report.check(
        "learned viewer preference selected dance_02 in the real Brain loop",
        eighth["request"]["skill_id"] == "dance_02"
        and eighth_tick.action is not None
        and eighth_tick.action.action.value == "DANCE",
        {
            "executing_skill_id": eighth["request"]["skill_id"],
            "action": eighth_tick.action.action.value if eighth_tick.action else None,
        },
        gate="preference_affects_decision",
    )
    report.check(
        "planner received the bounded personalized ranking",
        planner_ranking.get("dance_02", {}).get("personalization_bias") is not None
        and planner_ranking["dance_02"]["advisory_score"]
        > planner_ranking["dance_warmup"]["advisory_score"],
        planner_ranking,
        gate="preference_affects_decision",
    )

    viewer_a_after_eight = _viewer(second, VIEWER_A)
    report.record("viewer_a_after_eight", viewer_a_after_eight)
    payload_a8 = viewer_a_after_eight["viewer"]
    report.check(
        "sustained return interactions promoted viewer A to REGULAR",
        payload_a8["interaction_count"] == 8
        and payload_a8["relationship_level"] == "REGULAR",
        {
            "interaction_count": payload_a8["interaction_count"],
            "relationship_level": payload_a8["relationship_level"],
            "relationship_score": payload_a8["relationship_score"],
            "familiarity": payload_a8["familiarity"],
            "engagement_score": payload_a8["engagement_score"],
        },
        gate="relationship_progression",
    )
    report.check(
        "TRUSTED is not reachable from eight successful exchanges",
        payload_a8["relationship_level"] != "TRUSTED",
        payload_a8["relationship_level"],
    )

    # ------------------------------------------------------------------ #
    # viewer B isolation (checked before its first interaction)
    # ------------------------------------------------------------------ #
    viewer_b_before = _viewer(second, VIEWER_B)
    report.record("viewer_b_before_first_interaction", viewer_b_before)
    personalization_b = viewer_b_before["personalization"]
    report.check(
        "new viewer has no inherited relationship or preferences",
        viewer_b_before["viewer"] is None
        and not personalization_b["known_viewer"]
        and not personalization_b["returning"]
        and personalization_b["preferred_topics"] == []
        and personalization_b["preferred_skills"] == []
        and personalization_b["greeting"] is None,
        personalization_b,
        gate="viewer_isolation",
    )
    report.check(
        "viewer A context never leaks into viewer B",
        "dance" not in json.dumps(personalization_b)
        and VIEWER_A not in json.dumps(personalization_b),
        personalization_b,
        gate="viewer_isolation",
    )

    # ------------------------------------------------------------------ #
    # personalized greeting (viewer B never gets one)
    # ------------------------------------------------------------------ #
    before_b_prompts = len(chat_llm.prompts)
    greeting_b = _request(
        second, clock, VIEWER_B, "hello Lacrimosa", message_id="phase19_b_01"
    )
    prompt_b = chat_llm.prompts[before_b_prompts] if len(chat_llm.prompts) > before_b_prompts else ""
    report.record("viewer_b_greeting_prompt", prompt_b)
    report.check(
        "new viewer prompt carries no personalized context",
        "suggested returning-viewer greeting" not in prompt_b
        and "Viewer context (advisory only)" not in prompt_b
        and "dance_02" not in prompt_b,
        prompt_b,
        gate="viewer_isolation",
    )
    report.check(
        "new viewer keeps the base speech style line",
        "Speech style:" in prompt_b,
        prompt_b.splitlines()[:8],
    )
    report.check(
        "viewer B greeting stayed NEW with no inherited preferences",
        greeting_b["personalization"] is not None
        and not greeting_b["personalization"].known
        and greeting_b["personalization"].greeting is None,
        greeting_b["personalization"].to_dict()
        if greeting_b["personalization"]
        else None,
        gate="viewer_isolation",
    )
    _complete(second)
    viewer_b_after = _viewer(second, VIEWER_B)["viewer"]
    report.record("viewer_b_after_first_interaction", viewer_b_after)
    report.check(
        "viewer B first interaction stays NEW and preference-free",
        viewer_b_after is not None
        and viewer_b_after["interaction_count"] == 1
        and viewer_b_after["relationship_level"] == "NEW"
        and viewer_b_after["preferred_topics"] == []
        and viewer_b_after["preferred_skills"] == [],
        viewer_b_after,
        gate="viewer_isolation",
    )

    # ------------------------------------------------------------------ #
    # counterfactual: the same dance request in an equally fresh bucket
    # ------------------------------------------------------------------ #
    viewer_b_dance = _dance_exchange(
        second,
        clock,
        DECISIVE_DANCE_TEXT,
        message_id="phase19_b_02",
        viewer_id=VIEWER_B,
        snapshot=_snapshot("Floor"),
    )
    viewer_b_ranking = {
        item.get("skill_id"): item
        for item in (
            second.planner.requests[-1]["context"].get("learned_ranking", [])
            if second.planner.requests
            else []
        )
        if isinstance(item, dict)
        and item.get("skill_id") in {"dance_02", "dance_warmup"}
    }
    report.record(
        "viewer_b_counterfactual",
        {
            "text": DECISIVE_DANCE_TEXT,
            "context_key": viewer_b_dance["request"]["context_key"],
            "executing_skill_id": viewer_b_dance["request"]["skill_id"],
            "ranking_passed_to_planner": viewer_b_ranking,
        },
    )
    report.check(
        "viewer B ran the same request in an equally fresh floor bucket",
        viewer_b_dance["request"]["context_key"] == FLOOR_CONTEXT_KEY_00
        and viewer_b_ranking.get("dance_02", {}).get("advisory_score")
        == viewer_b_ranking.get("dance_warmup", {}).get("advisory_score"),
        viewer_b_ranking,
        gate="viewer_isolation",
    )
    report.check(
        "the same request lands on the generic skill without viewer preferences",
        viewer_b_dance["request"]["skill_id"] == "dance_warmup"
        and viewer_b_dance["request"]["personalization"] is not None
        and "dance" not in viewer_b_dance["request"]["personalization"].preferred_topics
        and "dance_02"
        not in viewer_b_dance["request"]["personalization"].preferred_skills
        and all(
            "viewer preference"
            not in str(item.get("personalization_reason"))
            for item in viewer_b_ranking.values()
        ),
        {
            "executing_skill_id": viewer_b_dance["request"]["skill_id"],
            "personalization": viewer_b_dance["request"]["personalization"].to_dict()
            if viewer_b_dance["request"]["personalization"]
            else None,
            "ranking": viewer_b_ranking,
        },
        gate="preference_affects_decision",
    )
    report.check(
        "viewer preference, not stable personality, tipped viewer A's dance choice",
        personalized_view["dance_02"]["advisory_score"]
        > viewer_b_ranking["dance_02"]["advisory_score"]
        and "viewer preference"
        in str(personalized_view["dance_02"]["personalization_reason"])
        and "viewer preference"
        not in str(viewer_b_ranking["dance_02"]["personalization_reason"]),
        {
            "viewer_a_dance_02": personalized_view["dance_02"]["advisory_score"],
            "viewer_b_dance_02": viewer_b_ranking["dance_02"]["advisory_score"],
            "viewer_a_reason": personalized_view["dance_02"]["personalization_reason"],
            "viewer_b_reason": viewer_b_ranking["dance_02"]["personalization_reason"],
        },
        gate="preference_affects_decision",
    )

    before_a_prompts = len(chat_llm.prompts)
    greeting_a = _request(
        second,
        clock,
        VIEWER_A,
        "hello Lacrimosa",
        message_id="phase19_a_09",
        gap=40.0,
    )
    prompt_a = chat_llm.prompts[before_a_prompts] if len(chat_llm.prompts) > before_a_prompts else ""
    report.record("viewer_a_greeting_prompt", prompt_a)
    report.record(
        "viewer_a_greeting_context",
        greeting_a["personalization"].to_dict()
        if greeting_a["personalization"]
        else None,
    )
    report.check(
        "returning viewer A greeting reaches the production chat prompt",
        "suggested returning-viewer greeting" in prompt_a
        and greeting_a["personalization"] is not None
        and greeting_a["personalization"].returning
        and greeting_a["personalization"].greeting is not None
        and "REGULAR" in prompt_a,
        prompt_a,
        gate="personalized_greeting",
    )
    _complete(second)

    # The same greeting is suppressed while the social speech cooldown runs:
    # a fresh message is queued 6s later, still inside the 12s speech cooldown.
    clock.advance(6.0)
    cooldown_submission = second.service.chat_service.submit(
        VIEWER_A, "good evening Lacrimosa", message_id="phase19_a_10_cooldown"
    )
    queued_message = second.service.chat_service.queue.peek()
    blocked_greeting = second.service.brain._personalized_context(  # noqa: SLF001
        queued_message
    )
    report.record(
        "greeting_during_social_cooldown",
        {
            "accepted": cooldown_submission.accepted,
            "queued": queued_message.raw_text if queued_message else None,
            "personalization": blocked_greeting.to_dict() if blocked_greeting else None,
        },
    )
    report.check(
        "greeting reuses the existing social speech cooldown",
        cooldown_submission.accepted
        and queued_message is not None
        and blocked_greeting is not None
        and blocked_greeting.known
        and blocked_greeting.returning
        and blocked_greeting.greeting is None,
        {
            "accepted": cooldown_submission.accepted,
            "personalization": blocked_greeting.to_dict()
            if blocked_greeting
            else None,
        },
        gate="personalized_greeting",
    )
    second.service.chat_service.discard_next("phase19_probe_cooldown_drain")

    # ------------------------------------------------------------------ #
    # topic continuity
    # ------------------------------------------------------------------ #
    related = _request(
        second,
        clock,
        VIEWER_A,
        "Lacrimosa, please dance again soon.",
        message_id="phase19_a_10",
    )
    report.record(
        "topic_continuity_related",
        related["personalization"].to_dict() if related["personalization"] else None,
    )
    report.check(
        "recent repeated topic continues for a related message",
        related["personalization"] is not None
        and related["personalization"].topic_continuity == "dance",
        related["personalization"].topic_continuity
        if related["personalization"]
        else None,
        gate="topic_continuity",
    )
    _complete(second)

    unrelated = _request(
        second,
        clock,
        VIEWER_A,
        "what is your favourite colour?",
        message_id="phase19_a_11",
    )
    report.record(
        "topic_continuity_unrelated",
        unrelated["personalization"].to_dict()
        if unrelated["personalization"]
        else None,
    )
    report.check(
        "unrelated conversation is not forced into the old topic",
        unrelated["personalization"] is not None
        and unrelated["personalization"].topic_continuity is None,
        unrelated["personalization"].topic_continuity
        if unrelated["personalization"]
        else None,
        gate="topic_continuity",
    )
    _complete(second)

    # ------------------------------------------------------------------ #
    # personality stability across a negative event
    # ------------------------------------------------------------------ #
    negative = _dance_exchange(
        second,
        clock,
        "Lacrimosa, please dance again now.",
        message_id="phase19_a_12",
        response="negative",
    )
    report.record(
        "negative_exchange",
        {
            "skill_id": negative["request"]["skill_id"],
            "completion": negative["completion"],
        },
    )
    personality_at_end = personality_traits(second.service.brain.personality)
    persisted_at_end = personality_traits(second.memory.load_personality_state())
    report.record("personality_at_end", personality_at_end)
    report.check(
        "personality traits are identical before and after the whole probe",
        personality_at_end == personality_at_start
        and persisted_at_end == personality_at_start,
        {
            "start": personality_at_start,
            "end": personality_at_end,
            "persisted": persisted_at_end,
            "diagnostics": second.memory.personality_diagnostics(),
        },
        gate="personality_stability",
    )

    # ------------------------------------------------------------------ #
    # failure fallback: broken viewer store must not stop the Brain
    # ------------------------------------------------------------------ #
    broken_memory = MemoryService(
        FailingMemoryRepository(),
        MemoryConfig(database_path=PROJECT_ROOT / "Logs/phase19_broken.db"),
    )
    broken_chat = ChatService(
        CapturingChatLLM(),
        ChatServiceConfig(max_messages_per_rate_window=50),
        clock=clock,
    )
    broken_planner = CapturingBrainProvider()
    broken_brain = AutonomousBrain(
        broken_planner,
        broken_chat,
        BrainConfig(decision_interval_seconds=0.0),
        clock=clock,
        memory_service=broken_memory,
    )
    clock.advance(40.0)
    broken_chat.submit("phase19_viewer_03", "hello Lacrimosa", message_id="phase19_c_01")
    broken_tick = broken_brain.tick(_snapshot_from_dict(_snapshot()))
    report.record(
        "degraded_tick",
        {
            "status": broken_tick.status.value,
            "message": broken_tick.message,
            "personalization": broken_brain.last_personalization.to_dict()
            if broken_brain.last_personalization
            else None,
        },
    )
    report.check(
        "personalization failure falls back to generic context",
        broken_tick.status in {BrainStatus.EXECUTING, BrainStatus.WAITING}
        and broken_brain.last_personalization is not None
        and not broken_brain.last_personalization.known
        and broken_brain.last_personalization.preferred_topics == ()
        and broken_brain.last_personalization.greeting is None,
        {
            "status": broken_tick.status.value,
            "personalization": broken_brain.last_personalization.to_dict()
            if broken_brain.last_personalization
            else None,
        },
        gate="failure_fallback",
    )
    report.check(
        "personality falls back to configuration when the store is broken",
        personality_traits(broken_memory.load_personality_state()) == personality_at_start
        and broken_memory.personality_diagnostics()["source"] == "configuration",
        broken_memory.personality_diagnostics(),
    )

    # ------------------------------------------------------------------ #
    # final persistence picture
    # ------------------------------------------------------------------ #
    viewer_a_final = _viewer(second, VIEWER_A)
    report.record("viewer_a_final", viewer_a_final)
    report.record("sqlite_after_probe", _sqlite_evidence())
    report.check(
        "viewer A evidence kept accumulating without personality drift",
        viewer_a_final["viewer"]["interaction_count"] >= 12
        and viewer_a_final["personalization"]["reason"] != "",
        {
            "interaction_count": viewer_a_final["viewer"]["interaction_count"],
            "relationship_level": viewer_a_final["viewer"]["relationship_level"],
            "reason": viewer_a_final["personalization"]["reason"],
            "top_preferences": {
                "topics": viewer_a_final["viewer"]["preferred_topics"][:3],
                "skills": viewer_a_final["viewer"]["preferred_skills"][:3],
            },
        },
        gate="returning_viewer_retention",
    )

    second.close()
    return report


def _snapshot_from_dict(value: dict[str, Any]):
    from backend.brain.world_state import build_world_state

    return build_world_state(value)


def main() -> int:
    report = run()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("PHASE 19 production probe")
    print(f"report: {REPORT_PATH}")
    for item in payload["checks"]:
        marker = "PASS" if item["ok"] else "FAIL"
        gate = f" [{item['gate']}]" if "gate" in item else ""
        print(f"  {marker} {item['check']}{gate}")
    summary = payload["summary"]
    print(
        f"summary: {summary['passed']}/{summary['checks']} checks passed, "
        f"failed={summary['failed']}"
    )
    if summary["failed_checks"]:
        print("failed checks: " + ", ".join(summary["failed_checks"]))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
