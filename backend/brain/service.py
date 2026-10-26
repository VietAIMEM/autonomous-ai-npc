"""Thread-safe service boundary between HTTP, brain, and Unity."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ..chat.service import ChatService
from ..runtime import instrument
from .goals import initiative_goal_context
from .loop import AutonomousBrain, BrainTickResult, _bounded_token
from .models import ActionResult, ActionType, ExecutionStatus
from .spatial import (
    TARGET_CAMERA,
    is_spatial_plan,
    plan_anchor,
    spatial_observation,
    spatial_relationship,
)
from .world_state import build_world_state


class BrainService:
    def __init__(
        self,
        brain: AutonomousBrain,
        chat_service: ChatService,
        *,
        clock=time.monotonic,
    ) -> None:
        self.brain = brain
        self.chat_service = chat_service
        self.memory_service = getattr(brain, "memory", None)
        self.speech_service = getattr(brain, "speech", None)
        self._clock = clock
        self._started_at = self._clock()
        self._started_wall = datetime.now(timezone.utc)
        self._lock = threading.RLock()
        self._last_world_dict: dict[str, Any] = {}
        self._last_result: Optional[ActionResult] = None
        self._tick_count = 0
        self._last_tick_monotonic: Optional[float] = None
        self._last_tick_wall: Optional[datetime] = None
        self._tick_in_progress = False

    def tick(self, snapshot: Mapping[str, Any], result_value: Mapping[str, Any] | None) -> BrainTickResult:
        started = self._clock()
        #: PA29-003: ``brain.tick.actions`` counts how many ticks *selected* an
        #: action to dispatch (``BrainTickResult.action``). It is a planning
        #: counter only - an action that is later dispatched, interrupted or
        #: verified is instrumented by the executor, never here. Reading a
        #: non-existent ``outcome.actions`` used to pin this metric at zero.
        selected_action = False
        with self._lock:
            if self._tick_in_progress:
                # A tick is already planning. Never run two Brain loops; the
                # caller (Unity) retries on its next decision interval.
                raise ValueError("brain_busy")
            self._tick_in_progress = True
        try:
            with self._lock:
                result = _parse_result(result_value)
                world = build_world_state(
                    snapshot,
                    viewer_count_floor=self._recent_viewer_count(),
                )
                self._last_world_dict = dict(world.to_dict())
                self._last_result = result
            # Planning can take tens of seconds with a real provider. It must
            # not hold the service lock, or /status and /diagnostics starve.
            outcome = self.brain.tick(world, result)
            selected_action = outcome.action is not None
            with self._lock:
                self._tick_count += 1
                self._last_tick_monotonic = started
                self._last_tick_wall = datetime.now(timezone.utc)
            instrument.count("brain.ticks")
            instrument.count("brain.tick.actions", 1 if selected_action else 0)
            return outcome
        except Exception:
            instrument.count("brain.tick.errors")
            instrument.record_error("brain", "tick_failed")
            raise
        finally:
            elapsed_ms = (self._clock() - started) * 1000.0
            instrument.observe_ms("brain.tick.latency_ms", elapsed_ms)
            with self._lock:
                self._tick_in_progress = False

    def _recent_viewer_count(self) -> int:
        """Real audience floor for the world snapshot.

        Unity reports its own local viewer list, which nothing populates in
        production. The chat/social runtime is the real owner of "who has been
        watching recently", so its count floors the snapshot value.
        """
        social = getattr(self.brain, "social", None)
        counter = getattr(social, "recent_viewer_count", None)
        if not callable(counter):
            return 0
        try:
            return max(0, int(counter()))
        except Exception:
            return 0

    # ---------------------------------------------------------------- runtime

    def uptime_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def last_tick_age_seconds(self) -> Optional[float]:
        """Seconds since the last tick, or ``None`` when no tick ever arrived."""
        with self._lock:
            last = self._last_tick_monotonic
        if last is None:
            return None
        return max(0.0, self._clock() - last)

    def runtime_metrics(self) -> dict[str, Any]:
        """Cheap, diagnostics-safe view of the single Brain loop."""
        with self._lock:
            last_tick = self._last_tick_wall
            payload = {
                "enabled": True,
                "tick_count": self._tick_count,
                "tick_in_progress": self._tick_in_progress,
                "uptime_seconds": round(self.uptime_seconds(), 3),
                "last_tick_age_seconds": (
                    None
                    if self._last_tick_monotonic is None
                    else round(max(0.0, self._clock() - self._last_tick_monotonic), 3)
                ),
                "last_tick_at": (
                    last_tick.astimezone(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                    if last_tick
                    else None
                ),
                "started_at": self._started_wall.astimezone(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            }
        instrument.gauge("brain.uptime_seconds", payload["uptime_seconds"])
        return payload


    def get_state(self) -> dict[str, Any]:
        with self._lock:
            runtime = self.brain.state
            pending = runtime.pending_action
            return {
                "status": runtime.status.value,
                "goal": runtime.current_goal.to_dict() if runtime.current_goal else None,
                "plan": runtime.current_plan.to_dict() if runtime.current_plan else None,
                "step_index": runtime.step_index,
                "pending_action": pending.to_dict() if pending else None,
                "last_failure": runtime.last_failure,
                "last_result": self._last_result.to_dict() if self._last_result else None,
                "world": self._last_world_dict,
                "chat": self.chat_service.get_state(),
                "memory": self.memory_service.get_state() if self.memory_service else {"enabled": False},
                "decision_interval_seconds": self.brain.config.decision_interval_seconds,
                "max_plan_steps": self.brain.planner.config.max_plan_steps,
                "social": self.brain.social.get_state() if getattr(self.brain, "social", None) else {"enabled": False},
                "behavior": self.get_behavior_state(),
                "spatial": self.get_spatial_state(),
                "personalization": self.get_personalization_state(),
                "skills": self.get_skills_runtime(),
                "speech": self.get_speech_summary(),
                "conversation": self.get_conversation_state(),
                "autonomy": self.get_autonomy_state(),
            }

    def get_autonomy_state(self) -> dict[str, Any]:
        """PHASE 27A bounded autonomy diagnostics.

        One flat read-model over the single Brain loop's initiative state:
        whether autonomy is enabled, the bounded goal it is currently
        carrying (only when that goal is an initiative), why the last
        initiative decision was reached, which activity it selected, and how
        the last real autonomous activity ended. Every field already exists on
        the Brain runtime -- there is no second diagnostics service.

        PHASE 27B adds the bounded decision evidence behind those fields: the
        candidates that passed and the per-candidate rejection reasons, the
        decision-time context the evaluator read, the last autonomous activity
        and why the loop is replanning, plus the provenance of the goal/plan in
        flight (so an injected plan is never reported as an autonomous
        decision). ``knowledge_contributions`` decomposes each ranked
        candidate's score into the part the existing verified
        ``SkillKnowledge`` record actually earned, so an operator can tell
        uniform evidence from evidence that decided the ordering. Every
        addition is additive: the Phase 27A keys stay present with their
        original meaning.
        """
        runtime = self.brain.state
        goal = runtime.current_goal
        context = initiative_goal_context(goal)
        return {
            "autonomy_enabled": bool(
                getattr(self.brain.config, "autonomy_enabled", False)
            ),
            "current_goal": goal.to_dict() if context is not None else None,
            "goal_status": goal.status.value if context is not None else None,
            "goal_reason": goal.reason if context is not None else None,
            "initiative_reason": runtime.initiative_reason,
            "selected_activity": runtime.selected_activity,
            "last_outcome": runtime.last_outcome,
            "candidate_activity": runtime.candidate_activity,
            "eligible_candidates": list(runtime.eligible_candidates),
            "candidate_rejections": [
                {"activity": activity, "reason": reason}
                for activity, reason in runtime.candidate_rejections
            ],
            "candidate_scores": [
                {"activity": activity, "score": score}
                for activity, score in runtime.candidate_scores
            ],
            "knowledge_contributions": [
                {"activity": activity, "contribution": contribution}
                for activity, contribution in runtime.knowledge_contributions
            ],
            "last_activity": self._last_autonomy_activity(runtime),
            "replanning_reason": runtime.replanning_reason,
            "next_decision_eligibility": runtime.next_decision_eligibility,
            "decision_source": runtime.decision_source,
            "initiative_context": runtime.initiative_context,
        }

    @staticmethod
    def _last_autonomy_activity(runtime: Any) -> Optional[str]:
        """The most recent completed/cancelled autonomous activity, if any.

        Read from the existing bounded journal and the last outcome, so the
        field never introduces new state: an outcome of ``"dance:failed"``
        names ``dance``.
        """
        if runtime.recent_autonomy_activities:
            return str(runtime.recent_autonomy_activities[-1][0])
        outcome = runtime.last_outcome
        if isinstance(outcome, str) and ":" in outcome:
            return outcome.split(":", 1)[0]
        return None

    def get_conversation_state(self) -> dict[str, Any]:
        """PHASE 26A turn-taking diagnostics for the current conversation.

        One flat, bounded read-model: what turn we are on, what the viewer
        wanted, what is still owed, and why anything was interrupted. It is
        derived from the existing chat context, Brain runtime and speech
        summary -- no transcripts beyond the bounded continuity fields, and no
        provider calls.
        """
        runtime = self.brain.state
        try:
            chat_payload = self.chat_service.get_state()
        except Exception:  # pragma: no cover - defensive
            chat_payload = {}
        if not isinstance(chat_payload, Mapping):
            chat_payload = {}
        chat_state = chat_payload.get("conversation")
        if not isinstance(chat_state, Mapping):
            chat_state = {}

        active = runtime.active_message
        if active is not None:
            current_intent = active.intent.value
        elif chat_state.get("previous_intent"):
            current_intent = chat_state.get("previous_intent")
        else:
            head = None
            queue = getattr(self.chat_service, "queue", None)
            if queue is not None:
                try:
                    head = queue.peek()
                except Exception:  # pragma: no cover - defensive
                    head = None
            current_intent = head.intent.value if head is not None else None

        pending = runtime.pending_action
        speech = self.get_speech_summary()
        # PHASE 26B: the same bounded expression decision that shaped this
        # turn stays visible to operators. Nothing here is transcript or
        # provider data; it is the derived tone/intensity/gesture triple.
        expression = getattr(runtime, "expression_decision", None)
        return {
            "conversation_state": chat_state.get("conversation_status"),
            "current_intent": current_intent,
            "current_topic": chat_state.get("current_topic"),
            "response_status": chat_state.get("response_status"),
            "active_viewer": chat_state.get("active_viewer"),
            "turn_count": chat_state.get("turn_count", 0),
            "pending_question": chat_state.get("pending_question"),
            "pending_action": (
                pending.action.value if pending is not None else chat_state.get("pending_action")
            ),
            "pending_action_target": (
                pending.target if pending is not None else None
            ),
            "interruption_reason": runtime.interruption_reason,
            "speech_state": speech.get("state") if isinstance(speech, Mapping) else None,
            "response_tone": _expression_field(expression, "tone"),
            "reaction_intensity": _expression_field(expression, "reaction_intensity"),
            "active_gesture": getattr(runtime, "active_gesture", None),
            "gesture_reason": _expression_field(expression, "gesture_reason"),
            "behavior_state": getattr(self.brain.behavior.state, "value", None),
            "previous_intent": chat_state.get("previous_intent"),
            "previous_response": chat_state.get("previous_response"),
            "last_interaction_at": chat_state.get("last_interaction_at"),
            "response_policy": chat_payload.get("last_response_policy"),
        }

    def get_spatial_state(self) -> dict[str, Any]:
        """PHASE 25 bounded spatial diagnostics.

        Derived from the same reported transforms the Brain reasons over, so
        operators see exactly what the verifier saw. No prompts, transcripts,
        or credentials are exposed.
        """
        world = getattr(self.brain, "world", None)
        runtime = self.brain.state
        verification = runtime.spatial_verification
        plan = runtime.current_plan
        anchor: Optional[str] = None
        if plan is not None and is_spatial_plan(plan):
            anchor = plan_anchor(plan)
        elif isinstance(verification, Mapping) and isinstance(
            verification.get("anchor"), str
        ):
            anchor = str(verification["anchor"])
        relationship = (
            spatial_relationship(world, target=anchor or TARGET_CAMERA)
            if world is not None
            else None
        )
        payload = (
            spatial_observation(relationship)
            if relationship is not None
            else {"transform_based": False, "target": anchor or TARGET_CAMERA}
        )
        payload["requested_desired_state"] = runtime.spatial_desired_state
        payload["spatial_plan_active"] = bool(plan is not None and is_spatial_plan(plan))
        payload["verification"] = verification
        payload["verification_result"] = (
            verification.get("result") if isinstance(verification, Mapping) else None
        )
        return payload

    def get_behavior_state(self) -> dict[str, Any]:
        """Lightweight, deterministic behavior diagnostics for the runtime."""
        behavior = getattr(self.brain, "behavior", None)
        if behavior is None:
            return {"enabled": False}
        character = self._last_world_dict.get("character")
        current_action = character.get("current_action") if isinstance(character, dict) else None
        skill_getter = getattr(self.brain, "current_skill_id", None)
        return behavior.to_dict(
            current_goal=self.brain.state.current_goal,
            current_skill=skill_getter() if callable(skill_getter) else None,
            current_action=current_action,
        )

    # --------------------------------------------------------- speech (Phase 11)

    def get_speech_summary(self) -> dict[str, Any]:
        speech = self.speech_service
        return speech.summary() if speech else {"enabled": False}

    def get_speech_state(self) -> dict[str, Any]:
        speech = self.speech_service
        return speech.get_state() if speech else {"enabled": False}

    def get_speech_health(self) -> dict[str, Any]:
        speech = self.speech_service
        return speech.health() if speech else {"status": "disabled", "enabled": False}

    def get_speech_request(self, request_id: str) -> dict[str, Any]:
        speech = self.speech_service
        if speech is None:
            raise KeyError("speech_disabled")
        request = speech.get(request_id)
        if request is None:
            raise KeyError("speech_request_not_found")
        return request.to_dict()

    def get_speech_audio(self, request_id: str) -> Optional[tuple[bytes, str, str]]:
        speech = self.speech_service
        if speech is None:
            return None
        return speech.audio(request_id)

    def mark_speech_playing(self, request_id: str) -> bool:
        speech = self.speech_service
        return bool(speech and speech.mark_playing(request_id))

    def mark_speech_completed(self, request_id: str) -> bool:
        speech = self.speech_service
        return bool(speech and speech.mark_completed(request_id))

    def cancel_speech(self, request_id: str, reason: str = "cancelled") -> bool:
        speech = self.speech_service
        return bool(speech and speech.cancel(request_id, reason))

    def get_skills(self) -> dict[str, Any]:
        executor = getattr(self.brain, "skill_executor", None)
        if executor is None:
            return {"enabled": False, "skills": []}
        return {
            "enabled": True,
            "skills": [skill.to_dict() for skill in executor.library.list()],
        }

    def get_skill(self, skill_id: str) -> dict[str, Any]:
        executor = getattr(self.brain, "skill_executor", None)
        if executor is None:
            raise KeyError("skills_disabled")
        skill = executor.library.get(skill_id)
        if skill is None:
            raise KeyError("skill_not_found")
        return {"skill": skill.to_dict(include_steps=True)}

    def get_skill_availability(self, skill_id: str) -> dict[str, Any]:
        executor = getattr(self.brain, "skill_executor", None)
        if executor is None:
            raise KeyError("skills_disabled")
        world = self.brain.world or build_world_state({})
        return executor.availability(skill_id, world).to_dict()

    def get_skills_runtime(self) -> dict[str, Any]:
        executor = getattr(self.brain, "skill_executor", None)
        if executor is None:
            return {"enabled": False, "runtime": None}
        return {"enabled": True, "runtime": executor.runtime_state()}

    def get_learning_state(self) -> dict[str, Any]:
        learning = getattr(self.brain, "learning", None)
        return learning.get_state() if learning else {"enabled": False}

    def get_learning_skill(self, skill_id: str) -> dict[str, Any]:
        learning = getattr(self.brain, "learning", None)
        if learning is None:
            raise KeyError("learning_disabled")
        return learning.get_skill_state(skill_id)

    def get_learning_context(self, context_key: str) -> dict[str, Any]:
        learning = getattr(self.brain, "learning", None)
        if learning is None:
            raise KeyError("learning_disabled")
        return learning.get_context_state(context_key)

    def get_learning_compositions(self) -> list[dict[str, Any]]:
        learning = getattr(self.brain, "learning", None)
        if learning is None:
            return []
        return learning.get_compositions()

    def reset_learning(self) -> None:
        learning = getattr(self.brain, "learning", None)
        if learning is not None:
            learning.reset()

    def get_social_state(self) -> dict[str, Any]:
        social = getattr(self.brain, "social", None)
        return social.get_state() if social else {"enabled": False}

    def get_goal(self) -> dict[str, Any]:
        goal = self.brain.state.current_goal
        return {"goal": goal.to_dict() if goal else None}

    def get_plan(self) -> dict[str, Any]:
        plan = self.brain.state.current_plan
        return {"plan": plan.to_dict() if plan else None}

    def get_status(self) -> dict[str, Any]:
        return {"status": self.brain.state.status.value}

    def get_memory_state(self) -> dict[str, Any]:
        if self.memory_service is None:
            return {"enabled": False}
        return self.memory_service.get_state()

    def get_viewer_memory(self, viewer_id: str) -> dict[str, Any]:
        if self.memory_service is None:
            return {"enabled": False, "viewer": None, "recent_interactions": []}
        return self.memory_service.get_viewer(viewer_id)

    def get_viewer_list(self, limit: int = 50) -> dict[str, Any]:
        """Bounded operational viewer list (PHASE 23); read-only."""
        if self.memory_service is None:
            return {"enabled": False, "viewers": [], "count": 0}
        lister = getattr(self.memory_service, "list_viewers", None)
        if not callable(lister):
            return {"enabled": True, "viewers": [], "count": 0, "reason": "not_supported"}
        payload = lister(limit)
        if isinstance(payload, Mapping):
            resolved = dict(payload)
            resolved.setdefault("enabled", True)
            return resolved
        return {"enabled": True, "viewers": [], "count": 0}

    def get_personality_state(self) -> dict[str, Any]:
        """PHASE 19 personality diagnostics; never exposes credentials."""
        profile = getattr(self.brain, "personality", None)
        traits = {}
        if profile is not None:
            try:
                from .personality import personality_speech_style, personality_traits

                traits = personality_traits(profile)
                style = personality_speech_style(profile)
            except Exception:
                traits, style = {}, None
        else:
            style = None
        persisted: dict[str, Any] = {}
        if self.memory_service is not None:
            try:
                persisted = self.memory_service.personality_diagnostics()
            except Exception:
                persisted = {}
        return {
            "profile": traits,
            "speech_style": style,
            "source": persisted.get("source", "configuration"),
            "persisted": persisted.get("profile", {}),
            "rule": "stable configuration; never drifts from runtime events",
        }

    def get_personalization_state(self) -> dict[str, Any]:
        """Lightweight explanation of the most recent personalized decision."""
        context = getattr(self.brain, "last_personalization", None)
        if context is None:
            return {"active": False}
        try:
            payload = context.to_dict()
        except Exception:
            return {"active": False}
        payload["active"] = True
        return payload


def _expression_field(expression: Any, name: str) -> Any:
    """Read one bounded field from the Phase 26B expression decision."""
    if expression is None:
        return None
    if isinstance(expression, Mapping):
        return expression.get(name)
    return getattr(expression, name, None)


def _parse_result(value: Mapping[str, Any] | None) -> ActionResult | None:
    if not value:
        return None
    try:
        action = ActionType(str(value.get("action")))
        status = ExecutionStatus(str(value.get("status")))
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid_action_result") from exc
    reason = value.get("reason", "reported")
    reached = value.get("reached")
    viewer_response = value.get("viewer_response")
    return ActionResult(
        action=action,
        status=status,
        reason=reason if isinstance(reason, str) else "reported",
        reached=reached if isinstance(reached, bool) else None,
        viewer_response=(
            viewer_response.strip().casefold()
            if isinstance(viewer_response, str) and viewer_response.strip()
            else None
        ),
        # PHASE 34: only an identifier-shaped value is accepted; a missing,
        # empty, or malformed id becomes ``None`` so it can never match an
        # id-bearing pending dispatch.
        dispatch_id=_bounded_token(value.get("dispatch_id")),
    )
