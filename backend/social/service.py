"""Phase 08 service boundary for social attention and conversation state."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from ..brain.models import ActionResult, WorldState
from ..chat.models import ChatIntent, ChatMessage, utc_now
from ..runtime.stream_latch import StreamStopLatch
from .attention import assess_camera_framing, SocialAttention
from .context import SocialContext
from .conversation import ConversationManager
from .models import (
    AttentionDecision,
    SocialConfig,
    SocialDecision,
    SocialSpeakingPolicy,
    ViewerSocialContext,
)


#: Only these Unity viewer responses become bounded social reactions.
VIEWER_REACTION_KINDS = frozenset({"positive", "negative", "neutral"})


class SocialAutonomyService:
    def __init__(
        self,
        config: SocialConfig | None = None,
        clock: Callable[[], datetime] = utc_now,
        *,
        stream_latch: StreamStopLatch | None = None,
    ) -> None:
        self.config = config or SocialConfig()
        self.clock = clock
        self.context = SocialContext(self.config, stop_latch=stream_latch)
        self.conversation = ConversationManager(self.config)
        self.attention = SocialAttention(self.config, self.conversation)

    def process_message(
        self,
        message: ChatMessage,
        world: WorldState | None,
        memory_service=None,
    ) -> SocialDecision:
        now = self.clock()
        viewer_context = self.load_viewer_context(message.viewer_id, memory_service)
        assessment = self.attention.evaluate(
            message,
            self.context,
            world,
            viewer_context,
            now,
        )
        self.context.record_message(message, now)
        decision = SocialDecision(
            decision=assessment.decision,
            speaking_policy=assessment.speaking_policy,
            viewer_id=message.viewer_id,
            topic=assessment.topic,
            reason=assessment.reason,
            priority=assessment.priority,
            attention_score=assessment.score,
            message=message,
        )
        self._apply_decision(decision, now)
        return decision

    def notify_processed(
        self,
        message: ChatMessage,
        speech: str | None,
        requested_action: str | None = None,
        now: datetime | None = None,
    ) -> None:
        now = now or self.clock()
        if message.intent in {ChatIntent.FAREWELL, ChatIntent.CONVERSATION_END}:
            self.context.conversation_state = self.context.conversation_state.ENDING
        # PHASE 26B: the viewer's own wording is the only honest source of a
        # sentiment reaction on the chat path. Recording the already-classified
        # kind keeps behavior selection explainable without a second model call.
        reaction = self.conversation.classify_reaction(message)
        if reaction in {"positive", "negative"}:
            self.context.note_reaction(message.viewer_id, reaction, now)
        if speech:
            self.context.record_speech(message.viewer_id, speech, now)
        if requested_action:
            self.context.note_reaction(message.viewer_id, "request_followup", now)

    def update_on_result(
        self,
        result: ActionResult,
        now: datetime | None = None,
        viewer_id: str | None = None,
    ) -> None:
        now = now or self.clock()
        self.context.update_on_action_result(
            result.action.value,
            result.status.value == "SUCCESS",
            now,
            viewer_id,
        )
        self.record_viewer_reaction(result.viewer_response, viewer_id, now)

    def record_viewer_reaction(
        self,
        response: str | None,
        viewer_id: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Fold one Unity viewer response into the existing reaction buffer.

        ``none``, empty, and unknown values are ignored so the Brain never sees
        a fabricated reaction. A missing viewer id stays inside the bounded
        ``unknown`` bucket instead of inventing an identity.
        """
        kind = response.strip().casefold() if isinstance(response, str) else ""
        if kind not in VIEWER_REACTION_KINDS:
            return False
        viewer = (viewer_id or "").strip()[:120] or "unknown"
        self.context.note_reaction(viewer, kind, now or self.clock())
        return True

    def load_viewer_context(
        self,
        viewer_id: str,
        memory_service=None,
    ) -> ViewerSocialContext:
        if memory_service is None:
            return ViewerSocialContext(viewer_id)
        try:
            state = memory_service.get_viewer(viewer_id)
            return ViewerSocialContext.from_memory(viewer_id, state.get("viewer"))
        except Exception:
            # Memory must inform social context, never make autonomy unavailable.
            return ViewerSocialContext(viewer_id)

    def camera_awareness(self, world: WorldState | None) -> str:
        return assess_camera_framing(world)

    def recent_viewer_count(self, now: datetime | None = None) -> int:
        """Distinct real viewers seen inside the conversation window.

        PHASE 24: the Brain's audience estimate must come from the chat
        traffic the runtime actually observed. Unity's local viewer list is the
        Phase 04 simulation and stays empty in production, which left the
        Brain's ``viewer_count`` pinned at 0 on a live stream.
        """
        moment = now or self.clock()
        timeout = float(self.config.conversation_timeout_seconds)
        return sum(
            1
            for stamp in self.context.viewer_last_seen.values()
            if isinstance(stamp, datetime)
            and (moment - stamp).total_seconds() <= timeout
        )

    def get_state(self) -> dict[str, object]:
        camera_awareness = self.camera_awareness(getattr(self, "_last_world", None))
        return {
            "config": {
                "attend_now_threshold": self.config.attend_now_threshold,
                "attention_threshold": self.config.attention_threshold,
                "speech_cooldown_seconds": self.config.speech_cooldown_seconds,
                "conversation_timeout_seconds": self.config.conversation_timeout_seconds,
            },
            "camera_awareness": camera_awareness,
            "context": self.context.snapshot(),
        }

    def update_world(self, world: WorldState | None) -> None:
        self._last_world = world

    def _apply_decision(self, decision: SocialDecision, now: datetime) -> None:
        message = decision.message
        if message is None:
            return

        if decision.decision == AttentionDecision.MERGE:
            # The deferred interaction is now being consumed by the brain, so
            # do not keep it in the pending diagnostic list as well.
            self.context.release_pending(message.message_id)

        if decision.decision in {
            AttentionDecision.ATTEND_NOW,
            AttentionDecision.ATTEND_SOON,
            AttentionDecision.MERGE,
        }:
            state = self.context.conversation_state
            if message.intent == ChatIntent.GREETING:
                state = state.GREETING
            self.context.activate(message, decision.topic, now, state)
        elif decision.decision == AttentionDecision.DEFER:
            self.context.defer(
                message,
                decision.topic,
                decision.reason,
                decision.attention_score,
                now,
            )
            self.context.update_on_decision(decision, now)
            return
        self.context.update_on_decision(decision, now)
