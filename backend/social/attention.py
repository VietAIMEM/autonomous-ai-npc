"""Deterministic social attention on top of Phase 05 priority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..brain.models import WorldState
from ..chat.models import ChatMessage
from .context import SocialContext
from .conversation import ConversationManager
from .models import (
    AttentionDecision,
    SocialConfig,
    SocialSpeakingPolicy,
    ViewerSocialContext,
)


BUSY_ACTIONS = {
    "dance",
    "dancing",
    "wave",
    "waving",
    "run",
    "running",
    "move",
    "moving",
    "walk",
    "walking",
    "turn",
    "turning",
    "pick_up_phone",
    "place_phone",
}
DEFAULT_CAMERA_FRAMING_DISTANCE = 5.5


@dataclass(frozen=True)
class AttentionAssessment:
    decision: AttentionDecision
    score: float
    priority: float
    topic: Optional[str]
    reason: str
    speaking_policy: SocialSpeakingPolicy


def assess_camera_framing(world: WorldState | None) -> str:
    """Use known transforms only; this is deliberately not computer vision."""
    if world is None:
        return "UNKNOWN"
    if not world.stream_active:
        return "STREAM_INACTIVE"
    if not world.stream_camera_available or not world.stream_camera_active:
        return "CAMERA_INACTIVE"
    if world.phone_state != "Placed":
        return "CAMERA_MOVES_WITH_PHONE"

    camera_target = world.target("CAMERA")
    camera_position = camera_target.position if camera_target else world.phone_position
    character = world.character_position
    distance = math.sqrt(
        (character[0] - camera_position[0]) ** 2
        + (character[2] - camera_position[2]) ** 2
    )
    if distance > DEFAULT_CAMERA_FRAMING_DISTANCE:
        return "OUTSIDE_FRAME"
    return "INSIDE_FRAME"


class SocialAttention:
    def __init__(
        self,
        config: SocialConfig | None = None,
        conversation: ConversationManager | None = None,
    ) -> None:
        self.config = config or SocialConfig()
        self.conversation = conversation or ConversationManager(self.config)

    def evaluate(
        self,
        message: ChatMessage,
        context: SocialContext,
        world: WorldState | None = None,
        viewer_context: ViewerSocialContext | None = None,
        now: datetime | None = None,
    ) -> AttentionAssessment:
        topic = self.conversation.infer_topic(message, context.active_topic)
        score = self.score(message, context, viewer_context, topic)
        busy = self._is_busy(world)
        can_speak = self.conversation.can_speak(now or message.timestamp, context.last_speech_at)

        if context.is_recent_duplicate(message) or message.repeat_count > 1:
            # The original queued item is still processed once; duplicate evidence
            # is retained without producing another identical response.
            if score >= self.config.attention_threshold:
                return AttentionAssessment(
                    AttentionDecision.MERGE,
                    score,
                    message.priority,
                    topic,
                    "repetition_evidence",
                    SocialSpeakingPolicy.CONTINUE_CONVERSATION,
                )
            return AttentionAssessment(
                AttentionDecision.IGNORE,
                score,
                message.priority,
                topic,
                "low_relevance_repetition",
                SocialSpeakingPolicy.IGNORE,
            )

        if score < self.config.attention_threshold:
            return AttentionAssessment(
                AttentionDecision.IGNORE,
                score,
                message.priority,
                topic,
                "below_social_attention_threshold",
                SocialSpeakingPolicy.IGNORE,
            )

        if busy and score >= self.config.busy_defer_threshold:
            return AttentionAssessment(
                AttentionDecision.DEFER,
                score,
                message.priority,
                topic,
                "character_busy",
                SocialSpeakingPolicy.DEFER,
            )

        if not can_speak and score < self.config.attend_now_threshold + 0.18:
            if score >= self.config.attention_threshold:
                return AttentionAssessment(
                    AttentionDecision.DEFER,
                    score,
                    message.priority,
                    topic,
                    "speech_cooldown",
                    SocialSpeakingPolicy.DEFER,
                )
            return AttentionAssessment(
                AttentionDecision.IGNORE,
                score,
                message.priority,
                topic,
                "speech_cooldown_low_relevance",
                SocialSpeakingPolicy.IGNORE,
            )

        if score >= self.config.attend_now_threshold:
            policy = (
                SocialSpeakingPolicy.RESPOND
                if not self.conversation.is_topic_change(topic, context.active_topic)
                else SocialSpeakingPolicy.CHANGE_TOPIC
            )
            return AttentionAssessment(
                AttentionDecision.ATTEND_NOW,
                score,
                message.priority,
                topic,
                "high_social_attention",
                policy,
            )

        return AttentionAssessment(
            AttentionDecision.ATTEND_SOON,
            score,
            message.priority,
            topic,
            "moderate_social_attention",
            SocialSpeakingPolicy.RESPOND,
        )

    def score(
        self,
        message: ChatMessage,
        context: SocialContext,
        viewer_context: ViewerSocialContext | None,
        topic: str | None,
    ) -> float:
        score = 0.25 + 0.65 * max(0.0, min(1.0, message.priority))

        if message.viewer_id == context.active_viewer_id:
            score += 0.08
        # PHASE 26A: continuity, not literal equality. A follow-up that moves
        # out of a meta frame (a greeting) into a real subject is the same
        # conversation, so it keeps the active-conversation bonus instead of
        # being deferred like a cold topic switch.
        if context.active_topic and self.conversation.is_topic_continuation(
            topic, context.active_topic
        ):
            score += 0.05
        if message.viewer_id in self._recent_viewer_ids(context):
            score += 0.04
        if viewer_context is not None:
            score += 0.07 * max(0.0, min(1.0, viewer_context.relationship_score))
            if viewer_context.preferences:
                overlap = set(viewer_context.preferences) & {message.target or "", topic or ""}
                if overlap:
                    score += 0.04

        if message.intent.value.startswith("REQUEST"):
            score += 0.04
        return max(0.0, min(1.0, score))

    @staticmethod
    def _recent_viewer_ids(context: SocialContext) -> set[str]:
        return {message.viewer_id for message in context.recent_messages[-4:]}

    @staticmethod
    def _is_busy(world: WorldState | None) -> bool:
        if world is None:
            return False
        normalized = world.current_action.strip().casefold().replace("-", "_").replace(" ", "_")
        return normalized in BUSY_ACTIONS
