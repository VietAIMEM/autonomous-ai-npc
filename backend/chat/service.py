"""High-level chat processing service."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from itertools import count
from typing import Callable, Mapping, Optional

from ..llm.base import LLMProvider, LLMProviderError
from ..runtime import instrument
from .context import ConversationContext
from .dialogue import (
    CharacterContext,
    build_prompt,
    make_request,
    parse_response,
)
from .filter import ChatFilter, FilterConfig, FilterDecision
from .intent import extract_intent, refine_intent
from .models import (
    ChatMessage,
    ChatStatus,
    DialogueResponse,
    WorldContext,
    utc_now,
)
from .normalization import normalize_text
from .policy import ResponsePolicy, ResponsePolicyEngine
from .priority import score_priority
from .queue import AttentionQueue


@dataclass(frozen=True)
class ChatServiceConfig:
    queue_capacity: int = 100
    #: Stale entries are dropped instead of being replayed minutes late. The
    #: Twitch channel can burst far faster than the Brain's decision interval.
    max_queue_age_seconds: float = 120.0
    max_text_length: int = 500
    duplicate_window_seconds: int = 30
    duplicate_similarity: float = 0.94
    rate_window_seconds: int = 10
    max_messages_per_rate_window: int = 5
    max_context_turns: int = 8
    #: PHASE 26A: a quiet conversation expires back to a coherent listening
    #: state instead of holding a stale topic forever. Mirrors the Phase 08
    #: social conversation timeout.
    conversation_timeout_seconds: float = 180.0

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ChatServiceConfig":
        """Overlay operator overrides for the bounded attention queue.

        The queue stays bounded; these knobs only size the attention window to
        the Brain's real decision interval so a genuine viewer request is not
        expired before the next decision can read it.
        """
        env = os.environ if environ is None else environ
        defaults = cls()
        return cls(
            queue_capacity=_as_positive_int(
                env.get("CHAT_QUEUE_CAPACITY"), defaults.queue_capacity
            ),
            max_queue_age_seconds=_as_float(
                env.get("CHAT_MAX_QUEUE_AGE_SECONDS"), defaults.max_queue_age_seconds
            ),
        )


@dataclass(frozen=True)
class SubmissionResult:
    message: ChatMessage
    accepted: bool
    merged_message: Optional[ChatMessage] = None


@dataclass(frozen=True)
class ProcessedResult:
    message: ChatMessage
    response: DialogueResponse
    fallback_reason: Optional[str] = None


class ChatService:
    def __init__(
        self,
        provider: LLMProvider,
        config: ChatServiceConfig | None = None,
        clock: Callable[[], datetime] = utc_now,
        world_context: WorldContext | None = None,
        character_context: CharacterContext | None = None,
        topic_resolver: Callable[[ChatMessage, Optional[str]], Optional[str]] | None = None,
    ) -> None:
        config = config or ChatServiceConfig()
        self.config = config
        self.clock = clock
        self.queue = AttentionQueue(
            config.queue_capacity,
            max_age_seconds=config.max_queue_age_seconds,
            clock=self.clock,
            # PA29-004: the queue's own terminal dispositions are published in
            # the established telemetry registry. Each dropped message is
            # counted once, at the moment the queue drops it, so the counters
            # can never double-count an eviction or an expiration.
            on_evict=lambda count: instrument.count("chat.queue_evictions", count),
            on_expire=lambda count: instrument.count("chat.queue_expired", count),
        )
        self.filter = ChatFilter(
            FilterConfig(
                max_text_length=config.max_text_length,
                duplicate_window_seconds=config.duplicate_window_seconds,
                duplicate_similarity=config.duplicate_similarity,
                rate_window_seconds=config.rate_window_seconds,
                max_messages_per_rate_window=config.max_messages_per_rate_window,
            )
        )
        self.conversation = ConversationContext(
            config.max_context_turns,
            timeout_seconds=config.conversation_timeout_seconds,
        )
        self.policy = ResponsePolicyEngine()
        #: PHASE 26A: the social layer owns the wider topic vocabulary. When a
        #: Brain wires that resolver in, both layers agree on the active topic;
        #: standalone chat keeps the small local fallback.
        self._topic_resolver = topic_resolver
        self.provider = provider
        self.world_context = world_context or WorldContext()
        self.character_context = character_context or CharacterContext()
        self._id_sequence = count(1)
        self._last_processed: Optional[ProcessedResult] = None
        self._last_policy: Optional[ResponsePolicy] = None
        #: PHASE 26A: at most one bounded observer of accepted messages. The
        #: Brain uses it to evaluate an arriving important message against
        #: work that is already in flight; intake must never block on it.
        self._message_listener: Optional[Callable[[ChatMessage], None]] = None

    def set_topic_resolver(
        self,
        resolver: Callable[[ChatMessage, Optional[str]], Optional[str]] | None,
    ) -> None:
        self._topic_resolver = resolver

    def set_message_listener(
        self,
        listener: Optional[Callable[[ChatMessage], None]],
    ) -> None:
        """PHASE 26A: observe accepted messages without owning the queue.

        Called after a message is queued, so the observer sees exactly what
        the Brain will attend to. A listener fault is counted and dropped:
        intake stays the authoritative path.
        """
        self._message_listener = listener

    def _notify_message_listener(self, message: ChatMessage) -> None:
        listener = self._message_listener
        if listener is None:
            return
        try:
            listener(message)
        except Exception:
            instrument.count("chat.listener_error")

    def submit(
        self,
        viewer_id: str,
        raw_text: str,
        message_id: str | None = None,
        timestamp: datetime | None = None,
        *,
        platform: str | None = None,
        viewer_name: str | None = None,
        channel: str | None = None,
    ) -> SubmissionResult:
        if not viewer_id or not viewer_id.strip():
            raise ValueError("viewer_id is required")
        if not isinstance(raw_text, str):
            raise ValueError("raw_text must be a string")

        instrument.count("chat.submitted")
        now = timestamp or self.clock()
        normalized = normalize_text(raw_text)
        message = ChatMessage(
            message_id=message_id or f"msg_{next(self._id_sequence):06d}",
            viewer_id=viewer_id.strip(),
            raw_text=raw_text,
            normalized_text=normalized,
            timestamp=now,
            status=ChatStatus.RECEIVED,
            platform=platform,
            viewer_name=viewer_name,
            channel=channel,
        )

        decision = self.filter.evaluate(message)
        if not decision.accepted:
            message.status = ChatStatus.FILTERED
            message.filtered_reason = decision.reason
            self.filter.record(message, accepted=False)
            instrument.count("chat.filtered")
            return SubmissionResult(message, False)

        duplicate = self.filter.find_duplicate(message)
        if duplicate is not None:
            message.status = ChatStatus.FILTERED
            message.filtered_reason = "duplicate"
            instrument.count("chat.filtered")
            merged = self.queue.find(duplicate.message_id)
            if merged is not None:
                merged.repeat_count += 1
                merged.priority = score_priority(merged)
                message.repeat_count = merged.repeat_count
                self.filter.record(message, accepted=False)
                return SubmissionResult(message, False, merged)
            self.filter.record(message, accepted=False)
            return SubmissionResult(message, False)

        # PHASE 26A: the deterministic extractor stays authoritative; this one
        # extra pass only resolves follow-up/topic/acknowledgement continuity
        # against the bounded context before the message is scored.
        intent_result = refine_intent(
            message,
            extract_intent(message.normalized_text),
            self.conversation,
        )
        message.intent = intent_result.intent
        message.target = intent_result.target
        message.desired_state = intent_result.desired_state
        message.priority = score_priority(message)
        message.status = ChatStatus.QUEUED

        if not self.queue.enqueue(message):
            message.status = ChatStatus.REJECTED
            message.filtered_reason = "queue_full"
            self.filter.record(message, accepted=False)
            instrument.count("chat.rejected")
            instrument.gauge("queue.chat.depth", float(len(self.queue)))
            return SubmissionResult(message, False)

        # PA29-006: record when the local queue took the message, and how old
        # the upstream event already was at that moment. A long "message age"
        # measured at attention time is then attributable: upstream staleness
        # shows up in ``chat.upstream_age_ms`` at intake, local backlog shows
        # up in ``chat.queue_wait_ms`` at attention. The viewer's original
        # timestamp is never rewritten.
        message.enqueued_at = self.clock()
        if message.timestamp is not None:
            upstream_age = _elapsed_ms(message.enqueued_at, message.timestamp)
            if upstream_age is not None:
                instrument.observe_ms("chat.upstream_age_ms", upstream_age)

        # Continuity is only updated once the turn is genuinely attending:
        # a full queue must not move the active viewer or the active topic.
        self.conversation.observe_message(message, topic=self._resolve_topic(message))
        self.filter.record(message, accepted=True)
        self._notify_message_listener(message)
        instrument.count("chat.accepted")
        instrument.gauge("queue.chat.depth", float(len(self.queue)))
        return SubmissionResult(message, True)

    def process_next(
        self,
        *,
        personalization: "Mapping[str, object] | None" = None,
    ) -> Optional[ProcessedResult]:
        message = self.queue.dequeue()
        if message is None:
            return None

        started = time.monotonic()
        # Phase 24: how late are we answering? A saturated queue once made the
        # Brain reply to the oldest surviving entry every time, so freshness is
        # measured at the single point where a message actually gets attention.
        # PA29-006 splits that single number into its two honest parts:
        # end-to-end age since the viewer sent it, and the local queue wait
        # since intake. Neither is derived from the other.
        now = self.clock()
        if message.timestamp is not None:
            age_ms = _elapsed_ms(now, message.timestamp)
            if age_ms is not None:
                instrument.observe_ms("chat.message_age_ms", age_ms)
        if message.enqueued_at is not None:
            wait_ms = _elapsed_ms(now, message.enqueued_at)
            if wait_ms is not None:
                instrument.observe_ms("chat.queue_wait_ms", wait_ms)
        character = self._character_context(personalization)
        self.conversation.begin_response(message)
        policy = self.policy.select(
            message,
            self.conversation,
            character,
            personalization,
        )
        self._last_policy = policy
        prompt = build_prompt(
            message,
            self.conversation,
            self.world_context,
            character,
            policy=policy,
        )
        request = make_request(prompt, message, self.get_state()["queued_count"])

        try:
            raw_response = self.provider.generate(request)
        except LLMProviderError as exc:
            fallback, _ = parse_response("", message)
            response = replace(fallback, speech=_provider_failure_speech())
            reason = type(exc).__name__
            instrument.count("chat.provider_errors")
            instrument.record_error("chat", exc)
        else:
            response, reason = parse_response(raw_response, message)

        message.status = ChatStatus.PROCESSED
        self.conversation.observe_response(
            message,
            response,
            topic=self._resolve_topic(message),
            fallback_reason=reason,
        )
        result = ProcessedResult(message, response, reason)
        self._last_processed = result
        instrument.count("chat.processed")
        instrument.observe_ms("chat.process_ms", (time.monotonic() - started) * 1000.0)
        instrument.gauge("queue.chat.depth", float(len(self.queue)))
        return result

    def _character_context(
        self,
        personalization: "Mapping[str, object] | None",
    ) -> CharacterContext:
        """Fold bounded personalization into the existing character context.

        Only derived, already-bounded fields are used; the raw viewer history
        never reaches the prompt. Invalid or missing data leaves the chat
        prompt exactly as Phase 12 built it.
        """
        if not isinstance(personalization, Mapping):
            return self.character_context

        updates: dict[str, object] = {}
        speech_style = personalization.get("speech_style")
        if isinstance(speech_style, str) and speech_style.strip():
            updates["speech_style"] = speech_style.strip()[:160]
        # PHASE 26B: the bounded behavioral/expression hint rides the same
        # prompt field Phase 19 established for tone. It is derived text only
        # (tone, reaction level, one short reason); the base character context
        # object is never mutated.
        expression = personalization.get("expression")
        if isinstance(expression, Mapping):
            hint = expression.get("hint")
            if isinstance(hint, str) and hint.strip():
                updates["speech_style"] = hint.strip()[:160]

        lines: list[str] = []
        if personalization.get("returning"):
            level = _safe_label(personalization.get("relationship_level"), "NEW")
            count = _safe_int(personalization.get("interaction_count"))
            lines.append(f"returning viewer, relationship {level}, {count} prior interactions")
        topics = _safe_labels(personalization.get("preferred_topics"), 3)
        if topics:
            lines.append("topics with repeat interest: " + ", ".join(topics))
        skills = _safe_labels(personalization.get("preferred_skills"), 3)
        if skills:
            lines.append("skills with repeat interest: " + ", ".join(skills))
        continuity = personalization.get("topic_continuity")
        if isinstance(continuity, str) and continuity.strip():
            lines.append("relevant recent topic: " + continuity.strip()[:60])
        greeting = personalization.get("greeting")
        if isinstance(greeting, str) and greeting.strip():
            lines.append("suggested returning-viewer greeting: " + greeting.strip()[:120])
        if lines:
            updates["viewer_context"] = tuple(lines)

        if not updates:
            return self.character_context
        return replace(self.character_context, **updates)

    def get_character_context(self) -> CharacterContext:
        """Diagnostics view of the base (unpersonalized) prompt context."""
        return self.character_context

    def discard_next(self, reason: str = "social_attention_ignore") -> Optional[ChatMessage]:
        """Remove a low-value queued message without spending an LLM request."""
        message = self.queue.dequeue()
        if message is None:
            return None
        # PA29-006: a discard is still an attention decision, so the local
        # queue wait is measured here with the same definition as processing.
        if message.enqueued_at is not None:
            wait_ms = _elapsed_ms(self.clock(), message.enqueued_at)
            if wait_ms is not None:
                instrument.observe_ms("chat.queue_wait_ms", wait_ms)
        message.status = ChatStatus.FILTERED
        message.filtered_reason = reason
        self.conversation.mark_discarded(reason)
        self._last_processed = None
        instrument.count("chat.discarded")
        instrument.gauge("queue.chat.depth", float(len(self.queue)))
        return message

    def set_last_response(self, speech: str) -> None:
        """Publish one authoritative utterance for the current processed turn.

        The Brain calls this only after plan normalization has chosen the exact
        SPEAK text. It updates the already-visible response; it never creates a
        second turn or bypasses normal response validation.
        """
        if self._last_processed is None or not isinstance(speech, str):
            return
        normalized = speech.strip()
        if not normalized:
            return
        response = replace(self._last_processed.response, speech=normalized)
        self._last_processed = replace(self._last_processed, response=response)
        self.conversation.previous_response = response.speech

    def get_queue(self) -> list[ChatMessage]:
        return self.queue.snapshot()

    def get_conversation_state(self) -> dict[str, object]:
        """PHASE 26A bounded continuity state (also used by diagnostics)."""
        return self.conversation.to_state(self.clock())

    def get_state(self) -> dict[str, object]:
        """Queue accounting as ``/chat/state`` reports it.

        ``queued_count`` is work still waiting for attention. ``queue_evictions``
        and ``queue_expired`` are terminal dispositions of messages that were
        already *accepted* into the queue (dropped to admit a better message,
        or dropped as stale). Accepted is therefore the sum of the terminal
        dispositions plus whatever is still queued::

            chat.accepted = chat.processed + chat.discarded
                            + queue_evictions + queue_expired + queued_count

        ``chat.submitted`` counts everything offered, so ``submitted -
        accepted`` is the filtered/rejected side and never overlaps the
        processing counters.
        """
        self.queue.expire()
        last_response = self._last_processed.response.to_dict() if self._last_processed else None
        return {
            "queued_count": len(self.queue),
            "queue_capacity": self.queue.capacity,
            "queue_evictions": self.queue.evicted_count,
            "queue_expired": self.queue.expired_count,
            "conversation_turns": len(self.conversation.snapshot()),
            "last_response": last_response,
            "conversation": self.get_conversation_state(),
            "last_response_policy": (
                self._last_policy.to_dict() if self._last_policy else None
            ),
        }

    def _resolve_topic(self, message: ChatMessage) -> Optional[str]:
        if self._topic_resolver is None:
            return None
        try:
            topic = self._topic_resolver(message, self.conversation.current_topic)
        except Exception:
            return None
        return topic if isinstance(topic, str) and topic.strip() else None


def _provider_failure_speech() -> str:
    return "I'm here, but I need a moment before I can answer."


def _elapsed_ms(later: datetime, earlier: datetime) -> Optional[float]:
    """Milliseconds between two stamps, UTC-normalizing naive values.

    Returns ``None`` when the pair cannot be compared, so a malformed stamp
    can never break intake or a tick; it only loses one latency sample.
    """
    if later.tzinfo is None:
        later = later.replace(tzinfo=timezone.utc)
    if earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=timezone.utc)
    try:
        return max(0.0, (later - earlier).total_seconds() * 1000.0)
    except (TypeError, ValueError):
        return None


def _safe_label(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().upper()[:24]
    return fallback


def _safe_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _safe_labels(value: object, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    cleaned = [
        item.strip()[:40]
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return list(dict.fromkeys(cleaned))[:limit]


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
