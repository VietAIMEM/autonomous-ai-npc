"""Deterministic Phase 05 intent and target extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import ChatIntent, ChatMessage

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from .context import ConversationContext


_PUNCTUATION = re.compile(r"[^\w\s']", re.UNICODE)


@dataclass(frozen=True)
class IntentResult:
    intent: ChatIntent
    target: str | None = None
    # PHASE 25: the desired *spatial* state behind a camera/phone request, e.g.
    # FACING, IN_FRAME, CLOSER, NEAR. Structured intent only; it never becomes a
    # Unity call by itself.
    desired_state: str | None = None


_ACTION_TARGETS = {
    "table": "table",
    "bench": "bench",
    "center": "center",
    "camera": "camera",
    "room": "room",
    "floor": "floor",
    "shelf": "shelf",
    "dance": "dance",
    "dancing": "dance",
    "wave": "wave",
    "waving": "wave",
    "look": "look",
    "looking": "look",
    "sit": "sit",
    "sitting": "sit",
    "stand": "stand",
    "standing": "stand",
    "run": "run",
    "running": "run",
}

_LOCATION_TARGETS = {
    "table": "table",
    "bench": "bench",
    "center": "center",
    "floor": "floor",
    "wall": "wall",
    "phone": "phone",
    "shelf": "shelf",
}

_STREAM_START_PATTERN = re.compile(
    r"\b(?:start|begin|turn\s+on|go\s+live\s+with)\s+(?:the\s+)?(?:livestream|stream)\b",
    re.IGNORECASE,
)
_STREAM_STOP_PATTERN = re.compile(
    r"\b(?:stop|end|turn\s+off|shut\s+down|finish)\s+(?:the\s+)?(?:livestream|stream)\b",
    re.IGNORECASE,
)
_PHONE_PICKUP_PATTERN = re.compile(
    r"\b(?:pick\s+up|grab|take)\s+(?:the\s+)?phone\b",
    re.IGNORECASE,
)
_PHONE_PLACEMENT_PATTERN = re.compile(
    r"\b(?:place|put|set)\s+(?:the\s+)?phone\b|\bput\s+(?:the\s+)?phone\s+down\b",
    re.IGNORECASE,
)

_REQUEST_PATTERNS = (
    re.compile(r"\b(?:can|could|will|would)\s+you\b", re.IGNORECASE),
    re.compile(r"\b(?:please)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:go|come|move|turn|look|dance|wave|sit|stand|run)\b", re.IGNORECASE),
    re.compile(r"^\s*lacrimosa\s*,?\s*(?:go|come|move|turn|look|dance|wave|sit|stand|run)\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:table|center|camera|room|bench|phone|floor)\b", re.IGNORECASE),
)

_MOVEMENT_PATTERNS = (
    re.compile(r"\b(?:go|come|move|walk|run)\s+(?:to|toward|towards|over\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(?:turn)\s+(?:to|toward|towards|around)\b", re.IGNORECASE),
)

_LOOK_PATTERNS = (
    re.compile(r"\b(?:look|looking)\s+(?:at|toward|towards)\b", re.IGNORECASE),
    re.compile(r"\b(?:look|looks)\s+(?:here|over\s+here|behind)\b", re.IGNORECASE),
)

#: PHASE 25 spatial phrases. Checked before the generic movement/look/action
#: matchers so "stand in front of the phone" normalizes to a physical target
#: with a desired state instead of the legacy bare "stand"/"location" labels.
#: Values are plain intent vocabulary; target resolution and any physical
#: consequence stay in the Brain.
_SPATIAL_PATTERNS: tuple[tuple[re.Pattern[str], str, str, ChatIntent], ...] = (
    (
        re.compile(
            r"\b(?:stand|go|move|walk|get)\s+"
            r"(?:in\s+front\s+of|to\s+the\s+front\s+of)\s+(?:the\s+)?(?:phone|smartphone)\b"
        ),
        "phone",
        "IN_FRAME",
        ChatIntent.REQUEST_MOVEMENT,
    ),
    (
        re.compile(
            r"\b(?:stand|go|move|walk|get)\s+"
            r"(?:in\s+front\s+of|to\s+the\s+front\s+of)\s+(?:the\s+)?"
            r"(?:camera|stream\s+camera|stream)\b"
        ),
        "camera",
        "IN_FRAME",
        ChatIntent.REQUEST_MOVEMENT,
    ),
    (
        re.compile(
            r"\b(?:come|move|step|walk|get)\s+(?:a\s+bit\s+)?(?:closer|near(?:er)?)\s+"
            r"(?:to\s+)?(?:the\s+)?(?:camera|stream\s+camera)\b"
        ),
        "camera",
        "CLOSER",
        ChatIntent.REQUEST_MOVEMENT,
    ),
    # PHASE 26A: a phone-anchored closer request must keep the phone anchor the
    # Phase 25 spatial resolver knows about instead of the generic camera one.
    (
        re.compile(
            r"\b(?:come|move|step|walk|get)\s+(?:a\s+bit\s+)?(?:closer|near(?:er)?)\s+"
            r"(?:to\s+)?(?:the\s+)?(?:phone|smartphone)\b"
        ),
        "phone",
        "CLOSER",
        ChatIntent.REQUEST_MOVEMENT,
    ),
    (
        re.compile(r"\b(?:come|move|step|walk)\s+(?:a\s+bit\s+)?(?:closer|near)\b"),
        "camera",
        "CLOSER",
        ChatIntent.REQUEST_MOVEMENT,
    ),
    (
        re.compile(r"\bturn\s+(?:toward|towards|to)\s+(?:the\s+)?(?:camera|stream\s+camera)\b"),
        "camera",
        "FACING",
        ChatIntent.REQUEST_ACTION,
    ),
    # "Turn toward me" is a viewer-direction request. The stream camera is the
    # only known proxy for the viewer, so it resolves to the camera target.
    (
        re.compile(r"\bturn\s+(?:toward|towards|to)\s+me\b"),
        "camera",
        "FACING",
        ChatIntent.REQUEST_ACTION,
    ),
    (
        re.compile(r"\b(?:look|looking)\s+(?:at|toward|towards)\s+(?:the\s+)?(?:camera|stream\s+camera)\b"),
        "camera",
        "FACING",
        ChatIntent.REQUEST_ACTION,
    ),
    (
        re.compile(r"\bface\s+(?:the\s+)?(?:camera|stream\s+camera)\b"),
        "camera",
        "FACING",
        ChatIntent.REQUEST_ACTION,
    ),
    (
        re.compile(r"\bface\s+me\b"),
        "camera",
        "FACING",
        ChatIntent.REQUEST_ACTION,
    ),
)

_GREETINGS = {
    "hello", "hi", "hey", "yo", "good morning", "good afternoon", "good evening",
    "hello there", "hi there", "hey there",
}
_FAREWELL_PATTERNS = (
    re.compile(r"^(?:bye|goodbye|good\s+night|see\s+ya|see\s+you)\b.*$", re.IGNORECASE),
)
_COMMENTS = {
    "you look cute",
    "you are cute",
    "you look great",
    "you look great today",
    "you are great",
    "nice",
    "cool",
}
_SHORT_REACTIONS = {"ha", "lol", "lmao", "omg", "wow", "ok", "okay"}


def extract_intent(normalized_text: str) -> IntentResult:
    text = normalized_text.casefold()
    compact = _PUNCTUATION.sub(" ", text)
    compact = re.sub(r"\s+", " ", compact).strip()

    if not compact:
        return IntentResult(ChatIntent.UNKNOWN)

    if any(compact == greeting for greeting in _GREETINGS):
        return IntentResult(ChatIntent.GREETING)
    if any(compact.startswith(f"{greeting} lacrimosa") for greeting in _GREETINGS):
        return IntentResult(ChatIntent.GREETING)
    if any(compact.endswith(f"lacrimosa {greeting}") for greeting in _GREETINGS):
        return IntentResult(ChatIntent.GREETING)
    if any(pattern.match(compact) for pattern in _FAREWELL_PATTERNS):
        return IntentResult(ChatIntent.FAREWELL)

    if _STREAM_START_PATTERN.search(compact):
        return IntentResult(ChatIntent.REQUEST_ACTION, "stream")
    if _STREAM_STOP_PATTERN.search(compact):
        return IntentResult(ChatIntent.REQUEST_ACTION, "stream")
    if _PHONE_PICKUP_PATTERN.search(compact):
        return IntentResult(ChatIntent.REQUEST_ACTION, "phone")
    if _PHONE_PLACEMENT_PATTERN.search(compact):
        return IntentResult(ChatIntent.REQUEST_ACTION, "phone")

    spatial = _match_spatial(compact)
    if spatial is not None:
        return spatial

    movement = _match_movement(compact)
    if movement is not None:
        return IntentResult(ChatIntent.REQUEST_MOVEMENT, movement)

    look_target = _match_look(compact)
    if look_target is not None:
        return IntentResult(ChatIntent.REQUEST_ACTION, look_target)

    action = _match_action(compact)
    if action is not None:
        return IntentResult(ChatIntent.REQUEST_ACTION, action)

    if "?" in normalized_text or _starts_question(compact):
        return IntentResult(ChatIntent.QUESTION, _extract_question_topic(compact))

    if compact in _COMMENTS:
        return IntentResult(ChatIntent.COMMENT)

    if len(set(compact)) == 1 or compact.rstrip("!.,?") in _SHORT_REACTIONS:
        return IntentResult(ChatIntent.COMMENT, "noise")

    return IntentResult(ChatIntent.UNKNOWN)


def refine_intent(
    message: ChatMessage,
    base: IntentResult,
    context: "ConversationContext | None" = None,
) -> IntentResult:
    """PHASE 26A continuity refinement over the existing classifier.

    The deterministic extractor stays exactly as Phase 05 built it. This pass
    only answers the continuity questions:

    * is this turn a follow-up to the previous one?
    * does it continue the active topic?
    * is it a bare acknowledgement or an explicit end of conversation?
    * is it a spatial continuation ("Closer.") that must reuse the Phase 25
      target/desired state instead of falling through to UNKNOWN?

    Without an active conversation the base result is returned untouched, so a
    cold ``Closer.`` never invents context that does not exist.
    """
    if context is None:
        return base

    compact = _compact_text(message.normalized_text)
    if not compact:
        return base

    if context.has_active_conversation() and context.pending_action_is_movement:
        if compact in _SPATIAL_CONTINUATIONS or _SPATIAL_CONTINUATION_PATTERN.match(
            compact
        ):
            if base.intent in {ChatIntent.UNKNOWN, ChatIntent.COMMENT}:
                # The continuation asks for its own desired state; only the
                # anchor (camera unless the active request named the phone) is
                # inherited from the Phase 25 turn before it.
                return IntentResult(
                    ChatIntent.REQUEST_MOVEMENT,
                    context.pending_action_target or "camera",
                    "CLOSER",
                )

    if base.intent in {ChatIntent.UNKNOWN, ChatIntent.COMMENT} and (
        compact in _CONVERSATION_END_PHRASES
    ):
        return IntentResult(ChatIntent.CONVERSATION_END)

    # PHASE 26A: a bare acknowledgement is continuity, not a reaction: it only
    # means "I am still here, keep going" while a conversation is active. The
    # phrase table is deliberately the whole scope -- nothing wider is inferred
    # and a cold "okay" keeps its Phase 05 COMMENT reading.
    if (
        base.intent in {ChatIntent.UNKNOWN, ChatIntent.COMMENT}
        and context.has_active_conversation()
        and compact in _ACKNOWLEDGEMENTS
    ):
        return IntentResult(ChatIntent.ACKNOWLEDGEMENT)

    topic = context.current_topic
    if base.intent == ChatIntent.QUESTION and context.has_active_conversation():
        words = compact.split()
        # A message that is still queued behind this one already leaves its
        # question in the context, so continuity works even when two viewer
        # messages arrive inside the same decision interval.
        has_prior = (
            context.previous_intent is not None or context.pending_question is not None
        )
        if (
            has_prior
            and len(words) <= _MAX_FOLLOW_UP_WORDS
            and _is_follow_up_probe(compact)
        ):
            return IntentResult(ChatIntent.FOLLOW_UP, topic)
        if (
            has_prior
            and base.target is None
            and topic
            and topic not in {"greeting", "farewell", "general_conversation"}
            and len(words) <= _MAX_TOPIC_CONTINUATION_WORDS
        ):
            return IntentResult(ChatIntent.TOPIC_CONTINUATION, topic)

    return base


#: Bare spatial continuations that only mean something inside an active
#: camera/movement turn ("Stand in front of the camera." -> "Closer.").
_SPATIAL_CONTINUATIONS = {
    "closer",
    "closer please",
    "a bit closer",
    "a little closer",
    "slightly closer",
    "nearer",
    "more",
    "a bit more",
    "a little more",
    "a little bit more",
    "bit more",
    "closer to the camera",
    "closer to the stream camera",
    "nearer to the camera",
    "move it a bit closer",
    "go a little closer",
}
_SPATIAL_CONTINUATION_PATTERN = re.compile(
    r"^(?:a\s+)?(?:bit|little|touch|slightly)\s+more$|^come\s+a\s+(?:bit|little)\s+more$"
)

#: Explicit "we are done" phrasing. Kept apart from farewell words so
#: "bye Lacrimosa" keeps its pinned FAREWELL intent.
_CONVERSATION_END_PHRASES = {
    "that's all",
    "that's it",
    "that is all",
    "that is it",
    "that's all for now",
    "we're done",
    "we are done",
    "i'm done",
    "im done",
    "nothing else",
    "no more questions",
    "no more for now",
    "end of conversation",
    "let's stop here",
    "we can stop now",
}

#: Short acknowledgements. The LLM still writes the words; this only tells the
#: response policy that the turn needs a very short reply.
_ACKNOWLEDGEMENTS = {
    "ok",
    "okay",
    "k",
    "sure",
    "sure thing",
    "got it",
    "gotcha",
    "alright",
    "all right",
    "yeah",
    "yep",
    "yup",
    "yes",
    "right",
    "cool",
    "nice",
    "thanks",
    "thank you",
    "thanks lacrimosa",
    "thank you lacrimosa",
}

#: Probes that only mean something as a continuation of the previous turn.
_FOLLOW_UP_PREFIXES = (
    "why",
    "how",
    "what kind",
    "what sort",
    "which",
    "what else",
    "what about",
    "what do you mean",
    "what does that mean",
    "when",
    "where",
    "and then",
    "and you",
    "and what",
    "go on",
    "continue",
    "tell me more",
    "really",
    "more",
)
#: Fragments that are a probe by themselves, with no prefix to strip.
_FOLLOW_UP_FRAGMENTS = {"why not", "how come", "how so", "and", "then", "go on"}

_MAX_FOLLOW_UP_WORDS = 6
_MAX_TOPIC_CONTINUATION_WORDS = 8
#: A probe may carry a couple of extra words ("what kind of dancing?") before
#: it stops being a fragment and becomes a question of its own.
_MAX_PROBE_REMAINDER_WORDS = 2

#: Auxiliaries that turn the remainder of a probe into a full question with its
#: own subject ("where are you going?" is a question, not a bare "where?").
_FULL_QUESTION_START = re.compile(
    r"^(?:are|is|am|do|does|did|can|could|will|would|was|were|have|has|had)\b"
)


def _is_follow_up_probe(compact: str) -> bool:
    if compact in _FOLLOW_UP_FRAGMENTS:
        return True
    for prefix in _FOLLOW_UP_PREFIXES:
        if not compact.startswith(prefix):
            continue
        remainder = compact[len(prefix):].strip()
        if not remainder:
            return True
        if len(remainder.split()) > _MAX_PROBE_REMAINDER_WORDS:
            return False
        return not _FULL_QUESTION_START.match(remainder)
    return False


def _compact_text(normalized_text: str) -> str:
    # Typographic apostrophes are folded to the ASCII form the phrase tables
    # below use; the punctuation filter intentionally keeps apostrophes.
    text = normalized_text.casefold().replace("\u2019", "'")
    text = _PUNCTUATION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _starts_question(text: str) -> bool:
    return bool(
        re.match(
            r"^(?:what|where|when|who|why|how|which|is|are|do|does|did|can|could)\b",
            text,
            re.IGNORECASE,
        )
    )


def _match_spatial(text: str) -> IntentResult | None:
    for pattern, target, desired_state, intent in _SPATIAL_PATTERNS:
        if pattern.search(text):
            return IntentResult(intent, target, desired_state)
    return None


def _match_movement(text: str) -> str | None:
    for pattern in _MOVEMENT_PATTERNS:
        if pattern.search(text):
            target = _match_any_target(text, _LOCATION_TARGETS)
            return target or "location"
    return None


def _match_look(text: str) -> str | None:
    for pattern in _LOOK_PATTERNS:
        match = pattern.search(text)
        if match:
            if re.search(r"\bcamera\b", text, re.IGNORECASE):
                return "camera"
            target = _match_any_target(text, _LOCATION_TARGETS)
            return target or "target"
    return None


def _match_action(text: str) -> str | None:
    if not any(pattern.search(text) for pattern in _REQUEST_PATTERNS):
        return None
    return _match_any_target(text, _ACTION_TARGETS)


def _extract_question_topic(text: str) -> str | None:
    if re.search(r"\bwhat\s+are\s+you\s+doing\b", text, re.IGNORECASE):
        return "current_activity"
    if re.search(r"\bwhere\s+are\s+you\s+going\b", text, re.IGNORECASE):
        return "destination"
    if re.search(r"\bwhat\s+is\s+(?:your|the)\s+name\b", text, re.IGNORECASE):
        return "name"
    return None


def _match_any_target(text: str, targets: dict[str, str]) -> str | None:
    matches = [(text.find(target), canonical) for target, canonical in targets.items() if target in text]
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]
