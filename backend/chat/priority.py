"""Deterministic normalized priority scoring."""

from __future__ import annotations

from .models import ChatIntent, ChatMessage


def score_priority(message: ChatMessage) -> float:
    score = 0.18

    if message.intent == ChatIntent.REQUEST_ACTION:
        score = 0.74
    elif message.intent == ChatIntent.REQUEST_MOVEMENT:
        score = 0.70
    # PHASE 26A continuity intents sit inside the existing bands: a follow-up
    # is a real question about the active topic, a topic continuation is the
    # same class one notch lower, an acknowledgement is light social traffic,
    # and an explicit end of conversation ranks with a farewell.
    elif message.intent == ChatIntent.FOLLOW_UP:
        score = 0.50
    elif message.intent == ChatIntent.TOPIC_CONTINUATION:
        score = 0.46
    elif message.intent == ChatIntent.QUESTION:
        score = 0.46
    elif message.intent == ChatIntent.GREETING:
        score = 0.34
    elif message.intent == ChatIntent.FAREWELL:
        score = 0.30
    elif message.intent == ChatIntent.CONVERSATION_END:
        score = 0.30
    elif message.intent == ChatIntent.ACKNOWLEDGEMENT:
        score = 0.24
    elif message.intent == ChatIntent.COMMENT:
        score = 0.22
    else:
        score = 0.20

    if _addresses_character(message.normalized_text):
        score += 0.14
    if message.target:
        score += 0.08
    if len(message.normalized_text.split()) >= 3:
        score += 0.03

    repeat_penalty = min(0.30, 0.08 * (message.repeat_count - 1))
    score -= repeat_penalty
    return max(0.0, min(1.0, score))


def _addresses_character(text: str) -> bool:
    lowered = text.casefold()
    return "lacrimosa" in lowered or lowered.startswith(("hey", "hi ", "hello "))
