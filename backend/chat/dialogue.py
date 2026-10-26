"""Prompt construction, provider response parsing and safe fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..llm.base import LLMRequest
from .context import ConversationContext
from .models import (
    ChatIntent,
    ChatMessage,
    DialogueResponse,
    WorldContext,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .policy import ResponsePolicy


_MAX_PROMPT_CHARS = 4000
_MAX_SPEECH_CHARS = 1000
_MAX_ACTION_CHARS = 80
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class CharacterContext:
    name: str = "Lacrimosa"
    identity: str = "autonomous virtual character"
    personality: tuple[str, ...] = ("calm", "gentle", "curious", "playful")
    # PHASE 19 additions are appended last with defaults so every existing
    # construction keeps its meaning. Both are advisory prompt text only.
    speech_style: Optional[str] = None
    viewer_context: tuple[str, ...] = ()

    def to_prompt_lines(self) -> list[str]:
        lines = [
            f"Name: {self.name}",
            f"Identity: {self.identity}",
            f"Personality: {', '.join(self.personality)}",
        ]
        if self.speech_style:
            lines.append(f"Speech style: {self.speech_style}")
        if self.viewer_context:
            lines.append("Viewer context (advisory only): " + " | ".join(self.viewer_context))
        return lines


def build_prompt(
    message: ChatMessage,
    conversation: ConversationContext,
    world: WorldContext,
    character: CharacterContext,
    *,
    policy: Optional["ResponsePolicy"] = None,
) -> str:
    lines = [
        "You are generating a short livestream reply for a virtual character.",
        "The reply must not execute actions. It only describes intent for a separate system.",
        "",
        "Character:",
        *character.to_prompt_lines(),
        "",
        "World:",
        *world.to_prompt_lines(),
        "",
        "Recent conversation:",
    ]
    for turn in conversation.snapshot():
        lines.append(f"{turn.speaker}: {turn.text}")
    lines.extend(
        [
            "",
            "Selected chat:",
            f"{message.viewer_id}: {message.raw_text}",
            f"Detected intent: {message.intent.value}",
            f"Detected target: {message.target or 'none'}",
            f"Priority: {message.priority:.2f}",
        ]
    )
    if policy is not None:
        # PHASE 26A: the structured constraint sits before the JSON contract so
        # the tail truncation below can never drop it.
        lines.extend(["", "Response policy:", f"  {policy.constraint}"])
    lines.extend(
        [
            "",
            "Return strict JSON with exactly these keys:",
            '{"speech": string, "intent": string, "requested_action": string|null, "priority": number}',
            "Intent must be one of: "
            + ", ".join(intent.value for intent in ChatIntent)
            + ".",
        ]
    )
    prompt = "\n".join(lines)
    return prompt[-_MAX_PROMPT_CHARS:]


def make_request(prompt: str, message: ChatMessage, viewer_count: int) -> LLMRequest:
    return LLMRequest(
        prompt=prompt,
        context={
            "character": "Lacrimosa",
            "message_id": message.message_id,
            "viewer_id": message.viewer_id,
            "intent": message.intent.value,
            "target": message.target,
            "priority": message.priority,
            "viewer_count": viewer_count,
        },
    )


def parse_response(
    raw_response: str,
    message: ChatMessage,
) -> tuple[DialogueResponse, Optional[str]]:
    """Parse provider output; never allow malformed fields into future systems."""
    try:
        payload = _parse_json_payload(raw_response)
        speech = _clean_text(payload.get("speech"), _MAX_SPEECH_CHARS)
        intent_value = str(payload.get("intent", "")).upper()
        intent = ChatIntent(intent_value)
        requested_action = _clean_optional_text(
            payload.get("requested_action"), _MAX_ACTION_CHARS
        )
        if not speech:
            raise ValueError("missing speech")
        if intent in (ChatIntent.REQUEST_ACTION, ChatIntent.REQUEST_MOVEMENT) and requested_action is None:
            raise ValueError("missing requested_action")
        response = DialogueResponse(
            speech=speech,
            intent=intent,
            requested_action=requested_action,
            priority=message.priority,
            message_id=message.message_id,
            viewer_id=message.viewer_id,
        )
        return response, None
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _fallback_response(message), "invalid_llm_response"


def _parse_json_payload(raw_response: str) -> dict[str, Any]:
    payload = json.loads(raw_response)
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not an object")
    return payload


def _clean_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = _CONTROL_CHARS.sub(" ", value).strip()
    return cleaned[:limit]


def _clean_optional_text(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    cleaned = _clean_text(value, limit)
    return cleaned or None


def _fallback_response(message: ChatMessage) -> DialogueResponse:
    return DialogueResponse(
        speech="I'm here, but I need a moment to understand that.",
        intent=ChatIntent.UNKNOWN,
        requested_action=None,
        priority=message.priority,
        message_id=message.message_id,
        viewer_id=message.viewer_id,
    )
