"""Deterministic mock provider for tests and local API use."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .base import LLMProvider, LLMProviderError, LLMRequest


@dataclass
class MockLLMProvider(LLMProvider):
    fail_with: str | None = None

    def generate(self, request: LLMRequest) -> str:
        if self.fail_with == "timeout":
            raise LLMProviderError("provider timeout")
        if self.fail_with == "provider":
            raise LLMProviderError("provider unavailable")
        if self.fail_with == "invalid_json":
            return "not-json"
        if self.fail_with == "missing_speech":
            return json.dumps({"intent": "QUESTION", "requested_action": None, "priority": 0.5})
        if self.fail_with == "empty":
            return ""
        context = request.context
        target = context.get("target")
        intent_value = str(context.get("intent"))
        if intent_value == "REQUEST_ACTION" and target:
            speech = f"Of course, I can {target} in a moment."
            requested_action = target
        elif intent_value == "REQUEST_MOVEMENT" and target:
            speech = f"All right, I can head toward the {target}."
            requested_action = f"move_to:{target}"
        # PHASE 26A continuity intents. The deterministic provider only has to
        # prove the pipeline carries them; wording stays minimal and distinct.
        elif intent_value == "FOLLOW_UP":
            label = str(target or "that").replace("_", " ")
            speech = f"Still the same topic - {label} - so here is a little more."
            requested_action = None
        elif intent_value == "TOPIC_CONTINUATION":
            label = str(target or "that").replace("_", " ")
            speech = f"Continuing with {label}, since that is what we were on."
            requested_action = None
        elif intent_value == "ACKNOWLEDGEMENT":
            speech = "Okay."
            requested_action = None
        elif intent_value == "CONVERSATION_END":
            speech = "Alright, I will be here when you want to talk again."
            requested_action = None
        elif intent_value == "QUESTION":
            speech = "I'm here in the room and taking a quiet look around."
            requested_action = None
        elif intent_value == "GREETING":
            speech = "Hello! It's good to see you."
            requested_action = None
        elif intent_value == "FAREWELL":
            speech = "Goodbye for now. Take care."
            requested_action = None
        else:
            speech = "Thank you for saying that."
            requested_action = None
        return json.dumps(
            {
                "speech": speech,
                "intent": intent_value,
                "requested_action": requested_action,
                "priority": float(context.get("priority", 0.0)),
            }
        )
