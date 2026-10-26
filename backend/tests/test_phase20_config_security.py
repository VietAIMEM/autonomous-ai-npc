"""PHASE 20 focused tests: production configuration and secret protection.

These exercise the real runtime facade (``backend.runtime.config``), the real
redaction helpers and the real provider error paths. Every secret-looking
value below is a synthetic marker that must never survive redaction.
"""

from __future__ import annotations

import io
import json
import threading
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from backend.chat.providers.runner import ChatSourceRunner
from backend.llm.base import LLMProviderError, LLMRequest
from backend.llm.config import (
    LLMConfig,
    LLMConfigError,
    PROVIDER_MOCK,
    PROVIDER_OPENROUTER,
)
from backend.llm.openrouter import OpenRouterConfig, OpenRouterProvider
from backend.memory.service import MemoryService
from backend.runtime.config import (
    ConfigurationError,
    ENV_DEVELOPMENT,
    ENV_PRODUCTION,
    ENV_TEST,
    RuntimeConfig,
    normalize_environment,
)
from backend.runtime.redaction import (
    REDACTED,
    contains_secret,
    redact_structure,
    redact_text,
    safe_error,
)


FAKE_LLM_KEY = "sk-live-phase20secret0000000000"
FAKE_STREAM_KEY = "live-phase20streamsecret"
FAKE_CHAT_TOKEN = "oauth:phase20chattokensecret"


class ProductionProviderTests(unittest.TestCase):
    """Production must never silently accept a mock provider."""

    def test_production_rejects_mock_provider(self) -> None:
        config = RuntimeConfig(environment=ENV_PRODUCTION)

        self.assertTrue(config.is_production)
        self.assertFalse(config.mocks_allowed)
        with self.assertRaises(ConfigurationError) as ctx:
            config.enforce_provider("llm", "mock")
        self.assertIn("llm_mock_provider_not_allowed_in_production", str(ctx.exception))

    def test_production_accepts_real_provider_names(self) -> None:
        config = RuntimeConfig(environment=ENV_PRODUCTION)

        config.enforce_provider("llm", "openrouter")
        config.enforce_provider("tts", "fish_audio")
        config.enforce_provider("chat", "twitch")
        config.enforce_provider("stream", "ffmpeg")

    def test_development_and_test_allow_mocks(self) -> None:
        for environment in (ENV_DEVELOPMENT, ENV_TEST):
            with self.subTest(environment=environment):
                config = RuntimeConfig(environment=environment)
                self.assertTrue(config.mocks_allowed)
                config.enforce_provider("llm", "mock")

    def test_explicit_override_is_the_only_production_mock_path(self) -> None:
        config = RuntimeConfig(environment=ENV_PRODUCTION, allow_mock_providers=True)

        self.assertTrue(config.mocks_allowed)
        config.enforce_provider("llm", "mock")

    def test_environment_facade_defaults_production_to_real_providers(self) -> None:
        config = RuntimeConfig.from_environment({"ROOM_AGENT_ENV": "production"})

        self.assertTrue(config.is_production)
        self.assertIs(config.allow_mock_providers, False)
        self.assertFalse(config.mocks_allowed)
        with self.assertRaises(ConfigurationError):
            config.enforce_provider("llm", "mock")

    def test_unknown_environment_value_degrades_to_development_with_note(self) -> None:
        config = RuntimeConfig.from_environment({"ROOM_AGENT_ENV": "banana"})

        self.assertEqual(config.environment, ENV_DEVELOPMENT)
        self.assertIn(
            "unrecognized_environment_value_treated_as_development", config.notes
        )

    def test_normalize_environment_variants(self) -> None:
        cases = {
            "prod": ENV_PRODUCTION,
            "PRODUCTION": ENV_PRODUCTION,
            "live": ENV_PRODUCTION,
            "test": ENV_TEST,
            "ci": ENV_TEST,
            "dev": ENV_DEVELOPMENT,
            "": ENV_DEVELOPMENT,
            None: ENV_DEVELOPMENT,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_environment(raw), expected)

    def test_invalid_environment_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ConfigurationError):
            RuntimeConfig(environment="staging")

    def test_runtime_config_to_dict_is_credential_free(self) -> None:
        config = RuntimeConfig.from_environment(
            {
                "ROOM_AGENT_ENV": "production",
                # A hostile environment: values that must never be echoed.
                "LLM_API_KEY": FAKE_LLM_KEY,
                "STREAM_KEY": FAKE_STREAM_KEY,
            }
        )

        described = config.to_dict()
        serialized = json.dumps(described)
        self.assertNotIn(FAKE_LLM_KEY, serialized)
        self.assertNotIn(FAKE_STREAM_KEY, serialized)
        self.assertFalse(contains_secret(serialized))
        for marker in ("api_key", "token", "password", "secret", "credential"):
            self.assertNotIn(marker, serialized)


class RedactionTests(unittest.TestCase):
    def test_redact_text_removes_bearer_tokens(self) -> None:
        text = f"Authorization: Bearer {FAKE_LLM_KEY}"

        self.assertTrue(contains_secret(text))
        cleaned = redact_text(text)
        self.assertNotIn(FAKE_LLM_KEY, cleaned)
        self.assertIn(REDACTED, cleaned)

    def test_redact_text_removes_key_value_credentials(self) -> None:
        samples = (
            f"api_key={FAKE_LLM_KEY}",
            f"api-key={FAKE_LLM_KEY}",
            f"stream_key: {FAKE_STREAM_KEY}",
            f"token={FAKE_CHAT_TOKEN}",
            "password=hunter2phase20",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                cleaned = redact_text(sample)
                self.assertIn(REDACTED, cleaned)
                for fragment in (
                    FAKE_LLM_KEY,
                    FAKE_STREAM_KEY,
                    FAKE_CHAT_TOKEN,
                    "hunter2phase20",
                ):
                    self.assertNotIn(fragment, cleaned)

    def test_redact_text_removes_stream_url_credentials(self) -> None:
        url = f"rtmp://broadcaster:{FAKE_STREAM_KEY}@live.example.com/app"

        cleaned = redact_text(url)
        self.assertNotIn(FAKE_STREAM_KEY, cleaned)
        self.assertIn(REDACTED, cleaned)

    def test_safe_error_redacts_and_bounds_exception_text(self) -> None:
        message = safe_error(Exception(f"connect failed: stream_key={FAKE_STREAM_KEY}"))

        self.assertNotIn(FAKE_STREAM_KEY, message)
        self.assertTrue(message.startswith("Exception: "))
        self.assertLessEqual(len(message), 320)

    def test_redact_structure_replaces_secret_keys_recursively(self) -> None:
        payload = {
            "provider": "openrouter",
            "nested": {
                "api_key": FAKE_LLM_KEY,
                "stream_key": FAKE_STREAM_KEY,
                "items": [{"access_token": FAKE_CHAT_TOKEN, "viewer": "viewer-1"}],
            },
        }

        cleaned = redact_structure(payload)
        serialized = json.dumps(cleaned)
        self.assertNotIn(FAKE_LLM_KEY, serialized)
        self.assertNotIn(FAKE_STREAM_KEY, serialized)
        self.assertNotIn(FAKE_CHAT_TOKEN, serialized)
        self.assertEqual(cleaned["nested"]["api_key"], REDACTED)
        self.assertEqual(cleaned["nested"]["items"][0]["access_token"], REDACTED)
        self.assertEqual(cleaned["nested"]["items"][0]["viewer"], "viewer-1")


class ProviderErrorRedactionTests(unittest.TestCase):
    def test_openrouter_error_body_is_redacted(self) -> None:
        body = json.dumps(
            {
                "error": {"message": "Authentication failed"},
                "request": {"Authorization": f"Bearer {FAKE_LLM_KEY}"},
                "api_key": FAKE_LLM_KEY,
            }
        ).encode("utf-8")

        def failing_urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            raise urllib.error.HTTPError(
                url="https://openrouter.ai/api/v1/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(body),
            )

        provider = OpenRouterProvider(
            OpenRouterConfig(),
            api_key=FAKE_LLM_KEY,
            urlopen=failing_urlopen,
        )

        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="hello"))

        detail = str(ctx.exception)
        self.assertTrue(detail.startswith("http_401"))
        self.assertNotIn(FAKE_LLM_KEY, detail)
        self.assertIn(REDACTED, detail)

    def test_memory_degraded_state_never_stores_key_material(self) -> None:
        stub = SimpleNamespace(_degraded=False, _last_error=None)

        MemoryService._mark_degraded(
            stub, Exception(f"write failed: api_key={FAKE_LLM_KEY}"), "save_episode"
        )

        self.assertTrue(stub._degraded)
        self.assertNotIn(FAKE_LLM_KEY, stub._last_error)
        self.assertTrue(stub._last_error.startswith("save_episode:"))
        self.assertLessEqual(len(stub._last_error), 260)

    def test_chat_source_error_is_redacted_before_exposure(self) -> None:
        stub = SimpleNamespace(
            _counters={"provider_errors": 0},
            _lock=threading.RLock(),
            _degraded=False,
            _last_error=None,
        )

        ChatSourceRunner._record_error(
            stub, f"connect failed: stream_key={FAKE_STREAM_KEY}"
        )

        self.assertTrue(stub._degraded)
        self.assertNotIn(FAKE_STREAM_KEY, stub._last_error)
        self.assertIn(REDACTED, stub._last_error)
        self.assertEqual(stub._counters["provider_errors"], 1)


class LLMConfigTests(unittest.TestCase):
    def test_unknown_llm_provider_is_rejected_not_silently_mocked(self) -> None:
        with self.assertRaises(LLMConfigError):
            LLMConfig.from_environment({"LLM_PROVIDER": "totally-unknown"})

    def test_missing_key_file_falls_back_to_explicit_mock_in_development(self) -> None:
        with TemporaryDirectory() as directory:
            key_path = Path(directory) / "missing-key.txt"
            config = LLMConfig.from_environment({"LLM_API_KEY_PATH": str(key_path)})

        self.assertEqual(config.provider, PROVIDER_MOCK)
        self.assertTrue(config.mock)

    def test_real_key_file_selects_real_provider_without_leaking_it(self) -> None:
        with TemporaryDirectory() as directory:
            key_path = Path(directory) / "api_key.txt"
            key_path.write_text(FAKE_LLM_KEY, encoding="utf-8")
            config = LLMConfig.from_environment({"LLM_API_KEY_PATH": str(key_path)})

            self.assertEqual(config.provider, PROVIDER_OPENROUTER)
            self.assertTrue(config.configured)
            described = config.to_dict()

        self.assertNotIn(FAKE_LLM_KEY, json.dumps(described))
        self.assertEqual(described["api_key_path"], "api_key.txt")
        self.assertFalse(described["mock"])


if __name__ == "__main__":
    unittest.main()
