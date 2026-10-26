"""Focused tests for the local Ollama provider and its local-only routing.

Everything here is offline: the HTTP transport is replaced with a deterministic
stub, so no request reaches Ollama, NVIDIA, OpenRouter or any other endpoint.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from backend.api.brain import resolve_llm_provider
from backend.llm.base import LLMProviderError, LLMRequest
from backend.llm.config import (
    LLMConfig,
    PROVIDER_NVIDIA,
    PROVIDER_OLLAMA,
    build_provider,
    describe_provider,
)
from backend.llm.failure_diagnostics import failure_record
from backend.llm.ollama import OllamaConfig, OllamaProvider, _is_loopback
from backend.runtime.config import RuntimeConfig, ConfigurationError


def _chat_body(content: str) -> bytes:
    return json.dumps(
        {
            "model": "qwen3.5:4b",
            "message": {"role": "assistant", "content": content},
            "done": True,
        }
    ).encode("utf-8")


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _capturing_urlopen(outcomes):
    """A urlopen stub that records every request and pops ``outcomes``."""
    calls: list[dict] = []

    def urlopen(request, timeout=None):  # noqa: ANN001 - transport stub
        calls.append(
            {
                "url": request.full_url,
                "data": json.loads(request.data.decode("utf-8")),
                "headers": dict(request.headers),
                "timeout": timeout,
            }
        )
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    return urlopen, calls


class LoopbackTests(unittest.TestCase):
    def test_loopback_hosts_accepted(self) -> None:
        for url in (
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://[::1]:11434",
            "http://127.8.9.10:11434/",
        ):
            self.assertTrue(_is_loopback(url), url)

    def test_non_loopback_hosts_rejected(self) -> None:
        for url in (
            "http://0.0.0.0:11434",
            "http://192.168.1.10:11434",
            "https://integrate.api.nvidia.com/v1",
            "http://ollama.internal:11434",
            "ftp://127.0.0.1:11434",
            "",
        ):
            self.assertFalse(_is_loopback(url), url)

    def test_provider_refuses_non_loopback_endpoint(self) -> None:
        # Fail closed: a non-local base URL must not even construct a provider.
        with self.assertRaises(LLMProviderError):
            OllamaProvider(
                OllamaConfig(base_url="http://203.0.113.7:11434"),
                urlopen=lambda *a, **k: None,
            )


class SelectionTests(unittest.TestCase):
    def test_environment_resolves_to_ollama_and_qwen(self) -> None:
        env = {"LLM_PROVIDER": "ollama"}
        config = LLMConfig.from_environment(env)
        self.assertEqual(config.provider, PROVIDER_OLLAMA)
        self.assertEqual(config.model, "qwen3.5:4b")
        self.assertEqual(config.base_url, "http://127.0.0.1:11434")
        self.assertEqual(config.num_ctx, 8192)
        self.assertTrue(config.configured)  # local: no credential required

    def test_environment_model_override_still_wins(self) -> None:
        env = {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3.5:4b"}
        self.assertEqual(LLMConfig.from_environment(env).model, "qwen3.5:4b")

    def test_build_provider_returns_ollama_provider(self) -> None:
        provider = build_provider(LLMConfig.from_environment({"LLM_PROVIDER": "ollama"}))
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.name, "ollama")
        self.assertEqual(provider.config.model, "qwen3.5:4b")
        self.assertEqual(provider.config.base_url, "http://127.0.0.1:11434")
        self.assertEqual(provider.config.num_ctx, 8192)

    def test_describe_provider_exposes_local_endpoint(self) -> None:
        provider = build_provider(LLMConfig.from_environment({"LLM_PROVIDER": "ollama"}))
        described = describe_provider(provider)
        self.assertEqual(described["kind"], "ollama")
        self.assertFalse(described["mock"])
        self.assertTrue(described["configured"])
        self.assertEqual(described["model"], "qwen3.5:4b")
        self.assertEqual(described["base_url"], "http://127.0.0.1:11434")

    def test_resolve_llm_provider_is_ollama(self) -> None:
        runtime = RuntimeConfig(environment="production")
        provider = resolve_llm_provider(runtime, {"LLM_PROVIDER": "ollama"})
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.name, "ollama")

    def test_unknown_provider_is_rejected_not_downgraded(self) -> None:
        with self.assertRaises(Exception):
            LLMConfig.from_environment({"LLM_PROVIDER": "not-a-provider"})

    def test_ollama_is_allowed_in_production(self) -> None:
        # Should not raise: ollama is a real (non-mock) provider.
        RuntimeConfig(environment="production").enforce_provider("llm", PROVIDER_OLLAMA)

    def test_mock_is_refused_in_production(self) -> None:
        with self.assertRaises(ConfigurationError):
            RuntimeConfig(environment="production").enforce_provider("llm", "mock")


class RequestShapeTests(unittest.TestCase):
    def test_request_uses_native_chat_with_num_ctx_and_no_auth(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        out = provider.generate(LLMRequest(prompt="plan this"))

        self.assertEqual(out, "{}")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:11434/api/chat")
        self.assertFalse(call["data"]["stream"])
        self.assertEqual(call["data"]["options"]["num_ctx"], 8192)
        self.assertIn("num_predict", call["data"]["options"])
        self.assertEqual(call["data"]["messages"][1]["content"], "plan this")
        # No credential may be attached to a local request.
        self.assertNotIn("Authorization", call["headers"])

    def test_num_ctx_is_configurable(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(num_ctx=4096), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        self.assertEqual(calls[0]["data"]["options"]["num_ctx"], 4096)

    def test_thinking_is_disabled_by_default_in_the_request(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        # The native ``think`` switch must be sent off so the model answers
        # directly instead of spending the budget in ``message.thinking``.
        self.assertIs(calls[0]["data"]["think"], False)

    def test_thinking_can_be_requested_on(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(think=True), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        self.assertIs(calls[0]["data"]["think"], True)

    def test_generation_ceiling_is_bounded_and_reported_separately(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(num_predict=1024), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        options = calls[0]["data"]["options"]
        # Context and generation are distinct knobs.
        self.assertEqual(options["num_ctx"], 8192)
        self.assertEqual(options["num_predict"], 1024)

    def test_shared_max_tokens_caps_generation(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(
            OllamaConfig(num_predict=1024, max_tokens=256), urlopen=urlopen
        )
        provider.generate(LLMRequest(prompt="x"))
        self.assertEqual(calls[0]["data"]["options"]["num_predict"], 256)

    def test_json_mode_is_sent_for_planner_requests(self) -> None:
        # The reasoning path sets ``brain`` in the request context; that is the
        # signal that lets the transport constrain output to JSON.
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x", context={"brain": True}))
        self.assertEqual(calls[0]["data"].get("format"), "json")

    def test_json_mode_is_not_sent_for_chat_requests(self) -> None:
        # The chat path shares this provider but must stay free-form prose.
        urlopen, calls = _capturing_urlopen([_chat_body("hi there")])
        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x", context={"character": "Lacrimosa"}))
        self.assertNotIn("format", calls[0]["data"])

    def test_json_mode_can_be_disabled(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(json_mode=None), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x", context={"brain": True}))
        self.assertNotIn("format", calls[0]["data"])

    def test_num_gpu_is_omitted_by_default(self) -> None:
        # Absent config must preserve Ollama's automatic placement exactly.
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        self.assertNotIn("num_gpu", calls[0]["data"]["options"])

    def test_num_gpu_is_sent_when_configured(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(num_gpu=24), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        options = calls[0]["data"]["options"]
        self.assertEqual(options["num_gpu"], 24)
        # Existing options must be unchanged.
        self.assertEqual(options["num_ctx"], 8192)
        self.assertEqual(options["num_predict"], 1024)

    def test_num_gpu_zero_is_sent(self) -> None:
        # ``0`` is a legitimate explicit "all layers on CPU" request.
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(num_gpu=0), urlopen=urlopen)
        provider.generate(LLMRequest(prompt="x"))
        self.assertEqual(calls[0]["data"]["options"]["num_gpu"], 0)

    def test_environment_num_gpu_resolution(self) -> None:
        base = {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3.5:4b"}
        self.assertIsNone(LLMConfig.from_environment(base).num_gpu)
        self.assertEqual(
            LLMConfig.from_environment({**base, "LLM_NUM_GPU": "24"}).num_gpu, 24
        )
        # Unparseable / negative values must not pin a count.
        self.assertIsNone(
            LLMConfig.from_environment({**base, "LLM_NUM_GPU": "abc"}).num_gpu
        )
        self.assertIsNone(
            LLMConfig.from_environment({**base, "LLM_NUM_GPU": "-5"}).num_gpu
        )

    def test_build_provider_threads_num_gpu_from_environment(self) -> None:
        provider = build_provider(
            LLMConfig.from_environment(
                {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen3.5:4b", "LLM_NUM_GPU": "24"}
            )
        )
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.config.num_gpu, 24)

    def test_follow_up_turns_are_forwarded(self) -> None:
        urlopen, calls = _capturing_urlopen([_chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        provider.generate(
            LLMRequest(
                prompt="p",
                context={
                    "assistant_message": {"role": "assistant", "content": "hi"},
                    "follow_up": "and then?",
                },
            )
        )
        roles = [m["role"] for m in calls[0]["data"]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])


class ResponseHandlingTests(unittest.TestCase):
    def test_openai_shaped_body_is_not_accepted(self) -> None:
        # The native endpoint shape is required; the OpenAI ``choices`` shape
        # must fail rather than silently pass through.
        body = json.dumps(
            {"choices": [{"message": {"content": "{}"}}]}
        ).encode("utf-8")
        provider = OllamaProvider(OllamaConfig(), urlopen=lambda *a, **k: _Response(body))
        with self.assertRaises(LLMProviderError):
            provider.generate(LLMRequest(prompt="x"))

    def test_malformed_json_is_rejected(self) -> None:
        provider = OllamaProvider(
            OllamaConfig(), urlopen=lambda *a, **k: _Response(b"not json")
        )
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        self.assertIn("invalid_provider_json", str(ctx.exception))

    def test_error_body_becomes_fixed_reason_without_leak(self) -> None:
        secret = "model 'qwen3.5:4b' not found; token=supersecret"
        body = json.dumps({"error": secret}).encode("utf-8")
        provider = OllamaProvider(
            OllamaConfig(), urlopen=lambda *a, **k: _Response(body)
        )
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        self.assertEqual(str(ctx.exception), "ollama_error")
        self.assertNotIn("supersecret", str(ctx.exception))

    def test_empty_content_is_rejected(self) -> None:
        body = json.dumps({"message": {"role": "assistant", "content": "  "}}).encode()
        provider = OllamaProvider(OllamaConfig(), urlopen=lambda *a, **k: _Response(body))
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        self.assertIn("empty_provider_content", str(ctx.exception))

    def test_reasoning_only_reply_is_flagged_distinctly(self) -> None:
        body = json.dumps(
            {"message": {"role": "assistant", "content": "", "thinking": "..."}}
        ).encode()
        provider = OllamaProvider(OllamaConfig(), urlopen=lambda *a, **k: _Response(body))
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        self.assertIn("empty_content_with_reasoning_details", str(ctx.exception))


class FailureModeTests(unittest.TestCase):
    def test_connection_refused_raises_and_never_falls_back(self) -> None:
        def urlopen(_request, timeout=None):  # noqa: ANN001
            raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        # Still the ollama provider object; no remote provider was consulted.
        self.assertEqual(provider.name, "ollama")
        info = getattr(ctx.exception, "transport_info", None)
        self.assertIsNotNone(info)
        self.assertEqual(info.attempt_count, 1)  # transport errors are not retried
        self.assertFalse(info.retries_exhausted)

    def test_read_timeout_is_one_attempt_not_exhaustion(self) -> None:
        def urlopen(_request, timeout=None):  # noqa: ANN001
            raise TimeoutError("timed out")

        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        info = ctx.exception.transport_info
        self.assertEqual(info.attempt_count, 1)
        self.assertFalse(info.retries_exhausted)
        record = failure_record(ctx.exception, provider=provider, elapsed_ms=120000)
        self.assertEqual(record["failure_category"], "timeout")
        self.assertEqual(record["transport_phase"], "read")

    def test_retry_absorbs_a_transient_server_error(self) -> None:
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=503,
            msg="unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"loading"}'),
        )
        urlopen, calls = _capturing_urlopen([err, _chat_body("{}")])
        provider = OllamaProvider(OllamaConfig(retry_attempts=1), urlopen=urlopen)
        sleeps: list[float] = []
        with mock.patch("backend.llm.openrouter.time.sleep", side_effect=sleeps.append):
            out = provider.generate(LLMRequest(prompt="x"))
        self.assertEqual(out, "{}")
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_exhausted_retry_reports_real_attempt_count(self) -> None:
        def urlopen(_request, timeout=None):  # noqa: ANN001
            raise urllib.error.HTTPError(
                url="http://127.0.0.1:11434/api/chat",
                code=503,
                msg="unavailable",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"busy"}'),
            )

        provider = OllamaProvider(OllamaConfig(retry_attempts=2), urlopen=urlopen)
        sleeps: list[float] = []
        with mock.patch("backend.llm.openrouter.time.sleep", side_effect=sleeps.append):
            with self.assertRaises(LLMProviderError) as ctx:
                provider.generate(LLMRequest(prompt="x"))
        info = ctx.exception.transport_info
        self.assertEqual(info.attempt_count, 3)
        self.assertEqual(info.configured_max_attempts, 3)
        self.assertTrue(info.retries_exhausted)
        record = failure_record(ctx.exception, provider=provider, elapsed_ms=1000)
        self.assertEqual(record["failure_category"], "provider_server_error")
        self.assertEqual(record["attempt_count"], 3)

    def test_diagnostic_logging_failure_does_not_replace_exception(self) -> None:
        from backend.llm import failure_diagnostics

        def urlopen(_request, timeout=None):  # noqa: ANN001
            raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

        provider = OllamaProvider(OllamaConfig(), urlopen=urlopen)
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="x"))
        original = ctx.exception
        with mock.patch.object(
            failure_diagnostics, "failure_record", side_effect=RuntimeError("boom")
        ):
            # Must swallow the diagnostics failure and leave the caller-facing
            # exception untouched.
            failure_diagnostics.log_planning_failure(original, provider=provider)
        self.assertIs(ctx.exception, original)

    def test_logged_event_never_contains_prompt_or_body(self) -> None:
        sentinel = "PROMPT_SENTINEL_9f3a"
        secret_body = "BODY_SENTINEL_ab12"
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(secret_body.encode()),
        )
        provider = OllamaProvider(OllamaConfig(), urlopen=lambda *a, **k: (_ for _ in ()).throw(err))
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt=sentinel))
        record = failure_record(ctx.exception, provider=provider, elapsed_ms=10)
        blob = json.dumps(record, default=str)
        self.assertNotIn(sentinel, blob)
        self.assertNotIn(secret_body, blob)


class DiagnosticsSecurityTests(unittest.TestCase):
    def test_failure_record_has_no_credential_or_raw_header_fields(self) -> None:
        provider = OllamaProvider(
            OllamaConfig(), urlopen=lambda *a, **k: _Response(b"{}")
        )
        err = LLMProviderError("http_429:x")
        record = failure_record(err, provider=provider, elapsed_ms=5)
        self.assertNotIn("authorization", json.dumps(record).lower())
        self.assertNotIn("api_key", json.dumps(record).lower())
        for key in record:
            self.assertNotIn("headers_raw", key)

    def test_ollama_provider_description_never_carries_a_key(self) -> None:
        provider = build_provider(LLMConfig.from_environment({"LLM_PROVIDER": "ollama"}))
        blob = json.dumps(describe_provider(provider))
        self.assertNotIn("api_key", blob.lower())
        self.assertNotIn("bearer", blob.lower())


if __name__ == "__main__":
    unittest.main()
