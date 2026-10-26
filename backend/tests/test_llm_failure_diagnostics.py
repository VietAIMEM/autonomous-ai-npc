"""Focused tests for privacy-safe LLM provider failure diagnostics.

The production capture on 2026-10-18 showed
``last_failure = planning_failed:LLMProviderError`` in 16 of 18 samples with no
corresponding log line. These tests pin the replacement: one bounded,
structured, secret-free event per failed planning request, with the original
planning outcome unchanged.

Every test is mocked and deterministic; no network call, credential or quota is
used.
"""

from __future__ import annotations

import io
import json
import time
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from backend.brain.models import Goal, GoalStatus, GoalType, PersonalityProfile
from backend.brain.planner import BrainPlanner
from backend.brain.reasoning import BrainReasoner
from backend.brain.world_state import build_world_state
from backend.llm.base import LLMProvider, LLMProviderError, LLMRequest
from backend.llm.failure_diagnostics import (
    CATEGORY_AUTHENTICATION,
    CATEGORY_CONNECTION,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_PROVIDER_SERVER_ERROR,
    CATEGORY_RATE_LIMIT_OR_QUOTA,
    CATEGORY_TIMEOUT,
    CATEGORY_UNKNOWN,
    EVENT_LLM_REQUEST_FAILED,
    PHASE_CONNECT,
    PHASE_READ,
    cause_class,
    classify_failure,
    configured_timeout_seconds,
    failure_record,
    is_retryable,
    log_planning_failure,
    provider_error_code,
    provider_request_id,
    safe_token,
    transport_phase,
)


def http_error(code: int, *, headers=None, body: bytes = b'{"error":"x"}'):
    """A structured ``HTTPError`` exactly as ``urllib`` raises it."""
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=code,
        msg="provider said no",
        hdrs=headers,
        fp=io.BytesIO(body),
    )


def wrapped(cause: BaseException | None, message: str = "provider failed"):
    """An ``LLMProviderError`` chained to ``cause``, as the transports raise it."""
    exc = LLMProviderError(message)
    if cause is not None:
        exc.__cause__ = cause
    return exc


class _RaisingProvider(LLMProvider):
    """Minimal provider stub: records nothing, raises the given error."""

    name = "test-provider"

    def __init__(self, error: BaseException, model: str = "vendor/model-1") -> None:
        self._error = error
        self.config = SimpleNamespace(model=model)

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - stub
        raise self._error


def world():
    return build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": {
                "CENTER": {"position": [0, 0, 0], "navigation_target": True},
            },
        }
    )


def goal() -> Goal:
    return Goal("goal", GoalType.EXPLORE_ROOM, 0.5, "test", status=GoalStatus.ACTIVE)


class TimeoutClassificationTests(unittest.TestCase):
    def test_timeout_classification(self):
        self.assertEqual(
            classify_failure(wrapped(TimeoutError("timed out"))), CATEGORY_TIMEOUT
        )
        # URLError whose reason is a timeout is still a timeout, not a drop.
        self.assertEqual(
            classify_failure(wrapped(urllib.error.URLError(TimeoutError()))),
            CATEGORY_TIMEOUT,
        )

        class ProviderTimeout(Exception):
            pass

        self.assertEqual(
            classify_failure(wrapped(ProviderTimeout())), CATEGORY_TIMEOUT
        )
        self.assertTrue(is_retryable(CATEGORY_TIMEOUT))

    def test_connection_classification(self):
        self.assertEqual(
            classify_failure(
                wrapped(urllib.error.URLError(ConnectionRefusedError("refused")))
            ),
            CATEGORY_CONNECTION,
        )
        self.assertEqual(
            classify_failure(wrapped(ConnectionResetError("reset"))),
            CATEGORY_CONNECTION,
        )
        self.assertTrue(is_retryable(CATEGORY_CONNECTION))


class HttpStatusClassificationTests(unittest.TestCase):
    def test_authentication_statuses(self):
        for code in (401, 403):
            with self.subTest(code=code):
                self.assertEqual(
                    classify_failure(wrapped(http_error(code), f"http_{code}:x")),
                    CATEGORY_AUTHENTICATION,
                )
        self.assertFalse(is_retryable(CATEGORY_AUTHENTICATION))

    def test_throttle_status_is_not_split_into_rate_limit_or_quota(self):
        for code in (429, 402):
            with self.subTest(code=code):
                record = failure_record(
                    wrapped(http_error(code), f"http_{code}:x")
                )
                self.assertEqual(
                    record["failure_category"], CATEGORY_RATE_LIMIT_OR_QUOTA
                )
                self.assertEqual(record["http_status"], code)
                # The provider cannot distinguish a transient throttle from an
                # exhausted quota, so no field claims which one it was, and
                # retryability stays explicitly unstated.
                self.assertIsNone(record["retryable"])
                for forbidden in ("rate_limit", "rate_limited", "quota", "quota_exhausted"):
                    self.assertNotIn(forbidden, record)

    def test_provider_server_errors(self):
        for code in (500, 502, 503, 504):
            with self.subTest(code=code):
                self.assertEqual(
                    classify_failure(wrapped(http_error(code), f"http_{code}:x")),
                    CATEGORY_PROVIDER_SERVER_ERROR,
                )
        self.assertTrue(is_retryable(CATEGORY_PROVIDER_SERVER_ERROR))

    def test_client_statuses_are_invalid_request(self):
        for code in (400, 404, 422):
            with self.subTest(code=code):
                self.assertEqual(
                    classify_failure(wrapped(http_error(code), f"http_{code}:x")),
                    CATEGORY_INVALID_REQUEST,
                )
        self.assertFalse(is_retryable(CATEGORY_INVALID_REQUEST))

    def test_unknown_exception_handling(self):
        # The local pre-flight refusal has no structured metadata to classify.
        self.assertEqual(
            classify_failure(LLMProviderError("api_key_unavailable")),
            CATEGORY_UNKNOWN,
        )
        self.assertEqual(classify_failure(RuntimeError("something")), CATEGORY_UNKNOWN)
        self.assertIsNone(is_retryable(CATEGORY_UNKNOWN))
        record = failure_record(LLMProviderError("api_key_unavailable"))
        self.assertEqual(record["failure_category"], CATEGORY_UNKNOWN)
        self.assertIsNone(record["http_status"])


class StructuredExtractionTests(unittest.TestCase):
    def test_structured_error_code_and_request_id_are_extracted(self):
        class SdkError(Exception):
            pass

        sdk = SdkError("boom")
        sdk.error_code = "insufficient_quota"
        sdk.request_id = "req-9f8e7d-abc"
        exc = wrapped(sdk)
        self.assertEqual(provider_error_code(exc), "insufficient_quota")
        self.assertEqual(provider_request_id(exc), "req-9f8e7d-abc")

    def test_request_id_read_from_structured_headers(self):
        cause = http_error(503, headers={"x-request-id": "nvcf-12345-abc"})
        self.assertEqual(provider_request_id(wrapped(cause)), "nvcf-12345-abc")

    def test_unsafe_or_oversized_extracted_values_are_dropped(self):
        class SdkError(Exception):
            pass

        sdk = SdkError("boom")
        sdk.error_code = "Bearer sk-live-should-never-be-logged"
        sdk.request_id = "x" * 200
        exc = wrapped(sdk)
        self.assertIsNone(provider_error_code(exc))
        self.assertIsNone(provider_request_id(exc))

    def test_safe_token_rules(self):
        self.assertEqual(safe_token("nvcf-req_1"), "nvcf-req_1")
        self.assertEqual(safe_token(" vendor/model-1 ", allow_namespace=True), "vendor/model-1")
        for bad in ("", "   ", None, 17, "has space", "a" * 65, "semi;colon"):
            with self.subTest(value=bad):
                self.assertIsNone(safe_token(bad))


class PrivacyTests(unittest.TestCase):
    def test_secrets_prompts_and_headers_never_reach_the_record(self):
        secret = "sk-live-DEADBEEF0123456789"
        prompt_fragment = "viewer asked: please dance for me"
        body = json.dumps(
            {
                "error": {"message": "invalid request"},
                "authorization": f"Bearer {secret}",
                "prompt": prompt_fragment,
            }
        ).encode("utf-8")
        cause = http_error(
            500,
            headers={
                "x-request-id": f"Bearer {secret}",
                "authorization": f"Bearer {secret}",
            },
            body=body,
        )
        # The transport message embeds the (previously redacted) body; the
        # diagnostic must not depend on that and must never echo it.
        exc = wrapped(cause, message=f"http_500:{body.decode()} {secret} {prompt_fragment}")

        record = failure_record(exc, provider=_RaisingProvider(exc))
        serialized = json.dumps(record)
        self.assertNotIn(secret, serialized)
        self.assertNotIn("Bearer", serialized)
        self.assertNotIn("authorization", serialized.casefold())
        self.assertNotIn("dance", serialized)
        self.assertNotIn(prompt_fragment, serialized)
        # A non-token header value is refused rather than truncated or escaped.
        self.assertIsNone(record["provider_request_id"])
        # Only the structured status survives.
        self.assertEqual(record["http_status"], 500)

    def test_emitted_log_event_is_secret_free_and_well_formed(self):
        secret = "sk-live-SECRETVALUE0123456789"
        exc = wrapped(
            http_error(503, headers={"x-request-id": "safe-req-1"}),
            message=f"http_503:{secret}",
        )
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            log_planning_failure(exc, provider=_RaisingProvider(exc))
        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.call_args
        self.assertEqual(args[1], EVENT_LLM_REQUEST_FAILED)
        self.assertNotIn(secret, json.dumps(kwargs))
        self.assertEqual(kwargs["failure_category"], CATEGORY_PROVIDER_SERVER_ERROR)
        self.assertEqual(kwargs["http_status"], 503)
        self.assertEqual(kwargs["provider"], "test-provider")
        self.assertEqual(kwargs["model"], "vendor/model-1")
        self.assertEqual(kwargs["provider_request_id"], "safe-req-1")
        self.assertEqual(kwargs["retryable"], True)
        self.assertEqual(kwargs["operation"], "planning")
        self.assertEqual(kwargs["exception_class"], "LLMProviderError")


class ErrorHandlingTests(unittest.TestCase):
    def test_logging_failure_does_not_raise(self):
        with mock.patch(
            "backend.llm.failure_diagnostics.log_event",
            side_effect=RuntimeError("logger exploded"),
        ):
            # Must swallow everything: diagnostics never mask the provider error.
            log_planning_failure(LLMProviderError("x"))

    def test_boundary_outcome_is_unchanged_when_logging_explodes(self):
        error = wrapped(http_error(500), "http_500:x")
        reasoner = BrainReasoner(_RaisingProvider(error), BrainPlanner(), None)
        with mock.patch(
            "backend.llm.failure_diagnostics.log_event",
            side_effect=RuntimeError("logger exploded"),
        ):
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "LLMProviderError")


class CompatibilityTests(unittest.TestCase):
    def test_boundary_returns_exception_class_name_once_per_failure(self):
        error = wrapped(TimeoutError("slow"), "transport_error:TimeoutError")
        provider = _RaisingProvider(error)
        reasoner = BrainReasoner(provider, BrainPlanner(), None)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        # Unchanged caller-facing contract: the loop turns this into
        # f"planning_failed:{reason}", i.e. the same
        # "planning_failed:LLMProviderError" the runtime already reports.
        self.assertEqual(reason, "LLMProviderError")
        self.assertEqual(f"planning_failed:{reason}", "planning_failed:LLMProviderError")
        self.assertEqual(spy.call_count, 1)

    def test_llm_provider_error_type_and_chaining_are_preserved(self):
        cause = http_error(429)
        exc = wrapped(cause, "http_429:x")
        self.assertIsInstance(exc, LLMProviderError)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIs(exc.__cause__, cause)
        self.assertEqual(str(exc), "http_429:x")
        # The mixin never mutates the exception it classifies.
        failure_record(exc)
        self.assertEqual(str(exc), "http_429:x")
        self.assertIs(exc.__cause__, cause)


class _SlowRaisingProvider(_RaisingProvider):
    """Raises the given error after a real, measured delay."""

    def __init__(self, error: BaseException, *, delay: float, model: str = "vendor/model-1"):
        super().__init__(error, model=model)
        self._delay = delay

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - stub
        time.sleep(self._delay)
        raise self._error


class TransportMetadataTests(unittest.TestCase):
    """The transport-level fields added for the timeout root-cause audit."""

    def test_direct_timeout_reports_cause_class_and_read_phase(self):
        exc = wrapped(TimeoutError("timed out"))
        record = failure_record(exc)
        self.assertEqual(record["failure_category"], CATEGORY_TIMEOUT)
        self.assertEqual(record["cause_class"], "TimeoutError")
        # A bare TimeoutError comes from the unwrapped getresponse()/read.
        self.assertEqual(record["transport_phase"], PHASE_READ)

    def test_connection_url_error_with_nested_timeout_reason(self):
        # urllib wraps connect/send failures -- including a socket timeout
        # during connect -- in URLError, with the OS error kept as ``reason``.
        cause = urllib.error.URLError(TimeoutError("connect timed out"))
        exc = wrapped(cause, "transport_error:URLError")
        record = failure_record(exc)
        self.assertEqual(record["failure_category"], CATEGORY_TIMEOUT)
        self.assertEqual(record["transport_phase"], PHASE_CONNECT)
        # The deepest chained class is still the underlying timeout.
        self.assertEqual(record["cause_class"], "TimeoutError")

    def test_cause_class_differs_from_the_wrapper_type(self):
        class SdkTransportError(Exception):
            pass

        record = failure_record(wrapped(SdkTransportError("boom")))
        self.assertEqual(record["exception_class"], "LLMProviderError")
        self.assertEqual(record["cause_class"], "SdkTransportError")
        # No phase is claimed for a transport this module has not verified.
        self.assertIsNone(record["transport_phase"])

    def test_missing_or_malformed_cause_metadata_is_omitted(self):
        # No chained cause at all (the local pre-flight refusal).
        record = failure_record(LLMProviderError("api_key_unavailable"))
        self.assertIsNone(record["cause_class"])
        self.assertIsNone(record["transport_phase"])

        # A cause class name that is not an identifier is dropped, not repaired.
        weird = type("Bad Class!", (Exception,), {})
        self.assertIsNone(failure_record(wrapped(weird())).get("cause_class"))

    def test_elapsed_ms_is_validated_and_never_negative(self):
        cases = (
            (0, 0),
            (12.4, 12),
            (120_000, 120000),
            (-1, None),
            (-0.5, None),
            (float("nan"), None),
            (float("inf"), None),
            (float("1e12"), None),
            (True, None),
            ("120", None),
            (None, None),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                record = failure_record(wrapped(TimeoutError()), elapsed_ms=value)
                self.assertEqual(record["elapsed_ms"], expected)

    def test_the_reasoner_measures_the_failed_request_duration(self):
        error = wrapped(TimeoutError("slow"), "transport_error:TimeoutError")
        reasoner = BrainReasoner(_SlowRaisingProvider(error, delay=0.15), BrainPlanner(), None)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "LLMProviderError")
        elapsed = spy.call_args.kwargs["elapsed_ms"]
        self.assertIsInstance(elapsed, int)
        self.assertGreaterEqual(elapsed, 100)
        self.assertLess(elapsed, 5_000)

    def test_timeout_seconds_comes_from_the_effective_config(self):
        provider = _RaisingProvider(TimeoutError())
        provider.config = SimpleNamespace(model="vendor/model-1", timeout_seconds=120.0)
        self.assertEqual(configured_timeout_seconds(provider), 120.0)
        self.assertEqual(
            failure_record(TimeoutError(), provider=provider)["timeout_seconds"], 120.0
        )
        # A missing, malformed or nonsensical budget is omitted, not guessed.
        for bad in (None, "120", True, -1.0, float("inf"), object()):
            with self.subTest(value=bad):
                provider.config = SimpleNamespace(model="m", timeout_seconds=bad)
                self.assertIsNone(configured_timeout_seconds(provider))
                self.assertIsNone(
                    failure_record(TimeoutError(), provider=provider)["timeout_seconds"]
                )

    def test_unclassifiable_transport_phase_is_omitted(self):
        # An HTTP error is an HTTP response, never a connect failure, even
        # though HTTPError subclasses URLError.
        self.assertIsNone(transport_phase(wrapped(http_error(503))))
        # A reset cannot be attributed to connect or read from the structure.
        self.assertIsNone(transport_phase(wrapped(ConnectionResetError("reset"))))
        self.assertIsNone(transport_phase(wrapped(OSError("broken"))))
        self.assertIsNone(transport_phase(LLMProviderError("api_key_unavailable")))

    def test_transport_fields_never_carry_sensitive_text(self):
        secret = "sk-live-SECRETVALUE0123456789"
        prompt = "viewer asked: please dance for me"
        cause = urllib.error.URLError(TimeoutError(f"{secret} {prompt}"))
        exc = wrapped(
            cause,
            message=(
                f"transport_error:URLError {secret} {prompt} "
                f"https://example.invalid/v1/chat?api_key={secret}&q={prompt}"
            ),
        )
        provider = _RaisingProvider(exc)
        provider.config = SimpleNamespace(model="vendor/model-1", timeout_seconds=120.0)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            log_planning_failure(exc, provider=provider, elapsed_ms=12.4)
        kwargs = spy.call_args.kwargs
        serialized = json.dumps(kwargs)
        for forbidden in (secret, "Bearer", "dance", prompt, "api_key", "example.invalid", "?"):
            self.assertNotIn(forbidden, serialized)
        # Only the verified class names and numbers survive.
        self.assertEqual(kwargs["cause_class"], "TimeoutError")
        self.assertEqual(kwargs["transport_phase"], PHASE_CONNECT)
        self.assertEqual(kwargs["elapsed_ms"], 12)
        self.assertEqual(kwargs["timeout_seconds"], 120.0)

    def test_existing_fields_and_semantics_are_unchanged(self):
        record = failure_record(
            wrapped(TimeoutError("slow")), provider=_RaisingProvider(TimeoutError())
        )
        for key in (
            "operation",
            "provider",
            "model",
            "exception_class",
            "failure_category",
            "http_status",
            "provider_error_code",
            "retryable",
            "provider_request_id",
        ):
            self.assertIn(key, record)
        self.assertEqual(record["operation"], "planning")
        self.assertEqual(record["exception_class"], "LLMProviderError")
        self.assertEqual(record["failure_category"], CATEGORY_TIMEOUT)
        self.assertTrue(record["retryable"])

    def test_logging_failure_never_replaces_the_provider_error(self):
        error = wrapped(TimeoutError("slow"), "transport_error:TimeoutError")
        reasoner = BrainReasoner(
            _SlowRaisingProvider(error, delay=0.01), BrainPlanner(), None
        )
        with mock.patch(
            "backend.llm.failure_diagnostics.log_event",
            side_effect=RuntimeError("logger exploded"),
        ):
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(f"planning_failed:{reason}", "planning_failed:LLMProviderError")

    def test_classification_helpers_are_pure(self):
        # The helpers must not mutate the exception they describe.
        cause = urllib.error.URLError(TimeoutError("x"))
        exc = wrapped(cause, "transport_error:URLError")
        self.assertEqual(cause_class(exc), "TimeoutError")
        self.assertEqual(transport_phase(exc), PHASE_CONNECT)
        self.assertEqual(str(exc), "transport_error:URLError")
        self.assertIs(exc.__cause__, cause)


if __name__ == "__main__":
    unittest.main()
