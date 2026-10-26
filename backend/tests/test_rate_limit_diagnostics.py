"""PHASE 39: rate-limit header and retry-attempt diagnostics.

Observability only. These tests pin what ``brain.llm_request_failed`` reports
about (a) the rate-limit response headers NVIDIA actually returned, drawn only
from the ten-name allowlist the Phase 24 probe already validated, and (b) how
many HTTP attempts the transport really made before giving up.

They must not change -- and here assert the absence of change in -- retry
counts, the retryable status set, the backoff formula, the ``Retry-After``
clamp, the provider timeout, or the failure taxonomy. Everything is mocked and
offline; no request reaches a provider.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from backend.llm.base import LLMProviderError, LLMRequest
from backend.llm.failure_diagnostics import (
    RATE_LIMIT_HEADER_NAMES,
    RETRY_AFTER_HEADER,
    classify_failure,
    failure_record,
    is_retryable,
    log_planning_failure,
    provider_request_id,
    rate_limit_headers,
)
from backend.llm.nvidia import NvidiaConfig, NvidiaProvider


def _ok_body() -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": '{"ok":true}'}}]}
    ).encode("utf-8")


def _http_error(code: int, *, headers=None, body: bytes = b'{"status":429}'):
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=code,
        msg="err",
        hdrs=headers,
        fp=io.BytesIO(body),
    )


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _provider(urlopen) -> NvidiaProvider:
    return NvidiaProvider(NvidiaConfig(), api_key="test-key", urlopen=urlopen)


def _drive(outcomes):
    """Run one generate() against a scripted sequence; return (error, calls)."""
    calls: list[int] = []
    queue = list(outcomes)

    def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
        calls.append(1)
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)

    provider = _provider(urlopen)
    error = None
    with mock.patch("backend.llm.openrouter.time.sleep"):
        try:
            provider.generate(LLMRequest(prompt="hi"))
        except LLMProviderError as exc:  # the only expected failure type
            error = exc
    return provider, error, calls


class AllowlistSourceTests(unittest.TestCase):
    def test_allowlist_matches_the_phase24_probe_exactly(self) -> None:
        # The ten names the Phase 24 probe validated against the live endpoint.
        self.assertEqual(
            RATE_LIMIT_HEADER_NAMES,
            (
                "retry-after",
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-tokens",
                "nvidia-ratelimit-limit",
                "nvidia-ratelimit-remaining",
            ),
        )
        self.assertIn(RETRY_AFTER_HEADER, RATE_LIMIT_HEADER_NAMES)


class HeaderCaptureTests(unittest.TestCase):
    def test_every_allowlisted_header_is_handled_when_present(self) -> None:
        raw = {name: "7" for name in RATE_LIMIT_HEADER_NAMES}
        validated, unparsed = rate_limit_headers(raw)
        self.assertEqual(set(validated), set(RATE_LIMIT_HEADER_NAMES))
        self.assertEqual(unparsed, ())
        self.assertTrue(all(value == "7" for value in validated.values()))

    def test_unknown_headers_are_never_included(self) -> None:
        validated, unparsed = rate_limit_headers(
            {
                "x-ratelimit-limit": "40",
                "server": "nginx",
                "x-secret-thing": "1",
                "x-envoy-upstream-service-time": "12",
            }
        )
        self.assertEqual(validated, {"x-ratelimit-limit": "40"})
        self.assertEqual(unparsed, ())

    def test_auth_cookie_and_token_headers_cannot_leak(self) -> None:
        hostile = {
            "authorization": "Bearer sk-live-LEAK",
            "cookie": "session=LEAK",
            "set-cookie": "session=LEAK",
            "x-api-key": "LEAK",
            "proxy-authorization": "Basic LEAK",
        }
        validated, unparsed = rate_limit_headers(hostile)
        self.assertEqual(validated, {})
        self.assertEqual(unparsed, ())
        self.assertNotIn("LEAK", json.dumps([validated, unparsed]))

    def test_missing_headers_produce_the_empty_representation(self) -> None:
        for empty in (None, {}, object(), [], 17):
            with self.subTest(value=empty):
                self.assertEqual(rate_limit_headers(empty), ({}, ()))

    def test_unsafe_and_malformed_values_are_dropped_by_name(self) -> None:
        for bad in ("abc", "1;2", "1e9", "-5", "+5", "nan", "NaN", "inf",
                    "-inf", "1 2", "a" * 200, "", "   ", "0x10"):
            with self.subTest(value=bad):
                validated, unparsed = rate_limit_headers({"x-ratelimit-limit": bad})
                self.assertEqual(validated, {})
                self.assertEqual(unparsed, ("x-ratelimit-limit",))
                self.assertNotIn(bad.strip() or "x", json.dumps(validated))

    def test_non_string_values_are_dropped(self) -> None:
        validated, unparsed = rate_limit_headers(
            {"x-ratelimit-limit": 40, "x-ratelimit-remaining": ["1"]}
        )
        self.assertEqual(validated, {})
        self.assertEqual(unparsed, ())

    def test_numeric_values_are_normalised_and_bounded(self) -> None:
        validated, unparsed = rate_limit_headers(
            {
                "x-ratelimit-limit": "40",
                "x-ratelimit-remaining": " 39 ",
                "x-ratelimit-reset": "39.500",
                "nvidia-ratelimit-remaining": "9999999999999",  # > 1e12
                "nvidia-ratelimit-limit": "9" * 21,             # too many digits
            }
        )
        self.assertEqual(validated["x-ratelimit-limit"], "40")
        self.assertEqual(validated["x-ratelimit-remaining"], "39")
        self.assertEqual(validated["x-ratelimit-reset"], "39.5")
        self.assertNotIn("nvidia-ratelimit-remaining", validated)
        self.assertNotIn("nvidia-ratelimit-limit", validated)
        self.assertEqual(
            set(unparsed), {"nvidia-ratelimit-remaining", "nvidia-ratelimit-limit"}
        )

    def test_retry_after_accepts_only_the_transports_representations(self) -> None:
        # The transport parses Retry-After with float(); the diagnostic accepts
        # exactly that, and reports anything else by name without its text.
        validated, unparsed = rate_limit_headers({RETRY_AFTER_HEADER: " 30 "})
        self.assertEqual(validated, {RETRY_AFTER_HEADER: "30"})
        self.assertEqual(unparsed, ())

        validated, unparsed = rate_limit_headers({RETRY_AFTER_HEADER: "2.5"})
        self.assertEqual(validated, {RETRY_AFTER_HEADER: "2.5"})
        self.assertEqual(unparsed, ())

        http_date = "Wed, 21 Oct 2026 07:28:00 GMT"
        validated, unparsed = rate_limit_headers({RETRY_AFTER_HEADER: http_date})
        self.assertEqual(validated, {})
        self.assertEqual(unparsed, (RETRY_AFTER_HEADER,))
        self.assertNotIn("2026", json.dumps([validated, list(unparsed)]))

    def test_logging_retry_after_does_not_change_the_retry_delay(self) -> None:
        exc = _http_error(429, headers={"Retry-After": "99"})
        provider = _provider(lambda *a, **k: None)
        before = provider._retry_delay(exc, 0)
        rate_limit_headers(exc.headers)
        after = provider._retry_delay(exc, 0)
        self.assertEqual(before, after)
        self.assertEqual(after, 2.0)  # unchanged 2s clamp

    def test_correlation_ids_stay_bounded(self) -> None:
        good = _http_error(429)
        good.headers = {"nvcf-reqid": "abc-123_XYZ"}
        self.assertEqual(provider_request_id(good), "abc-123_XYZ")

        for bad in ("x" * 200, "has space", "semi;colon", "", 17):
            with self.subTest(value=bad):
                exc = _http_error(429)
                exc.headers = {"nvcf-reqid": bad}
                self.assertIsNone(provider_request_id(exc))


class RetryAttemptDiagnosticsTests(unittest.TestCase):
    def test_exhausted_429_reports_three_attempts_and_exhaustion(self) -> None:
        provider, error, calls = _drive([_http_error(429)] * 3)
        self.assertIsNotNone(error)
        self.assertEqual(len(calls), 3)
        record = failure_record(error, provider=provider, elapsed_ms=2310)
        self.assertEqual(record["attempt_count"], 3)
        self.assertEqual(record["configured_max_attempts"], 3)
        self.assertIs(record["retries_exhausted"], True)
        self.assertEqual(record["http_status"], 429)
        self.assertEqual(record["failure_category"], "rate_limit_or_quota")
        self.assertIsNone(record["retryable"])

    def test_non_retryable_first_attempt_reports_one_attempt(self) -> None:
        provider, error, calls = _drive([_http_error(400, headers=None)])
        self.assertEqual(len(calls), 1)
        record = failure_record(error, provider=provider)
        self.assertEqual(record["attempt_count"], 1)
        self.assertIs(record["retries_exhausted"], False)
        self.assertEqual(record["failure_category"], "invalid_request")
        self.assertIs(record["retryable"], False)

    def test_read_timeout_reports_one_attempt_and_no_exhaustion(self) -> None:
        provider, error, calls = _drive([TimeoutError("read timed out")])
        self.assertEqual(len(calls), 1)
        record = failure_record(error, provider=provider, elapsed_ms=120171)
        self.assertEqual(record["attempt_count"], 1)
        self.assertIs(record["retries_exhausted"], False)
        self.assertEqual(record["failure_category"], "timeout")
        self.assertIs(record["retryable"], True)
        self.assertEqual(record["transport_phase"], "read")

    def test_exhausted_5xx_reports_the_actual_count(self) -> None:
        for code in (502, 503, 504):
            with self.subTest(code=code):
                provider, error, calls = _drive([_http_error(code)] * 3)
                self.assertEqual(len(calls), 3)
                record = failure_record(error, provider=provider)
                self.assertEqual(record["attempt_count"], 3)
                self.assertIs(record["retries_exhausted"], True)
                self.assertEqual(record["failure_category"], "provider_server_error")

    def test_attempt_count_matches_mocked_transport_invocations(self) -> None:
        cases = [
            ([_http_error(429), _ok_body()], 2, None),
            ([_http_error(429), _http_error(429), _http_error(429)], 3, 3),
            ([_http_error(429), _http_error(400)], 2, 2),
            ([_http_error(503), _ok_body()], 2, None),
            ([_http_error(401)], 1, 1),
        ]
        for outcomes, expected_calls, expected_attempts in cases:
            with self.subTest(outcomes=len(outcomes), expected=expected_calls):
                provider, error, calls = _drive(outcomes)
                self.assertEqual(len(calls), expected_calls)
                if expected_attempts is None:
                    self.assertIsNone(error)
                else:
                    record = failure_record(error, provider=provider)
                    self.assertEqual(record["attempt_count"], expected_attempts)
                    self.assertEqual(record["attempt_count"], len(calls))

    def test_rate_limit_headers_reach_the_event(self) -> None:
        headers = {
            "Retry-After": "30",
            "x-ratelimit-limit": "40",
            "x-ratelimit-remaining": "39",
            "X-RateLimit-Reset": "12.5",
            "authorization": "Bearer sk-live-LEAK",
        }
        provider, error, _ = _drive([_http_error(429, headers=headers)] * 3)
        record = failure_record(error, provider=provider)
        self.assertEqual(
            record["rate_limit_headers"],
            {
                "retry-after": "30",
                "x-ratelimit-limit": "40",
                "x-ratelimit-remaining": "39",
                "x-ratelimit-reset": "12.5",
            },
        )
        self.assertIsNone(record["rate_limit_headers_unparsed"])
        self.assertNotIn("LEAK", json.dumps(record))

    def test_missing_headers_omit_the_field_entirely(self) -> None:
        provider, error, _ = _drive([_http_error(429)] * 3)
        record = failure_record(error, provider=provider)
        self.assertIsNone(record["rate_limit_headers"])
        self.assertIsNone(record["rate_limit_headers_unparsed"])

    def test_unparsed_headers_are_reported_by_name_only(self) -> None:
        headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
        provider, error, _ = _drive([_http_error(429, headers=headers)] * 3)
        record = failure_record(error, provider=provider)
        self.assertIsNone(record["rate_limit_headers"])
        self.assertEqual(record["rate_limit_headers_unparsed"], ["retry-after"])
        self.assertNotIn("2026", json.dumps(record))

    def test_bogus_attached_metadata_is_revalidated(self) -> None:
        provider, error, _ = _drive([_http_error(429)] * 3)
        error.transport_info = type(error.transport_info)(
            attempt_count=-3,
            configured_max_attempts="lots",
            retries_exhausted="yes",
            rate_limit_headers={"authorization": "Bearer sk-live-LEAK"},
            rate_limit_headers_unparsed=("not-an-allowlisted-name",),
        )
        record = failure_record(error, provider=provider)
        self.assertIsNone(record["attempt_count"])
        self.assertIsNone(record["configured_max_attempts"])
        self.assertIsNone(record["retries_exhausted"])
        self.assertIsNone(record["rate_limit_headers"])
        self.assertIsNone(record["rate_limit_headers_unparsed"])
        self.assertNotIn("LEAK", json.dumps(record))

    def test_event_field_set_is_exactly_the_expected_superset(self) -> None:
        provider, error, _ = _drive([_http_error(429)] * 3)
        record = failure_record(error, provider=provider, elapsed_ms=2310)
        self.assertEqual(
            set(record),
            {
                # pre-existing fields, unchanged
                "operation", "provider", "model", "exception_class",
                "failure_category", "http_status", "provider_error_code",
                "retryable", "provider_request_id", "cause_class",
                "elapsed_ms", "timeout_seconds", "transport_phase",
                # PHASE 39 additions
                "attempt_count", "configured_max_attempts", "retries_exhausted",
                "rate_limit_headers", "rate_limit_headers_unparsed",
            },
        )

    def test_taxonomy_is_unchanged_for_each_class(self) -> None:
        cases = [
            (_http_error(429), "rate_limit_or_quota", None),
            (_http_error(402), "rate_limit_or_quota", None),
            (_http_error(401), "authentication", False),
            (_http_error(503), "provider_server_error", True),
            (TimeoutError("t"), "timeout", True),
        ]
        for exc, category, retryable in cases:
            with self.subTest(exc=type(exc).__name__, code=getattr(exc, "code", None)):
                self.assertEqual(classify_failure(exc), category)
                self.assertIs(is_retryable(category), retryable)

    def test_diagnostics_failure_never_masks_the_provider_error(self) -> None:
        provider, error, _ = _drive([_http_error(429)] * 3)
        with mock.patch(
            "backend.llm.failure_diagnostics.log_event",
            side_effect=RuntimeError("logger exploded"),
        ):
            # Must not raise, and must not alter the exception being handled.
            log_planning_failure(error, provider=provider, elapsed_ms=1)
        self.assertIsInstance(error, LLMProviderError)
        self.assertTrue(str(error).startswith("http_429"))

    def test_caller_still_receives_the_same_exception_type_and_message(self) -> None:
        _provider_obj, error, _ = _drive([_http_error(429)] * 3)
        self.assertIsInstance(error, LLMProviderError)
        self.assertTrue(str(error).startswith("http_429:"))
        self.assertIsInstance(error.__cause__, urllib.error.HTTPError)

    def test_response_body_and_credentials_never_reach_the_record(self) -> None:
        secret = "Bearer sk-live-SECRETVALUE0123456789"
        body = json.dumps({"detail": secret, "status": 429}).encode("utf-8")
        provider, error, _ = _drive(
            [_http_error(429, headers={"authorization": secret}, body=body)] * 3
        )
        record = failure_record(error, provider=provider)
        rendered = json.dumps(record)
        self.assertNotIn("sk-live", rendered)
        self.assertNotIn("SECRETVALUE", rendered)

    def test_emitted_event_shape_and_the_known_token_name_redaction(self) -> None:
        """The emitted line, not just ``failure_record``.

        Documents a deliberate interaction: the shared structure redactor
        replaces the value of any nested key containing "token", which is a
        false positive for two allowlisted header names
        (``*-limit-tokens``/``*-remaining-tokens``). The *name* still appears,
        so presence is observable; only the number is masked. The allowlist and
        key names are deliberately NOT renamed to dodge the redactor.
        """
        headers = {
            "Retry-After": "30",
            "x-ratelimit-limit": "40",
            "x-ratelimit-remaining-tokens": "7",
            "authorization": "Bearer sk-live-LEAK",
        }
        _provider_obj, error, _ = _drive(
            [_http_error(429, headers=headers)] * 3
        )
        emitted: dict = {}

        def _capture(_logger, event, **fields):
            emitted["event"] = event
            emitted.update(
                {k: v for k, v in fields.items() if v is not None}
            )

        with mock.patch(
            "backend.llm.failure_diagnostics.log_event", side_effect=_capture
        ):
            log_planning_failure(
                error, provider=_provider_obj, elapsed_ms=2300
            )

        self.assertEqual(emitted["event"], "brain.llm_request_failed")
        self.assertEqual(emitted["attempt_count"], 3)
        self.assertEqual(emitted["configured_max_attempts"], 3)
        self.assertIs(emitted["retries_exhausted"], True)
        self.assertEqual(emitted["http_status"], 429)
        self.assertEqual(emitted["failure_category"], "rate_limit_or_quota")
        self.assertNotIn("LEAK", json.dumps(emitted))
        self.assertNotIn("authorization", json.dumps(emitted))


class Phase24RetryBehaviourUnchangedTests(unittest.TestCase):
    """The diagnostics must not have moved the retry policy."""

    def test_backoff_schedule_and_attempt_budget_unchanged(self) -> None:
        sleeps: list[float] = []
        calls: list[int] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            calls.append(1)
            raise _http_error(429)

        provider = _provider(urlopen)
        with mock.patch(
            "backend.llm.openrouter.time.sleep", side_effect=sleeps.append
        ):
            with self.assertRaises(LLMProviderError):
                provider.generate(LLMRequest(prompt="hi"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_openrouter_still_never_retries(self) -> None:
        from backend.llm.openrouter import OpenRouterConfig, OpenRouterProvider

        calls: list[int] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            calls.append(1)
            raise _http_error(429)

        provider = OpenRouterProvider(
            OpenRouterConfig(), api_key="test-key", urlopen=urlopen
        )
        with self.assertRaises(LLMProviderError):
            provider.generate(LLMRequest(prompt="hi"))
        self.assertEqual(len(calls), 1)

    def test_retry_after_is_still_clamped_to_two_seconds(self) -> None:
        sleeps: list[float] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            raise _http_error(429, headers={"Retry-After": "99"})

        provider = _provider(urlopen)
        with mock.patch(
            "backend.llm.openrouter.time.sleep", side_effect=sleeps.append
        ):
            with self.assertRaises(LLMProviderError):
                provider.generate(LLMRequest(prompt="hi"))
        self.assertEqual(sleeps, [2.0, 2.0])

    def test_provider_timeout_configuration_unchanged(self) -> None:
        self.assertEqual(NvidiaConfig().timeout_seconds, 120.0)
        self.assertEqual(NvidiaConfig().retry_attempts, 2)
        self.assertEqual(NvidiaConfig().retry_backoff_seconds, 0.5)
        self.assertEqual(NvidiaConfig().retry_max_backoff_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
