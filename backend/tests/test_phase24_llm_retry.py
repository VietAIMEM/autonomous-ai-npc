"""PHASE 24 focused tests for the bounded provider retry.

The real NVIDIA NIM endpoint answers a burst with a bare HTTP 429 without a
``Retry-After`` header (see ``nvidia_rate_limit_probe`` evidence), so the
shared transport gains a small opt-in retry. These tests are deterministic and
never touch the network.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from backend.llm.base import LLMProviderError, LLMRequest
from backend.llm.nvidia import NvidiaConfig, NvidiaProvider
from backend.llm.openrouter import OpenRouterConfig, OpenRouterProvider


def _ok_body() -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": '{"ok":true}'}}]}
    ).encode("utf-8")


def _throttle(headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs=headers,
        fp=io.BytesIO(b'{"status":429,"title":"Too Many Requests"}'),
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


class RetryTransportTests(unittest.TestCase):
    def test_openrouter_default_never_retries(self) -> None:
        calls: list[int] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            calls.append(1)
            raise _throttle()

        provider = OpenRouterProvider(
            OpenRouterConfig(), api_key="test-key", urlopen=urlopen
        )
        with self.assertRaises(LLMProviderError):
            provider.generate(LLMRequest(prompt="hi"))
        self.assertEqual(len(calls), 1)

    def test_nvidia_retries_a_bare_429_until_success(self) -> None:
        outcomes: list[object] = [_throttle(), _ok_body()]
        calls: list[int] = []
        sleeps: list[float] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            calls.append(1)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return _Response(outcome)

        provider = NvidiaProvider(
            NvidiaConfig(), api_key="test-key", urlopen=urlopen
        )
        with mock.patch(
            "backend.llm.openrouter.time.sleep", side_effect=sleeps.append
        ):
            raw = provider.generate(LLMRequest(prompt="hi"))

        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_nvidia_gives_up_after_bounded_attempts(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            calls.append(1)
            raise _throttle()

        provider = NvidiaProvider(
            NvidiaConfig(), api_key="test-key", urlopen=urlopen
        )
        with mock.patch(
            "backend.llm.openrouter.time.sleep", side_effect=sleeps.append
        ):
            with self.assertRaises(LLMProviderError) as ctx:
                provider.generate(LLMRequest(prompt="hi"))

        self.assertTrue(str(ctx.exception).startswith("http_429"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_retry_after_header_wins_and_is_clamped(self) -> None:
        sleeps: list[float] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            raise _throttle(headers={"Retry-After": "99"})

        provider = NvidiaProvider(
            NvidiaConfig(), api_key="test-key", urlopen=urlopen
        )
        with mock.patch(
            "backend.llm.openrouter.time.sleep", side_effect=sleeps.append
        ):
            with self.assertRaises(LLMProviderError):
                provider.generate(LLMRequest(prompt="hi"))

        # The provider asked for 99s; the transport clamps to its 2s ceiling.
        self.assertEqual(sleeps, [2.0, 2.0])

    def test_non_retryable_status_still_fails_immediately(self) -> None:
        calls: list[int] = []

        def urlopen(_request, timeout=None):  # noqa: ANN001 - transport stub
            calls.append(1)
            raise urllib.error.HTTPError(
                url="https://example.invalid",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b"{}"),
            )

        provider = NvidiaProvider(
            NvidiaConfig(), api_key="test-key", urlopen=urlopen
        )
        with self.assertRaises(LLMProviderError) as ctx:
            provider.generate(LLMRequest(prompt="hi"))
        self.assertTrue(str(ctx.exception).startswith("http_401"))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
