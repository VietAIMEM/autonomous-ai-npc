"""Privacy-safe diagnostics for LLM provider request failures.

The planning boundary in :mod:`backend.brain.reasoning` catches
``LLMProviderError`` and previously reported only the exception class name, so
the *kind* of provider failure (timeout, connectivity, auth, throttling,
server error) was unrecoverable from production logs.

This module adds one structured event per failed request. Two rules shape it:

1. **Only structured metadata is read.** Everything logged is taken from
   exception attributes the current providers actually set on the original,
   chained cause (see ``openrouter.py`` / ``nvidia.py``, which always raise
   ``LLMProviderError(...) from exc``):

   * ``urllib.error.HTTPError`` — ``.code`` (int) and ``.headers``
   * ``urllib.error.URLError``  — ``.reason``
   * ``TimeoutError`` / ``ConnectionError`` / ``OSError`` subclasses

2. **The provider message is never read.** The transport embeds a redacted
   copy of the HTTP response body in its exception message, and this module
   deliberately does not depend on that redaction being perfect. Exception
   text, prompts, credentials and response bodies are never logged here.

Anything extracted from provider-controlled data (an error code or a request
id) is validated against a strict character set and a length bound before it is
eligible to be logged. Logging is best-effort and never raises.
"""

from __future__ import annotations

import logging
import math
import re
import urllib.error
from typing import Any, Optional

from .base import TransportFailureInfo
from ..runtime.obs import log_event

_LOGGER = logging.getLogger(__name__)

#: The single event name emitted for a failed LLM request.
EVENT_LLM_REQUEST_FAILED = "brain.llm_request_failed"

#: Bounded failure taxonomy. ``rate_limit_or_quota`` deliberately does not
#: distinguish a transient throttle from exhausted quota: the live NIM endpoint
#: answers a burst with a bare HTTP 429 and no distinguishing metadata
#: (see ``nvidia.py``), so a more specific claim would not be justified.
CATEGORY_TIMEOUT = "timeout"
CATEGORY_CONNECTION = "connection"
CATEGORY_AUTHENTICATION = "authentication"
CATEGORY_RATE_LIMIT_OR_QUOTA = "rate_limit_or_quota"
CATEGORY_PROVIDER_SERVER_ERROR = "provider_server_error"
CATEGORY_INVALID_REQUEST = "invalid_request"
CATEGORY_UNKNOWN = "unknown"

FAILURE_CATEGORIES = frozenset(
    {
        CATEGORY_TIMEOUT,
        CATEGORY_CONNECTION,
        CATEGORY_AUTHENTICATION,
        CATEGORY_RATE_LIMIT_OR_QUOTA,
        CATEGORY_PROVIDER_SERVER_ERROR,
        CATEGORY_INVALID_REQUEST,
        CATEGORY_UNKNOWN,
    }
)

#: HTTP statuses with a specifically verified meaning.
_AUTHENTICATION_STATUSES = frozenset({401, 403})
_RATE_LIMIT_OR_QUOTA_STATUSES = frozenset({429, 402})
_INVALID_REQUEST_STATUSES = frozenset({400, 404, 405, 406, 409, 413, 415, 422})

#: Longest token this module will emit for a provider-supplied value.
MAX_TOKEN_LENGTH = 64

_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")
#: Model aliases legitimately contain a namespace separator.
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9._:\-/]{1,64}$")

#: Header names a provider may use to return a correlation id.
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "nvcf-reqid", "x-amzn-requestid")

#: Optional structured attributes a provider/SDK may set. Absent in the
#: current transports, which put the code inside the (unread) response body.
_ERROR_CODE_ATTRIBUTES = ("provider_error_code", "error_code")
_REQUEST_ID_ATTRIBUTES = ("request_id", "provider_request_id", "x_request_id")

#: PHASE 39: the exact response-header allowlist already validated against the
#: live NIM endpoint by the Phase 24 probe
#: (``AgentScripts/phase24/evidence/nvidia_rate_limit_probe.py``). Reused
#: verbatim rather than guessed from current API assumptions, so the diagnostic
#: can only ever name headers that were deliberately checked before this change
#: existed. Nothing outside this tuple is ever read or emitted.
RATE_LIMIT_HEADER_NAMES = (
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
)

#: ``Retry-After`` is the one header the transport itself consumes, so it
#: accepts the same representations the transport accepts (any finite float).
#: Every other allowlisted name is a plain count, so only integer/decimal text
#: is accepted -- no sign, exponent, ``nan``, ``inf`` or arbitrary string.
RETRY_AFTER_HEADER = "retry-after"
_NUMERIC_HEADER_PATTERN = re.compile(r"^[0-9]{1,15}(?:\.[0-9]{1,6})?$")

#: Largest magnitude published for a numeric rate-limit header. A larger value
#: is a broken/absurd header, not data worth carrying.
MAX_RATE_LIMIT_VALUE = 1e12

#: Most response headers inspected while looking for the allowlist.
MAX_HEADER_SCAN = 64

#: Largest attempt count published for one ``generate()`` call.
MAX_ATTEMPT_COUNT = 64

#: Transport phases this module is willing to name. The urllib stack pins them:
#: ``AbstractHTTPHandler.do_open`` wraps failures from
#: ``HTTPConnection.request()`` -- DNS, TCP and TLS establishment plus the send
#: -- as ``URLError``, while ``getresponse()`` and the body read in
#: ``openrouter.generate`` are left unwrapped, so a socket timeout there
#: surfaces as a bare ``TimeoutError``. Nothing here infers a phase from timing.
PHASE_CONNECT = "connect"
PHASE_READ = "read"

#: Longest timing value this module will publish, in milliseconds (24 h). A
#: larger measured value means the clock is wrong, not that a request ran.
MAX_ELAPSED_MS = 86_400_000


def _valid_duration_ms(value: Any) -> Optional[int]:
    """``value`` as a finite, non-negative whole millisecond count, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    if number > MAX_ELAPSED_MS:
        return None
    return int(round(number))


def _valid_status(value: Any) -> Optional[int]:
    """Return ``value`` as an HTTP status int, or ``None`` if implausible."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 100 <= value <= 599:
        return value
    return None


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    """``exc`` plus its chained cause/context, bounded and cycle-safe."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and len(chain) < 4:
        if id(current) in seen:
            break
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_timeout(exc: BaseException) -> bool:
    """True when any link in the chain is a timeout, by type or type name."""
    for candidate in _exception_chain(exc):
        if isinstance(candidate, TimeoutError):
            return True
        if isinstance(candidate, urllib.error.URLError) and isinstance(
            candidate.reason, TimeoutError
        ):
            return True
        name = type(candidate).__name__.casefold()
        if "timeout" in name:
            return True
    return False


def _chain_with_reasons(exc: BaseException) -> tuple[BaseException, ...]:
    """``_exception_chain`` plus any nested ``reason`` exceptions.

    ``urllib`` wraps a connect/send failure as ``URLError(err)``. When urllib
    raises it itself the OS error also lands in ``__context__``, but a
    ``URLError`` that was constructed (or re-raised) directly keeps the cause
    *only* in ``.reason``. Both shapes must be followed for the class of the
    real transport failure to be reachable. ``HTTPError.reason`` is a plain
    string (its status message), so it is skipped.
    """
    chain = list(_exception_chain(exc))
    seen = {id(item) for item in chain}
    index = 0
    while index < len(chain) and len(chain) < 6:
        reason = getattr(chain[index], "reason", None)
        if isinstance(reason, BaseException) and id(reason) not in seen:
            seen.add(id(reason))
            chain.append(reason)
        index += 1
    return tuple(chain)


def http_status(exc: BaseException) -> Optional[int]:
    """The HTTP status carried by the chain, if any.

    ``urllib.error.HTTPError`` exposes the real status as ``.code``; some SDKs
    use ``status``/``status_code`` instead. Only a plausible integer is
    accepted, and only from structured attributes — never parsed from text.
    """
    for candidate in _exception_chain(exc):
        if isinstance(candidate, urllib.error.HTTPError):
            status = _valid_status(getattr(candidate, "code", None))
            if status is not None:
                return status
        for attribute in ("status_code", "status"):
            status = _valid_status(getattr(candidate, attribute, None))
            if status is not None:
                return status
    return None


def classify_failure(exc: BaseException) -> str:
    """Map an exception onto the bounded failure taxonomy."""
    if _is_timeout(exc):
        return CATEGORY_TIMEOUT

    status = http_status(exc)
    if status is not None:
        if status in _AUTHENTICATION_STATUSES:
            return CATEGORY_AUTHENTICATION
        if status in _RATE_LIMIT_OR_QUOTA_STATUSES:
            return CATEGORY_RATE_LIMIT_OR_QUOTA
        if status in _INVALID_REQUEST_STATUSES:
            return CATEGORY_INVALID_REQUEST
        if 500 <= status <= 599:
            return CATEGORY_PROVIDER_SERVER_ERROR
        if 400 <= status <= 499:
            # Any other client-side status is still a request problem, but it
            # is not one this code has specifically verified.
            return CATEGORY_INVALID_REQUEST
        return CATEGORY_UNKNOWN

    if isinstance(exc, ConnectionError):
        return CATEGORY_CONNECTION
    for candidate in _exception_chain(exc):
        if isinstance(candidate, urllib.error.URLError):
            # Not a timeout (checked above), so it is a transport failure.
            return CATEGORY_CONNECTION
        if isinstance(candidate, ConnectionError):
            return CATEGORY_CONNECTION
    return CATEGORY_UNKNOWN


def is_retryable(category: str) -> Optional[bool]:
    """Whether the failure is transient, or ``None`` when not established.

    ``None`` is the honest answer for throttling: a retry helps a transient
    throttle but not an exhausted quota, and the current providers cannot tell
    those apart.
    """
    if category in (
        CATEGORY_TIMEOUT,
        CATEGORY_CONNECTION,
        CATEGORY_PROVIDER_SERVER_ERROR,
    ):
        return True
    if category in (CATEGORY_AUTHENTICATION, CATEGORY_INVALID_REQUEST):
        return False
    return None


def safe_token(value: Any, *, allow_namespace: bool = False) -> Optional[str]:
    """A bounded, character-validated token, or ``None``.

    Provider-controlled values are only eligible to be logged if they are a
    short string of identifier characters; anything else (including any text
    that could carry a secret or a prompt fragment) is dropped.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_TOKEN_LENGTH:
        return None
    pattern = _MODEL_PATTERN if allow_namespace else _TOKEN_PATTERN
    return text if pattern.match(text) else None


def _numeric_header_text(value: Any, *, allow_float_form: bool) -> Optional[str]:
    """A normalised numeric string, or ``None`` when not safely representable.

    ``allow_float_form`` is true only for ``Retry-After``, because that is the
    one header the transport itself parses with ``float()``. Every other
    allowlisted header is matched against an integer/decimal pattern first, so
    ``nan``, ``inf``, ``1e9``, signs and arbitrary text are rejected rather than
    normalised into something that looks meaningful.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > MAX_TOKEN_LENGTH:
        return None
    if not allow_float_form and not _NUMERIC_HEADER_PATTERN.match(text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or abs(number) > MAX_RATE_LIMIT_VALUE:
        return None
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _lowercase_header_items(headers: Any) -> dict[str, str]:
    """Case-folded ``{name: value}`` for a real header object, bounded.

    Handles both ``email.message.Message`` (what urllib actually produces) and
    a plain mapping, including the ``None`` a mocked ``HTTPError`` may carry.
    Anything that is not a string name/string value is skipped.
    """
    items = getattr(headers, "items", None)
    if not callable(items):
        return {}
    try:
        pairs = list(items())
    except Exception:  # pragma: no cover - defensive
        return {}
    found: dict[str, str] = {}
    for pair in pairs[:MAX_HEADER_SCAN]:
        try:
            key, value = pair
        except (TypeError, ValueError):
            continue
        if isinstance(key, str) and isinstance(value, str):
            found[key.strip().lower()] = value
    return found


def rate_limit_headers(headers: Any) -> tuple[dict[str, str], tuple[str, ...]]:
    """``(validated, unparsed)`` for the allowlisted rate-limit headers.

    Only the ten Phase 24 names are ever considered. A name that is present but
    whose value is not safely representable is reported by *name* in
    ``unparsed`` -- so an HTTP-date ``Retry-After`` is visible as "the header
    was there" without its text ever being logged. A name that is simply absent
    appears in neither collection, which is what lets the caller omit the field
    entirely instead of implying a limit of zero.
    """
    validated: dict[str, str] = {}
    unparsed: list[str] = []
    found = _lowercase_header_items(headers)
    for name in RATE_LIMIT_HEADER_NAMES:
        raw = found.get(name)
        if raw is None:
            continue
        value = _numeric_header_text(raw, allow_float_form=(name == RETRY_AFTER_HEADER))
        if value is None:
            unparsed.append(name)
        else:
            validated[name] = value
    return validated, tuple(unparsed)


def transport_failure_info(exc: BaseException) -> Optional[TransportFailureInfo]:
    """The bounded transport metadata attached to the exception chain, if any."""
    for candidate in _exception_chain(exc):
        info = getattr(candidate, "transport_info", None)
        if isinstance(info, TransportFailureInfo):
            return info
    return None


def _valid_attempt_count(value: Any) -> Optional[int]:
    """A plausible HTTP attempt count. Rejects bools, non-ints and nonsense."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 1 <= value <= MAX_ATTEMPT_COUNT:
        return value
    return None


def _valid_flag(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def provider_error_code(exc: BaseException) -> Optional[str]:
    """A validated provider error code, when the provider exposes one.

    ``HTTPError.code`` is the HTTP status and is deliberately not used here.
    The current transports carry their provider code inside the response body,
    which this module never reads, so in production this is normally absent.
    """
    for candidate in _exception_chain(exc):
        for attribute in _ERROR_CODE_ATTRIBUTES:
            token = safe_token(getattr(candidate, attribute, None))
            if token is not None:
                return token
    return None


def provider_request_id(exc: BaseException) -> Optional[str]:
    """A validated correlation id from a structured attribute or header."""
    for candidate in _exception_chain(exc):
        for attribute in _REQUEST_ID_ATTRIBUTES:
            token = safe_token(getattr(candidate, attribute, None))
            if token is not None:
                return token
        headers = getattr(candidate, "headers", None)
        getter = getattr(headers, "get", None)
        if not callable(getter):
            continue
        for header in _REQUEST_ID_HEADERS:
            try:
                token = safe_token(getter(header))
            except Exception:  # pragma: no cover - defensive
                token = None
            if token is not None:
                return token
    return None


def provider_identity(provider: Any) -> tuple[Optional[str], Optional[str]]:
    """``(provider name, model alias)`` from the configured provider object."""
    name = safe_token(getattr(provider, "name", None))
    config = getattr(provider, "config", None)
    model = safe_token(getattr(config, "model", None), allow_namespace=True)
    return name, model


def cause_class(exc: BaseException) -> Optional[str]:
    """Class name of the deepest chained cause, excluding the top exception.

    The transports always raise ``LLMProviderError(...) from <original>``, so
    the interesting class is the chained one, not the wrapper. Only the class
    *name* is read -- never the message, ``args`` or any attribute of it -- and
    it is validated as a bounded identifier before it can be logged. A nested
    ``URLError.reason`` is followed too, so a connect timeout reports the
    underlying ``TimeoutError`` rather than the ``URLError`` wrapper.
    """
    chain = _chain_with_reasons(exc)
    for candidate in reversed(chain[1:]):
        token = safe_token(type(candidate).__name__)
        if token is not None:
            return token
    return None


def transport_phase(exc: BaseException) -> Optional[str]:
    """Which phase of the urllib request failed, when the stack pins it.

    Returns ``"connect"`` when the chained transport exception is a
    ``urllib.error.URLError`` that is not an ``HTTPError`` (urllib raises that
    only from ``HTTPConnection.request``: DNS, TCP/TLS establishment or send),
    and ``"read"`` when it is a bare ``TimeoutError`` (raised while waiting for
    the response headers or body). Anything else is left unclassified rather
    than guessed.
    """
    chain = _exception_chain(exc)
    if len(chain) < 2:
        return None
    failure = chain[1]
    # HTTPError subclasses URLError but is an HTTP response, not a connect
    # failure, so it must be excluded before the URLError branch.
    if isinstance(failure, urllib.error.HTTPError):
        return None
    if isinstance(failure, urllib.error.URLError):
        return PHASE_CONNECT
    if isinstance(failure, TimeoutError):
        return PHASE_READ
    return None


def configured_timeout_seconds(provider: Any) -> Optional[float]:
    """The effective transport timeout of the configured provider, if stated."""
    config = getattr(provider, "config", None)
    value = getattr(config, "timeout_seconds", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 3)


def failure_record(
    exc: BaseException,
    *,
    provider: Any = None,
    elapsed_ms: Any = None,
) -> dict[str, Any]:
    """The complete, privacy-safe field set for one failed request.

    Field provenance:

    * ``operation``        — constant ``"planning"`` (the only caller today).
    * ``provider``/``model`` — the configured provider object, token-validated.
    * ``exception_class``  — ``type(exc).__name__``, token-validated.
    * ``failure_category`` — this module's taxonomy.
    * ``http_status``      — structured ``.code``/``.status`` only.
    * ``provider_error_code`` — structured attribute only; normally absent.
    * ``retryable``        — derived from the category, ``None`` if unclear.
    * ``provider_request_id`` — structured attribute or response header.
    * ``cause_class``      — class name of the deepest chained cause only.
    * ``elapsed_ms``       — caller-measured duration of the failed request.
    * ``timeout_seconds``  — the configured transport timeout, when stated.
    * ``transport_phase``  — ``connect``/``read`` only when the urllib stack
      pins the phase; omitted rather than guessed.
    * ``attempt_count``/``configured_max_attempts``/``retries_exhausted`` —
      actual transport history, taken from the ``TransportFailureInfo`` the
      transport attaches to the exception. Counts real requests, never inferred
      from elapsed time. ``retries_exhausted`` is true only when at least one
      retry was configured and the final attempt still returned a retryable
      status; a first-attempt non-retryable error and an un-retried transport
      timeout both report ``False``.
    * ``rate_limit_headers``/``rate_limit_headers_unparsed`` — only the
      allowlisted rate-limit response headers, value-validated and normalised.
      Omitted entirely when no allowlisted header was present.

    Every value that is unavailable or unverifiable is ``None``, which
    ``log_event`` omits from the emitted line rather than guessing a default.
    """
    category = classify_failure(exc)
    provider_name, model = provider_identity(provider)
    info = transport_failure_info(exc)
    # Re-validate rather than trusting the attached mapping: this module owns
    # the guarantee that nothing outside the allowlist can reach the event.
    headers, unparsed = rate_limit_headers(getattr(info, "rate_limit_headers", None))
    carried = getattr(info, "rate_limit_headers_unparsed", ())
    if isinstance(carried, (list, tuple)):
        for name in carried:
            if (
                isinstance(name, str)
                and name in RATE_LIMIT_HEADER_NAMES
                and name not in unparsed
            ):
                unparsed = unparsed + (name,)
    return {
        "operation": "planning",
        "provider": provider_name,
        "model": model,
        "exception_class": safe_token(type(exc).__name__),
        "failure_category": category,
        "http_status": http_status(exc),
        "provider_error_code": provider_error_code(exc),
        "retryable": is_retryable(category),
        "provider_request_id": provider_request_id(exc),
        "cause_class": cause_class(exc),
        "elapsed_ms": _valid_duration_ms(elapsed_ms),
        "timeout_seconds": configured_timeout_seconds(provider),
        "transport_phase": transport_phase(exc),
        "attempt_count": _valid_attempt_count(getattr(info, "attempt_count", None)),
        "configured_max_attempts": _valid_attempt_count(
            getattr(info, "configured_max_attempts", None)
        ),
        "retries_exhausted": _valid_flag(getattr(info, "retries_exhausted", None)),
        "rate_limit_headers": headers or None,
        "rate_limit_headers_unparsed": list(unparsed) or None,
    }


def log_planning_failure(
    exc: BaseException,
    *,
    provider: Any = None,
    elapsed_ms: Any = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Emit one ``brain.llm_request_failed`` event. Never raises.

    A diagnostics failure must never replace or mask the provider failure the
    caller is already handling, so every step is guarded.
    """
    try:
        record = failure_record(exc, provider=provider, elapsed_ms=elapsed_ms)
        log_event(logger or _LOGGER, EVENT_LLM_REQUEST_FAILED, **record)
    except Exception:  # pragma: no cover - diagnostics must never break a tick
        pass


__all__ = [
    "CATEGORY_AUTHENTICATION",
    "CATEGORY_CONNECTION",
    "CATEGORY_INVALID_REQUEST",
    "CATEGORY_PROVIDER_SERVER_ERROR",
    "CATEGORY_RATE_LIMIT_OR_QUOTA",
    "CATEGORY_TIMEOUT",
    "CATEGORY_UNKNOWN",
    "EVENT_LLM_REQUEST_FAILED",
    "FAILURE_CATEGORIES",
    "MAX_ELAPSED_MS",
    "MAX_ATTEMPT_COUNT",
    "MAX_RATE_LIMIT_VALUE",
    "MAX_TOKEN_LENGTH",
    "PHASE_CONNECT",
    "PHASE_READ",
    "RATE_LIMIT_HEADER_NAMES",
    "RETRY_AFTER_HEADER",
    "cause_class",
    "classify_failure",
    "configured_timeout_seconds",
    "failure_record",
    "http_status",
    "is_retryable",
    "log_planning_failure",
    "provider_error_code",
    "provider_identity",
    "provider_request_id",
    "rate_limit_headers",
    "safe_token",
    "transport_failure_info",
    "transport_phase",
]
