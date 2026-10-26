"""HTTP surface for the PHASE 13 real livestream transport.

Served from the Brain API port, which is the one Unity already talks to for
speech. ``/stream/start`` and ``/stream/stop`` stay downstream of the existing
``START_STREAM`` / ``STOP_STREAM`` actions; ``/stream/media/*`` accept captured
bytes from Unity. The routing logic lives in :func:`handle_stream_request` so it
is testable without opening a socket.

Nothing here can return a stream key: the config view is the sanitized
``StreamingConfig.to_dict()``, and diagnostics only carry counters and states.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Mapping, Optional
from urllib.parse import parse_qs

from ..runtime.stream_latch import StreamStopLatch
from ..streaming.service import StreamingService, build_streaming_service


#: Hard ceiling for one media POST. Frames are capped far below this by the
#: provider; the number exists so a confused client cannot pin the server.
MAX_MEDIA_BODY_BYTES = 2_000_000


def disabled_state() -> dict[str, Any]:
    """Answer Unity gets when no real transport is configured.

    ``ok`` stays true for logical commands: PHASE 01-12 behaviour, where the
    livestream is a simulation, must not start failing because transport is off.
    """
    return {
        "enabled": False,
        "provider": "disabled",
        "platform": "none",
        "state": "STOPPED",
        "live": False,
        "transport_enabled": False,
        "real_transport": False,
        "ok": True,
        "reason": "TRANSPORT_DISABLED",
    }


def handle_stream_request(
    streaming: Optional[StreamingService],
    *,
    path: str,
    method: str = "GET",
    query: Mapping[str, Any] | None = None,
    body: bytes = b"",
    latch: Optional[StreamStopLatch] = None,
) -> tuple[HTTPStatus, dict[str, Any]]:
    """Route one ``/stream/*`` request. Returns a JSON-serialisable payload.

    ``latch`` is optional so transport-less tests keep the old call shape. When
    it is wired in, an explicit control-plane stop latches the deliberate stop
    and an explicit control-plane start clears it, matching the Brain's own
    ``STOP_STREAM`` / ``START_STREAM`` latch.
    """
    params = _flatten(query or {})
    kind = (path or "").rstrip("/").lower()

    if kind in {"/stream/state", "/stream/health"}:
        if streaming is None:
            return HTTPStatus.OK, disabled_state()
        if kind == "/stream/health":
            payload = streaming.health()
            payload["enabled"] = True
            return HTTPStatus.OK, payload
        return HTTPStatus.OK, streaming.get_state()

    if streaming is None:
        # A transport-less backend still answers commands politely so Unity can
        # back off instead of retrying every tick.
        payload = disabled_state()
        payload["transport"] = dict(payload)
        return HTTPStatus.OK, payload

    if kind == "/stream/start" and method == "POST":
        wait = _as_float(params.get("wait"), streaming.config.live_timeout_seconds)
        if latch is not None:
            # PHASE 24: the operator's start is current intent, so it must
            # release the deliberate stop before the transport wait. The
            # control plane cannot feed frames itself; clearing first is what
            # lets the Brain's stream-maintenance goal re-arm Unity's capture
            # pump during this same start (and after a failed one), instead of
            # leaving the stream wedged behind an engaged latch.
            latch.clear(source="api:/stream/start")
        result = streaming.request_start(camera=params.get("camera", ""), wait_seconds=wait)
        health = streaming.health()
        return HTTPStatus.OK, {
            "ok": result.ok,
            "state": result.state.value,
            "reason": result.reason,
            "waited_ms": result.waited_ms,
            "transport_enabled": result.transport_enabled,
            "live": health["live"],
            "provider": streaming.provider.name,
            "platform": getattr(streaming.provider, "platform", "unknown"),
            "ingest": streaming.ingest_profile(),
        }

    if kind == "/stream/stop" and method == "POST":
        wait = _as_float(params.get("wait"), 5.0)
        result = streaming.request_stop(wait_seconds=wait)
        if latch is not None:
            # An explicit control-plane stop is deliberate even when the
            # encoder needs another moment; shutdown calls ``stop()`` directly
            # and never reaches this route, so a graceful exit cannot latch.
            latch.record_stop(source="api:/stream/stop")
        return HTTPStatus.OK, {
            "ok": result.ok,
            "state": result.state.value,
            "reason": result.reason,
            "waited_ms": result.waited_ms,
            "transport_enabled": result.transport_enabled,
            "live": False,
        }

    if kind == "/stream/media/video" and method == "POST":
        status, complaint = _check_body(body, streaming.provider.max_frame_bytes)
        if status is not None:
            return status, complaint
        sequence = _as_int(params.get("seq"), 0)
        result = streaming.submit_video(body, sequence=sequence, camera=params.get("camera", ""))
        return _media_reply(result)

    if kind == "/stream/media/audio" and method == "POST":
        status, complaint = _check_body(body, streaming.provider.max_audio_bytes)
        if status is not None:
            return status, complaint
        sequence = _as_int(params.get("seq"), 0)
        result = streaming.submit_audio(
            body,
            sequence=sequence,
            sample_rate=_as_int(params.get("rate"), 0) or None,
            channels=_as_int(params.get("channels"), 0) or None,
        )
        return _media_reply(result)

    return HTTPStatus.NOT_FOUND, {"error": "not_found"}


def is_stream_path(path: str) -> bool:
    return (path or "").startswith("/stream/")


def _media_reply(result: Any) -> tuple[HTTPStatus, dict[str, Any]]:
    # ``NOT_STREAMING`` is an instruction for Unity to stop capturing, not an
    # error: answering 200 keeps a stopped transport from looking like an outage.
    status = HTTPStatus.OK if (result.accepted or result.reason in {"NOT_STREAMING", "DUPLICATE"}) else HTTPStatus.BAD_REQUEST
    return status, result.to_dict()


def _check_body(body: bytes, limit: int) -> tuple[Optional[HTTPStatus], dict[str, Any]]:
    if len(body) > MAX_MEDIA_BODY_BYTES:
        return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_too_large"}
    if not body:
        return HTTPStatus.BAD_REQUEST, {"error": "empty_body"}
    if len(body) > limit:
        return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "media_too_large", "limit": limit}
    return None, {}


def _flatten(query: Mapping[str, Any]) -> dict[str, str]:
    if hasattr(query, "getlist"):
        return {key: values[0] for key, values in query.items() if values}  # type: ignore[union-attr]
    return {
        key: (value[0] if isinstance(value, (list, tuple)) and value else str(value))
        for key, value in query.items()
    }


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def build_streaming_source() -> Optional[StreamingService]:
    """Create the configured transport service, or ``None`` when it is off."""
    return build_streaming_service()


__all__ = [
    "MAX_MEDIA_BODY_BYTES",
    "build_streaming_source",
    "disabled_state",
    "handle_stream_request",
    "is_stream_path",
]
