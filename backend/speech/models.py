"""Typed models for the Phase 11 speech layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..chat.models import format_utc, utc_now
from ..tts.base import TTSAudio


class SpeechState(str, Enum):
    SPEECH_REQUESTED = "SPEECH_REQUESTED"
    SYNTHESIZING = "SYNTHESIZING"
    READY = "READY"
    PLAYING = "PLAYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset(
    {SpeechState.COMPLETED, SpeechState.FAILED, SpeechState.CANCELLED}
)
# A request in one of these states already owns usable (or reserved) audio, so
# an identical request must never trigger a second provider generation.
IN_FLIGHT_STATES = frozenset(
    {SpeechState.SPEECH_REQUESTED, SpeechState.SYNTHESIZING, SpeechState.READY, SpeechState.PLAYING}
)


@dataclass
class SpeechMetrics:
    """Timestamps only for stages that really happened. No faked streaming."""

    requested_at: Optional[datetime] = None
    synthesis_started_at: Optional[datetime] = None
    audio_ready_at: Optional[datetime] = None
    playback_started_at: Optional[datetime] = None
    playback_completed_at: Optional[datetime] = None

    @staticmethod
    def _ms(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
        if start is None or end is None:
            return None
        return round(max(0.0, (end - start).total_seconds()) * 1000.0, 1)

    @property
    def time_to_first_audio_ms(self) -> Optional[float]:
        return self._ms(self.requested_at, self.audio_ready_at)

    @property
    def synthesis_ms(self) -> Optional[float]:
        return self._ms(self.synthesis_started_at, self.audio_ready_at)

    @property
    def playback_ms(self) -> Optional[float]:
        return self._ms(self.playback_started_at, self.playback_completed_at)

    @property
    def total_ms(self) -> Optional[float]:
        return self._ms(self.requested_at, self.playback_completed_at)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_at": format_utc(self.requested_at) if self.requested_at else None,
            "synthesis_started_at": (
                format_utc(self.synthesis_started_at) if self.synthesis_started_at else None
            ),
            "audio_ready_at": format_utc(self.audio_ready_at) if self.audio_ready_at else None,
            "playback_started_at": (
                format_utc(self.playback_started_at) if self.playback_started_at else None
            ),
            "playback_completed_at": (
                format_utc(self.playback_completed_at) if self.playback_completed_at else None
            ),
            "time_to_first_audio_ms": self.time_to_first_audio_ms,
            "synthesis_ms": self.synthesis_ms,
            "playback_ms": self.playback_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class SpeechRequest:
    request_id: str
    text: str
    text_key: str
    dedup_key: str
    source: str = "brain"
    plan_id: Optional[str] = None
    step_index: Optional[int] = None
    voice: str = ""
    model: str = ""
    output_format: str = "mp3"
    priority: float = 0.5
    interruptible: bool = True
    created_at: datetime = field(default_factory=utc_now)
    state: SpeechState = SpeechState.SPEECH_REQUESTED
    attempt: int = 1
    provider: str = "unknown"
    error: Optional[str] = None
    generation_id: Optional[str] = None
    reused_from: Optional[str] = None
    audio: Optional[TTSAudio] = None
    metrics: SpeechMetrics = field(default_factory=SpeechMetrics)

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def byte_length(self) -> int:
        return self.audio.byte_length if self.audio else 0

    @property
    def duration_seconds(self) -> Optional[float]:
        return self.audio.duration_seconds if self.audio else None

    @property
    def audio_format(self) -> Optional[str]:
        return self.audio.format if self.audio else None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "state": self.state.value,
            "source": self.source,
            "plan_id": self.plan_id,
            "step_index": self.step_index,
            "text": self.text[:200],
            "text_length": len(self.text),
            "provider": self.provider,
            "model": self.model,
            "voice": self.voice,
            "format": self.audio_format or self.output_format,
            "priority": round(self.priority, 3),
            "interruptible": self.interruptible,
            "attempt": self.attempt,
            "created_at": format_utc(self.created_at),
            "error": self.error,
            "reused_from": self.reused_from,
            "byte_length": self.byte_length,
            "duration_seconds": (
                round(self.duration_seconds, 3) if self.duration_seconds is not None else None
            ),
            "metrics": self.metrics.to_dict(),
        }
