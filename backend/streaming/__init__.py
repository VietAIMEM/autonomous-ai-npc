"""PHASE 13 real livestream transport.

The package holds the *transport* half of livestreaming. Unity's
``LivestreamSystem`` stays the single owner of logical stream state, the Brain
keeps issuing ``START_STREAM`` / ``STOP_STREAM``, and this layer turns the
frames and audio Unity already produces into a real stream.
"""

from .base import (
    AudioChunk,
    StreamHealth,
    StreamState,
    StreamingProvider,
    StreamingProviderError,
    VideoFrame,
)
from .config import StreamingConfig, build_provider
from .mock import MockStreamingProvider
from .service import StreamingService, build_streaming_service

__all__ = [
    "AudioChunk",
    "MockStreamingProvider",
    "StreamHealth",
    "StreamState",
    "StreamingConfig",
    "StreamingProvider",
    "StreamingProviderError",
    "StreamingService",
    "VideoFrame",
    "build_provider",
    "build_streaming_service",
]
