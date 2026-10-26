"""PHASE 22 viewer-facing web surface.

One small same-origin presentation layer in front of the existing runtime:

* :mod:`backend.viewer_web.server` serves the livestream page and a narrow,
  whitelist-built public projection of the real Brain/Chat/Stream state, and
  forwards viewer messages into the existing ``ChatService``.
* :mod:`backend.viewer_web.relay` owns the smallest browser-compatible
  presentation path for the real SRT livestream (ffmpeg stream-copy to HLS).

No second chat backend, no second stream pipeline, no viewer relationship
logic here: everything either reads the real runtime or calls the one existing
chat intake.
"""

from .relay import HlsRelay
from .server import ViewerWebServer

__all__ = ["HlsRelay", "ViewerWebServer"]
