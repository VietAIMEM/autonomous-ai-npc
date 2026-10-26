"""Dedicated Motius inference worker.

Reads one JSON request from stdin and emits one JSON response on stdout.
This module runs only inside the isolated Motius Python environment.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import time
import traceback


RESULT_PREFIX = "MOTIUS_RESULT "


def _ram_bytes() -> int:
    if sys.platform == "win32":
        class _MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)

    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _read_request() -> tuple[str, int, str]:
    try:
        request = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise ValueError("INVALID_REQUEST_JSON") from exc
    if not isinstance(request, dict):
        raise ValueError("INVALID_REQUEST")

    prompt = request.get("prompt")
    frame_count = request.get("frame_count")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("INVALID_PROMPT")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("INVALID_FRAME_COUNT")
    device = request.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("INVALID_DEVICE")
    return prompt.strip(), frame_count, device.strip()


def _finite_bounds(motion) -> dict[str, object]:
    values = [float(value) for row in motion for value in row]
    if not values:
        raise ValueError("EMPTY_MOTION")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("NONFINITE_MOTION")
    return {
        "finite": True,
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _run() -> None:
    import torch
    from motius import Pipeline

    prompt, frame_count, device = _read_request()
    checkpoint = os.environ.get(
        "ROOM_AGENT_MOTIUS_CHECKPOINT",
        r"D:\motius\model-cache\Motius-T2M-GPT-HumanML3D",
    )

    ram_before = _ram_bytes()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(checkpoint, bundle_kwargs={"device": device})
    model_load_s = time.perf_counter() - load_started

    generation_started = time.perf_counter()
    motions = pipeline.infer_text_to_motion([prompt], lengths=[frame_count])
    generation_s = time.perf_counter() - generation_started
    motion = motions[0]
    _finite_bounds(motion)
    ram_after = _ram_bytes()

    resource: dict[str, object] = {
        "model_load_s": model_load_s,
        "generation_s": generation_s,
        "ram_before_bytes": ram_before,
        "ram_after_bytes": ram_after,
        "device": device,
    }
    if device == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        resource.update(
            {
                "cuda_allocated_after_bytes": int(torch.cuda.memory_allocated()),
                "cuda_reserved_after_bytes": int(torch.cuda.memory_reserved()),
                "cuda_used_after_bytes": int(total - free),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )

    response = {
        "status": "OK",
        "motion": motion.tolist(),
        "resource": resource,
        "pipeline": type(pipeline).__name__,
    }
    print(RESULT_PREFIX + json.dumps(response, allow_nan=False), flush=True)


def main() -> int:
    try:
        _run()
        return 0
    except Exception as exc:
        response = {
            "status": "ERROR",
            "reason": type(exc).__name__,
            "detail": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(RESULT_PREFIX + json.dumps(response, allow_nan=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
