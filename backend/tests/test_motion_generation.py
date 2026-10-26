"""PHASE 15 motion generation tests. Everything here runs offline."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional

from backend.api.brain import create_brain_server
from backend.learning.models import (
    MotionDefinition,
    MotionSource,
    MotionValidationStatus,
    SkeletonCompatibility,
)
from backend.motion_generation import (
    MotionGenerationConfig,
    MotionGenerationError,
    MotionGenerationRequest,
    MotionGenerationResult,
    MotionGenerationStatus,
    MotionGenerationService,
    MockMotionGenerator,
    MotiusT2MGPTGenerator,
    MotionGenerator,
    JSONMotionGenerationMetadataStore,
    validate_motion_payload,
)


PROVIDER_MOCK = "mock"


def _result(
    request: MotionGenerationRequest,
    *,
    motion: Any = None,
    motion_id: str = "motion_generated",
) -> MotionGenerationResult:
    if motion is None:
        motion = tuple(
            tuple(0.0 for _ in range(263))
            for _ in range(round(request.duration_seconds * 20))
        )
    return MotionGenerationResult(
        request=request,
        motion_id=motion_id,
        duration_seconds=len(motion) / 20.0,
        format="humanml3d-263",
        motion=motion,
    )


class FakeMotionGenerator(MotionGenerator):
    name = "fake"

    def __init__(
        self,
        result: Optional[MotionGenerationResult] = None,
        error: Optional[BaseException] = None,
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        self.result = result
        self.error = error
        self.delay_seconds = delay_seconds
        self.calls: list[MotionGenerationRequest] = []
        self.active = 0
        self.max_active = 0

    def generate(self, request: MotionGenerationRequest) -> MotionGenerationResult:
        self.calls.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            if self.error is not None:
                raise self.error
            return self.result or _result(request)
        finally:
            self.active -= 1


class MotionGeneratorTests(unittest.TestCase):
    def test_motion_generator_abstraction_is_shared(self) -> None:
        self.assertTrue(issubclass(MockMotionGenerator, MotionGenerator))
        self.assertTrue(issubclass(MotiusT2MGPTGenerator, MotionGenerator))

        motius = MotiusT2MGPTGenerator(MotionGenerationConfig(provider="motius"))
        described = motius.describe()
        for key in ("provider", "model", "checkpoint", "device", "loaded", "representation"):
            self.assertIn(key, described)
        self.assertFalse(described["loaded"])

    def test_motius_provider_runs_in_isolated_subprocess(self) -> None:
        motius = MotiusT2MGPTGenerator(
            MotionGenerationConfig(
                provider="motius",
                device="cpu",
                timeout_seconds=1.0,
            )
        )
        response = {
            "status": "OK",
            "motion": [[0.0] * 263 for _ in range(50)],
            "resource": {"generation_s": 1.2},
        }
        process = unittest.mock.Mock()
        process.communicate.return_value = (
            f"MOTIUS_RESULT {json.dumps(response)}",
            "",
        )
        process.returncode = 0

        with unittest.mock.patch(
            "backend.motion_generation.motius.subprocess.Popen",
            return_value=process,
        ) as popen:
            result = motius.generate(
                MotionGenerationRequest(
                    prompt="a person raises one hand and waves",
                    duration_seconds=2.5,
                )
            )

        self.assertEqual(result.generation_status, MotionGenerationStatus.GENERATED)
        self.assertEqual(result.format, "humanml3d-263")
        self.assertEqual(len(result.motion), 50)
        self.assertEqual(popen.call_count, 1)
        command = popen.call_args.args[0]
        self.assertIn(
            "motius_worker.py",
            [Path(part).name for part in command],
        )
        self.assertIn("motius", popen.call_args.kwargs["cwd"])
        self.assertIn("a person raises one hand and waves", process.communicate.call_args.kwargs["input"])
        self.assertTrue(result.metadata["resource"])

    def test_motius_provider_timeout_fails_safely(self) -> None:
        motius = MotiusT2MGPTGenerator(
            MotionGenerationConfig(provider="motius", timeout_seconds=0.1)
        )
        process = unittest.mock.Mock()
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("motius", 0.1),
            ("", ""),
        ]

        with unittest.mock.patch(
            "backend.motion_generation.motius.subprocess.Popen",
            return_value=process,
        ):
            with self.assertRaises(MotionGenerationError) as context:
                motius.generate(MotionGenerationRequest(prompt="a person waves"))

        self.assertEqual(context.exception.reason, "timeout")
        self.assertEqual(process.communicate.call_count, 2)

    def test_motius_provider_worker_failure_fails_safely(self) -> None:
        motius = MotiusT2MGPTGenerator(MotionGenerationConfig(provider="motius"))
        response = {
            "status": "ERROR",
            "reason": "MODEL_UNAVAILABLE",
        }
        process = unittest.mock.Mock()
        process.communicate.return_value = (
            f"MOTIUS_RESULT {json.dumps(response)}",
            "",
        )
        process.returncode = 1

        with unittest.mock.patch(
            "backend.motion_generation.motius.subprocess.Popen",
            return_value=process,
        ):
            with self.assertRaises(MotionGenerationError) as context:
                motius.generate(MotionGenerationRequest(prompt="a person waves"))

        self.assertEqual(context.exception.reason, "generation_failed")

    def test_mock_generator_returns_valid_motion(self) -> None:
        generator = MockMotionGenerator(MotionGenerationConfig(provider=PROVIDER_MOCK))
        result = generator.generate(
            MotionGenerationRequest(
                prompt="a person raises one hand and waves",
                duration_seconds=2.5,
            )
        )

        self.assertEqual(result.generation_status, MotionGenerationStatus.GENERATED)
        self.assertEqual(result.format, "humanml3d-263")
        self.assertEqual(len(result.motion), 50)
        self.assertTrue(all(len(row) == 263 for row in result.motion))
        self.assertAlmostEqual(result.duration_seconds, 2.5)
        validate_motion_payload(
            result.motion,
            frame_rate=20.0,
            expected_duration_seconds=2.5,
        )

    def test_motion_payload_validation_rejects_invalid_output(self) -> None:
        with self.assertRaises(MotionGenerationError) as context:
            validate_motion_payload([])
        self.assertEqual(context.exception.reason, "EMPTY_MOTION_OUTPUT")

        with self.assertRaises(MotionGenerationError) as context:
            validate_motion_payload([[0.0] * 262])
        self.assertEqual(context.exception.reason, "INVALID_MOTION_DIMENSIONS")

        with self.assertRaises(MotionGenerationError) as context:
            validate_motion_payload([[math.nan] + [0.0] * 262])
        self.assertEqual(context.exception.reason, "INVALID_MOTION_VALUE")

        with self.assertRaises(MotionGenerationError) as context:
            validate_motion_payload([[math.inf] + [0.0] * 262])
        self.assertEqual(context.exception.reason, "INVALID_MOTION_VALUE")


class MotionGenerationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _service(
        self,
        *,
        config: MotionGenerationConfig | None = None,
        generator: MotionGenerator | None = None,
        metadata_store: Any = None,
    ) -> MotionGenerationService:
        return MotionGenerationService(
            generator or FakeMotionGenerator(),
            config or MotionGenerationConfig(provider=PROVIDER_MOCK),
            metadata_store=metadata_store,
            start_worker=False,
        )

    def test_request_validation_rejects_unsafe_and_unbounded_requests(self) -> None:
        service = self._service(
            config=MotionGenerationConfig(provider=PROVIDER_MOCK, max_prompt_length=20)
        )

        cases = [
            ("", "empty_prompt"),
            ("a" * 21, "prompt_too_long"),
            ("import os and dance", "unsafe_prompt"),
            ("dance", "duration_out_of_bounds"),
            ("dance", "speed_out_of_bounds"),
        ]
        durations = [2.0, 2.0, 2.0, 6.0, 2.0]
        speeds = [1.0, 1.0, 1.0, 1.0, 0.0]
        for index, (prompt, expected_error) in enumerate(cases):
            job = service.request(prompt, duration_seconds=durations[index], speed=speeds[index])
            self.assertEqual(job.error, expected_error)
            self.assertEqual(job.state, MotionGenerationStatus.FAILED)

    def test_queue_bounds_reject_overflow(self) -> None:
        service = self._service(
            config=MotionGenerationConfig(provider=PROVIDER_MOCK, max_queue_depth=1)
        )

        first = service.request("a person waves")
        second = service.request("a person walks")
        self.assertEqual(service.pending_count(), 1)
        self.assertEqual(second.state, MotionGenerationStatus.FAILED)
        self.assertEqual(second.error, "generation_queue_full")
        self.assertTrue(first.state in {MotionGenerationStatus.QUEUED, MotionGenerationStatus.GENERATING})

    def test_single_concurrency_is_enforced(self) -> None:
        generator = FakeMotionGenerator(delay_seconds=0.01)
        service = self._service(
            config=MotionGenerationConfig(provider=PROVIDER_MOCK, max_queue_depth=2),
            generator=generator,
        )

        service.request("a person waves")
        service.request("a person walks")
        service.run_pending_once()
        service.run_pending_once()

        self.assertEqual(generator.max_active, 1)
        self.assertEqual(len(generator.calls), 2)

    def test_generation_timeout_fails_safely(self) -> None:
        generator = FakeMotionGenerator(error=MotionGenerationError("timeout"))
        service = self._service(generator=generator)

        job = service.request("a person waves")
        service.run_pending_once()

        refreshed = service.get(job.request_id)
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed.state, MotionGenerationStatus.FAILED)
        self.assertEqual(refreshed.error, "timeout")
        self.assertIsNone(refreshed.result)

    def test_provider_failure_fails_safely(self) -> None:
        generator = FakeMotionGenerator(error=MotionGenerationError("model_unavailable"))
        service = self._service(generator=generator)

        job = service.request("a person waves")
        service.run_pending_once()

        refreshed = service.get(job.request_id)
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed.state, MotionGenerationStatus.FAILED)
        self.assertEqual(refreshed.error, "model_unavailable")
        self.assertIsNone(refreshed.result)

    def test_invalid_output_fails_safely(self) -> None:
        request = MotionGenerationRequest(prompt="a person waves", duration_seconds=0.1)
        invalid_motion = tuple(tuple(0.0 for _ in range(262)) for _ in range(2))
        generator = FakeMotionGenerator(_result(request, motion=invalid_motion))
        service = self._service(generator=generator)

        job = service.request("a person waves")
        service.run_pending_once()

        refreshed = service.get(job.request_id)
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed.state, MotionGenerationStatus.FAILED)
        self.assertEqual(refreshed.error, "INVALID_MOTION_DIMENSIONS")
        self.assertIsNone(refreshed.result)

    def test_metadata_persistence_does_not_store_raw_motion(self) -> None:
        metadata_path = self.directory / "generated_motions.json"
        service = self._service(
            metadata_store=JSONMotionGenerationMetadataStore(metadata_path)
        )

        job = service.request("a person waves")
        service.run_pending_once()

        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1)
        self.assertNotIn("motion", records[0])
        self.assertEqual(records[0]["format"], "humanml3d-263")
        self.assertEqual(records[0]["validation_status"], "VALID")
        self.assertEqual(service.get(job.request_id).result.motion_id, records[0]["motion_id"])

    def test_versioning_creates_new_generation_for_repeat(self) -> None:
        service = MockMotionGenerator(MotionGenerationConfig(provider=PROVIDER_MOCK))

        first = service.generate(MotionGenerationRequest(prompt="a person waves"))
        second = service.generate(MotionGenerationRequest(prompt="a person waves"))

        self.assertNotEqual(first.motion_id, second.motion_id)
        self.assertEqual(first.duration_seconds, second.duration_seconds)

    def test_diagnostics_report_required_fields_without_secrets(self) -> None:
        service = self._service()

        job = service.request("a person waves")
        service.run_pending_once()

        state = service.get_state()
        health = service.health()
        for key in (
            "generator",
            "model",
            "device",
            "loaded",
            "queue_depth",
            "max_concurrent_generations",
            "last_generation_ms",
            "last_motion_length",
            "last_error",
        ):
            self.assertIn(key, state)
        self.assertIn("status", health)
        self.assertIn("last_error", health)

        encoded = json.dumps(state, ensure_ascii=False)
        self.assertNotIn("api_key", encoded)
        self.assertNotIn("Authorization", encoded)

    def test_motion_definition_boundary_is_preserved(self) -> None:
        service = self._service()

        job = service.request("a person waves")
        service.run_pending_once()
        result = service.get(job.request_id).result

        motion = MotionDefinition(
            motion_id=result.motion_id,
            name="Generated Wave",
            description="Generated wave motion.",
            animation_reference="humanml3d-263",
            duration_seconds=result.duration_seconds,
            source=MotionSource.GENERATED,
            skeleton_status=SkeletonCompatibility.UNKNOWN,
            validation_status=MotionValidationStatus.PENDING,
            enabled=False,
        )

        self.assertEqual(motion.source, MotionSource.GENERATED)
        self.assertEqual(motion.skeleton_status, SkeletonCompatibility.UNKNOWN)
        self.assertEqual(motion.validation_status, MotionValidationStatus.PENDING)
        self.assertFalse(motion.enabled)


class MotionGenerationDiagnosticsRouteTests(unittest.TestCase):
    def test_diagnostics_route_exposes_state_and_health(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        service = MotionGenerationService(
            FakeMotionGenerator(),
            MotionGenerationConfig(provider=PROVIDER_MOCK),
            metadata_store=Path(directory.name) / "generated_motions.json",
            start_worker=False,
        )
        service.request("a person waves")
        service.run_pending_once()

        server = create_brain_server(None, "127.0.0.1", 0, motion_generation=service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # LIFO cleanups: stop serving, join, then close the socket.
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        host, port = server.server_address
        base = f"http://{host}:{port}"

        with urllib.request.urlopen(f"{base}/motion-generation/state", timeout=2) as response:
            state = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base}/motion-generation/health", timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))

        self.assertEqual(state["generator"], "fake")
        self.assertIn("queue_depth", state)
        self.assertEqual(health["generator"], "fake")
        self.assertIn("status", health)


if __name__ == "__main__":
    unittest.main()
