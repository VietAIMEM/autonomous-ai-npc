"""PHASE 20 health vocabulary, unified diagnostics and bounded-buffer tests.

Everything here is offline: probes read local state only, and the unified
status surface is checked for both accuracy and secret-free output.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.api.chat import chat_health
from backend.api.diagnostics import build_diagnostics, build_health, build_status, queue_depths
from backend.chat.models import ChatStatus
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.motion_generation import (
    JSONMotionGenerationMetadataStore,
    MockMotionGenerator,
    MotionGenerationConfig,
    MotionGenerationService,
)
from backend.runtime.config import RuntimeConfig
from backend.runtime.health import (
    STATUS_DEGRADED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    build_report,
    classify,
    component,
    from_legacy,
    normalize_status,
    overall,
    sqlite_probe,
    worst_of,
)
from backend.runtime.metrics import MetricsRegistry


SECRET = "sk-live-phase20diagsecret000000"


def make_config(directory: Path, environment: str = "test") -> RuntimeConfig:
    return RuntimeConfig(
        environment=environment,
        data_dir=directory / "data",
        log_dir=directory / "logs",
        temp_dir=directory / "temp",
        backup_dir=directory / "backups",
        state_dir=directory / "state",
    )


class HealthVocabularyTests(unittest.TestCase):
    def test_classify_keeps_configured_reachable_and_healthy_apart(self) -> None:
        self.assertEqual(
            classify(configured=False, intentional=True), STATUS_UNAVAILABLE
        )
        self.assertEqual(classify(configured=True, reachable=False), STATUS_UNAVAILABLE)
        self.assertEqual(
            classify(configured=True, reachable=True, healthy=False), STATUS_DEGRADED
        )
        self.assertEqual(
            classify(configured=True, reachable=True, healthy=True), STATUS_READY
        )
        # Reachable but never probe-verified is degraded, not a silent success.
        self.assertEqual(classify(configured=True), STATUS_DEGRADED)

    def test_normalize_maps_legacy_words(self) -> None:
        self.assertEqual(normalize_status("OK"), STATUS_READY)
        self.assertEqual(normalize_status("LIVE"), STATUS_READY)
        self.assertEqual(normalize_status("reconnecting"), STATUS_DEGRADED)
        self.assertEqual(normalize_status("stopped"), STATUS_UNAVAILABLE)
        self.assertIsNone(normalize_status("nonsense"))
        self.assertIsNone(normalize_status(None))

    def test_component_payload_records_every_question(self) -> None:
        entry = component(
            "tts",
            configured=True,
            reachable=True,
            healthy=False,
            required=False,
            intentional=False,
            detail={"queue_length": 3},
            reason="provider_error",
        )

        self.assertEqual(entry["status"], STATUS_DEGRADED)
        self.assertTrue(entry["configured"])
        self.assertTrue(entry["reachable"])
        self.assertFalse(entry["healthy"])
        self.assertFalse(entry["required"])
        self.assertEqual(entry["detail"], {"queue_length": 3})
        self.assertEqual(entry["reason"], "provider_error")

    def test_from_legacy_adapts_subsystem_payloads(self) -> None:
        ready = from_legacy("tts", {"status": "ok", "enabled": True})
        self.assertEqual(ready["status"], STATUS_READY)

        stopped = from_legacy("stream", {"status": "stopped", "enabled": True})
        self.assertEqual(stopped["status"], STATUS_UNAVAILABLE)
        self.assertFalse(stopped["intentional"])

        disabled = from_legacy("tts", {"enabled": False, "status": "disabled"})
        self.assertEqual(disabled["status"], STATUS_UNAVAILABLE)
        self.assertFalse(disabled["configured"])

        missing = from_legacy("motion", None, default_configured=False)
        self.assertEqual(missing["status"], STATUS_UNAVAILABLE)
        self.assertEqual(missing["reason"], "no_health_payload")

    def test_from_legacy_reads_real_state_machine_payloads(self) -> None:
        # Shape of the live Twitch chat source probe (``state``, not ``status``).
        connected = from_legacy(
            "chat",
            {
                "ok": True,
                "provider": "twitch",
                "platform": "twitch",
                "state": "CONNECTED",
                "degraded": False,
                "reconnect_count": 0,
                "last_error": None,
            },
        )
        self.assertEqual(connected["status"], STATUS_READY)
        self.assertNotIn("reason", connected)

        # A live stream keeps the stale ``NO_MEDIA`` error from before its
        # first frame; the healthy verdict must not surface it as a reason.
        live_stream = from_legacy(
            "stream",
            {
                "state": "LIVE",
                "ok": True,
                "transport_enabled": True,
                "encoder_fps": 15.0,
                "dropped_frames": 0,
                "last_error": "NO_MEDIA",
            },
            enabled_key="transport_enabled",
        )
        self.assertEqual(live_stream["status"], STATUS_READY)
        self.assertNotIn("reason", live_stream)
        self.assertEqual(live_stream["detail"]["last_error"], "NO_MEDIA")

        idle_stream = from_legacy(
            "stream",
            {"state": "IDLE", "ok": False, "transport_enabled": True, "last_error": "NO_MEDIA"},
            enabled_key="transport_enabled",
        )
        self.assertEqual(idle_stream["status"], STATUS_DEGRADED)
        self.assertEqual(idle_stream["reason"], "NO_MEDIA")

        retrying = from_legacy(
            "chat",
            {
                "ok": True,
                "state": "RETRYING",
                "degraded": True,
                "last_error": "connection reset",
            },
        )
        self.assertEqual(retrying["status"], STATUS_DEGRADED)
        self.assertEqual(retrying["reason"], "connection reset")

        degraded_connected = from_legacy(
            "chat", {"ok": True, "state": "CONNECTED", "degraded": True}
        )
        self.assertEqual(degraded_connected["status"], STATUS_DEGRADED)

        disconnected = from_legacy("chat", {"ok": False, "state": "DISCONNECTED"})
        self.assertEqual(disconnected["status"], STATUS_UNAVAILABLE)

        stopped = from_legacy(
            "stream",
            {"state": "STOPPED", "ok": False, "transport_enabled": True},
            enabled_key="transport_enabled",
        )
        self.assertEqual(stopped["status"], STATUS_UNAVAILABLE)

        # ``enabled: false`` stays unavailable even if the probe also says ok.
        disabled = from_legacy(
            "stream",
            {"state": "LIVE", "ok": True, "transport_enabled": False},
            enabled_key="transport_enabled",
        )
        self.assertEqual(disabled["status"], STATUS_UNAVAILABLE)
        self.assertFalse(disabled["configured"])

    def test_overall_ignores_optional_components_and_worst_of_orders(self) -> None:
        components = {
            "llm": component("llm", configured=True, reachable=True, healthy=True),
            "unity": component(
                "unity",
                configured=False,
                required=False,
                intentional=True,
                status=STATUS_UNAVAILABLE,
            ),
        }
        self.assertEqual(overall(components), STATUS_READY)

        components["tts"] = component(
            "tts", configured=True, reachable=True, healthy=False
        )
        self.assertEqual(overall(components), STATUS_DEGRADED)
        self.assertEqual(
            worst_of([STATUS_READY, STATUS_UNAVAILABLE, STATUS_DEGRADED]),
            STATUS_UNAVAILABLE,
        )

    def test_build_report_counts_issues_and_ready_flag(self) -> None:
        components = {
            "backend": component("backend", configured=True, reachable=True, healthy=True),
            "chat": component(
                "chat",
                configured=True,
                reachable=True,
                healthy=False,
                reason="queue_full",
            ),
        }

        report = build_report(
            components, environment="test", uptime_seconds=12.5
        )

        self.assertEqual(report["status"], STATUS_DEGRADED)
        self.assertFalse(report["ready"])
        self.assertEqual(report["counts"][STATUS_READY], 1)
        self.assertEqual(report["counts"][STATUS_DEGRADED], 1)
        self.assertEqual(report["issues"], [{"component": "chat", "status": STATUS_DEGRADED, "reason": "queue_full"}])
        self.assertEqual(report["uptime_seconds"], 12.5)


class SqliteProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def _database(self, name: str = "memory.db") -> Path:
        path = self.directory / name
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, body TEXT)")
            connection.execute("INSERT INTO episodes (body) VALUES ('hello')")
            connection.commit()
        finally:
            connection.close()
        return path

    def test_missing_database_is_unavailable_not_ready(self) -> None:
        probe = sqlite_probe(self.directory / "absent.db")

        self.assertEqual(probe["status"], STATUS_UNAVAILABLE)
        self.assertFalse(probe["configured"])
        self.assertEqual(probe["reason"], "database_missing")

    def test_healthy_database_probes_ready_with_cheap_read(self) -> None:
        probe = sqlite_probe(self._database())

        self.assertEqual(probe["status"], STATUS_READY)
        self.assertTrue(probe["reachable"])
        self.assertTrue(probe["healthy"])
        self.assertGreaterEqual(probe["detail"]["schema_version"], 1)
        self.assertGreater(probe["detail"]["size_bytes"], 0)

    def test_quick_check_reports_integrity(self) -> None:
        probe = sqlite_probe(self._database(), quick_check=True)

        self.assertEqual(probe["status"], STATUS_READY)
        self.assertEqual(probe["detail"]["quick_check"], "ok")

    def test_corrupt_file_is_unavailable_not_a_fake_success(self) -> None:
        path = self.directory / "broken.db"
        path.write_bytes(b"this is not a sqlite database at all")

        probe = sqlite_probe(path)

        self.assertEqual(probe["status"], STATUS_UNAVAILABLE)
        self.assertFalse(probe["reachable"])


class MetricsBoundsTests(unittest.TestCase):
    def test_series_limit_drops_new_names_but_keeps_existing_ones(self) -> None:
        registry = MetricsRegistry(max_series=2)

        registry.counter("chat.submitted")
        registry.counter("chat.submitted")
        registry.counter("tts.requests")
        self.assertEqual(registry.counter("overflow.series"), 0)

        self.assertEqual(registry.counter_value("chat.submitted"), 2)
        self.assertEqual(registry.counter_value("overflow.series"), 0)
        self.assertEqual(registry.snapshot()["dropped_series"], 1)

    def test_latency_history_is_a_fixed_size_window(self) -> None:
        registry = MetricsRegistry(max_samples=4)
        for value in range(1, 11):
            registry.observe("brain.tick_ms", float(value))

        summary = registry.snapshot()["latency_ms"]["brain.tick_ms"]

        self.assertEqual(summary["count"], 4.0)
        self.assertEqual(summary["min"], 7.0)
        self.assertEqual(summary["max"], 10.0)
        self.assertEqual(summary["last"], 10.0)

    def test_error_ledger_is_bounded_and_redacted(self) -> None:
        registry = MetricsRegistry(max_errors=3)
        for index in range(5):
            registry.record_error("llm", f"attempt {index} failed: api_key={SECRET}")

        snapshot = registry.snapshot()

        self.assertEqual(snapshot["errors_by_component"]["llm"], 5)
        self.assertEqual(len(snapshot["recent_errors"]), 3)
        serialized = json.dumps(snapshot)
        self.assertNotIn(SECRET, serialized)
        self.assertIn("[redacted]", serialized)

    def test_reset_clears_counters_and_limits(self) -> None:
        registry = MetricsRegistry(max_series=2)
        registry.counter("a")
        registry.counter("b")
        registry.counter("c")
        registry.reset()

        snapshot = registry.snapshot()

        self.assertEqual(snapshot["counter_count"], 0)
        self.assertEqual(snapshot["dropped_series"], 0)
        self.assertEqual(registry.counter("a"), 1)


class _MemoryStub:
    def __init__(self, state: dict) -> None:
        self._state = state

    def get_state(self) -> dict:
        return dict(self._state)

    def personality_diagnostics(self) -> dict:
        return {"dominant_trait": "curious"}


class _ProviderWithSecret:
    """A provider object that *holds* a key; diagnostics must not echo it."""

    has_api_key = True
    api_key = SECRET
    name = "openrouter"

    def __init__(self) -> None:
        self.config = SimpleNamespace(model="openai/gpt-5-nano")


class DiagnosticsPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.config = make_config(self.directory)
        self.addCleanup(self._directory.cleanup)

    def _service(
        self,
        provider=None,
        memory_state: dict | None = None,
        runtime: dict | None = None,
        unity_age: float | None = None,
    ) -> SimpleNamespace:
        metrics = {
            "enabled": True,
            "tick_count": 4,
            "tick_in_progress": False,
            "last_tick_age_seconds": 0.5,
        }
        if runtime:
            metrics.update(runtime)
        service = SimpleNamespace(
            brain=SimpleNamespace(provider=provider or MockLLMProvider()),
            memory_service=_MemoryStub(memory_state) if memory_state else None,
            runtime_metrics=lambda: dict(metrics),
        )
        if unity_age is not None:
            service.last_tick_age_seconds = lambda: float(unity_age)
        return service

    def test_status_is_compact_and_never_raises_with_a_bare_service(self) -> None:
        payload = build_status(self._service(), config=self.config)

        self.assertIn(payload["status"], {STATUS_READY, STATUS_DEGRADED, STATUS_UNAVAILABLE})
        self.assertEqual(payload["process"]["brain_port"], self.config.brain_port)
        self.assertEqual(payload["brain"]["tick_count"], 4)
        self.assertIn("queues", payload)
        self.assertIn("recent_errors", payload)
        self.assertEqual(payload["health"]["backend"], STATUS_READY)

    def test_status_headline_matches_health_when_stream_is_intentionally_stopped(self) -> None:
        streaming = SimpleNamespace(
            health=lambda: {
                "state": "STOPPED",
                "ok": False,
                "transport_enabled": True,
                "last_error": "NO_MEDIA",
            },
            get_state=lambda: {
                "state": "STOPPED",
                "queues": {},
                "transport_enabled": True,
            },
        )
        service = self._service()
        status = build_status(service, config=self.config, streaming=streaming)
        health = build_health(service, config=self.config, streaming=streaming)

        # The deliberate stop stays visible per-subsystem...
        self.assertEqual(status["health"]["stream"], STATUS_UNAVAILABLE)
        # ...but it is not a runtime fault on either surface.
        self.assertEqual(status["status"], health["status"])
        entry = health["components"]["stream"]
        self.assertFalse(entry["required"])
        self.assertTrue(entry["intentional"])
        # The stale start failure stays visible as an error, never as the
        # current reason for a stream the operator stopped on purpose.
        self.assertEqual(entry["reason"], "intentional_stop")
        self.assertEqual(entry["detail"]["last_error"], "NO_MEDIA")

    def test_durable_stop_latch_keeps_idle_transport_intentional_after_restart(self) -> None:
        # After a restart the transport boots IDLE even though the durable
        # stop-latch still records the operator's deliberate stop. The intent,
        # not the transient transport word, decides whether the stream is a
        # required subsystem (REAL: post-restart health read DEGRADED forever
        # while the latch was correctly honouring the stop).
        streaming = SimpleNamespace(
            health=lambda: {
                "state": "IDLE",
                "ok": False,
                "transport_enabled": True,
                "reason": "TIMEOUT",
            },
            get_state=lambda: {
                "state": "IDLE",
                "queues": {},
                "transport_enabled": True,
            },
        )
        latch = {
            "stopped": True,
            "persistent": True,
            "loaded": True,
            "source": "api:/stream/stop",
        }
        health = build_health(
            self._service(), config=self.config, streaming=streaming, stream_latch=latch
        )
        entry = health["components"]["stream"]
        # "Off on purpose" reads the same as a fresh stop request, so the
        # Admin Dashboard never shows a deliberate shutdown as an impairment.
        self.assertEqual(entry["status"], STATUS_UNAVAILABLE)
        self.assertTrue(entry["intentional"])
        self.assertFalse(entry["required"])
        self.assertEqual(entry["reason"], "intentional_stop")
        self.assertEqual(entry["detail"]["last_error"], "TIMEOUT")
        self.assertEqual(health["status"], STATUS_READY)

        # A healthy latch-less idle transport must still be a real fault.
        unlatched = build_health(self._service(), config=self.config, streaming=streaming)
        self.assertFalse(unlatched["components"]["stream"]["intentional"])
        self.assertTrue(unlatched["components"]["stream"]["required"])

    def test_unity_stale_threshold_tolerates_planner_bound_tick_cadence(self) -> None:
        # REAL: planning p95 measured at ~26-31 s plus the 5 s Unity decision
        # interval (max observed tick age 41 s), so the old 30 s threshold
        # flapped a genuinely ticking Unity to DEGRADED.
        ready = build_health(self._service(unity_age=45.0), config=self.config)
        self.assertEqual(ready["components"]["unity"]["status"], STATUS_READY)
        self.assertEqual(
            ready["components"]["unity"]["detail"]["stale_after_seconds"], 60.0
        )

        stale = build_health(self._service(unity_age=61.0), config=self.config)
        self.assertEqual(stale["components"]["unity"]["status"], STATUS_DEGRADED)
        self.assertEqual(stale["components"]["unity"]["reason"], "tick_stale")

    def test_brain_health_ignores_routine_in_flight_tick_and_flags_stuck_ones(self) -> None:
        in_flight = build_health(
            self._service(
                runtime={"tick_in_progress": True, "last_tick_age_seconds": 24.0}
            ),
            config=self.config,
        )
        brain = in_flight["components"]["brain"]
        self.assertEqual(brain["status"], STATUS_READY)
        self.assertTrue(brain["detail"]["tick_in_progress"])

        stuck = build_health(
            self._service(
                runtime={"tick_in_progress": True, "last_tick_age_seconds": 120.0}
            ),
            config=self.config,
        )
        brain = stuck["components"]["brain"]
        self.assertEqual(brain["status"], STATUS_DEGRADED)
        self.assertEqual(brain["reason"], "tick_stuck")

    def test_health_marks_configured_llm_ready_without_paid_probe(self) -> None:
        payload = build_health(
            self._service(provider=_ProviderWithSecret()), config=self.config
        )

        llm = payload["components"]["llm"]
        self.assertEqual(llm["status"], STATUS_READY)
        self.assertTrue(llm["configured"])
        self.assertEqual(llm["reason"], "configured_skip_paid_probe")
        self.assertNotIn(SECRET, json.dumps(payload))

    def test_mock_provider_is_visible_degraded_in_development(self) -> None:
        payload = build_health(self._service(), config=self.config)

        llm = payload["components"]["llm"]
        self.assertEqual(llm["status"], STATUS_DEGRADED)
        self.assertEqual(llm["reason"], "mock_provider")
        self.assertTrue(llm["configured"])

    def test_diagnostics_never_leak_credentials_from_nested_state(self) -> None:
        service = self._service(
            provider=_ProviderWithSecret(),
            memory_state={
                "enabled": True,
                "degraded": True,
                "last_error": f"save failed: api_key={SECRET}",
                "counts": {"episodes": 3},
            },
        )

        payload = build_diagnostics(service, config=self.config)
        serialized = json.dumps(payload)

        self.assertNotIn(SECRET, serialized)
        self.assertIn("[redacted]", serialized)
        self.assertEqual(payload["config"], self.config.to_dict())
        self.assertEqual(payload["providers"]["llm"]["kind"], "openrouter")
        self.assertNotIn("api_key", json.dumps(payload["providers"]))

    def test_queue_depths_read_existing_bounded_queues(self) -> None:
        chat = ChatService(MockLLMProvider(), ChatServiceConfig(queue_capacity=7))
        service = SimpleNamespace(chat_service=chat, speech_service=None)

        depths = queue_depths(service)

        self.assertEqual(depths["chat"], {"queued_count": 0, "queue_capacity": 7})

    def test_chat_health_reports_disabled_provider_as_intentional(self) -> None:
        chat = ChatService(MockLLMProvider(), ChatServiceConfig())

        result = chat_health(chat, None)

        self.assertEqual(result["status"], STATUS_UNAVAILABLE)
        self.assertTrue(result["component"]["intentional"])
        self.assertEqual(result["component"]["reason"], "chat_provider_disabled")

    def test_chat_health_follows_the_provider_state_machine(self) -> None:
        chat = ChatService(MockLLMProvider(), ChatServiceConfig())

        connected = chat_health(
            chat,
            SimpleNamespace(
                health=lambda: {
                    "state": "CONNECTED",
                    "provider": "twitch",
                    "platform": "twitch",
                    "reconnect_count": 0,
                    "degraded": False,
                }
            ),
        )
        self.assertEqual(connected["status"], STATUS_READY)

        retrying = chat_health(
            chat,
            SimpleNamespace(
                health=lambda: {
                    "state": "RETRYING",
                    "provider": "twitch",
                    "last_error": "connection reset",
                }
            ),
        )
        self.assertEqual(retrying["status"], STATUS_DEGRADED)

        broken = chat_health(
            chat,
            SimpleNamespace(health=lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        self.assertEqual(broken["status"], STATUS_UNAVAILABLE)


class QueueAndBufferBoundsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_chat_queue_overflow_evicts_lowest_value_not_unbounded(self) -> None:
        """PHASE 24: the queue stays bounded but admits the fresher message."""
        service = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(queue_capacity=2, max_messages_per_rate_window=50),
        )
        # Stay inside the freshness window while keeping a deterministic order.
        start = datetime.now(timezone.utc) - timedelta(seconds=5)
        for index in range(2):
            accepted = service.submit(
                f"viewer_{index}",
                f"unique message {index}",
                timestamp=start + timedelta(seconds=index),
            )
            self.assertTrue(accepted.accepted)

        overflow = service.submit(
            "viewer_99", "one message too many", timestamp=start + timedelta(seconds=2)
        )

        # Same priority class, fresher stamp: the incoming message evicts the
        # oldest survivor instead of being locked out. The bound still holds.
        self.assertTrue(overflow.accepted)
        self.assertEqual(overflow.message.status, ChatStatus.QUEUED)
        queued = service.get_queue()
        self.assertEqual(len(queued), 2)
        self.assertEqual(service.get_state()["queue_evictions"], 1)
        self.assertEqual(
            [message.raw_text for message in queued],
            ["one message too many", "unique message 1"],
        )

    def test_motion_job_table_evicts_oldest_terminal_jobs(self) -> None:
        config = MotionGenerationConfig(
            provider="mock", max_queue_depth=8, max_retained_jobs=2
        )
        service = MotionGenerationService(
            MockMotionGenerator(config), config, start_worker=False
        )
        self.addCleanup(service.shutdown)

        for index in range(3):
            service.request(f"dance step {index}", dedup_key=f"key-{index}")
        for _ in range(3):
            self.assertTrue(service.run_pending_once(timeout=0.5))

        latest = service.request("dance step three", dedup_key="key-3")

        self.assertLessEqual(len(service._jobs), 2)
        self.assertIsNotNone(service.get(latest.request_id))
        counters = service.get_state()["counters"]
        self.assertEqual(counters["requested"], 4)

    def test_motion_metadata_store_keeps_only_the_newest_records(self) -> None:
        store = JSONMotionGenerationMetadataStore(
            self.directory / "motion_metadata.json", max_records=2
        )
        for index in range(5):
            store.save(SimpleNamespace(to_dict=lambda index=index: {"motion_id": f"m{index}"}))

        records = store.list()

        self.assertEqual([item["motion_id"] for item in records], ["m3", "m4"])
        self.assertEqual(store.count(), 2)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
