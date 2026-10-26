from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from backend.chat.context import ConversationContext
from backend.chat.dialogue import CharacterContext, build_prompt, parse_response
from backend.chat.filter import ChatFilter
from backend.chat.intent import extract_intent
from backend.chat.models import ChatIntent, ChatMessage, ChatStatus, WorldContext, utc_now
from backend.chat.normalization import normalize_text
from backend.chat.priority import score_priority
from backend.chat.queue import AttentionQueue
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.runtime import instrument
from backend.runtime.metrics import MetricsRegistry


def make_message(
    text: str,
    viewer_id: str = "viewer_01",
    timestamp: datetime | None = None,
    message_id: str | None = None,
) -> ChatMessage:
    normalized = normalize_text(text)
    return ChatMessage(
        message_id=message_id or text.replace(" ", "_"),
        viewer_id=viewer_id,
        raw_text=text,
        normalized_text=normalized,
        timestamp=timestamp or utc_now(),
    )


class ChatPipelineTests(unittest.TestCase):
    def test_normalization_preserves_raw_text(self) -> None:
        raw = "  Lacrimosa,   can\t you  dance?  "
        normalized = normalize_text(raw)
        self.assertEqual(normalized, "Lacrimosa, can you dance?")

    def test_basic_intents(self) -> None:
        cases = {
            "hello": ChatIntent.GREETING,
            "hello Lacrimosa": ChatIntent.GREETING,
            "Lacrimosa hello": ChatIntent.GREETING,
            "what are you doing?": ChatIntent.QUESTION,
            "can you dance?": ChatIntent.REQUEST_ACTION,
            "look at the camera": ChatIntent.REQUEST_ACTION,
            "sit down": ChatIntent.REQUEST_ACTION,
            "Lacrimosa, dance!": ChatIntent.REQUEST_ACTION,
            "where are you going?": ChatIntent.QUESTION,
            "go to the table": ChatIntent.REQUEST_MOVEMENT,
            "bye Lacrimosa": ChatIntent.FAREWELL,
            "you look cute": ChatIntent.COMMENT,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = extract_intent(normalize_text(text))
                self.assertEqual(result.intent, expected)

    def test_target_extraction(self) -> None:
        self.assertEqual(extract_intent(normalize_text("can you dance?")).target, "dance")
        self.assertEqual(extract_intent(normalize_text("Lacrimosa, dance!")).target, "dance")
        self.assertEqual(extract_intent(normalize_text("go to the table")).target, "table")
        self.assertEqual(extract_intent(normalize_text("look at the camera")).target, "camera")

    def test_stream_and_phone_action_target_extraction(self) -> None:
        cases = {
            "Lacrimosa, can you start the stream?": "stream",
            "Lacrimosa, can you stop the stream?": "stream",
            "Lacrimosa, please pick up the phone.": "phone",
            "Lacrimosa, please place the phone on the table.": "phone",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                result = extract_intent(normalize_text(text))
                self.assertEqual(result.intent, ChatIntent.REQUEST_ACTION)
                self.assertEqual(result.target, expected)

    def test_priority_prefers_actionable_direct_requests(self) -> None:
        texts = [
            "hello",
            "what are you doing?",
            "Lacrimosa can you dance?",
            "bye",
        ]
        messages = [make_message(text) for text in texts]
        for message in messages:
            message.intent = extract_intent(message.normalized_text).intent
            message.target = extract_intent(message.normalized_text).target
            message.priority = score_priority(message)
        by_text = {message.raw_text: message.priority for message in messages}
        self.assertGreater(by_text["Lacrimosa can you dance?"], by_text["what are you doing?"])
        self.assertGreater(by_text["what are you doing?"], by_text["hello"])
        self.assertGreater(by_text["hello"], by_text["bye"])
        self.assertTrue(all(0.0 <= message.priority <= 1.0 for message in messages))

    def test_empty_and_long_messages_are_filtered(self) -> None:
        message_filter = ChatFilter()
        empty = make_message("   ")
        long = make_message("a" * 501)
        self.assertFalse(message_filter.evaluate(empty).accepted)
        self.assertFalse(message_filter.evaluate(long).accepted)

    def test_rate_limiting(self) -> None:
        message_filter = ChatFilter()
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        accepted = 0
        for index in range(6):
            message = make_message(f"unique message {index}", timestamp=start + timedelta(seconds=index))
            if message_filter.evaluate(message).accepted:
                accepted += 1
                message_filter.record(message, True)
        self.assertEqual(accepted, 5)

    def test_queue_orders_and_burst_protection(self) -> None:
        queue = AttentionQueue(capacity=2)
        low = make_message("hello", message_id="low")
        high = make_message("Lacrimosa can you dance?", message_id="high")
        low.intent = extract_intent(low.normalized_text).intent
        high.intent = extract_intent(high.normalized_text).intent
        low.priority = score_priority(low)
        high.priority = score_priority(high)
        self.assertTrue(queue.enqueue(low))
        self.assertTrue(queue.enqueue(high))
        overflow = make_message("you look cute", message_id="overflow")
        self.assertFalse(queue.enqueue(overflow))
        self.assertEqual(queue.dequeue().message_id, "high")
        self.assertEqual(queue.dequeue().message_id, "low")

    def test_queue_evicts_lowest_ranked_instead_of_dropping_a_better_one(self) -> None:
        """PHASE 21 / PART 2: a full queue must not lock out the best message."""
        queue = AttentionQueue(capacity=2)
        filler = make_message("hello", message_id="filler")
        filler.intent = extract_intent(filler.normalized_text).intent
        filler.priority = 0.1
        important = make_message("Lacrimosa can you dance for me?", message_id="important")
        important.intent = extract_intent(important.normalized_text).intent
        important.priority = 0.95

        self.assertTrue(queue.enqueue(filler))
        self.assertTrue(queue.enqueue(important))
        self.assertEqual(len(queue), 2)

        request = make_message("Can you dance for me right now?", message_id="request")
        request.intent = extract_intent(request.normalized_text).intent
        request.priority = 0.9
        self.assertTrue(queue.enqueue(request))
        self.assertEqual(queue.evicted_count, 1)
        self.assertEqual(len(queue), 2)
        self.assertIsNone(queue.find("filler"))
        self.assertEqual(queue.dequeue().message_id, "important")
        self.assertEqual(queue.dequeue().message_id, "request")

        lower = make_message("you look cute", message_id="lower")
        lower.priority = 0.05
        queue.clear()
        queue.enqueue(important)
        mid = make_message("nice stream", message_id="mid")
        mid.priority = 0.5
        queue.enqueue(mid)
        self.assertFalse(queue.enqueue(lower))
        self.assertEqual(queue.evicted_count, 1)
        self.assertEqual(queue.dequeue().message_id, "important")

    def test_queue_answers_the_freshest_message_within_a_priority_class(self) -> None:
        """PHASE 24: a saturated real queue must not replay its oldest survivor."""
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        queue = AttentionQueue(capacity=5)
        for index in range(3):
            message = make_message(
                f"question {index}?",
                message_id=f"q{index}",
                timestamp=start + timedelta(seconds=index),
            )
            message.intent = ChatIntent.QUESTION
            message.priority = 0.46
            self.assertTrue(queue.enqueue(message))
        self.assertEqual(queue.peek().message_id, "q2")
        self.assertEqual(
            [queue.dequeue().message_id for _ in range(3)], ["q2", "q1", "q0"]
        )

    def test_priority_still_outranks_freshness(self) -> None:
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        queue = AttentionQueue(capacity=5)
        request = make_message(
            "Lacrimosa can you dance?", message_id="request", timestamp=start
        )
        request.priority = 0.9
        chatter = make_message(
            "nice stream", message_id="chatter", timestamp=start + timedelta(seconds=30)
        )
        chatter.priority = 0.22
        self.assertTrue(queue.enqueue(chatter))
        self.assertTrue(queue.enqueue(request))
        self.assertEqual(queue.dequeue().message_id, "request")

    def test_full_queue_evicts_the_stalest_low_value_entry(self) -> None:
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        queue = AttentionQueue(capacity=2)
        for index in range(2):
            filler = make_message(
                f"filler {index}",
                message_id=f"filler{index}",
                timestamp=start + timedelta(seconds=index),
            )
            filler.priority = 0.2
            self.assertTrue(queue.enqueue(filler))
        fresh = make_message(
            "fresh comment", message_id="fresh", timestamp=start + timedelta(seconds=2)
        )
        fresh.priority = 0.2
        self.assertTrue(queue.enqueue(fresh))
        self.assertEqual(queue.evicted_count, 1)
        self.assertIsNone(queue.find("filler0"))
        self.assertEqual(queue.dequeue().message_id, "fresh")
        self.assertEqual(queue.dequeue().message_id, "filler1")

    def test_queue_expires_stale_backlog_without_replaying_it(self) -> None:
        """PHASE 21 / PART 2: a busy channel must not be answered minutes late."""
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        now = {"value": start}
        queue = AttentionQueue(
            capacity=5,
            max_age_seconds=60.0,
            clock=lambda: now["value"],
        )
        stale = make_message("old news", message_id="stale", timestamp=start)
        self.assertTrue(queue.enqueue(stale))

        now["value"] = start + timedelta(seconds=61)
        fresh = make_message(
            "Lacrimosa can you dance?",
            message_id="fresh",
            timestamp=now["value"],
        )
        self.assertTrue(queue.enqueue(fresh))
        self.assertEqual(queue.dequeue().message_id, "fresh")
        self.assertEqual(queue.expired_count, 1)
        self.assertEqual(len(queue), 0)


class ChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChatServiceConfig(max_messages_per_rate_window=10)
        self.service = ChatService(MockLLMProvider(), self.config)

    def test_message_age_is_measured_when_processing(self) -> None:
        # Freshness is observed where the message actually gets attention:
        # a saturated queue answering the oldest survivor shows up here as a
        # large age instead of staying invisible.
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            now = {"value": datetime(2026, 9, 9, tzinfo=timezone.utc)}
            service = ChatService(MockLLMProvider(), clock=lambda: now["value"])
            service.submit(
                "viewer_01", "Can you dance?", message_id="age_probe", timestamp=now["value"]
            )
            now["value"] = now["value"] + timedelta(seconds=30)
            self.assertIsNotNone(service.process_next())
            summary = registry.snapshot()["latency_ms"]["chat.message_age_ms"]
        finally:
            instrument.set_registry(None)
        self.assertEqual(summary["count"], 1)
        self.assertAlmostEqual(summary["last"], 30000.0, delta=50.0)

    def test_submission_status_and_processing(self) -> None:
        result = self.service.submit("viewer_01", "Can you dance?")
        self.assertTrue(result.accepted)
        self.assertEqual(result.message.status, ChatStatus.QUEUED)
        self.assertEqual(result.message.intent, ChatIntent.REQUEST_ACTION)
        processed = self.service.process_next()
        self.assertIsNotNone(processed)
        self.assertEqual(processed.message.status, ChatStatus.PROCESSED)
        self.assertEqual(processed.response.intent, ChatIntent.REQUEST_ACTION)
        self.assertEqual(processed.response.requested_action, "dance")

    def test_duplicate_messages_are_merged_not_reprocessed(self) -> None:
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        self.service.clock = lambda: start
        first = self.service.submit("viewer_01", "dance please", "msg_001", start)
        self.assertTrue(first.accepted)
        second = self.service.submit("viewer_01", "DANCE PLEASE!", "msg_002", start + timedelta(seconds=2))
        self.assertFalse(second.accepted)
        self.assertEqual(second.message.filtered_reason, "duplicate")
        self.assertIsNotNone(second.merged_message)
        self.assertEqual(second.merged_message.repeat_count, 2)
        self.assertEqual(len(self.service.get_queue()), 1)

    def test_rate_protection(self) -> None:
        service = ChatService(MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=2))
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        accepted = service.submit("v1", "one", timestamp=start)
        accepted2 = service.submit("v1", "two", timestamp=start + timedelta(seconds=1))
        blocked = service.submit("v1", "three", timestamp=start + timedelta(seconds=2))
        self.assertTrue(accepted.accepted)
        self.assertTrue(accepted2.accepted)
        self.assertFalse(blocked.accepted)
        self.assertEqual(blocked.message.filtered_reason, "rate_limited")

    def test_multi_viewer_integrity(self) -> None:
        for index, viewer in enumerate(("viewer_01", "viewer_02", "viewer_03"), 1):
            result = self.service.submit(viewer, f"Lacrimosa can you {index * 10} times?".replace("10", "dance").replace("20", "wave").replace("30", "run"))
            self.assertTrue(result.accepted)
            self.assertEqual(result.message.viewer_id, viewer)
        queued_ids = {message.viewer_id for message in self.service.get_queue()}
        self.assertEqual(queued_ids, {"viewer_01", "viewer_02", "viewer_03"})

    def test_conversation_context_is_bounded(self) -> None:
        service = ChatService(MockLLMProvider(), ChatServiceConfig(max_context_turns=4))
        first = service.submit("viewer_01", "What are you doing?")
        first_response = service.process_next()
        second = service.submit("viewer_01", "Where are you going?")
        second_response = service.process_next()
        turns = service.conversation.snapshot()
        self.assertEqual(len(turns), 4)
        self.assertIn(first.message.raw_text, [turn.text for turn in turns])
        self.assertIn(second.message.raw_text, [turn.text for turn in turns])
        self.assertIn("I'm here in the room", first_response.response.speech)
        self.assertIn("I'm here in the room", second_response.response.speech)

    def test_provider_failures_return_safe_fallback(self) -> None:
        for failure in ("timeout", "provider", "invalid_json", "missing_speech", "empty"):
            with self.subTest(failure=failure):
                service = ChatService(MockLLMProvider(fail_with=failure))
                submission = service.submit("viewer_01", "Can you dance?")
                processed = service.process_next()
                self.assertIsNotNone(processed)
                self.assertIsNotNone(processed.fallback_reason)
                self.assertIn("moment", processed.response.speech)
                self.assertFalse(processed.response.requested_action == "dance")

    def test_response_validation_rejects_bad_fields(self) -> None:
        message = make_message("hello")
        response, reason = parse_response(json.dumps({"intent": "NOT_AN_INTENT", "speech": "ok"}), message)
        self.assertEqual(response.intent, ChatIntent.UNKNOWN)
        self.assertEqual(reason, "invalid_llm_response")


class ContextAndPromptTests(unittest.TestCase):
    def test_prompt_is_small_and_structured(self) -> None:
        message = make_message("Can you dance?")
        message.intent = ChatIntent.REQUEST_ACTION
        message.target = "dance"
        prompt = build_prompt(
            message,
            ConversationContext(),
            WorldContext(),
            CharacterContext(),
        )
        self.assertLessEqual(len(prompt), 4000)
        self.assertIn("Name: Lacrimosa", prompt)
        self.assertIn("Detected target: dance", prompt)
        self.assertIn("Return strict JSON", prompt)

    def test_parse_accepts_valid_json(self) -> None:
        message = make_message("Can you dance?")
        response, reason = parse_response(
            '{"speech":"Okay!","intent":"REQUEST_ACTION","requested_action":"dance","priority":0.9}',
            message,
        )
        self.assertIsNone(reason)
        self.assertEqual(response.speech, "Okay!")
        self.assertEqual(response.requested_action, "dance")


class ChatQueueTelemetryTests(unittest.TestCase):
    """PA29-004: queue evictions/expirations reach the telemetry registry.

    ``/chat/state`` already exposed ``queue_evictions``/``queue_expired``, but
    the ``/diagnostics`` counter registry did not, so an operator reading
    ``chat.accepted`` vs ``chat.processed`` saw messages silently disappear.
    """

    def _service(self, now: dict[str, datetime], capacity: int = 3) -> ChatService:
        return ChatService(
            MockLLMProvider(),
            ChatServiceConfig(
                queue_capacity=capacity,
                max_queue_age_seconds=60.0,
                max_messages_per_rate_window=100,
            ),
            clock=lambda: now["value"],
        )

    def test_evictions_and_expirations_are_registered_in_telemetry(self) -> None:
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            start = datetime(2026, 9, 9, tzinfo=timezone.utc)
            now = {"value": start}
            service = self._service(now, capacity=2)

            # Fill the queue, then let the oldest entry go stale.
            self.assertTrue(
                service.submit(
                    "viewer_a", "alpha message", "m1", start
                ).accepted
            )
            self.assertTrue(
                service.submit(
                    "viewer_b", "bravo message", "m2", start + timedelta(seconds=1)
                ).accepted
            )
            now["value"] = start + timedelta(seconds=61)
            self.assertEqual(service.get_state()["queue_expired"], 1)
            self.assertEqual(len(service.get_queue()), 1)  # m1 expired in place

            # Refill, then admit a fresher message of the same class: the
            # lowest-ranked survivor is evicted rather than the new one.
            self.assertTrue(
                service.submit(
                    "viewer_c", "charlie message", "m3", now["value"]
                ).accepted
            )
            self.assertTrue(
                service.submit(
                    "viewer_d", "delta message", "m4", now["value"]
                ).accepted
            )

            state = service.get_state()
            self.assertEqual(state["queue_expired"], 1)
            self.assertEqual(state["queue_evictions"], 1)
            self.assertEqual(registry.counter_value("chat.queue_expired"), 1)
            self.assertEqual(registry.counter_value("chat.queue_evictions"), 1)
        finally:
            instrument.set_registry(None)

    def test_every_accepted_message_has_exactly_one_terminal_disposition(
        self,
    ) -> None:
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            start = datetime(2026, 9, 9, tzinfo=timezone.utc)
            now = {"value": start}
            service = self._service(now, capacity=3)

            def stamp(seconds: float) -> datetime:
                return start + timedelta(seconds=seconds)

            # processed: accepted then dequeued for attention.
            self.assertTrue(
                service.submit("viewer_a", "alpha message", "m1", stamp(0)).accepted
            )
            self.assertTrue(
                service.submit("viewer_b", "bravo message", "m2", stamp(1)).accepted
            )
            now["value"] = stamp(2)
            self.assertIsNotNone(service.process_next())

            # discarded: accepted then dropped without spending a request.
            self.assertTrue(
                service.submit("viewer_c", "charlie message", "m3", stamp(2)).accepted
            )
            now["value"] = stamp(3)
            self.assertTrue(
                service.submit("viewer_d", "delta message", "m4", stamp(3)).accepted
            )
            self.assertIsNotNone(service.discard_next())

            # expired: accepted, then stale before attention.
            now["value"] = stamp(70)
            self.assertTrue(
                service.submit("viewer_e", "echo message", "m5", now["value"]).accepted
            )
            now["value"] = stamp(71)
            self.assertTrue(
                service.submit(
                    "viewer_f", "foxtrot message", "m6", stamp(71)
                ).accepted
            )
            now["value"] = stamp(72)
            self.assertTrue(
                service.submit(
                    "viewer_g", "golf message", "m7", now["value"]
                ).accepted
            )

            # rejected: the queue is full and the arrival is older than every
            # survivor, so it is refused outright.
            rejected = service.submit("viewer_h", "hotel message", "m8", stamp(50))
            self.assertFalse(rejected.accepted)
            self.assertEqual(rejected.message.filtered_reason, "queue_full")

            # evicted: accepted, then dropped to admit a fresher equal-class
            # message; the freshest three stay queued for the next decision.
            now["value"] = stamp(73)
            self.assertTrue(
                service.submit(
                    "viewer_i", "india message", "m9", now["value"]
                ).accepted
            )

            state = service.get_state()
            queued = state["queued_count"]
            self.assertEqual(queued, 3)

            accepted = registry.counter_value("chat.accepted")
            terminal = (
                registry.counter_value("chat.processed")
                + registry.counter_value("chat.discarded")
                + registry.counter_value("chat.queue_evictions")
                + registry.counter_value("chat.queue_expired")
                + queued
            )
            self.assertEqual(accepted, 8)
            self.assertEqual(terminal, accepted)
            self.assertEqual(
                registry.counter_value("chat.queue_evictions"), state["queue_evictions"]
            )
            self.assertEqual(
                registry.counter_value("chat.queue_expired"), state["queue_expired"]
            )
            self.assertEqual(
                registry.counter_value("chat.processed"), 1
            )
            self.assertEqual(registry.counter_value("chat.discarded"), 1)
            self.assertEqual(registry.counter_value("chat.queue_expired"), 2)
            self.assertEqual(registry.counter_value("chat.queue_evictions"), 1)
            # The rejected message is on the submission side exactly once.
            self.assertEqual(registry.counter_value("chat.rejected"), 1)
            self.assertEqual(registry.counter_value("chat.submitted"), 9)
            self.assertEqual(registry.counter_value("chat.filtered"), 0)
        finally:
            instrument.set_registry(None)


class ChatBacklogLatencyTests(unittest.TestCase):
    """PA29-006: upstream age, local queue wait and processing stay distinct.

    The startup backlog the audit measured (ages climbing to the 120s queue
    bound) was a *local* queue drain of genuinely current messages, not a
    replay of old upstream events. One conflated ``chat.message_age_ms`` could
    not tell those apart, so the two legs are now measured separately and the
    viewer's original timestamp is never rewritten.
    """

    def _service(self, now: dict[str, datetime]) -> ChatService:
        return ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=100),
            clock=lambda: now["value"],
        )

    def test_fresh_message_reports_small_intake_age_and_real_queue_wait(self) -> None:
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            start = datetime(2026, 9, 9, tzinfo=timezone.utc)
            now = {"value": start}
            service = self._service(now)

            service.submit("viewer_01", "fresh comment", "m1", start)
            now["value"] = start + timedelta(seconds=30)
            self.assertIsNotNone(service.process_next())

            latencies = registry.snapshot()["latency_ms"]
            self.assertAlmostEqual(
                latencies["chat.upstream_age_ms"]["last"], 0.0, delta=1.0
            )
            self.assertAlmostEqual(
                latencies["chat.queue_wait_ms"]["last"], 30_000.0, delta=50.0
            )
            self.assertAlmostEqual(
                latencies["chat.message_age_ms"]["last"], 30_000.0, delta=50.0
            )
        finally:
            instrument.set_registry(None)

    def test_replayed_old_message_is_not_recast_as_local_backlog(self) -> None:
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            start = datetime(2026, 9, 9, tzinfo=timezone.utc)
            now = {"value": start}
            service = self._service(now)

            # The viewer sent this ten minutes ago; it only reaches the local
            # queue now (a reconnect replay the provider freshness guard did
            # not catch, or a direct local injection).
            old_sent_at = start - timedelta(seconds=600)
            submitted = service.submit(
                "viewer_01", "old upstream event", "m1", old_sent_at
            )
            self.assertTrue(submitted.accepted)
            # Intake records the honest upstream age and the local arrival
            # time; the viewer's send time is never rewritten.
            self.assertEqual(submitted.message.timestamp, old_sent_at)
            self.assertEqual(submitted.message.enqueued_at, start)
            latencies = registry.snapshot()["latency_ms"]
            self.assertAlmostEqual(
                latencies["chat.upstream_age_ms"]["last"], 600_000.0, delta=100.0
            )

            # The 120s queue bound then drops it as stale on the next peek:
            # the replay is never answered late and never re-stamped.
            now["value"] = start + timedelta(seconds=2)
            processed = service.process_next()
            self.assertIsNone(processed)
            self.assertEqual(submitted.message.timestamp, old_sent_at)
            self.assertEqual(submitted.message.enqueued_at, start)
            self.assertEqual(registry.counter_value("chat.queue_expired"), 1)
            self.assertEqual(service.get_state()["queue_expired"], 1)
            # It was never attended, so no end-to-end age sample exists for it.
            self.assertNotIn("chat.message_age_ms", registry.snapshot()["latency_ms"])

            # A genuine current request is unaffected by the stale drop.
            now["value"] = start + timedelta(seconds=10)
            fresh = service.submit("viewer_02", "fresh request", "m2", now["value"])
            self.assertTrue(fresh.accepted)
            now["value"] = start + timedelta(seconds=12)
            answered = service.process_next()
            self.assertIsNotNone(answered)
            self.assertEqual(
                answered.message.timestamp, start + timedelta(seconds=10)
            )
            latencies = registry.snapshot()["latency_ms"]
            self.assertAlmostEqual(
                latencies["chat.queue_wait_ms"]["last"], 2_000.0, delta=50.0
            )
            self.assertAlmostEqual(
                latencies["chat.message_age_ms"]["last"], 2_000.0, delta=50.0
            )
            self.assertEqual(registry.counter_value("chat.processed"), 1)
        finally:
            instrument.set_registry(None)

    def test_discarded_attention_records_the_same_local_wait(self) -> None:
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            start = datetime(2026, 9, 9, tzinfo=timezone.utc)
            now = {"value": start}
            service = self._service(now)

            service.submit("viewer_01", "low value chatter", "m1", start)
            now["value"] = start + timedelta(seconds=15)
            self.assertIsNotNone(service.discard_next())

            latencies = registry.snapshot()["latency_ms"]
            self.assertAlmostEqual(
                latencies["chat.queue_wait_ms"]["last"], 15_000.0, delta=50.0
            )
        finally:
            instrument.set_registry(None)


if __name__ == "__main__":
    unittest.main()
