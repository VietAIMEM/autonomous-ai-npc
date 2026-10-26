"use strict";

/* Unit tests for the DOM-free viewer page state logic (app-state.js).
 * These cover UI state derivation only; they never talk to the runtime and
 * are not a substitute for the real browser/runtime E2E acceptance.
 */

const test = require("node:test");
const assert = require("node:assert/strict");

const state = require("../app-state.js");

const half = () => 0.5;

test("resolveViewerId prefers a valid URL viewer over stored identity", () => {
  assert.equal(state.resolveViewerId("?viewer=viewer_01", "viewer_99"), "viewer_01");
  assert.equal(state.resolveViewerId("?foo=bar&viewer=viewer_02", null), "viewer_02");
  assert.equal(state.resolveViewerId("", "viewer_99"), "viewer_99");
  assert.equal(state.resolveViewerId(null, "viewer_99"), "viewer_99");
  assert.equal(state.resolveViewerId(undefined, "viewer_99"), "viewer_99");
});

test("resolveViewerId rejects invalid identities and falls back safely", () => {
  for (const bad of ["bad id", "../etc", "!nope", "x".repeat(65), ""]) {
    assert.equal(state.resolveViewerId("?viewer=" + encodeURIComponent(bad), null), null, bad);
    assert.equal(state.resolveViewerId("", bad), null, bad);
  }
  assert.equal(state.resolveViewerId("?viewer=" + encodeURIComponent("../etc"), "viewer_99"), "viewer_99");
  assert.equal(state.resolveViewerId("?viewer=viewer_01", "not valid"), "viewer_01");
  assert.equal(state.isValidViewerId("viewer_01"), true);
  assert.equal(state.isValidViewerId("viewer."), true);
  assert.equal(state.isValidViewerId(" viewer_01 "), true);
  assert.equal(state.isValidViewerId(""), false);
  assert.equal(state.isValidViewerId(null), false);
});

test("generated identities match the backend patterns", () => {
  const viewer = state.createViewerId(half);
  assert.equal(viewer, "viewer_88888888");
  assert.equal(state.isValidViewerId(viewer), true);

  const session = state.createSessionId(half);
  assert.equal(state.isValidSessionId(session), true);
  assert.equal(session.length, "sess_".length + 16);

  const client = state.createClientId(half);
  assert.equal(state.CLIENT_ID_PATTERN.test(client), true);
  assert.equal(client.length, 2 + 16);
});

test("nextPollDelay backs off with failures and respects hidden tabs", () => {
  assert.equal(state.nextPollDelay({ failures: 0 }), state.POLL_BASE_MS);
  assert.equal(state.nextPollDelay({ failures: 1 }), state.POLL_BASE_MS * 2);
  assert.equal(state.nextPollDelay({ failures: 2 }), state.POLL_BASE_MS * 4);
  assert.equal(state.nextPollDelay({ failures: 9 }), state.POLL_MAX_MS);
  assert.equal(state.nextPollDelay({ failures: 0, hidden: true }), state.POLL_HIDDEN_MS);
  assert.equal(state.nextPollDelay({ failures: 2, hidden: true }), state.POLL_MAX_MS);
  assert.equal(state.nextPollDelay({}), state.POLL_BASE_MS);
});

test("deriveConnection reports failure states without ever claiming LIVE", () => {
  assert.equal(state.deriveConnection(null, { ok: false, failures: 1 }), state.STATUS_RECONNECTING);
  assert.equal(state.deriveConnection(null, { ok: false, failures: 3 }), state.STATUS_RECONNECTING);
  assert.equal(state.deriveConnection(null, { ok: false, failures: 4 }), state.STATUS_OFFLINE);
  assert.equal(state.deriveConnection(null, { ok: false, failures: 40 }), state.STATUS_OFFLINE);
  assert.equal(state.deriveConnection(null, {}), state.STATUS_RECONNECTING);
});

test("deriveConnection trusts only known stream states", () => {
  const ok = { ok: true };
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "LIVE" }), state.STATUS_LIVE);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "live" }), state.STATUS_LIVE);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "CONNECTING" }), state.STATUS_CONNECTING);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "RECONNECTING" }), state.STATUS_RECONNECTING);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "OFFLINE" }), state.STATUS_OFFLINE);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "DEGRADED" }), state.STATUS_DEGRADED);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "WHATEVER" }), state.STATUS_DEGRADED);
  assert.equal(state.deriveConnection(null, { ...ok, streamConnection: "" }), state.STATUS_DEGRADED);
});

test("applySendResult marks accepted sends as sent", () => {
  const pending = state.localMessage({
    clientId: "c_0000000000000001",
    viewerId: "viewer_01",
    text: "hello",
    at: "2026-10-02T10:00:00Z"
  });
  assert.equal(pending.delivery, state.DELIVERY_PENDING);
  assert.equal(pending.local, true);

  const sent = state.applySendResult(pending, {
    accepted: true,
    delivery: "accepted",
    reason: null,
    message: {
      id: "msg_000001",
      client_id: "c_0000000000000001",
      delivery: "accepted",
      status: "QUEUED"
    }
  });
  assert.equal(sent.delivery, state.DELIVERY_ACCEPTED);
  assert.equal(sent.id, "msg_000001");
  assert.equal(sent.local, false);
  assert.equal(state.deliveryLabel(sent), "sent");
});

test("applySendResult surfaces backend refusal instead of success", () => {
  const pending = state.localMessage({ clientId: "c_0000000000000002", text: "spam" });
  const refused = state.applySendResult(pending, {
    accepted: false,
    delivery: "failed",
    reason: "sending too fast, try again in a moment",
    message: { id: "msg_000002", delivery: "failed", status: "FILTERED" }
  });
  assert.equal(refused.delivery, state.DELIVERY_FAILED);
  assert.equal(refused.reason, "sending too fast, try again in a moment");
  assert.equal(state.deliveryLabel(refused), "sending too fast, try again in a moment");
  assert.equal(state.isTerminalDelivery(refused.delivery), true);

  const bogus = state.applySendResult(pending, { accepted: false, delivery: "accepted" });
  assert.equal(bogus.delivery, state.DELIVERY_FAILED);
  assert.equal(state.deliveryLabel(bogus), "not delivered");
});

test("applySendResult keeps a merged duplicate delivered", () => {
  const pending = state.localMessage({ clientId: "c_0000000000000003", text: "again" });
  const merged = state.applySendResult(pending, {
    accepted: false,
    delivery: "delivered",
    reason: "that looked like a repeat of your recent message",
    merged_message_id: "msg_000001",
    message: { id: "msg_000003", delivery: "delivered", reason: null, status: "FILTERED" }
  });
  assert.equal(merged.delivery, state.DELIVERY_DELIVERED);
  assert.equal(merged.reason, "that looked like a repeat of your recent message");
  assert.equal(state.deliveryLabel(merged), "delivered");
  assert.equal(state.isTerminalDelivery(merged.delivery), true);
});

test("applySendResult fails closed on a missing or broken response", () => {
  const pending = state.localMessage({ clientId: "c_0000000000000004", text: "hi" });
  const broken = state.applySendResult(pending, null);
  assert.equal(broken.delivery, state.DELIVERY_FAILED);
  assert.equal(broken.reason, "the message could not be sent");

  const empty = state.applySendResult(pending, {});
  assert.equal(empty.delivery, state.DELIVERY_FAILED);
  assert.equal(empty.local, false);
});

test("reconcileMessages merges optimistic bubbles by client id", () => {
  const localOne = state.localMessage({ clientId: "c_aaaa000000000001", text: "first", at: "2026-10-02T10:00:00Z" });
  const localTwo = state.localMessage({ clientId: "c_bbbb000000000002", text: "second", at: "2026-10-02T10:00:01Z" });
  const server = [
    {
      id: "msg_000010",
      client_id: "c_aaaa000000000001",
      role: "viewer",
      viewer_id: "viewer_01",
      text: "first",
      at: "2026-10-02T10:00:00Z",
      status: "QUEUED",
      delivery: "accepted",
      reason: null
    },
    {
      id: "resp:msg_000010",
      role: "lacrimosa",
      viewer_id: null,
      text: "Hello!",
      at: "2026-10-02T10:00:02Z",
      delivery: "delivered"
    }
  ];
  const merged = state.reconcileMessages([localOne, localTwo], server, 200);
  assert.equal(merged.length, 3);
  assert.equal(merged[0].id, "msg_000010");
  assert.equal(merged[0].client_id, "c_aaaa000000000001");
  assert.equal(merged[0].local, false);
  assert.equal(merged[0].delivery, "accepted");
  assert.equal(merged[1].role, "lacrimosa");
  assert.equal(merged[2].client_id, "c_bbbb000000000002");
  assert.equal(merged[2].local, true);
});

test("reconcileMessages bounds history and ignores malformed server rows", () => {
  const locals = [];
  const server = [];
  for (let index = 0; index < 12; index += 1) {
    server.push({ id: "msg_" + index, role: "viewer", text: "t" + index, delivery: "delivered" });
  }
  server.push({ role: "viewer", text: "no id" });
  const merged = state.reconcileMessages(locals, server, 5);
  assert.equal(merged.length, 5);
  assert.equal(merged[0].id, "msg_7");
  assert.equal(merged[4].id, "msg_11");

  const empty = state.reconcileMessages(null, null, 0);
  assert.equal(empty.length, 0);
});

test("reconcileMessages never freezes the transcript under feed churn", () => {
  let list = [];
  for (let poll = 0; poll < 300; poll += 1) {
    const server = [];
    for (let index = 0; index < 60; index += 1) {
      const id = "msg_" + (poll * 3 + index);
      server.push({ id, role: "viewer", viewer_id: "twitch:1", text: "t" + id, delivery: "sent" });
    }
    list = state.reconcileMessages(list, server, 200);
  }
  assert.equal(list.length, 60);
  assert.equal(list[list.length - 1].id, "msg_" + (299 * 3 + 59));
  const ids = new Set(list.map((item) => item.id));
  assert.equal(ids.size, list.length);
});

test("reconcileMessages keeps the viewer's own bubble after feed eviction", () => {
  const own = state.localMessage({
    clientId: "c_cccc000000000001",
    viewerId: "viewer_01",
    text: "hello Lacrimosa",
    at: "2026-10-02T10:00:00Z"
  });
  let list = state.reconcileMessages([own], [], 200);
  assert.equal(list.length, 1);
  assert.equal(list[0].delivery, state.DELIVERY_PENDING);

  for (let poll = 0; poll < 120; poll += 1) {
    const server = [];
    for (let index = 0; index < 60; index += 1) {
      server.push({ id: "msg_" + (poll * 4 + index), role: "viewer", text: "t", delivery: "sent" });
    }
    list = state.reconcileMessages(list, server, 200);
  }
  assert.equal(list.length, 61);
  assert.equal(list[0].delivery, "sent");
  assert.equal(list[list.length - 1].client_id, "c_cccc000000000001");
  assert.equal(list[list.length - 1].text, "hello Lacrimosa");
});

test("reconcileMessages bounds how many own bubbles it retains", () => {
  const mine = [];
  for (let index = 0; index < state.MAX_OWN_MESSAGES + 15; index += 1) {
    mine.push(
      state.localMessage({
        clientId: "c_eeee" + String(index).padStart(12, "0"),
        viewerId: "viewer_01",
        text: "m" + index,
        at: "2026-10-02T10:00:00Z"
      })
    );
  }
  const server = [];
  for (let index = 0; index < 60; index += 1) {
    server.push({ id: "msg_" + index, role: "viewer", text: "t" + index, delivery: "sent" });
  }
  const list = state.reconcileMessages(mine, server, 200);
  assert.equal(list.length, 60 + state.MAX_OWN_MESSAGES);
  assert.equal(list[60].text, "m15");
  assert.equal(list[list.length - 1].text, "m" + (state.MAX_OWN_MESSAGES + 14));
});

test("delivery labels cover every public status", () => {
  assert.equal(state.deliveryLabel({ delivery: "pending" }), "sending…");
  assert.equal(state.deliveryLabel({ delivery: "accepted" }), "sent");
  assert.equal(state.deliveryLabel({ delivery: "delivered" }), "delivered");
  assert.equal(state.deliveryLabel({ delivery: "failed", reason: "chat is busy right now, try again in a moment" }), "chat is busy right now, try again in a moment");
  assert.equal(state.deliveryLabel({ delivery: "failed" }), "not delivered");
  assert.equal(state.deliveryLabel({}), "");
  assert.equal(state.deliveryLabel(null), "");
});

test("activity dot classes are presentation only", () => {
  assert.equal(state.activityDotClass("Speaking"), "speaking");
  assert.equal(state.activityDotClass("Moving"), "moving");
  assert.equal(state.activityDotClass("Looking"), "moving");
  assert.equal(state.activityDotClass("Waving"), "moving");
  assert.equal(state.activityDotClass("Dancing"), "dancing");
  assert.equal(state.activityDotClass("Live"), "live");
  assert.equal(state.activityDotClass("Idle"), "");
  assert.equal(state.activityDotClass("unexpected"), "");
  assert.equal(state.activityDotClass(null), "");
});

test("connection notices stay public and never leak internal detail", () => {
  assert.equal(state.connectionNotice(state.STATUS_OFFLINE), "Lacrimosa is offline.");
  assert.equal(state.connectionNotice(state.STATUS_RECONNECTING), "Reconnecting to the runtime…");
  assert.equal(state.connectionNotice(state.STATUS_CONNECTING), "Connecting to the runtime…");
  assert.equal(state.connectionNotice(state.STATUS_LIVE), "");
  assert.equal(state.connectionNotice(null), "");
  for (const status of [state.STATUS_OFFLINE, state.STATUS_RECONNECTING, state.STATUS_CONNECTING]) {
    const text = state.connectionNotice(status);
    assert.equal(/[\\/]|http|Error|Traceback|:?\d{4,}/.test(text), false, text);
  }
});

test("offline threshold is shared with the poll loop", () => {
  assert.equal(state.OFFLINE_AFTER_FAILURES, 4);
  assert.equal(state.deriveConnection(state.STATUS_LIVE, { ok: false, failures: 3 }), state.STATUS_RECONNECTING);
  assert.equal(state.deriveConnection(state.STATUS_LIVE, { ok: false, failures: state.OFFLINE_AFTER_FAILURES }), state.STATUS_OFFLINE);
});
