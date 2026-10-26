"use strict";

/* Unit tests for the DOM-free admin dashboard state logic (app-state.js).
 * These cover UI derivation only: they never talk to the runtime and are not
 * a substitute for the real authenticated browser acceptance run.
 */

const test = require("node:test");
const assert = require("node:assert/strict");

const state = require("../app-state.js");

const EXPECTED_SECTIONS = [
  "overview",
  "runtime",
  "brain",
  "providers",
  "tts",
  "chat",
  "stream",
  "motion",
  "skills",
  "learning",
  "memory",
  "viewers",
  "personality",
  "diagnostics",
  "logs",
  "settings",
];

test("sections cover the required dashboard surface exactly once", () => {
  assert.deepEqual(
    state.SECTIONS.map((section) => section.id),
    EXPECTED_SECTIONS
  );
  const labels = new Set(state.SECTIONS.map((section) => section.label));
  assert.equal(labels.size, EXPECTED_SECTIONS.length);
  for (const section of state.SECTIONS) {
    assert.ok(section.requests.length >= 1, section.id);
    for (const path of section.requests) {
      assert.match(path, /^\/admin\/api\//, section.id);
    }
  }
  assert.equal(state.sectionById("motion").label, "Motion");
  assert.equal(state.sectionById("nope"), null);
});

test("navigation only accepts known sections", () => {
  let current = state.createState();
  current = state.reduce(current, { type: "navigate", section: "stream" });
  assert.equal(current.section, "stream");
  current = state.reduce(current, { type: "navigate", section: "../../etc/passwd" });
  assert.equal(current.section, "stream");
  current = state.reduce(current, { type: "navigate", section: null });
  assert.equal(current.section, "stream");
});

test("session transitions gate the dashboard", () => {
  let current = state.createState();
  assert.equal(current.authenticated, false);
  assert.equal(current.sessionChecked, false);

  current = state.reduce(current, {
    type: "session",
    authenticated: true,
    role: "admin",
    permissions: ["admin.read", "admin.write"],
  });
  assert.equal(current.authenticated, true);
  assert.equal(current.sessionChecked, true);
  assert.deepEqual(current.permissions, ["admin.read", "admin.write"]);

  current = state.reduce(current, {
    type: "data",
    section: "overview",
    payload: { "/admin/api/overview": { status: "RUNNING" } },
    at: "2026-10-03T00:00:00Z",
  });
  assert.ok(current.pages.overview);

  current = state.reduce(current, { type: "logout" });
  assert.equal(current.authenticated, false);
  assert.equal(current.sessionChecked, true);
  assert.deepEqual(current.pages, {});
  assert.equal(current.section, "overview");
});

test("data and error actions are tracked per section", () => {
  let current = state.createState();
  current = state.reduce(current, {
    type: "data",
    section: "logs",
    payload: { "/admin/api/logs": { count: 1 } },
    at: "2026-10-03T00:00:00Z",
  });
  assert.equal(current.failures, 0);
  assert.equal(current.lastRefreshAt, "2026-10-03T00:00:00Z");

  const problem = state.describeError(503, { error: "chat_unavailable" });
  current = state.reduce(current, { type: "error", section: "logs", error: problem });
  assert.equal(current.errors.logs.code, "chat_unavailable");

  current = state.reduce(current, {
    type: "data",
    section: "logs",
    payload: { "/admin/api/logs": { count: 2 } },
    at: "2026-10-03T00:00:01Z",
  });
  assert.equal(current.errors.logs, undefined);

  current = state.reduce(current, { type: "poll", failures: 3 });
  assert.equal(current.failures, 3);
  current = state.reduce(current, { type: "poll", failures: -5 });
  assert.equal(current.failures, 0);
});

test("login and command state is explicit", () => {
  let current = state.createState();
  current = state.reduce(current, { type: "login_pending" });
  assert.equal(current.login.pending, true);
  assert.equal(current.login.error, null);
  current = state.reduce(current, {
    type: "login_error",
    error: state.describeError(401, { error: "invalid_credentials" }),
  });
  assert.equal(current.login.pending, false);
  assert.equal(current.login.error.kind, "authentication");

  current = state.reduce(current, { type: "command", running: true, label: "Stop" });
  assert.deepEqual(current.command, { running: true, label: "Stop" });
  current = state.reduce(current, { type: "command", running: false });
  assert.deepEqual(current.command, { running: false, label: null });

  current = state.reduce(current, {
    type: "notice",
    notice: { tone: "ok", title: "ok", message: "" },
    result: { label: "Stop", ok: true },
  });
  assert.equal(current.notice.tone, "ok");
  assert.equal(current.commandResult.label, "Stop");
  current = state.reduce(current, { type: "navigate", section: "runtime" });
  assert.equal(current.notice, null);
});

test("describeError maps backend statuses to operator results", () => {
  assert.equal(state.describeError(0, null).kind, "network");
  assert.equal(state.describeError(200, { ok: true }).kind, "success");
  assert.equal(state.describeError(401, { error: "authentication_required" }).kind, "authentication");
  const permission = state.describeError(403, {
    error: "permission_denied",
    permission: "admin.write",
  });
  assert.equal(permission.kind, "permission");
  assert.match(permission.message, /admin\.write/);
  assert.equal(state.describeError(404, { error: "motion_request_not_found" }).kind, "notfound");
  assert.equal(
    state.describeError(409, { error: "instance_already_running" }).kind,
    "conflict"
  );
  assert.equal(state.describeError(400, { error: "invalid_motion_id" }).kind, "validation");
  assert.equal(state.describeError(413, { error: "request_too_large" }).kind, "validation");
  const throttled = state.describeError(429, {
    error: "login_throttled",
    retry_after_seconds: 30,
  });
  assert.equal(throttled.kind, "throttled");
  assert.match(throttled.message, /30/);
  assert.equal(state.describeError(503, { error: "tts_unavailable" }).kind, "unavailable");
  assert.equal(state.describeError(500, { error: "unavailable" }).kind, "server");

  // Any unclassified 4xx still lands in the validation bucket with safe text.
  const unknown = state.describeError(418, {});
  assert.equal(unknown.kind, "validation");
  assert.ok(!/at Object|stack/i.test(unknown.message));
});

test("status tones classify runtime vocabulary", () => {
  assert.equal(state.statusTone("RUNNING"), "ok");
  assert.equal(state.statusTone("READY"), "ok");
  assert.equal(state.statusTone("live"), "ok");
  assert.equal(state.statusTone("DEGRADED"), "warn");
  assert.equal(state.statusTone("requires restart"), "warn");
  assert.equal(state.statusTone("NOT_CONFIGURED"), "warn");
  assert.equal(state.statusTone("UNAVAILABLE"), "bad");
  assert.equal(state.statusTone("ERROR"), "bad");
  assert.equal(state.statusTone(""), "muted");
  assert.equal(state.statusTone(null), "muted");
});

test("buildQuery only emits bounded log filters", () => {
  assert.equal(state.buildQuery("overview", { limit: "50" }), "");
  assert.equal(state.buildQuery("logs", {}), "");
  assert.equal(
    state.buildQuery("logs", {
      severity: "ERROR",
      component: "brain",
      operation: "",
      since_minutes: "30",
      limit: "50",
      ignored: "drop me",
    }),
    "?severity=ERROR&component=brain&since_minutes=30&limit=50"
  );
  assert.equal(
    state.buildQuery("logs", { component: "a b&c=1" }),
    "?component=a%20b%26c%3D1"
  );
});

test("logs filters are stored per section and never leak between sections", () => {
  let current = state.createState();
  current = state.reduce(current, {
    type: "filters",
    section: "logs",
    values: { severity: "ERROR", limit: "100" },
  });
  assert.equal(current.filters.logs.severity, "ERROR");
  assert.equal(current.filters.logs.limit, "100");
  current = state.reduce(current, {
    type: "filters",
    section: "logs",
    values: { severity: "WARNING" },
  });
  assert.equal(current.filters.logs.severity, "WARNING");
  assert.equal(current.filters.logs.limit, "100");
});

test("credential-shaped strings are detected before rendering", () => {
  assert.deepEqual(state.findSecretExposure("all clear"), []);
  assert.deepEqual(state.findSecretExposure(""), []);
  assert.deepEqual(state.findSecretExposure(null), []);
  assert.ok(
    state.findSecretExposure('{"api_key": "sk-abcdefghijklmnop123456"}').length > 0
  );
  assert.ok(
    state.findSecretExposure("Authorization: Bearer abcdefghijklmnopqrstuvwxyz").length > 0
  );
  assert.ok(state.findSecretExposure('stream_key=0123456789abcdef').length > 0);
  // Short or unrelated tokens are not flagged, so ordinary logs still render.
  assert.deepEqual(state.findSecretExposure('{"token": "abc"}'), []);
  assert.deepEqual(state.findSecretExposure("token budget exceeded"), []);
});

test("polling backs off with failures and respects hidden tabs", () => {
  assert.equal(state.nextPollDelay({ failures: 0 }), state.POLL_BASE_MS);
  assert.equal(state.nextPollDelay({ failures: 1 }), state.POLL_BASE_MS * 2);
  assert.equal(state.nextPollDelay({ failures: 2 }), state.POLL_BASE_MS * 4);
  assert.equal(state.nextPollDelay({ failures: 9 }), state.POLL_MAX_MS);
  assert.equal(state.nextPollDelay({ failures: 0, hidden: true }), state.POLL_HIDDEN_MS);
  assert.equal(state.nextPollDelay({}), state.POLL_BASE_MS);
});

// ---------------------------------------------------------------- builders

function overviewPayload() {
  return {
    "/admin/api/overview": {
      generated_at: "2026-10-03T10:00:00Z",
      status: "RUNNING",
      environment: "production",
      process: { pid: 4242, uptime_seconds: 300, state: "ready", instance_id: "abc" },
      components: {
        brain: { status: "READY", detail: "tick 12", required: true },
        unity: { status: "UNAVAILABLE", reason: "editor closed", required: false },
      },
      health_summary: { status: "DEGRADED" },
      active_skill: "wave_hello",
      active_goal: "greet_new_viewer",
      behavior: { state: "IDLE", previous_state: "WAVE" },
      viewer_count: 3,
      queues: {
        chat: { queued_count: 1, queue_capacity: 100 },
        speech: { queue_length: 0, queue_capacity: 10 },
        chat_source: { state: "CONNECTED" },
      },
      stream: { state: "LIVE", live: true, provider: "srt", stop_latch: false },
      speech: { state: "IDLE" },
      brain: { tick_count: 12, tick_in_progress: false, last_tick_age_seconds: 0.4 },
      recent_errors: [
        { at: "2026-10-03T09:00:00Z", component: "tts", message: "provider timeout" },
      ],
    },
  };
}

test("overview view projects real runtime state", () => {
  const view = state.buildView("overview", overviewPayload());
  assert.equal(view.title, "Overview");
  const labels = view.summary.map((entry) => entry.label);
  assert.ok(labels.includes("Runtime"));
  assert.equal(view.summary.find((entry) => entry.label === "Runtime").value, "RUNNING");
  assert.equal(view.summary.find((entry) => entry.label === "Viewers").value, "3");
  const health = view.tables.find((table) => table.title === "Service health");
  assert.equal(health.rows.length, 2);
  assert.equal(view.commands.length, 0);
});

test("runtime view exposes lifecycle commands without inventing state", () => {
  const view = state.buildView("runtime", {
    "/admin/api/runtime": {
      runtime: {
        running: true,
        pid: 4242,
        lock: { held: true, role: "runtime", pid: 4242 },
        status_file: { state: "READY", environment: "production", brain_port: 8766 },
        stop_requested: false,
        http_status: { reachable: true },
      },
      dispatched: null,
      admin: { pid: 4243, port: 8771, state: "running" },
    },
    "/admin/api/runtime/health": {
      health: { status: "READY", components: { brain: { status: "READY" } } },
    },
  });
  assert.equal(view.title, "Runtime");
  assert.deepEqual(
    view.commands.map((command) => command.id),
    ["runtime-start", "runtime-stop", "runtime-restart"]
  );
  for (const command of view.commands) {
    assert.equal(command.request.method, "POST");
    assert.match(command.request.path, /^\/admin\/api\/runtime\/(start|stop|restart)$/);
  }
  const stop = view.commands.find((command) => command.id === "runtime-stop");
  assert.ok(stop.confirm, "stop must ask for confirmation");
});

test("provider, tts, chat and stream views stay read-first", () => {
  const providers = state.buildView("providers", {
    "/admin/api/providers": {
      llm: { provider: "nvidia", model: "llama", configured: true, reachable: true, latency: { last_ms: 120 } },
      tts: { provider: "kokoro", voice: "af_heart", configured: true, reachable: false },
      configured: true,
      secret_values_exposed: false,
    },
  });
  assert.equal(providers.title, "Models / Providers");
  const rows = providers.tables[0].rows.map((row) => row.cells.join("|")).join("\n");
  assert.match(rows, /nvidia/);
  assert.ok(!/secret/i.test(rows));

  const tts = state.buildView("tts", {
    "/admin/api/tts": {
      provider: "kokoro",
      model: "kokoro-v1",
      voice: "af_heart",
      state: "READY",
      enabled: true,
      queue: { length: 0, capacity: 10 },
      latency: { last_ms: 220 },
    },
  });
  assert.equal(tts.commands.length, 1);
  assert.equal(tts.commands[0].request.path, "/admin/api/tts/test");
  assert.equal(tts.commands[0].fields[0].name, "text");

  const chat = state.buildView("chat", {
    "/admin/api/chat": {
      provider: "twitch",
      state: "CONNECTED",
      queue: { queued_count: 0, queue_capacity: 50 },
    },
  });
  assert.equal(chat.commands[0].request.path, "/admin/api/chat/reconnect");

  const stream = state.buildView("stream", {
    "/admin/api/stream": {
      provider: "srt",
      state: { state: "STOPPED", live: false, viewer_count: 0 },
      health: { status: "STOPPED", reachable: false },
      stop_latch: { engaged: true, reason: "operator requested stop" },
    },
  });
  assert.equal(stream.commands.length, 3);
  const latchStat = stream.summary.find((entry) => entry.label === "Stop latch");
  assert.equal(latchStat.value, "ENGAGED");
  assert.equal(latchStat.tone, "warn");
  assert.ok(stream.notes.some((note) => /STOP_STREAM/.test(note.text)));
});

test("stop latch rendering follows the real runtime snapshot shape", () => {
  const engagedSnapshot = {
    stopped: true,
    persistent: true,
    source: "brain:STOP_STREAM",
    updated_at: "2026-10-03T17:40:17Z",
  };
  const releasedSnapshot = {
    stopped: false,
    persistent: true,
    source: "brain:START_STREAM",
    updated_at: "2026-10-03T17:40:17Z",
  };

  const engagedOverview = state.buildView("overview", {
    "/admin/api/overview": {
      ...overviewPayload()["/admin/api/overview"],
      stream: { state: "STOPPED", live: false, provider: "srt", stop_latch: engagedSnapshot },
    },
  });
  const overviewLatch = engagedOverview.kv
    .find((group) => group.title === "Stream")
    .items.find((entry) => entry.label === "Stop latch");
  assert.equal(overviewLatch.value, "ENGAGED");
  assert.equal(overviewLatch.tone, "warn");

  const releasedOverview = state.buildView("overview", {
    "/admin/api/overview": {
      ...overviewPayload()["/admin/api/overview"],
      stream: { state: "LIVE", live: true, provider: "srt", stop_latch: releasedSnapshot },
    },
  });
  assert.equal(
    releasedOverview.kv
      .find((group) => group.title === "Stream")
      .items.find((entry) => entry.label === "Stop latch").value,
    "released",
  );

  const stream = state.buildView("stream", {
    "/admin/api/stream": {
      provider: "srt",
      state: { state: "STOPPED", live: false },
      health: { status: "STOPPED" },
      stop_latch: engagedSnapshot,
    },
  });
  assert.equal(
    stream.summary.find((entry) => entry.label === "Stop latch").value,
    "ENGAGED",
  );
  const latchGroup = stream.kv.find((group) => group.title === "Stop latch");
  assert.equal(latchGroup.items.find((entry) => entry.label === "Engaged").value, "yes");
  assert.equal(
    latchGroup.items.find((entry) => entry.label === "Reason").value,
    "brain:STOP_STREAM",
  );

  const diagnostics = state.buildView("diagnostics", {
    "/admin/api/diagnostics": {
      runtime: { process: { state: "RUNNING" }, stream_stop_latch: releasedSnapshot },
    },
  });
  const runtimeGroup = diagnostics.kv.find((group) => group.title === "Runtime");
  assert.equal(
    runtimeGroup.items.find((entry) => entry.label === "Stop latch").value,
    "released",
  );
});

test("motion view exposes the full import and processing workflow", () => {
  const view = state.buildView("motion", {
    "/admin/api/motion": {
      gate: "VALID + COMPATIBLE is enforced by the backend.",
      import: { enabled: true, request: null },
      processing: {
        enabled: true,
        requests: [
          {
            request_id: "req-1",
            source_motion_id: "phase16_walking",
            processed_motion_id: "phase16_walking_processed_v1",
            state: "READY",
            validation_status: "VALID",
            compatibility_status: "COMPATIBLE",
            quality_score: 0.76,
            animation_reference: "WalkingProcessed",
          },
        ],
      },
      generation: { enabled: true, requests: [] },
      motions: [
        {
          motion_id: "phase15_wave",
          source: "generated",
          version: 1,
          duration: 2.5,
          fps: 30,
          quality_score: 0.8,
          skeleton_status: "COMPATIBLE",
          validation_status: "VALID",
          animation_reference: "MotiusPhase15Wave",
        },
      ],
    },
  });
  assert.equal(view.title, "Motion");
  const ids = view.commands.map((command) => command.id);
  for (const expected of [
    "motion-import",
    "motion-import-path",
    "motion-generate",
    "motion-process",
    "motion-approve",
    "motion-register",
    "motion-job",
  ]) {
    assert.ok(ids.includes(expected), expected);
  }
  const upload = view.commands.find((command) => command.id === "motion-import");
  assert.equal(upload.upload, true);
  assert.ok(upload.fields.some((field) => field.type === "file" && field.required));
  const table = view.tables.find((entry) => entry.rows.length > 0);
  assert.ok(table);
  const text = JSON.stringify(view);
  assert.ok(!/mock/i.test(text), "no mock data may be described in the UI");
});

test("skills view toggles entries but never bypasses the executor", () => {
  const view = state.buildView("skills", {
    "/admin/api/skills": {
      count: 2,
      enabled: true,
      runtime: { enabled: true, execution_count: 4, success_count: 4, failure_count: 0 },
      skills: [
        { skill_id: "wave", name: "Wave", category: "GESTURE", enabled: true, conditions: ["greeting"] },
        { skill_id: "walk", name: "Walk", category: "MOVEMENT", enabled: false, conditions: [] },
      ],
    },
  });
  const rows = view.tables[0].rows;
  assert.equal(rows[0].actions[0].request.path, "/admin/api/skills/disable");
  assert.equal(rows[1].actions[0].request.path, "/admin/api/skills/enable");
  assert.deepEqual(rows[1].actions[0].request.body, { skill_id: "walk" });
  assert.ok(view.notes.some((note) => /SkillExecutor/.test(note.text)));
});

test("learning view is read-only apart from an explicit destructive reset", () => {
  const view = state.buildView("learning", {
    "/admin/api/learning": {
      enabled: true,
      state: { motion_count: 3, knowledge_entries: 1, composite_count: 0, exploration_epsilon: 0.1 },
      knowledge: [
        {
          skill_id: "wave",
          context_key: "greeting|idle",
          trials: 1,
          successes: 1,
          failures: 0,
          success_rate: 1.0,
          average_score: 0.7,
          confidence: 0.5,
        },
      ],
      compositions: [],
    },
  });
  assert.equal(view.summary.find((entry) => entry.label === "Trials").value, "1");
  assert.equal(view.commands.length, 1);
  assert.equal(view.commands[0].id, "learning-reset");
  assert.ok(view.commands[0].confirm, "reset must be confirmed");
  assert.ok(view.notes.some((note) => /never fabricated/.test(note.text)));
});

test("memory and viewer views are read-only and bounded", () => {
  const memory = state.buildView("memory", {
    "/admin/api/memory": {
      episodic: [{ at: "2026-10-03T09:00:00Z", summary: "greeted viewer_01" }],
      semantic: [],
      viewers: [],
    },
  });
  assert.equal(memory.commands.length, 0);
  assert.ok(memory.tables.length >= 1);

  const viewers = state.buildView("viewers", {
    "/admin/api/viewers": {
      count: 1,
      limit: 50,
      viewers: [
        {
          viewer_id: "viewer_01",
          interaction_count: 5,
          relationship_level: 2,
          familiarity: 0.4,
          last_interaction: "2026-10-03T09:00:00Z",
        },
      ],
    },
  });
  assert.equal(viewers.commands.length, 0);
  assert.equal(viewers.tables[0].rows.length, 1);
});

test("personality, brain and diagnostics views render without mutations", () => {
  const brain = state.buildView("brain", {
    "/admin/api/status": {
      status: "RUNNING",
      brain: { tick_count: 9, tick_in_progress: true, last_tick_age_seconds: 1.2 },
      active_skill: "wave",
      queues: { chat: { queued_count: 0 }, speech: { queue_length: 1 } },
    },
    "/admin/api/personality": {
      goal: "greet viewers",
      behavior: { state: "IDLE", previous_state: "WAVE" },
      personality: { speech_style: "warm" },
    },
  });
  assert.equal(brain.title, "Brain");
  assert.equal(brain.commands.length, 0);
  assert.ok(brain.summary.some((entry) => entry.label === "Tick" && entry.value === "IN PROGRESS"));

  const personality = state.buildView("personality", {
    "/admin/api/personality": { traits: { warmth: 0.8 }, behavior: { state: "IDLE" } },
    "/admin/api/overview": overviewPayload()["/admin/api/overview"],
  });
  assert.equal(personality.commands.length, 0);

  const diagnostics = state.buildView("diagnostics", {
    "/admin/api/diagnostics": {
      generated_at: "2026-10-03T10:00:00Z",
      components: { brain: { status: "READY" } },
      checks: [{ name: "unity", status: "UNAVAILABLE", detail: "editor closed" }],
    },
  });
  assert.equal(diagnostics.commands.length, 0);
  const text = JSON.stringify(diagnostics);
  assert.ok(!/api_key|stream_key|password/i.test(text));
});

test("logs view renders bounded redacted entries with filters", () => {
  const view = state.buildView("logs", {
    "/admin/api/logs": {
      available: true,
      log_file: "backend/logs/room_agent.log",
      count: 1,
      matched: 1,
      limit: 200,
      scanned_bytes: 4096,
      truncated: false,
      entries: [
        {
          timestamp: "2026-10-03T10:00:00Z",
          level: "ERROR",
          component: "tts",
          message: "provider timeout for ***",
        },
      ],
    },
  });
  assert.equal(view.title, "Logs");
  assert.equal(view.filters.length, state.LOG_FILTERS.length);
  assert.equal(view.tables[0].rows.length, 1);
  assert.equal(view.commands.length, 0);
});

test("settings view saves through backend validation only", () => {
  const view = state.buildView("settings", {
    "/admin/api/settings": {
      values: { learning_enabled: true },
      updated_at: "2026-10-03T09:00:00Z",
      applied: { learning_enabled: true, log_level: "INFO" },
      secret_values_exposed: false,
      specs: [
        {
          key: "decision_interval_seconds",
          kind: "float",
          default: 5,
          requires_restart: true,
          minimum: 0,
          maximum: 600,
        },
        { key: "learning_enabled", kind: "bool", default: true, requires_restart: false },
        {
          key: "log_level",
          kind: "enum",
          default: "INFO",
          requires_restart: false,
          choices: ["DEBUG", "INFO", "WARNING", "ERROR"],
        },
      ],
    },
  });
  assert.equal(view.commands.length, 1);
  const save = view.commands[0];
  assert.equal(save.wrap, "values");
  assert.equal(save.request.path, "/admin/api/settings");
  assert.equal(save.fields.length, 3);
  assert.equal(save.fields[0].type, "number");
  assert.deepEqual(save.fields[1].options, ["", "true", "false"]);
  assert.deepEqual(save.fields[2].options, ["", "DEBUG", "INFO", "WARNING", "ERROR"]);
  const text = JSON.stringify(view);
  assert.ok(!/sk-|Bearer |api_key/i.test(text));
});

test("every section builds a complete view from an empty payload", () => {
  for (const section of EXPECTED_SECTIONS) {
    const view = state.buildView(section, {});
    assert.ok(view.title && view.title.length > 0, section);
    assert.ok(Array.isArray(view.summary), section);
    assert.ok(Array.isArray(view.kv), section);
    assert.ok(Array.isArray(view.tables), section);
    assert.ok(Array.isArray(view.notes), section);
    assert.ok(Array.isArray(view.commands), section);
    for (const command of view.commands) {
      assert.ok(command.id && command.label, section);
      assert.ok(command.request && command.request.path, section);
      assert.match(command.request.method, /^(GET|POST)$/, section);
    }
  }
  assert.equal(state.buildView("unknown", {}).title, "Unknown section");
});

test("buildView never invents values for missing runtime data", () => {
  const view = state.buildView("runtime", {});
  assert.equal(view.summary.find((entry) => entry.label === "Backend").value, "STOPPED");
  const overview = state.buildView("overview", {});
  assert.equal(overview.summary.find((entry) => entry.label === "Runtime").value, "-");
});
