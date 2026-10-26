/* Admin dashboard state logic: DOM-free, dependency-free, unit-testable.
 *
 * Loaded as a plain script in the browser (window.AdminAppState) and
 * require()d by the Node tests. It holds no business logic and no secrets: it
 * turns the authenticated /admin/api/* projections into the exact shapes the
 * renderer draws, and it describes backend errors as operator-readable
 * results. Authorization always happens in the backend process.
 */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AdminAppState = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var POLL_BASE_MS = 4000;
  var POLL_HIDDEN_MS = 15000;
  var POLL_MAX_MS = 30000;

  var SECTIONS = [
    { id: "overview", label: "Overview", requests: ["/admin/api/overview"] },
    {
      id: "runtime",
      label: "Runtime",
      requests: ["/admin/api/runtime", "/admin/api/runtime/health"]
    },
    {
      id: "brain",
      label: "Brain",
      requests: ["/admin/api/status", "/admin/api/personality"]
    },
    { id: "providers", label: "Models / Providers", requests: ["/admin/api/providers"] },
    { id: "tts", label: "TTS", requests: ["/admin/api/tts"] },
    { id: "chat", label: "Chat", requests: ["/admin/api/chat"] },
    { id: "stream", label: "Stream", requests: ["/admin/api/stream"] },
    { id: "motion", label: "Motion", requests: ["/admin/api/motion"] },
    { id: "skills", label: "Skills", requests: ["/admin/api/skills"] },
    { id: "learning", label: "Learning", requests: ["/admin/api/learning"] },
    { id: "memory", label: "Memory", requests: ["/admin/api/memory"] },
    { id: "viewers", label: "Viewers", requests: ["/admin/api/viewers"] },
    {
      id: "personality",
      label: "Personality / Behavior",
      requests: ["/admin/api/personality", "/admin/api/overview"]
    },
    { id: "diagnostics", label: "Diagnostics", requests: ["/admin/api/diagnostics"] },
    { id: "logs", label: "Logs", requests: ["/admin/api/logs"] },
    { id: "settings", label: "Settings", requests: ["/admin/api/settings"] }
  ];

  var LOG_FILTERS = [
    {
      name: "severity",
      label: "Severity",
      type: "select",
      options: ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    },
    { name: "component", label: "Component", type: "text", placeholder: "brain, chat, tts" },
    { name: "operation", label: "Operation", type: "text", placeholder: "skill, stream, recovery" },
    { name: "since_minutes", label: "Since (minutes)", type: "number", minimum: 1, maximum: 1440 },
    { name: "limit", label: "Limit", type: "select", options: ["50", "100", "200", "500"] }
  ];

  var LOG_QUERY_KEYS = [
    "severity",
    "component",
    "operation",
    "since_minutes",
    "limit"
  ];

  var BAD_TONES = [
    "ERROR",
    "FAILED",
    "FAILURE",
    "UNAVAILABLE",
    "OFFLINE",
    "BLOCKED",
    "INVALID",
    "INCOMPATIBLE",
    "REJECTED",
    "DENIED",
    "CRITICAL",
    "FATAL"
  ];

  var WARN_TONES = [
    "DEGRADED",
    "STARTING",
    "STOPPING",
    "STOPPED",
    "PENDING",
    "QUEUED",
    "IMPORTING",
    "PROCESSING",
    "VALIDATING",
    "SYNTHESIZING",
    "REQUESTED",
    "PAUSED",
    "APPROVED",
    "PARTIAL",
    "REQUIRES_RESTART",
    "REQUIRES RESTART",
    "UNKNOWN",
    "NOT_CONFIGURED",
    "UNCONFIGURED",
    "DISABLED",
    "MISSING",
    "STALE"
  ];

  var OK_TONES = [
    "READY",
    "RUNNING",
    "LIVE",
    "CONNECTED",
    "COMPLETED",
    "COMPLETE",
    "REGISTERED",
    "VALID",
    "COMPATIBLE",
    "HEALTHY",
    "OK",
    "PASS",
    "APPLIED",
    "SAVED",
    "ENABLED",
    "ACTIVE",
    "SUCCESS",
    "CONFIGURED",
    "PLAYING",
    "SPEAKING",
    "MOVING",
    "IDLE"
  ];

  var SECRET_PATTERNS = [
    { label: "authorization header", pattern: /\bBearer\s+[A-Za-z0-9._\-+/=]{12,}/ },
    {
      label: "credential field",
      pattern: /"?(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|passwd|stream[_-]?key|client[_-]?secret|private[_-]?key|credential)"?\s*[:=]\s*"?([A-Za-z0-9._\-+/=]{16,})/
    }
  ];

  // ------------------------------------------------------------------ helpers

  function isObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  function asObject(value) {
    return isObject(value) ? value : {};
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function stopLatchEngaged(latch) {
    // The real runtime snapshot is {stopped, source, updated_at, ...}; the
    // legacy engaged/stop_requested aliases stay accepted for older fixtures.
    var value = asObject(latch);
    return value.stopped === true || value.engaged === true || value.stop_requested === true;
  }

  function get(value, path, fallback) {
    var parts = String(path === undefined || path === null ? "" : path).split(".");
    var current = value;
    var i;
    var key;
    for (i = 0; i < parts.length; i += 1) {
      if (!parts[i]) {
        continue;
      }
      if (current === null || current === undefined || typeof current !== "object") {
        return fallback;
      }
      key = parts[i];
      if (!Object.prototype.hasOwnProperty.call(current, key)) {
        return fallback;
      }
      current = current[key];
    }
    return current === undefined ? fallback : current;
  }

  function pick(payload, path, fallback) {
    return get(payload, path, fallback);
  }

  function round(value, digits) {
    var factor = Math.pow(10, digits || 0);
    return Math.round(value * factor) / factor;
  }

  function sectionById(id) {
    var wanted = String(id || "");
    var i;
    for (i = 0; i < SECTIONS.length; i += 1) {
      if (SECTIONS[i].id === wanted) {
        return SECTIONS[i];
      }
    }
    return null;
  }

  function normalizeStatus(value) {
    if (value === null || value === undefined) {
      return "";
    }
    return String(value).trim().toUpperCase().replace(/[\s-]+/g, "_");
  }

  function statusTone(value) {
    var text = normalizeStatus(value);
    if (!text) {
      return "muted";
    }
    var i;
    for (i = 0; i < BAD_TONES.length; i += 1) {
      if (text.indexOf(BAD_TONES[i]) !== -1) {
        return "bad";
      }
    }
    for (i = 0; i < WARN_TONES.length; i += 1) {
      if (text.indexOf(WARN_TONES[i]) !== -1) {
        return "warn";
      }
    }
    for (i = 0; i < OK_TONES.length; i += 1) {
      if (text.indexOf(OK_TONES[i]) !== -1) {
        return "ok";
      }
    }
    return "muted";
  }

  function humanizeCode(code) {
    if (!code || typeof code !== "string") {
      return "";
    }
    var text = code.replace(/[_-]+/g, " ").trim();
    if (!text) {
      return "";
    }
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  function describeError(status, body) {
    var payload = asObject(body);
    var code =
      typeof payload.error === "string"
        ? payload.error
        : typeof payload.code === "string"
          ? payload.code
          : null;
    var num = Number(status);
    if (!isFinite(num) || num <= 0) {
      return {
        kind: "network",
        status: 0,
        code: code,
        title: "Runtime unreachable",
        message: "The admin API did not answer. The backend may be stopped or restarting."
      };
    }
    if (num < 400 && payload.ok !== false) {
      return {
        kind: "success",
        status: num,
        code: code,
        title: "Confirmed by the backend",
        message: typeof payload.note === "string" && payload.note ? payload.note : ""
      };
    }
    if (num === 401) {
      return {
        kind: "authentication",
        status: num,
        code: code || "authentication_required",
        title: "Sign-in required",
        message: "The admin session expired or was rejected. Sign in again."
      };
    }
    if (num === 403) {
      return {
        kind: "permission",
        status: num,
        code: code || "permission_denied",
        title: "Permission denied",
        message:
          "This admin role does not hold " +
          (typeof payload.permission === "string" && payload.permission
            ? payload.permission
            : "the required permission") +
          "."
      };
    }
    if (num === 404) {
      return {
        kind: "notfound",
        status: num,
        code: code || "not_found",
        title: "Not found",
        message: humanizeCode(code) || "The requested record does not exist."
      };
    }
    if (num === 409) {
      return {
        kind: "conflict",
        status: num,
        code: code || "conflict",
        title: "Rejected by the runtime",
        message: humanizeCode(code) || "The runtime is already in that state."
      };
    }
    if (num === 413) {
      return {
        kind: "validation",
        status: num,
        code: code || "request_too_large",
        title: "Input rejected",
        message: humanizeCode(code) || "The request is larger than the admin API accepts."
      };
    }
    if (num === 429) {
      return {
        kind: "throttled",
        status: num,
        code: code || "login_throttled",
        title: "Too many attempts",
        message:
          "Retry after " +
          formatDuration(Number(payload.retry_after_seconds)) +
          "."
      };
    }
    if (num === 503) {
      return {
        kind: "unavailable",
        status: num,
        code: code || "unavailable",
        title: "Dependency unavailable",
        message: humanizeCode(code) || "The backing service is not available in this runtime."
      };
    }
    if (num >= 500) {
      return {
        kind: "server",
        status: num,
        code: code || "server_error",
        title: "Backend error",
        message: "The backend reported a failure. Check Diagnostics and Logs."
      };
    }
    if (num >= 400) {
      return {
        kind: "validation",
        status: num,
        code: code || "invalid_request",
        title: "Rejected by validation",
        message: humanizeCode(code) || "The backend rejected this request."
      };
    }
    return {
      kind: "unknown",
      status: num,
      code: code,
      title: "Unhandled response",
      message: humanizeCode(code) || "The backend returned an unexpected response."
    };
  }

  function formatValue(value) {
    if (value === null || value === undefined || value === "") {
      return "-";
    }
    if (typeof value === "boolean") {
      return value ? "yes" : "no";
    }
    if (typeof value === "number") {
      return isFinite(value) ? String(round(value, 4)) : "-";
    }
    if (Array.isArray(value)) {
      if (!value.length) {
        return "-";
      }
      return value
        .map(function (item) {
          return isObject(item) ? JSON.stringify(item) : String(item);
        })
        .join(", ");
    }
    if (isObject(value)) {
      try {
        return JSON.stringify(value);
      } catch (error) {
        return "-";
      }
    }
    return String(value);
  }

  function formatTimestamp(value) {
    if (value === null || value === undefined || value === "") {
      return "-";
    }
    if (typeof value === "number" && isFinite(value)) {
      var ms = value < 1e12 ? value * 1000 : value;
      try {
        return new Date(ms).toISOString().replace(".000Z", "Z");
      } catch (error) {
        return String(value);
      }
    }
    if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value)) {
      return value.length > 19 ? value.slice(0, 19).replace("T", " ") : value;
    }
    return String(value);
  }

  function formatDuration(seconds) {
    var value = Number(seconds);
    if (!isFinite(value) || value < 0) {
      return "-";
    }
    if (value < 1) {
      return Math.round(value * 1000) + " ms";
    }
    if (value < 60) {
      return round(value, 1) + " s";
    }
    if (value < 3600) {
      return Math.floor(value / 60) + "m " + round(value % 60, 0) + "s";
    }
    return Math.floor(value / 3600) + "h " + Math.floor((value % 3600) / 60) + "m";
  }

  function nextPollDelay(options) {
    var opts = options || {};
    var failures = Math.max(0, Number(opts.failures) || 0);
    var base = opts.hidden ? POLL_HIDDEN_MS : POLL_BASE_MS;
    if (!failures) {
      return base;
    }
    return Math.min(POLL_MAX_MS, base * Math.pow(2, failures));
  }

  function findSecretExposure(text) {
    if (typeof text !== "string" || !text) {
      return [];
    }
    var found = [];
    var i;
    for (i = 0; i < SECRET_PATTERNS.length; i += 1) {
      if (SECRET_PATTERNS[i].pattern.test(text)) {
        found.push(SECRET_PATTERNS[i].label);
      }
    }
    return found;
  }

  // -------------------------------------------------------------------- state

  function createState() {
    return {
      authenticated: false,
      sessionChecked: false,
      role: null,
      permissions: [],
      section: "overview",
      loading: false,
      failures: 0,
      pages: {},
      errors: {},
      notice: null,
      command: { running: false, label: null },
      login: { pending: false, error: null },
      filters: { logs: { limit: "200" } },
      commandResult: null,
      lastRefreshAt: null
    };
  }

  function reduce(state, action) {
    var current = state || createState();
    var act = action || {};
    var next;
    var pages;
    var errors;
    var filters;
    switch (act.type) {
      case "session":
        next = Object.assign({}, current, {
          authenticated: act.authenticated === true,
          sessionChecked: true,
          role: act.role || null,
          permissions: asArray(act.permissions)
        });
        if (!next.authenticated) {
          next.pages = {};
          next.errors = {};
          next.commandResult = null;
        }
        return next;
      case "navigate":
        return Object.assign({}, current, {
          section: sectionById(act.section) ? act.section : current.section,
          notice: null
        });
      case "loading":
        return Object.assign({}, current, {
          loading: act.loading === true
        });
      case "data":
        pages = Object.assign({}, current.pages);
        pages[act.section] = { payload: asObject(act.payload), at: act.at || null };
        errors = Object.assign({}, current.errors);
        delete errors[act.section];
        return Object.assign({}, current, {
          pages: pages,
          errors: errors,
          failures: 0,
          lastRefreshAt: act.at || null
        });
      case "error":
        errors = Object.assign({}, current.errors);
        errors[act.section] = act.error || null;
        return Object.assign({}, current, { errors: errors });
      case "poll":
        return Object.assign({}, current, {
          failures: Math.max(0, Number(act.failures) || 0)
        });
      case "notice":
        return Object.assign({}, current, {
          notice: act.notice || null,
          commandResult: act.result === undefined ? current.commandResult : act.result
        });
      case "command":
        return Object.assign({}, current, {
          command: {
            running: act.running === true,
            label: act.running === true ? act.label || null : null
          }
        });
      case "login_pending":
        return Object.assign({}, current, {
          login: { pending: true, error: null }
        });
      case "login_error":
        return Object.assign({}, current, {
          login: { pending: false, error: act.error || null }
        });
      case "filters":
        filters = Object.assign({}, current.filters);
        filters[act.section] = Object.assign({}, filters[act.section] || {}, act.values || {});
        return Object.assign({}, current, { filters: filters });
      case "logout":
        return Object.assign(createState(), { sessionChecked: true });
      default:
        return current;
    }
  }

  function buildQuery(section, filters) {
    if (section !== "logs") {
      return "";
    }
    var values = asObject(filters);
    var parts = [];
    var i;
    var key;
    var value;
    for (i = 0; i < LOG_QUERY_KEYS.length; i += 1) {
      key = LOG_QUERY_KEYS[i];
      value = values[key];
      if (value === null || value === undefined || String(value).trim() === "") {
        continue;
      }
      parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(String(value).trim()));
    }
    return parts.length ? "?" + parts.join("&") : "";
  }

  // -------------------------------------------------------------- view build

  function stat(label, value, tone) {
    return {
      label: label,
      value: formatValue(value),
      tone: tone === undefined ? statusTone(value) : tone
    };
  }

  function item(label, value, tone) {
    return {
      label: label,
      value: formatValue(value),
      tone: tone === undefined ? statusTone(value) : tone
    };
  }

  function note(text, tone) {
    return { text: text, tone: tone || "muted" };
  }

  function table(title, columns, rows, empty) {
    return { title: title, columns: columns, rows: rows, empty: empty || "No records." };
  }

  function row(cells, actions) {
    var payload = { cells: cells };
    if (actions && actions.length) {
      payload.actions = actions;
    }
    return payload;
  }

  function action(label, method, path, body, options) {
    var opts = options || {};
    return {
      label: label,
      tone: opts.tone || "default",
      confirm: opts.confirm || null,
      request: { method: method, path: path, body: body || null }
    };
  }

  function field(name, label, type, options) {
    var opts = options || {};
    return {
      name: name,
      label: label,
      type: type || "text",
      placeholder: opts.placeholder || "",
      required: opts.required === true,
      value: opts.value === undefined ? "" : opts.value,
      options: asArray(opts.options),
      minimum: opts.minimum === undefined ? null : opts.minimum,
      maximum: opts.maximum === undefined ? null : opts.maximum
    };
  }

  function componentRows(components) {
    var source = asObject(components);
    var keys = Object.keys(source).sort();
    return keys.map(function (name) {
      var entry = asObject(source[name]);
      var detail = entry.detail;
      if (detail === undefined || detail === null || detail === "") {
        detail = entry.reason;
      }
      return row([
        name,
        formatValue(entry.status),
        formatValue(detail),
        entry.required === false ? "optional" : "required"
      ]);
    });
  }

  function errorRows(entries) {
    return asArray(entries).map(function (entry) {
      var record = asObject(entry);
      return row([
        formatTimestamp(record.at || record.timestamp || record.time),
        formatValue(record.component || record.source),
        formatValue(record.operation || record.event),
        formatValue(record.message || record.error)
      ]);
    });
  }

  function providerRow(role, detail) {
    var record = asObject(detail);
    var configured = record.configured;
    var reachable = record.reachable;
    var latency = record.latency;
    if (isObject(latency)) {
      latency = latency.last_ms || latency.average_ms || latency.p50_ms;
    }
    return row([
      role,
      formatValue(record.provider),
      formatValue(record.model || record.voice || record.platform),
      configured === undefined ? "-" : configured === true ? "configured" : "not configured",
      reachable === undefined ? "-" : reachable === true ? "reachable" : "unavailable",
      latency === undefined || latency === null ? "-" : formatDuration(latency)
    ]);
  }

  function kvGroup(title, items) {
    return { title: title, items: items };
  }

  function settingsFields(specs) {
    return asArray(specs).map(function (raw) {
      var spec = asObject(raw);
      var key = String(spec.key || "");
      var label = humanizeCode(key);
      if (spec.kind === "bool") {
        return field(key, label, "select", { options: ["", "true", "false"] });
      }
      if (spec.kind === "enum") {
        return field(key, label, "select", { options: [""].concat(asArray(spec.choices)) });
      }
      return field(key, label, "number", {
        minimum: spec.minimum,
        maximum: spec.maximum,
        placeholder:
          spec.default === null || spec.default === undefined
            ? "keeps the environment default"
            : String(spec.default)
      });
    });
  }

  function buildOverview(payload) {
    var d = asObject(pick(payload, "/admin/api/overview", {}));
    var brain = asObject(d.brain);
    var stream = asObject(d.stream);
    var speech = asObject(d.speech);
    var queues = asObject(d.queues);
    var chatQueue = asObject(queues.chat);
    var speechQueue = asObject(queues.speech);
    var behavior = asObject(d.behavior);
    var process = asObject(d.process);
    return {
      title: "Overview",
      subtitle: "Generated " + formatTimestamp(d.generated_at),
      summary: [
        stat("Runtime", d.status),
        stat("Environment", d.environment, "muted"),
        stat("Brain tick", brain.tick_in_progress === true ? "IN PROGRESS" : "IDLE"),
        stat("Last tick", formatDuration(brain.last_tick_age_seconds) === "-" ? "-" : formatDuration(brain.last_tick_age_seconds), "muted"),
        stat("Viewers", d.viewer_count),
        stat("Stream", stream.state),
        stat("Speech", speech.state)
      ],
      kv: [
        kvGroup("Process", [
          item("PID", process.pid),
          item("Uptime", formatDuration(process.uptime_seconds), "muted"),
          item("State", process.state),
          item("Instance", process.instance_id, "muted")
        ]),
        kvGroup("Brain", [
          item("Active skill", d.active_skill),
          item("Active goal", d.active_goal),
          item("Behavior", behavior.state),
          item("Previous behavior", behavior.previous_state, "muted"),
          item("Ticks", brain.tick_count, "muted"),
          item("Uptime", formatDuration(brain.uptime_seconds), "muted")
        ]),
        kvGroup("Queues", [
          item("Chat", chatQueue.queued_count, "muted"),
          item("Chat capacity", chatQueue.queue_capacity, "muted"),
          item("Speech", speechQueue.queue_length, "muted"),
          item("Speech capacity", speechQueue.queue_capacity, "muted"),
          item("Chat source", asObject(queues.chat_source).state, "muted")
        ]),
        kvGroup("Stream", [
          item("State", stream.state),
          item("Live", stream.live),
          item("Provider", stream.provider, "muted"),
          item(
            "Stop latch",
            stopLatchEngaged(stream.stop_latch || d.stream_stop_latch) ? "ENGAGED" : "released",
            stopLatchEngaged(stream.stop_latch || d.stream_stop_latch) ? "warn" : "muted"
          )
        ])
      ],
      tables: [
        table(
          "Service health",
          ["Component", "Status", "Detail", "Requirement"],
          componentRows(d.components),
          "No health components reported."
        ),
        table(
          "Recent errors",
          ["Time", "Component", "Operation", "Message"],
          errorRows(d.recent_errors),
          "No recent errors."
        )
      ],
      notes: [
        note("Health and state are read from the running backend on every refresh.", "muted")
      ],
      commands: []
    };
  }

  function buildRuntime(payload) {
    var d = asObject(pick(payload, "/admin/api/runtime", {}));
    var healthPayload = asObject(pick(payload, "/admin/api/runtime/health", {}));
    var runtime = asObject(d.runtime);
    var statusFile = asObject(runtime.status_file);
    var lock = asObject(runtime.lock);
    var admin = asObject(d.admin);
    var dispatched = asObject(d.dispatched);
    var health = asObject(healthPayload.health);
    var httpStatus = asObject(runtime.http_status);
    return {
      title: "Runtime",
      subtitle: "Lifecycle controller and instance lock",
      summary: [
        stat("Backend", runtime.running === true ? "RUNNING" : "STOPPED"),
        stat("State", statusFile.state || "unknown"),
        stat("PID", runtime.pid, "muted"),
        stat("Environment", statusFile.environment, "muted"),
        stat("Health", health.status),
        stat("Stop requested", runtime.stop_requested === true ? "YES" : "NO", runtime.stop_requested ? "warn" : "muted")
      ],
      kv: [
        kvGroup("Process", [
          item("PID", runtime.pid),
          item("Instance", statusFile.instance_id, "muted"),
          item("Host", statusFile.host, "muted"),
          item("Brain port", statusFile.brain_port, "muted"),
          item("Chat port", statusFile.chat_port, "muted"),
          item("Admin port", statusFile.admin_port === undefined ? admin.port : statusFile.admin_port, "muted")
        ]),
        kvGroup("Lock", [
          item("Lock held", lock.held),
          item("Lock role", lock.role, "muted"),
          item("Lock pid", lock.pid, "muted"),
          item("On-disk state", statusFile.on_disk_state, "muted"),
          item("HTTP probe", httpStatus.reachable === true ? "reachable" : "unreachable")
        ]),
        kvGroup("Admin dispatch", [
          item("Last action", dispatched.action, "muted"),
          item("Dispatch pid", dispatched.pid, "muted"),
          item("Dispatch running", dispatched.running),
          item("Dispatch log", dispatched.log, "muted"),
          item("Admin pid", admin.pid, "muted"),
          item("Admin state", admin.state, "muted")
        ])
      ],
      tables: [
        table(
          "Backend health",
          ["Component", "Status", "Detail", "Requirement"],
          componentRows(health.components),
          "No health report available."
        )
      ],
      notes: [
        note(
          "Start, stop and restart are dispatched to the existing lifecycle controller; the backend confirms the result.",
          "muted"
        )
      ],
      commands: [
        {
          id: "runtime-start",
          label: "Start",
          tone: "primary",
          request: { method: "POST", path: "/admin/api/runtime/start" }
        },
        {
          id: "runtime-stop",
          label: "Stop",
          tone: "danger",
          confirm: "Stop the running backend? Viewers and the stream will go offline.",
          request: { method: "POST", path: "/admin/api/runtime/stop" }
        },
        {
          id: "runtime-restart",
          label: "Restart",
          tone: "danger",
          confirm: "Restart the backend? The current process will stop and a new one will start.",
          request: { method: "POST", path: "/admin/api/runtime/restart" }
        }
      ]
    };
  }

  function buildBrain(payload) {
    var status = asObject(pick(payload, "/admin/api/status", {}));
    var personality = asObject(pick(payload, "/admin/api/personality", {}));
    var brain = asObject(status.brain);
    var queues = asObject(status.queues);
    var behavior = asObject(personality.behavior);
    return {
      title: "Brain",
      subtitle: "Decision loop, behavior and active selection",
      summary: [
        stat("Runtime", status.status),
        stat("Tick", brain.tick_in_progress === true ? "IN PROGRESS" : "IDLE"),
        stat("Ticks", brain.tick_count, "muted"),
        stat("Active skill", status.active_skill),
        stat("Behavior", behavior.state),
        stat("Goal", personality.goal)
      ],
      kv: [
        kvGroup("Tick", [
          item("Tick count", brain.tick_count, "muted"),
          item("Tick in progress", brain.tick_in_progress),
          item("Last tick", formatTimestamp(brain.last_tick_at), "muted"),
          item("Last tick age", formatDuration(brain.last_tick_age_seconds)),
          item("Uptime", formatDuration(brain.uptime_seconds), "muted")
        ]),
        kvGroup("Selection", [
          item("Active skill", status.active_skill),
          item("Current goal", personality.goal),
          item("Behavior", behavior.state),
          item("Previous behavior", behavior.previous_state, "muted"),
          item("Speech style", personality.personality ? personality.personality.speech_style : "-", "muted")
        ]),
        kvGroup("Queues", [
          item("Chat", asObject(queues.chat).queued_count, "muted"),
          item("Speech", asObject(queues.speech).queue_length, "muted"),
          item("Chat source", asObject(queues.chat_source).state, "muted")
        ]),
        kvGroup("Memory context", [
          item("Short-term turns", get(personality, "personalization.short_term_turns"), "muted"),
          item("Relationship context", get(personality, "personalization.relationship_level"), "muted")
        ])
      ],
      tables: [
        table(
          "Component status",
          ["Component", "Status", "Detail", "Requirement"],
          Object.keys(asObject(status.health)).map(function (name) {
            return row([name, formatValue(status.health[name])]);
          }),
          "No status report available."
        ),
        table(
          "Recent errors",
          ["Time", "Component", "Operation", "Message"],
          errorRows(status.recent_errors),
          "No recent errors."
        )
      ],
      notes: [
        note("Behavior, goal and skill are read from the live Brain; nothing is simulated here.", "muted")
      ],
      commands: []
    };
  }

  function buildProviders(payload) {
    var d = asObject(pick(payload, "/admin/api/providers", {}));
    var motion = asObject(d.motion);
    return {
      title: "Models / Providers",
      subtitle: "Configured providers reported by the backend",
      summary: [
        stat("LLM", get(d, "llm.provider")),
        stat("TTS", get(d, "tts.provider")),
        stat("Chat", get(d, "chat.provider")),
        stat("Stream", get(d, "stream.provider")),
        stat("Secrets exposed", d.secrets_exposed === true ? "YES" : "NO", d.secrets_exposed === true ? "bad" : "ok")
      ],
      kv: [
        kvGroup("LLM", [
          item("Model", get(d, "llm.model")),
          item("Endpoint", get(d, "llm.endpoint"), "muted"),
          item("Configured", get(d, "llm.configured")),
          item("Reachable", get(d, "llm.reachable")),
          item("Credentials", get(d, "llm.credentials"), "muted"),
          item("Mock provider", get(d, "llm.mock") === true ? "YES" : "NO", get(d, "llm.mock") === true ? "warn" : "ok")
        ]),
        kvGroup("TTS", [
          item("Model", get(d, "tts.model")),
          item("Voice", get(d, "tts.voice")),
          item("Configured", get(d, "tts.configured")),
          item("Reachable", get(d, "tts.reachable")),
          item("Latency", formatDuration(get(d, "tts.latency")))
        ]),
        kvGroup("Chat", [
          item("Provider", get(d, "chat.provider")),
          item("Configured", get(d, "chat.configured")),
          item("Channels", get(d, "chat.channels"))
        ]),
        kvGroup("Stream", [
          item("Provider", get(d, "stream.provider")),
          item("Platform", get(d, "stream.platform"), "muted"),
          item("Configured", get(d, "stream.configured")),
          item("Real transport", get(d, "stream.real_transport"))
        ]),
        kvGroup("Motion services", [
          item("Generation", motion.generation_enabled),
          item("Processing", motion.processing_enabled),
          item("Import", motion.import_enabled)
        ])
      ],
      tables: [
        table(
          "Provider matrix",
          ["Role", "Provider", "Model / Voice", "Configured", "Reachable", "Latency"],
          [
            providerRow("LLM", d.llm),
            providerRow("TTS", d.tts),
            providerRow("Chat", d.chat),
            providerRow("Stream", d.stream)
          ],
          "No providers reported."
        )
      ],
      notes: [
        note("Credentials are held by the backend process and are never returned to this browser.", "muted")
      ],
      commands: []
    };
  }

  function buildTts(payload) {
    var d = asObject(pick(payload, "/admin/api/tts", {}));
    var state = asObject(d.state);
    var health = asObject(d.health);
    var request = asObject(d.request);
    return {
      title: "TTS",
      subtitle: "Speech service state and provider health",
      summary: [
        stat("Enabled", state.enabled),
        stat("State", state.state),
        stat("Provider", health.provider || state.provider),
        stat("Voice", health.voice || state.voice),
        stat("Queue", state.queue_length, "muted"),
        stat("Latency", formatDuration(health.latency && (health.latency.last_ms || health.latency.average_ms)))
      ],
      kv: [
        kvGroup("Provider", [
          item("Provider", health.provider),
          item("Model", health.model),
          item("Voice", health.voice),
          item("Configured", health.configured),
          item("Reachable", health.reachable),
          item("Mock", health.mock === true ? "YES" : "NO", health.mock === true ? "warn" : "ok")
        ]),
        kvGroup("Queue", [
          item("Queue length", state.queue_length, "muted"),
          item("Queue capacity", state.queue_capacity, "muted"),
          item("Speaking", state.speaking),
          item("Current request", state.current_request_id, "muted")
        ]),
        kvGroup("Last lookup", [
          item("Request", request.request_id),
          item("State", request.state),
          item("Error", request.error, request.error ? "bad" : "muted")
        ])
      ],
      tables: [
        table(
          "Recent speech errors",
          ["Time", "Component", "Operation", "Message"],
          errorRows(d.recent_errors),
          "No recent speech errors."
        )
      ],
      notes: [
        note("A test utterance is synthesized by the real provider configured in the backend.", "muted")
      ],
      commands: [
        {
          id: "tts-test",
          label: "Speak test line",
          tone: "primary",
          note: "Runs through the existing speech queue.",
          fields: [
            field("text", "Text", "text", {
              required: true,
              placeholder: "Admin check",
              value: "Admin check"
            })
          ],
          request: { method: "POST", path: "/admin/api/tts/test" }
        }
      ]
    };
  }

  function buildChat(payload) {
    var d = asObject(pick(payload, "/admin/api/chat", {}));
    var queue = asObject(d.queue);
    var source = asObject(d.source);
    var health = asObject(d.health);
    return {
      title: "Chat",
      subtitle: "Message intake and queue",
      summary: [
        stat("Source", source.state || source.provider),
        stat("Enabled", source.enabled),
        stat("Queue", queue.queued_count, "muted"),
        stat("Capacity", queue.queue_capacity, "muted"),
        stat("Dropped", queue.dropped_count, queue.dropped_count ? "warn" : "muted"),
        stat("Reconnects", source.reconnect_count, "muted")
      ],
      kv: [
        kvGroup("Source", [
          item("Provider", source.provider || health.provider),
          item("Channel", source.channel, "muted"),
          item("State", source.state),
          item("Enabled", source.enabled),
          item("Connected at", formatTimestamp(source.connected_at), "muted"),
          item("Last error", source.last_error, source.last_error ? "bad" : "muted")
        ]),
        kvGroup("Queue", [
          item("Queued", queue.queued_count, "muted"),
          item("Capacity", queue.queue_capacity, "muted"),
          item("Processed", queue.processed_count, "muted"),
          item("Dropped", queue.dropped_count, "muted"),
          item("Rejected", queue.rejected_count, "muted"),
          item("Rate limit", queue.rate_limit_per_minute, "muted")
        ]),
        kvGroup("Health", [
          item("Status", health.status),
          item("Reachable", health.reachable),
          item("Last message", formatTimestamp(health.last_message_at), "muted")
        ])
      ],
      tables: [
        table(
          "Recent chat errors",
          ["Time", "Component", "Operation", "Message"],
          errorRows(d.recent_errors),
          "No recent chat errors."
        )
      ],
      notes: [
        note("Reconnect restarts the existing intake runner; no second chat pipeline is created.", "muted")
      ],
      commands: [
        {
          id: "chat-reconnect",
          label: "Reconnect",
          tone: "default",
          note: "Restarts the existing chat ingest runner.",
          request: { method: "POST", path: "/admin/api/chat/reconnect" }
        }
      ]
    };
  }

  function buildStream(payload) {
    var d = asObject(pick(payload, "/admin/api/stream", {}));
    var state = asObject(d.state);
    var health = asObject(d.health);
    var latch = asObject(d.stop_latch);
    var latchEngaged = stopLatchEngaged(latch);
    return {
      title: "Stream",
      subtitle: "Transport state and encoder",
      summary: [
        stat("State", state.state),
        stat("Live", state.live),
        stat("Provider", d.provider, "muted"),
        stat("Viewers", state.viewer_count, "muted"),
        stat("Camera", state.camera, "muted"),
        stat("Stop latch", latchEngaged ? "ENGAGED" : "released", latchEngaged ? "warn" : "muted")
      ],
      kv: [
        kvGroup("Transport", [
          item("State", state.state),
          item("Live", state.live),
          item("Provider", d.provider),
          item("Platform", state.platform, "muted"),
          item("Target", state.target || state.server_url, "muted"),
          item("Reconnects", state.reconnect_count, "muted")
        ]),
        kvGroup("Camera and encoder", [
          item("Camera", state.camera, "muted"),
          item("Camera state", state.camera_state, "muted"),
          item("Encoder", state.encoder, "muted"),
          item("Encoder state", state.encoder_state, "muted"),
          item("Bitrate", state.bitrate, "muted"),
          item("FPS", state.fps, "muted")
        ]),
        kvGroup("Health", [
          item("Status", health.status),
          item("Reachable", health.reachable),
          item("Last error", health.last_error || state.last_error, "muted")
        ]),
        kvGroup("Stop latch", [
          item("Engaged", latchEngaged, latchEngaged ? "warn" : "muted"),
          item("Reason", latch.reason || latch.source, "muted"),
          item("Requested at", formatTimestamp(latch.requested_at || latch.updated_at), "muted")
        ])
      ],
      tables: [],
      notes: [
        note(
          "Commands route through the existing stream handler, so an engaged STOP_STREAM latch is respected.",
          "muted"
        )
      ],
      commands: [
        {
          id: "stream-start",
          label: "Start",
          tone: "primary",
          fields: [field("camera", "Camera", "text", { placeholder: "default" })],
          request: { method: "POST", path: "/admin/api/stream/start" }
        },
        {
          id: "stream-stop",
          label: "Stop",
          tone: "danger",
          confirm: "Stop the stream? Viewers will lose the feed.",
          request: { method: "POST", path: "/admin/api/stream/stop" }
        },
        {
          id: "stream-reconnect",
          label: "Reconnect",
          tone: "default",
          confirm: "Reconnect the stream transport?",
          fields: [field("camera", "Camera", "text", { placeholder: "default" })],
          request: { method: "POST", path: "/admin/api/stream/reconnect" }
        }
      ]
    };
  }

  function buildMotion(payload) {
    var d = asObject(pick(payload, "/admin/api/motion", {}));
    var importState = asObject(d.import);
    var processing = asObject(d.processing);
    var generation = asObject(d.generation);
    var motions = asArray(d.motions);
    var rows = motions.map(function (raw) {
      var motion = asObject(raw);
      return row(
        [
          formatValue(motion.motion_id),
          formatValue(motion.source),
          formatValue(motion.version),
          formatDuration(motion.duration_seconds),
          formatValue(motion.fps),
          formatValue(motion.quality_score),
          formatValue(motion.skeleton_status),
          formatValue(motion.validation_status),
          formatValue(motion.animation_reference),
          formatValue(motion.processing_version),
          formatValue(motion.cleanup_status || (motion.last_error ? "FAILED" : "-"))
        ],
        [
          {
            label: "Process",
            tone: "default",
            request: {
              method: "POST",
              path: "/admin/api/motion/process",
              body: { source_motion_id: motion.motion_id }
            }
          }
        ]
      );
    });
    return {
      title: "Motion",
      subtitle: "Acquisition, retargeting and registration",
      summary: [
        stat("Motions", d.motion_count, "muted"),
        stat("Import service", importState.enabled),
        stat("Processing service", processing.enabled),
        stat("Generation service", generation.enabled),
        stat("Gate", "VALID + COMPATIBLE", "warn")
      ],
      kv: [
        kvGroup("Import", [
          item("Enabled", importState.enabled),
          item("State", importState.state),
          item("Queued", importState.queued_count, "muted"),
          item("Reason", importState.reason, "muted")
        ]),
        kvGroup("Processing", [
          item("Enabled", processing.enabled),
          item("State", processing.state),
          item("Jobs", processing.job_count, "muted"),
          item("Reason", processing.reason, "muted")
        ]),
        kvGroup("Generation", [
          item("Enabled", generation.enabled),
          item("State", generation.state),
          item("Jobs", generation.job_count, "muted"),
          item("Reason", generation.reason, "muted")
        ])
      ],
      tables: [
        table(
          "Motion library",
          [
            "Motion",
            "Source",
            "Version",
            "Duration",
            "FPS",
            "Quality",
            "Skeleton",
            "Validation",
            "Animation reference",
            "Processing",
            "Cleanup"
          ],
          rows,
          "No motion records."
        )
      ],
      notes: [
        note(String(d.gate || "VALID + COMPATIBLE is enforced by the backend."), "muted")
      ],
      commands: [
        {
          id: "motion-import",
          label: "Import motion",
          tone: "primary",
          note: "Source file is preserved; an existing motion id is never overwritten.",
          upload: true,
          fields: [
            field("file", "Motion file", "file", { required: true }),
            field("motion_id", "Motion id", "text", { required: true, placeholder: "phase23_walking" }),
            field("name", "Name", "text", { placeholder: "Admin import" }),
            field("clip_name", "Clip name", "text"),
            field("tags", "Tags", "text", { placeholder: "admin, walking" }),
            field("skill_category", "Skill category", "select", {
              options: ["", "MOVEMENT", "GESTURE", "EXPRESSION", "IDLE", "ACTION", "SOCIAL", "DANCE", "COMBO"]
            })
          ],
          request: { method: "POST", path: "/admin/api/motion/import" }
        },
        {
          id: "motion-import-path",
          label: "Import project file",
          tone: "default",
          note: "Uses an existing file inside the project root.",
          fields: [
            field("source_path", "Project path", "text", {
              required: true,
              placeholder: "unity/3d_unity/Assets/animation/walking/Walking.fbx"
            }),
            field("motion_id", "Motion id", "text", { required: true }),
            field("skill_category", "Skill category", "select", {
              options: ["", "MOVEMENT", "GESTURE", "EXPRESSION", "IDLE", "ACTION", "SOCIAL", "DANCE", "COMBO"]
            })
          ],
          request: { method: "POST", path: "/admin/api/motion/import" }
        },
        {
          id: "motion-generate",
          label: "Generate from text",
          tone: "default",
          note: "Runs the existing text to motion service.",
          fields: [
            field("prompt", "Prompt", "text", { required: true, placeholder: "wave hello" }),
            field("duration_seconds", "Duration (s)", "number", { minimum: 0.5, maximum: 10, value: "2.5" }),
            field("motion_type", "Motion type", "text", { placeholder: "general" }),
            field("style", "Style", "text"),
            field("speed", "Speed", "number", { minimum: 0.1, maximum: 3, value: "1" })
          ],
          request: { method: "POST", path: "/admin/api/motion/generate" }
        },
        {
          id: "motion-process",
          label: "Process / retarget",
          tone: "default",
          fields: [
            field("source_motion_id", "Source motion", "text", { required: true }),
            field("processed_motion_id", "Processed id", "text", { placeholder: "auto" }),
            field("processing_version", "Version", "number", { minimum: 1, value: "1" }),
            field("skill_category", "Skill category", "select", {
              options: ["", "MOVEMENT", "GESTURE", "EXPRESSION", "IDLE", "ACTION", "SOCIAL", "DANCE", "COMBO"]
            })
          ],
          request: { method: "POST", path: "/admin/api/motion/process" }
        },
        {
          id: "motion-approve",
          label: "Approve",
          tone: "default",
          note: "Approval is refused unless the job is READY, VALID and COMPATIBLE.",
          fields: [field("request_id", "Processing request", "text", { required: true })],
          request: { method: "POST", path: "/admin/api/motion/approve" }
        },
        {
          id: "motion-register",
          label: "Register",
          tone: "default",
          note: "Registration is refused unless the job is APPROVED.",
          fields: [field("request_id", "Processing request", "text", { required: true })],
          request: { method: "POST", path: "/admin/api/motion/register" }
        },
        {
          id: "motion-job",
          label: "Inspect job",
          tone: "default",
          fields: [
            field("kind", "Kind", "select", { options: ["processing", "import", "generation"], value: "processing" }),
            field("request_id", "Request id", "text", { required: true })
          ],
          request: { method: "GET", path: "/admin/api/motion/job" }
        }
      ]
    };
  }

  function buildSkills(payload) {
    var d = asObject(pick(payload, "/admin/api/skills", {}));
    var runtime = asObject(d.runtime);
    var rows = asArray(d.skills).map(function (raw) {
      var skill = asObject(raw);
      var learning = asObject(skill.learning);
      var enabled = skill.enabled === true;
      var skillId = String(skill.skill_id || "");
      return row(
        [
          skillId,
          formatValue(skill.name),
          formatValue(skill.category),
          enabled ? "enabled" : "disabled",
          formatValue(skill.conditions),
          formatValue(skill.animation_reference || skill.motion_id),
          formatValue(skill.source_motion_id),
          formatValue(learning.trials),
          formatValue(learning.successes),
          formatValue(learning.failures),
          formatValue(learning.average_score),
          formatTimestamp(learning.last_used)
        ],
        [
          enabled
            ? {
                label: "Disable",
                tone: "default",
                request: {
                  method: "POST",
                  path: "/admin/api/skills/disable",
                  body: { skill_id: skillId }
                }
              }
            : {
                label: "Enable",
                tone: "primary",
                request: {
                  method: "POST",
                  path: "/admin/api/skills/enable",
                  body: { skill_id: skillId }
                }
              }
        ]
      );
    });
    return {
      title: "Skills",
      subtitle: "Library, conditions and execution history",
      summary: [
        stat("Skills", d.count, "muted"),
        stat("Library", d.enabled),
        stat("Executor", runtime.enabled),
        stat("Executions", runtime.execution_count, "muted"),
        stat("Failures", runtime.failure_count, runtime.failure_count ? "warn" : "muted")
      ],
      kv: [
        kvGroup("Runtime", [
          item("Enabled", runtime.enabled),
          item("Executions", runtime.execution_count, "muted"),
          item("Successes", runtime.success_count, "muted"),
          item("Failures", runtime.failure_count, "muted"),
          item("Last skill", runtime.last_skill_id, "muted"),
          item("Last error", runtime.last_error, "muted")
        ])
      ],
      tables: [
        table(
          "Skill library",
          [
            "Skill",
            "Name",
            "Category",
            "State",
            "Conditions",
            "Animation",
            "Source motion",
            "Trials",
            "Successes",
            "Failures",
            "Score",
            "Last used"
          ],
          rows,
          "No skills registered."
        )
      ],
      notes: [
        note(
          "Enable and disable only toggle the existing library entry; conditions, cooldowns and the allowlist still run in the SkillExecutor.",
          "muted"
        )
      ],
      commands: []
    };
  }

  function buildLearning(payload) {
    var d = asObject(pick(payload, "/admin/api/learning", {}));
    var state = asObject(d.state);
    var knowledge = asArray(d.knowledge);
    var compositions = asArray(d.compositions);
    var trials = 0;
    var successes = 0;
    var failures = 0;
    knowledge.forEach(function (raw) {
      var record = asObject(raw);
      trials += Number(record.trials) || 0;
      successes += Number(record.successes) || 0;
      failures += Number(record.failures) || 0;
    });
    return {
      title: "Learning",
      subtitle: "SkillKnowledge, ranking and learned compositions",
      summary: [
        stat("Learning", d.enabled),
        stat("Knowledge rows", knowledge.length, "muted"),
        stat("Trials", trials, "muted"),
        stat("Successes", successes, "muted"),
        stat("Failures", failures, failures ? "warn" : "muted"),
        stat("Compositions", compositions.length, "muted")
      ],
      kv: [
        kvGroup("Service", [
          item("Enabled", d.enabled),
          item("Motions", state.motion_count, "muted"),
          item("Knowledge entries", state.knowledge_entries, "muted"),
          item("Composites", state.composite_count, "muted"),
          item("Exploration", state.exploration_epsilon, "muted"),
          item("Last context", state.last_context_key, "muted")
        ])
      ],
      tables: [
        table(
          "SkillKnowledge ranking",
          [
            "Skill",
            "Context",
            "Trials",
            "Successes",
            "Failures",
            "Interruptions",
            "Success rate",
            "Score",
            "Confidence",
            "Last used"
          ],
          knowledge.map(function (raw) {
            var record = asObject(raw);
            return row([
              formatValue(record.skill_id),
              formatValue(record.context_key),
              formatValue(record.trials),
              formatValue(record.successes),
              formatValue(record.failures),
              formatValue(record.interruptions),
              formatValue(record.success_rate),
              formatValue(record.average_score),
              formatValue(record.confidence),
              formatTimestamp(record.last_used)
            ]);
          }),
          "No learning knowledge recorded."
        ),
        table(
          "Learned compositions",
          ["Composition", "Context", "Skills", "Score", "Uses", "Last used"],
          compositions.map(function (raw) {
            var record = asObject(raw);
            return row([
              formatValue(record.composite_id || record.name),
              formatValue(record.context_key),
              formatValue(record.skill_ids || record.steps),
              formatValue(record.average_score || record.score),
              formatValue(record.uses),
              formatTimestamp(record.last_used)
            ]);
          }),
          "No learned compositions."
        )
      ],
      notes: [
        note("Learning evidence is read from the live service; it is never fabricated here.", "muted")
      ],
      commands: [
        {
          id: "learning-reset",
          label: "Reset learning",
          tone: "danger",
          confirm:
            "Reset all learned knowledge and compositions? This deletes real learning evidence and cannot be undone.",
          request: { method: "POST", path: "/admin/api/learning/reset" }
        }
      ]
    };
  }

  function buildMemory(payload) {
    var d = asObject(pick(payload, "/admin/api/memory", {}));
    var state = asObject(d.state);
    var shortTerm = asObject(d.short_term);
    var episodic = asArray(d.episodic);
    var semantic = asArray(d.semantic);
    var interactions = asArray(d.interactions);
    return {
      title: "Memory",
      subtitle: "Episodic, semantic and interaction history",
      summary: [
        stat("Memory", d.enabled),
        stat("Episodes", episodic.length, "muted"),
        stat("Semantic", semantic.length, "muted"),
        stat("Interactions", interactions.length, "muted"),
        stat("Short-term", shortTerm.turn_count, "muted")
      ],
      kv: [
        kvGroup("Service", [
          item("Enabled", d.enabled),
          item("Database", state.database, "muted"),
          item("Episodes", state.episode_count, "muted"),
          item("Semantic", state.semantic_count, "muted"),
          item("Viewers", state.viewer_count, "muted"),
          item("Short-term turns", shortTerm.turn_count, "muted")
        ])
      ],
      tables: [
        table(
          "Recent episodic memory",
          ["Viewer", "Topic", "Summary", "At"],
          episodic.map(function (raw) {
            var record = asObject(raw);
            return row([
              formatValue(record.viewer_id),
              formatValue(record.topic),
              formatValue(record.summary || record.text),
              formatTimestamp(record.created_at || record.at)
            ]);
          }),
          "No episodic memory."
        ),
        table(
          "Semantic memory",
          ["Subject", "Topic", "Value", "Confidence", "At"],
          semantic.map(function (raw) {
            var record = asObject(raw);
            return row([
              formatValue(record.subject),
              formatValue(record.topic),
              formatValue(record.value || record.text),
              formatValue(record.confidence),
              formatTimestamp(record.updated_at || record.created_at)
            ]);
          }),
          "No semantic memory."
        ),
        table(
          "Interaction history",
          ["Viewer", "Topic", "Message", "At"],
          interactions.map(function (raw) {
            var record = asObject(raw);
            return row([
              formatValue(record.viewer_id),
              formatValue(record.topic),
              formatValue(record.message || record.text),
              formatTimestamp(record.created_at || record.at)
            ]);
          }),
          "No interactions recorded."
        )
      ],
      notes: [note("Read-only projection. The dashboard cannot edit memory tables.", "muted")],
      commands: []
    };
  }

  function buildViewers(payload) {
    var d = asObject(pick(payload, "/admin/api/viewers", {}));
    var viewers = asArray(d.viewers);
    return {
      title: "Viewers",
      subtitle: "Operational viewer information",
      summary: [
        stat("Viewers", d.count === undefined ? viewers.length : d.count, "muted"),
        stat("Memory", d.enabled),
        stat("Limit", d.limit, "muted")
      ],
      kv: [],
      tables: [
        table(
          "Viewer relationships",
          ["Viewer", "Interactions", "Relationship", "Familiarity", "Preferences", "Last interaction"],
          viewers.map(function (raw) {
            var viewer = asObject(raw);
            return row([
              formatValue(viewer.viewer_id || viewer.id),
              formatValue(viewer.interaction_count),
              formatValue(viewer.relationship_level),
              formatValue(viewer.familiarity),
              formatValue(viewer.preferences),
              formatTimestamp(viewer.last_interaction_at || viewer.last_seen_at)
            ]);
          }),
          "No viewer records."
        )
      ],
      notes: [
        note("Viewer history is read-only; the dashboard cannot impersonate a viewer or rewrite history.", "muted")
      ],
      commands: []
    };
  }

  function buildPersonality(payload) {
    var d = asObject(pick(payload, "/admin/api/personality", {}));
    var overview = asObject(pick(payload, "/admin/api/overview", {}));
    var personality = asObject(d.personality);
    var profile = asObject(personality.profile);
    var behavior = asObject(d.behavior);
    var personalization = asObject(d.personalization);
    var traits = Object.keys(profile).sort();
    return {
      title: "Personality / Behavior",
      subtitle: "Stable traits and current behavior selection",
      summary: [
        stat("Behavior", behavior.state),
        stat("Previous", behavior.previous_state, "muted"),
        stat("Goal", d.goal),
        stat("Active skill", d.active_skill),
        stat("Source", personality.source, "muted"),
        stat("Viewers", overview.viewer_count, "muted")
      ],
      kv: [
        kvGroup("Traits", [
          item("Speech style", personality.speech_style, "muted")
        ].concat(
          traits.map(function (key) {
            return item(key, profile[key]);
          })
        )),
        kvGroup("Behavior", [
          item("State", behavior.state),
          item("Previous state", behavior.previous_state, "muted"),
          item("Current action", behavior.current_action, "muted"),
          item("Current skill", behavior.current_skill, "muted"),
          item("Current goal", behavior.current_goal, "muted")
        ]),
        kvGroup("Relationship context", [
          item("Relationship level", personalization.relationship_level, "muted"),
          item("Familiarity", personalization.familiarity, "muted"),
          item("Preferences", personalization.preferences, "muted")
        ])
      ],
      tables: [],
      notes: [
        note("Internal prompts are never returned to the browser.", "muted")
      ],
      commands: []
    };
  }

  function buildDiagnostics(payload) {
    var d = asObject(pick(payload, "/admin/api/diagnostics", {}));
    var runtime = asObject(d.runtime);
    var health = asObject(d.health);
    var config = asObject(d.config);
    var metrics = asObject(d.metrics);
    var storage = asObject(d.storage);
    var persistence = asObject(d.persistence);
    var providers = asObject(d.providers);
    return {
      title: "Diagnostics",
      subtitle: "Health, sanitized configuration and metrics",
      summary: [
        stat("Status", d.status),
        stat("Environment", d.environment, "muted"),
        stat("Probe", formatDuration(health.probe_ms)),
        stat("Uptime", formatDuration(health.uptime_seconds), "muted"),
        stat("PID", get(runtime, "process.pid"), "muted"),
        stat("Metrics", metrics.enabled)
      ],
      kv: [
        kvGroup("Configuration", [
          item("Host", config.host, "muted"),
          item("Brain port", config.brain_port, "muted"),
          item("Chat port", config.chat_port, "muted"),
          item("Viewer port", config.viewer_port, "muted"),
          item("Admin port", config.admin_port, "muted"),
          item("Data dir", config.data_dir, "muted"),
          item("Log level", config.log_level, "muted"),
          item("Mocks allowed", config.mocks_allowed),
          item("Log retention", config.log_max_age_days, "muted"),
          item("Backup retention", config.backup_retention, "muted")
        ]),
        kvGroup("Runtime", [
          item("State", get(runtime, "process.state")),
          item("Instance", get(runtime, "process.instance_id"), "muted"),
          item("Active skill", runtime.active_skill),
          item(
            "Stop latch",
            stopLatchEngaged(runtime.stream_stop_latch) ? "ENGAGED" : "released",
            stopLatchEngaged(runtime.stream_stop_latch) ? "warn" : "muted"
          )
        ]),
        kvGroup("Storage", [
          item("Backup", storage.backup ? "reported" : "-", "muted"),
          item("Retention", storage.retention ? "reported" : "-", "muted"),
          item("Databases", persistence.databases, "muted")
        ])
      ],
      tables: [
        table(
          "Component health",
          ["Component", "Status", "Detail", "Requirement"],
          componentRows(health.components),
          "No health report available."
        ),
        table(
          "Providers",
          ["Role", "Provider", "Model / Voice", "Configured", "Reachable", "Latency"],
          [
            providerRow("LLM", providers.llm),
            providerRow("TTS", providers.tts),
            providerRow("Chat", providers.chat),
            providerRow("Stream", providers.stream)
          ],
          "No providers reported."
        )
      ],
      notes: [
        note("Only sanitized configuration is shown; credentials and store keys stay in the backend.", "muted")
      ],
      commands: []
    };
  }

  function buildLogs(payload) {
    var d = asObject(pick(payload, "/admin/api/logs", {}));
    var entries = asArray(d.entries);
    return {
      title: "Logs",
      subtitle: "Bounded, redacted operational log tail",
      filters: LOG_FILTERS,
      summary: [
        stat("Entries", d.count, "muted"),
        stat("Matched", d.matched, "muted"),
        stat("Limit", d.limit, "muted"),
        stat("Scanned", formatValue(d.scanned_bytes), "muted"),
        stat("Truncated", d.truncated === true ? "YES" : "NO", d.truncated ? "warn" : "muted"),
        stat("File", d.available === false ? "missing" : d.log_file, d.available === false ? "warn" : "muted")
      ],
      kv: [],
      tables: [
        table(
          "Log entries",
          ["Time", "Level", "Component", "Message"],
          entries.map(function (raw) {
            var entry = asObject(raw);
            return row([
              formatTimestamp(entry.timestamp),
              formatValue(entry.level),
              formatValue(entry.component),
              formatValue(entry.message)
            ]);
          }),
          d.available === false ? "Log file not available." : "No log entries matched."
        )
      ],
      notes: [
        note("Messages are redacted by the backend before they leave the process.", "muted")
      ],
      commands: []
    };
  }

  function buildSettings(payload) {
    var d = asObject(pick(payload, "/admin/api/settings", {}));
    var values = asObject(d.values);
    var applied = asObject(d.applied);
    var specs = asArray(d.specs);
    return {
      title: "Settings",
      subtitle: "Validated, non-secret runtime settings",
      summary: [
        stat("Updated", formatTimestamp(d.updated_at), "muted"),
        stat("Stored", Object.keys(values).length, "muted"),
        stat("Learning", applied.learning_enabled),
        stat("Log level", applied.log_level, "muted"),
        stat("Secrets exposed", d.secret_values_exposed === true ? "YES" : "NO", d.secret_values_exposed === true ? "bad" : "ok")
      ],
      kv: [],
      tables: [
        table(
          "Settings",
          ["Setting", "Type", "Value", "Effective", "Effect"],
          specs.map(function (raw) {
            var spec = asObject(raw);
            var key = String(spec.key || "");
            var effective =
              Object.prototype.hasOwnProperty.call(applied, key) ? applied[key] : values[key];
            return row([
              key,
              formatValue(spec.kind),
              Object.prototype.hasOwnProperty.call(values, key) ? formatValue(values[key]) : "-",
              formatValue(effective),
              spec.requires_restart === true ? "requires restart" : "live"
            ]);
          }),
          "No settings available."
        )
      ],
      notes: [
        note("A live value is reported as applied only after the backend confirms it.", "muted")
      ],
      commands: [
        {
          id: "settings-save",
          label: "Save settings",
          tone: "primary",
          wrap: "values",
          fields: settingsFields(specs),
          request: { method: "POST", path: "/admin/api/settings" }
        }
      ]
    };
  }

  var BUILDERS = {
    overview: buildOverview,
    runtime: buildRuntime,
    brain: buildBrain,
    providers: buildProviders,
    tts: buildTts,
    chat: buildChat,
    stream: buildStream,
    motion: buildMotion,
    skills: buildSkills,
    learning: buildLearning,
    memory: buildMemory,
    viewers: buildViewers,
    personality: buildPersonality,
    diagnostics: buildDiagnostics,
    logs: buildLogs,
    settings: buildSettings
  };

  function buildView(section, payload) {
    var builder = BUILDERS[String(section || "")];
    if (!builder) {
      return {
        title: "Unknown section",
        subtitle: "",
        summary: [],
        kv: [],
        tables: [],
        notes: [],
        commands: []
      };
    }
    return builder(asObject(payload));
  }

  return {
    SECTIONS: SECTIONS,
    LOG_FILTERS: LOG_FILTERS,
    LOG_QUERY_KEYS: LOG_QUERY_KEYS,
    POLL_BASE_MS: POLL_BASE_MS,
    POLL_HIDDEN_MS: POLL_HIDDEN_MS,
    POLL_MAX_MS: POLL_MAX_MS,
    sectionById: sectionById,
    statusTone: statusTone,
    describeError: describeError,
    createState: createState,
    reduce: reduce,
    buildQuery: buildQuery,
    buildView: buildView,
    findSecretExposure: findSecretExposure,
    formatValue: formatValue,
    formatTimestamp: formatTimestamp,
    formatDuration: formatDuration,
    nextPollDelay: nextPollDelay
  };
});
