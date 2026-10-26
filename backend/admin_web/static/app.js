/* Room Agent admin dashboard controller.
 *
 * Talks only to the authenticated /admin/api/* surface, draws exactly what the
 * backend projects, and treats every payload as untrusted text: nothing is
 * inserted as HTML, no secret-shaped value is rendered, and no authorization
 * decision is made here. The backend is the only authority.
 */
(function () {
  "use strict";

  var S = window.AdminAppState;
  if (!S) {
    return;
  }

  var state = S.createState();
  var pollTimer = null;
  var els = {};

  // ------------------------------------------------------------- utilities

  function $(id) {
    return document.getElementById(id);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function option(value, label) {
    var node = document.createElement("option");
    node.value = value;
    node.textContent = label === undefined ? value || "(default)" : label;
    return node;
  }

  // --------------------------------------------------------------- transport

  function api(path, options) {
    var opts = options || {};
    var init = {
      method: opts.method || "GET",
      credentials: "same-origin",
      headers: { Accept: "application/json" }
    };
    if (opts.form !== undefined) {
      init.body = opts.form;
    } else if (opts.body !== undefined && opts.body !== null) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }
    return fetch(path, init).then(
      function (response) {
        return response.text().then(function (text) {
          var body = null;
          if (text) {
            try {
              body = JSON.parse(text);
            } catch (error) {
              body = null;
            }
          }
          var ok = response.ok && (!body || body.ok !== false);
          return { ok: ok, status: response.status, body: body };
        });
      },
      function () {
        return { ok: false, status: 0, body: null };
      }
    );
  }

  // ----------------------------------------------------------------- session

  function probeSession() {
    api("/admin/api/session").then(function (res) {
      var body = res.body || {};
      if (res.ok && body.authenticated === true) {
        dispatch({
          type: "session",
          authenticated: true,
          role: body.role,
          permissions: body.permissions
        });
        showApp();
        loadSection(state.section);
        return;
      }
      var message = "";
      if (!res.ok && res.status !== 401) {
        var problem = S.describeError(res.status, res.body);
        message = problem.title + ": " + problem.message;
      }
      dispatch({ type: "session", authenticated: false });
      showLogin(message);
    });
  }

  function showLogin(message) {
    cancelPoll();
    els.appView.hidden = true;
    els.loginView.hidden = false;
    setLoginError(message || "");
    window.setTimeout(function () {
      els.loginToken.focus();
    }, 0);
  }

  function showApp() {
    els.loginView.hidden = true;
    els.appView.hidden = false;
    renderSessionBadges();
    renderNav();
  }

  function renderSessionBadges() {
    var role = state.role ? "role: " + state.role : "role unknown";
    els.roleBadge.textContent = role;
    els.roleBadge.className = "badge badge-ok";
  }

  function handleUnauthorized() {
    dispatch({ type: "logout" });
    showLogin(
      S.describeError(401, { error: "session_expired" }).message
    );
  }

  function setLoginError(message) {
    if (!message) {
      els.loginError.hidden = true;
      els.loginError.textContent = "";
      return;
    }
    els.loginError.textContent = message;
    els.loginError.hidden = false;
  }

  function submitLogin(event) {
    event.preventDefault();
    var token = String(els.loginToken.value || "").trim();
    if (!token) {
      setLoginError("Enter the admin token printed by the runtime.");
      return;
    }
    dispatch({ type: "login_pending" });
    els.loginSubmit.disabled = true;
    setLoginError("");
    api("/admin/api/login", { method: "POST", body: { token: token } }).then(
      function (res) {
        els.loginSubmit.disabled = false;
        els.loginToken.value = "";
        if (res.ok && res.body && res.body.authenticated === true) {
          dispatch({
            type: "session",
            authenticated: true,
            role: res.body.role,
            permissions: res.body.permissions
          });
          dispatch({
            type: "notice",
            notice: {
              tone: "ok",
              title: "Signed in",
              message: "The backend issued an admin session."
            }
          });
          showApp();
          loadSection(state.section);
          return;
        }
        var problem = S.describeError(res.status, res.body);
        dispatch({ type: "login_error", error: problem });
        setLoginError(problem.title + " — " + problem.message);
      }
    );
  }

  function signOut() {
    cancelPoll();
    api("/admin/api/logout", { method: "POST", body: {} }).then(function () {
      dispatch({ type: "logout" });
      showLogin("");
    });
  }

  // -------------------------------------------------------------- navigation

  function buildNav() {
    els.nav.replaceChildren();
    S.SECTIONS.forEach(function (section) {
      var button = el("button", "nav-btn", section.label);
      button.type = "button";
      button.dataset.section = section.id;
      button.addEventListener("click", function () {
        navigate(section.id);
      });
      els.nav.appendChild(button);
    });
  }

  function renderNav() {
    Array.prototype.forEach.call(els.nav.children, function (button) {
      var active = button.dataset.section === state.section;
      button.setAttribute("aria-current", active ? "true" : "false");
    });
  }

  function navigate(sectionId) {
    dispatch({ type: "navigate", section: sectionId });
    renderNav();
    els.panel.replaceChildren(el("p", "muted", "Loading " + sectionId + "…"));
    els.sectionTitle.textContent = sectionLabel(sectionId);
    els.sectionSubtitle.textContent = "";
    loadSection(sectionId);
  }

  function sectionLabel(sectionId) {
    var section = S.sectionById(sectionId);
    return section ? section.label : "Admin";
  }

  // ------------------------------------------------------------ data loading

  function dispatch(action) {
    state = S.reduce(state, action);
  }

  function loadSection(sectionId) {
    var section = S.sectionById(sectionId || state.section);
    if (!section) {
      return;
    }
    var payload = {};
    var requests = section.requests.map(function (path) {
      var query = section.id === "logs" ? S.buildQuery("logs", state.filters.logs) : "";
      return api(path + query).then(function (res) {
        if (res.status === 401) {
          return { unauthorized: true };
        }
        if (!res.ok) {
          return { failure: res, path: path };
        }
        payload[path] = res.body;
        return {};
      });
    });

    Promise.all(requests).then(function (results) {
      var unauthorized = results.some(function (entry) {
        return entry.unauthorized === true;
      });
      if (unauthorized) {
        handleUnauthorized();
        return;
      }
      var failure = null;
      results.forEach(function (entry) {
        if (!failure && entry.failure) {
          failure = entry;
        }
      });
      var at = new Date().toISOString();
      if (failure) {
        var problem = S.describeError(failure.failure.status, failure.failure.body);
        dispatch({ type: "error", section: section.id, error: problem });
        dispatch({ type: "poll", failures: state.failures + 1 });
        renderBanner();
      } else {
        dispatch({ type: "data", section: section.id, payload: payload, at: at });
        dispatch({ type: "notice", notice: null });
        dispatch({ type: "poll", failures: 0 });
        renderBanner();
      }
      render();
      schedulePoll();
    });
  }

  function schedulePoll() {
    cancelPoll();
    if (!state.authenticated) {
      return;
    }
    var delay = S.nextPollDelay({
      failures: state.failures,
      hidden: document.hidden === true
    });
    pollTimer = window.setTimeout(function () {
      loadSection(state.section);
    }, delay);
  }

  function cancelPoll() {
    if (pollTimer !== null) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  // ---------------------------------------------------------------- guarding

  /* Defence in depth: the backend redacts, and the UI still refuses to draw a
   * payload that looks like it carries a credential. */
  function payloadExposure(payload) {
    var text;
    try {
      text = JSON.stringify(payload);
    } catch (error) {
      return ["unserializable payload"];
    }
    return S.findSecretExposure(text);
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    var section = S.sectionById(state.section);
    var page = state.pages[state.section];
    var problem = state.errors[state.section];

    els.lastRefresh.textContent = state.lastRefreshAt
      ? "updated " + S.formatTimestamp(state.lastRefreshAt)
      : "not refreshed yet";

    if (!page || !page.payload) {
      els.panel.replaceChildren(
        el(
          "p",
          "panel-empty",
          problem
            ? problem.title + " — " + problem.message
            : "Waiting for " + sectionLabel(state.section) + " data…"
        )
      );
      return;
    }

    var exposure = payloadExposure(page.payload);
    if (exposure.length) {
      els.panel.replaceChildren(
        el(
          "p",
          "panel-empty",
          "Response withheld: the payload matched a credential pattern (" +
            exposure.join(", ") +
            "). Report this to the backend owner."
        )
      );
      dispatch({
        type: "error",
        section: state.section,
        error: {
          kind: "server",
          status: 500,
          code: "secret_exposure_blocked",
          title: "Response withheld",
          message: "A payload field looked like a credential and was not displayed."
        }
      });
      renderBanner();
      return;
    }

    var view = S.buildView(state.section, page.payload);
    els.sectionTitle.textContent = view.title;
    els.sectionSubtitle.textContent = view.subtitle || "";

    var nodes = [];
    if (view.summary && view.summary.length) {
      nodes.push(renderSummary(view.summary));
    }
    if (view.filters && view.filters.length) {
      nodes.push(renderFilters(view.filters));
    }
    if (view.kv && view.kv.length) {
      nodes.push(renderKv(view.kv));
    }
    (view.tables || []).forEach(function (table) {
      nodes.push(renderTable(table));
    });
    if (view.commands && view.commands.length) {
      nodes.push(renderCommands(view.commands));
    }
    if (view.notes && view.notes.length) {
      nodes.push(renderNotes(view.notes));
    }
    els.panel.replaceChildren.apply(els.panel, nodes);
    renderResult();
  }

  function renderSummary(stats) {
    var grid = el("div", "summary-grid");
    stats.forEach(function (stat) {
      var card = el("div", "stat tone-" + (stat.tone || "muted"));
      card.appendChild(el("span", "stat-label", stat.label));
      card.appendChild(el("span", "stat-value value-" + (stat.tone || "muted"), stat.value));
      grid.appendChild(card);
    });
    return grid;
  }

  function renderKv(groups) {
    var grid = el("div", "kv-grid");
    groups.forEach(function (group) {
      var block = el("section", "kv-group");
      block.appendChild(el("h3", null, group.title));
      var list = el("dl", "kv-list");
      (group.items || []).forEach(function (entry) {
        list.appendChild(el("dt", null, entry.label));
        list.appendChild(el("dd", "value-" + (entry.tone || "muted"), entry.value));
      });
      block.appendChild(list);
      grid.appendChild(block);
    });
    return grid;
  }

  function renderTable(table) {
    var block = el("section", "table-block");
    block.appendChild(el("h3", null, table.title));
    var rows = table.rows || [];
    if (!rows.length) {
      block.appendChild(el("p", "muted small", table.empty || "No records."));
      return block;
    }
    var scroll = el("div", "table-scroll");
    var node = el("table");
    var head = el("thead");
    var headRow = el("tr");
    (table.columns || []).forEach(function (column) {
      headRow.appendChild(el("th", null, column));
    });
    head.appendChild(headRow);
    node.appendChild(head);
    var body = el("tbody");
    rows.forEach(function (record) {
      var tr = el("tr");
      (record.cells || []).forEach(function (cell) {
        tr.appendChild(el("td", null, cell));
      });
      if (record.actions && record.actions.length) {
        var actions = el("td", "cell-actions");
        record.actions.forEach(function (action) {
          actions.appendChild(buildActionButton(action));
        });
        tr.appendChild(actions);
      }
      body.appendChild(tr);
    });
    node.appendChild(body);
    scroll.appendChild(node);
    block.appendChild(scroll);
    return block;
  }

  function buildActionButton(action) {
    var button = el("button", "btn btn-mini" + toneClass(action.tone), action.label);
    button.type = "button";
    button.addEventListener("click", function () {
      if (action.confirm && !window.confirm(action.confirm)) {
        return;
      }
      runRequest(action.label, action.request, button);
    });
    return button;
  }

  function renderCommands(commands) {
    var section = el("section", "command-block-wrapper");
    section.appendChild(el("h3", null, "Operations"));
    var grid = el("div", "command-grid");
    commands.forEach(function (command) {
      grid.appendChild(renderCommand(command));
    });
    section.appendChild(grid);
    return section;
  }

  function renderCommand(command) {
    var block = el("form", "command-block command-" + (command.tone || "default"));
    var head = el("div", "command-head");
    head.appendChild(el("h3", null, command.label));
    block.appendChild(head);
    if (command.note) {
      block.appendChild(el("p", "command-note", command.note));
    }
    var fields = command.fields || [];
    if (fields.length) {
      var grid = el("div", "field-grid");
      fields.forEach(function (descriptor) {
        grid.appendChild(renderField(descriptor));
      });
      block.appendChild(grid);
    }
    var actions = el("div", "command-actions");
    var submit = el("button", "btn" + toneClass(command.tone), command.label);
    submit.type = "submit";
    actions.appendChild(submit);
    block.appendChild(actions);
    block.addEventListener("submit", function (event) {
      event.preventDefault();
      submitCommand(command, block, submit);
    });
    return block;
  }

  function renderField(descriptor) {
    var wrap = el("div", "field");
    var id = "field-" + descriptor.name + "-" + Math.random().toString(36).slice(2, 8);
    var label = el("label", null, descriptor.label + (descriptor.required ? " *" : ""));
    label.setAttribute("for", id);
    wrap.appendChild(label);
    var input;
    if (descriptor.type === "select") {
      input = document.createElement("select");
      (descriptor.options || []).forEach(function (value) {
        input.appendChild(option(value, value || "(default)"));
      });
      input.value = descriptor.value || "";
    } else {
      input = document.createElement("input");
      input.type =
        descriptor.type === "file"
          ? "file"
          : descriptor.type === "number"
            ? "number"
            : "text";
      if (descriptor.type === "number") {
        if (descriptor.minimum !== null && descriptor.minimum !== undefined) {
          input.min = String(descriptor.minimum);
        }
        if (descriptor.maximum !== null && descriptor.maximum !== undefined) {
          input.max = String(descriptor.maximum);
        }
      }
      if (descriptor.type === "file") {
        input.accept = ".fbx,.json,.bvh,.glb,.gltf,.dae";
      }
      input.placeholder = descriptor.placeholder || "";
      input.value = descriptor.value || "";
    }
    input.id = id;
    input.name = descriptor.name;
    if (descriptor.required) {
      input.required = true;
    }
    wrap.appendChild(input);
    return wrap;
  }

  function renderFilters(descriptors) {
    var form = el("form", "filters");
    descriptors.forEach(function (descriptor) {
      var wrap = el("div", "field");
      wrap.appendChild(el("label", null, descriptor.label));
      var input;
      if (descriptor.type === "select") {
        input = document.createElement("select");
        (descriptor.options || []).forEach(function (value) {
          input.appendChild(option(value, value || "any"));
        });
      } else {
        input = document.createElement("input");
        input.type = descriptor.type === "number" ? "number" : "text";
        input.placeholder = descriptor.placeholder || "";
        if (descriptor.minimum !== undefined && descriptor.minimum !== null) {
          input.min = String(descriptor.minimum);
        }
        if (descriptor.maximum !== undefined && descriptor.maximum !== null) {
          input.max = String(descriptor.maximum);
        }
      }
      input.name = descriptor.name;
      input.value =
        state.filters.logs[descriptor.name] === undefined
          ? ""
          : String(state.filters.logs[descriptor.name]);
      wrap.appendChild(input);
      form.appendChild(wrap);
    });
    var actions = el("div", "field");
    var submit = el("button", "btn btn-primary", "Apply filters");
    submit.type = "submit";
    actions.appendChild(submit);
    form.appendChild(actions);
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var values = {};
      Array.prototype.forEach.call(form.elements, function (input) {
        if (input.name) {
          values[input.name] = input.value;
        }
      });
      dispatch({ type: "filters", section: "logs", values: values });
      loadSection("logs");
    });
    return form;
  }

  function renderNotes(notes) {
    var list = el("ul", "notes");
    notes.forEach(function (entry) {
      list.appendChild(el("li", "value-" + (entry.tone || "muted"), entry.text));
    });
    return list;
  }

  function renderBanner() {
    var notice = state.notice;
    var problem = state.errors[state.section];
    var tone = "muted";
    var title = "";
    var message = "";
    if (notice) {
      tone = notice.tone || "muted";
      title = notice.title || "";
      message = notice.message || "";
    } else if (problem) {
      tone = toneForKind(problem.kind);
      title = problem.title;
      message = problem.message;
    }
    if (!title && !message) {
      els.banner.hidden = true;
      els.banner.replaceChildren();
      return;
    }
    var content = [];
    if (title) {
      content.push(el("strong", null, title));
    }
    if (message) {
      content.push(el("span", null, message));
    }
    els.banner.className = "banner banner-" + tone;
    els.banner.replaceChildren.apply(els.banner, content);
    els.banner.hidden = false;
  }

  function toneForKind(kind) {
    if (kind === "success") {
      return "ok";
    }
    if (kind === "network" || kind === "unavailable" || kind === "throttled") {
      return "warn";
    }
    return "bad";
  }

  function toneClass(tone) {
    if (tone === "primary") {
      return " btn-primary";
    }
    if (tone === "danger") {
      return " btn-danger";
    }
    return "";
  }

  // ----------------------------------------------------------------- command

  function submitCommand(command, form, button) {
    if (command.confirm && !window.confirm(command.confirm)) {
      return;
    }
    var fields = command.fields || [];
    var values = {};
    var missing = null;
    fields.forEach(function (descriptor) {
      if (descriptor.type === "file") {
        return;
      }
      var input = form.elements[descriptor.name];
      var raw = input ? String(input.value || "").trim() : "";
      if (!raw) {
        if (descriptor.required && !missing) {
          missing = descriptor.label;
        }
        return;
      }
      values[descriptor.name] = descriptor.type === "number" ? Number(raw) : raw;
    });
    if (missing) {
      setNotice("bad", "Missing input", missing + " is required.");
      return;
    }

    var request = { method: command.request.method, path: command.request.path };
    if (request.method === "GET") {
      request.body = values;
      runRequest(command.label, request, button);
      return;
    }
    if (command.upload) {
      var form_data = new FormData();
      Object.keys(values).forEach(function (key) {
        form_data.append(key, values[key]);
      });
      fields.forEach(function (descriptor) {
        if (descriptor.type !== "file") {
          return;
        }
        var input = form.elements[descriptor.name];
        if (input && input.files && input.files.length) {
          form_data.append(descriptor.name, input.files[0], input.files[0].name);
        }
      });
      if (!form_data.get("file")) {
        setNotice("bad", "Missing input", "Select a motion file to upload.");
        return;
      }
      runRequest(command.label, request, button, { form: form_data });
      return;
    }
    var body = command.wrap === "values" ? { values: values } : values;
    runRequest(command.label, request, button, { body: body });
  }

  function runRequest(label, request, button, options) {
    var opts = options || {};
    if (button) {
      button.disabled = true;
    }
    dispatch({ type: "command", running: true, label: label });
    var call =
      request.method === "GET"
        ? api(buildRequestUrl(request))
        : api(request.path, {
            method: request.method,
            body: opts.body,
            form: opts.form
          });
    call.then(function (res) {
      if (button) {
        button.disabled = false;
      }
      dispatch({ type: "command", running: false });
      if (res.status === 401) {
        handleUnauthorized();
        return;
      }
      var problem = res.ok ? null : S.describeError(res.status, res.body);
      dispatch({
        type: "notice",
        notice: res.ok
          ? { tone: "ok", title: label + ": confirmed", message: "" }
          : { tone: toneForKind(problem.kind), title: problem.title, message: problem.message },
        result: {
          label: label,
          ok: res.ok,
          status: res.status,
          error: problem,
          payload: res.body
        }
      });
      renderBanner();
      renderResult();
      loadSection(state.section);
    });
  }

  function buildRequestUrl(request) {
    var body = request.body || {};
    var parts = [];
    Object.keys(body).forEach(function (key) {
      if (body[key] === null || body[key] === undefined || String(body[key]).trim() === "") {
        return;
      }
      parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(String(body[key]).trim()));
    });
    return request.path + (parts.length ? "?" + parts.join("&") : "");
  }

  function setNotice(tone, title, message) {
    dispatch({
      type: "notice",
      notice: { tone: tone, title: title, message: message }
    });
    renderBanner();
  }

  // ------------------------------------------------------------------ result

  function renderResult() {
    var result = state.commandResult;
    if (!result) {
      els.result.hidden = true;
      els.result.replaceChildren();
      return;
    }
    var nodes = [];
    nodes.push(
      el("h3", null, result.ok ? "Confirmed: " + result.label : "Rejected: " + result.label)
    );
    var pairs = summarizeResult(result);
    if (pairs.length) {
      var list = el("dl", "kv-list");
      pairs.forEach(function (entry) {
        list.appendChild(el("dt", null, entry.label));
        list.appendChild(el("dd", "value-" + (entry.tone || "muted"), entry.value));
      });
      nodes.push(list);
    } else if (!result.ok) {
      nodes.push(
        el(
          "p",
          "muted small",
          result.error ? result.error.message : "The backend did not return details."
        )
      );
    }
    els.result.replaceChildren.apply(els.result, nodes);
    els.result.hidden = false;
  }

  function summarizeResult(result) {
    var payload = result.payload && typeof result.payload === "object" ? result.payload : {};
    var job = payload.job && typeof payload.job === "object" ? payload.job : null;
    var pairs = [];
    function add(label, value, tone) {
      if (value === null || value === undefined || value === "") {
        return;
      }
      pairs.push({
        label: label,
        value: S.formatValue(value),
        tone: tone === undefined ? S.statusTone(value) : tone
      });
    }
    if (job) {
      add("State", job.state);
      add("Request", job.request_id);
      add("Motion", job.motion_id || job.source_motion_id);
      add("Processed", job.processed_motion_id);
      add("Validation", job.validation_status || job.validation);
      add("Compatibility", job.compatibility_status || job.compatibility);
      add("Quality score", job.quality_score);
      add("Animation", job.animation_reference);
      add("Error", job.error, job.error ? "bad" : "muted");
      return pairs;
    }
    if (payload.results && typeof payload.results === "object") {
      Object.keys(payload.results).forEach(function (key) {
        var entry = payload.results[key] || {};
        var detail = entry.error
          ? entry.error
          : entry.requires_restart
            ? "requires restart"
            : entry.applied
              ? "applied"
              : "saved";
        add(key, entry.status ? entry.status + " (" + detail + ")" : detail);
      });
      return pairs;
    }
    ["action", "requested", "already_stopped", "dispatch", "state", "status", "count", "message"].forEach(
      function (key) {
        add(key, payload[key]);
      }
    );
    if (payload.runtime && typeof payload.runtime === "object") {
      add("Runtime running", payload.runtime.running);
    }
    return pairs;
  }

  // -------------------------------------------------------------------- init

  function bind() {
    els.loginForm.addEventListener("submit", submitLogin);
    els.logoutButton.addEventListener("click", signOut);
    els.refreshButton.addEventListener("click", function () {
      loadSection(state.section);
    });
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && state.authenticated) {
        loadSection(state.section);
      }
    });
  }

  function applyEnvironmentBadge() {
    var page = state.pages.overview;
    if (!page || !page.payload) {
      return;
    }
    var overview = page.payload["/admin/api/overview"] || {};
    if (!overview.environment) {
      return;
    }
    els.envBadge.textContent = "env: " + String(overview.environment);
    els.envBadge.className = "badge";
  }

  function init() {
    els = {
      loginView: $("login-view"),
      loginForm: $("login-form"),
      loginToken: $("login-token"),
      loginSubmit: $("login-submit"),
      loginError: $("login-error"),
      appView: $("app-view"),
      nav: $("nav"),
      roleBadge: $("role-badge"),
      envBadge: $("env-badge"),
      sectionTitle: $("section-title"),
      sectionSubtitle: $("section-subtitle"),
      lastRefresh: $("last-refresh"),
      refreshButton: $("refresh-button"),
      logoutButton: $("logout-button"),
      banner: $("banner"),
      result: $("result"),
      panel: $("panel")
    };
    buildNav();
    bind();
    var originalDispatch = dispatch;
    dispatch = function (action) {
      originalDispatch(action);
      if (action && action.type === "data" && action.section === "overview") {
        applyEnvironmentBadge();
      }
    };
    probeSession();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
