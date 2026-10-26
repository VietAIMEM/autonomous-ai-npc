/* DOM glue for the viewer page. Presentation only: it renders the public
 * state document and forwards chat into the one existing chat intake. No
 * relationship, memory or Brain logic lives here. */
(function () {
  "use strict";

  var State = window.ViewerAppState;
  var VIEWER_STORAGE_KEY = "room_agent.viewer_id";
  var SESSION_STORAGE_KEY = "room_agent.session_id";

  var elements = {
    badge: document.getElementById("live-badge"),
    viewerCount: document.getElementById("viewer-count"),
    video: document.getElementById("video"),
    overlay: document.getElementById("player-overlay"),
    overlayText: document.getElementById("overlay-text"),
    activityDot: document.getElementById("activity-dot"),
    activityLabel: document.getElementById("activity-label"),
    streamDetail: document.getElementById("stream-detail"),
    nowLine: document.getElementById("now-line"),
    chatConnection: document.getElementById("chat-connection"),
    messages: document.getElementById("messages"),
    thinking: document.getElementById("thinking"),
    composer: document.getElementById("composer"),
    input: document.getElementById("message-input"),
    sendButton: document.getElementById("send-button"),
    viewerId: document.getElementById("viewer-id"),
    sendState: document.getElementById("send-state")
  };
  if (!elements.messages || !elements.composer) {
    return;
  }

  // ------------------------------------------------------------- identity

  var storedViewer = null;
  try {
    storedViewer = window.localStorage.getItem(VIEWER_STORAGE_KEY);
  } catch (error) {
    storedViewer = null;
  }
  var viewerId = State.resolveViewerId(window.location.search, storedViewer);
  if (!viewerId) {
    viewerId = State.createViewerId();
  }
  try {
    window.localStorage.setItem(VIEWER_STORAGE_KEY, viewerId);
  } catch (error) {
    /* private mode: identity simply does not persist */
  }
  if (elements.viewerId) {
    elements.viewerId.textContent = viewerId;
  }

  var sessionId = null;
  try {
    sessionId = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  } catch (error) {
    sessionId = null;
  }
  if (!State.isValidSessionId(sessionId)) {
    sessionId = State.createSessionId();
    try {
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    } catch (error) {
      /* ignore */
    }
  }

  // --------------------------------------------------------------- state

  var snapshot = null;
  var connection = State.STATUS_CONNECTING;
  var failures = 0;
  var timerId = null;
  var fetching = false;
  var sending = false;
  var messages = [];
  var messageNodes = {};
  var sendStateTimer = null;

  // -------------------------------------------------------------- network

  function stateUrl() {
    return (
      "/viewer/api/state?viewer=" +
      encodeURIComponent(viewerId) +
      "&session=" +
      encodeURIComponent(sessionId)
    );
  }

  function schedule(delay) {
    if (timerId) {
      clearTimeout(timerId);
    }
    var wait =
      typeof delay === "number"
        ? delay
        : State.nextPollDelay({ failures: failures, hidden: document.hidden });
    timerId = setTimeout(fetchState, Math.max(300, wait));
  }

  function safeRender() {
    try {
      render();
    } catch (error) {
      /* one bad frame must never stop the poll loop */
    }
  }

  function finishFetch(timeoutId) {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    fetching = false;
    safeRender();
    schedule();
  }

  function fetchState() {
    if (fetching) {
      return;
    }
    fetching = true;
    var controller = null;
    var timeoutId = null;
    try {
      controller = new AbortController();
    } catch (error) {
      controller = null;
    }
    if (controller) {
      timeoutId = setTimeout(function () {
        try {
          controller.abort();
        } catch (error) {
          /* ignore */
        }
      }, 7000);
    }
    var options = { cache: "no-store", credentials: "same-origin" };
    if (controller) {
      options.signal = controller.signal;
    }
    window
      .fetch(stateUrl(), options)
      .then(function (response) {
        if (!response.ok) {
          throw new Error("state_unavailable");
        }
        return response.json();
      })
      .then(function (payload) {
        failures = 0;
        snapshot = payload && typeof payload === "object" ? payload : null;
        connection = State.deriveConnection(connection, {
          ok: true,
          streamConnection: snapshot && snapshot.stream ? snapshot.stream.connection : "",
          failures: 0
        });
      })
      .catch(function () {
        failures += 1;
        connection = State.deriveConnection(connection, { ok: false, failures: failures });
      })
      .then(function () {
        finishFetch(timeoutId);
      }, function () {
        finishFetch(timeoutId);
      });
  }

  function refreshNow() {
    if (timerId) {
      clearTimeout(timerId);
      timerId = null;
    }
    fetchState();
  }

  function postChat(clientId, text) {
    return window
      .fetch("/viewer/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          viewer_id: viewerId,
          text: text,
          client_message_id: clientId
        })
      })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return null;
          })
          .then(function (body) {
            if (!response.ok && !(body && body.delivery)) {
              throw new Error("chat_unavailable");
            }
            return body;
          });
      });
  }

  // -------------------------------------------------------------- sending

  function setSendState(text, kind) {
    if (!elements.sendState) {
      return;
    }
    elements.sendState.textContent = text || "";
    elements.sendState.className = "send-state" + (kind ? " " + kind : "");
    if (sendStateTimer) {
      clearTimeout(sendStateTimer);
    }
    if (text) {
      sendStateTimer = setTimeout(function () {
        elements.sendState.textContent = "";
        elements.sendState.className = "send-state";
      }, 6000);
    }
  }

  function submitMessage(text, clientId, isRetry) {
    if (sending || failures > 0) {
      return;
    }
    sending = true;
    if (!isRetry) {
      messages.push(
        State.localMessage({
          clientId: clientId,
          viewerId: viewerId,
          text: text,
          at: new Date().toISOString()
        })
      );
    } else {
      messages = messages.map(function (item) {
        if (item.client_id !== clientId) {
          return item;
        }
        var retried = Object.assign({}, item);
        retried.delivery = State.DELIVERY_PENDING;
        retried.reason = null;
        return retried;
      });
    }
    updateComposer();
    renderMessages();
    postChat(clientId, text)
      .then(function (response) {
        messages = messages.map(function (item) {
          return item.client_id === clientId ? State.applySendResult(item, response) : item;
        });
        if (response && response.duplicate_request) {
          setSendState("already counted");
        } else if (response && response.accepted) {
          setSendState("sent");
        } else {
          setSendState((response && response.reason) || "not accepted", "failed");
        }
        sending = false;
        updateComposer();
        renderMessages();
        refreshNow();
      })
      .catch(function () {
        messages = messages.map(function (item) {
          return item.client_id === clientId ? State.applySendResult(item, null) : item;
        });
        setSendState("could not send — click the message to retry", "failed");
        sending = false;
        updateComposer();
        renderMessages();
      });
  }

  elements.composer.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = elements.input ? elements.input.value.trim() : "";
    if (!text || sending || failures > 0) {
      return;
    }
    if (elements.input) {
      elements.input.value = "";
    }
    submitMessage(text, State.createClientId(), false);
  });

  elements.messages.addEventListener("click", function (event) {
    var node = event.target;
    while (node && node !== elements.messages && !(node.dataset && node.dataset.retry)) {
      node = node.parentNode;
    }
    if (!node || node === elements.messages || !node.dataset || !node.dataset.retry) {
      return;
    }
    var clientId = node.dataset.retry;
    var item = null;
    for (var i = 0; i < messages.length; i += 1) {
      if (messages[i].client_id === clientId) {
        item = messages[i];
        break;
      }
    }
    if (item && item.delivery === State.DELIVERY_FAILED) {
      submitMessage(item.text, clientId, true);
    }
  });

  function updateComposer() {
    var connected = failures === 0;
    if (elements.sendButton) {
      elements.sendButton.disabled = !connected || sending;
    }
    if (elements.input) {
      elements.input.disabled = !connected;
      elements.input.placeholder = connected
        ? "Say something to Lacrimosa…"
        : "Reconnecting to chat…";
    }
  }

  // ------------------------------------------------------------- rendering

  function clock(iso) {
    if (!iso) {
      return "";
    }
    var date = new Date(iso);
    if (isNaN(date.getTime())) {
      return "";
    }
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function shortViewer(id) {
    if (!id || id === viewerId) {
      return "You";
    }
    var text = String(id);
    return text.length > 16 ? text.slice(0, 15) + "…" : text;
  }

  function nowSentence(label) {
    var value = String(label || "Idle").toLowerCase();
    return "Lacrimosa is " + value + ".";
  }

  function setBadge(status) {
    if (!elements.badge) {
      return;
    }
    elements.badge.textContent = status.charAt(0) + status.slice(1).toLowerCase();
    elements.badge.className = "badge badge-" + status.toLowerCase();
  }

  function renderMessages() {
    var list = State.reconcileMessages(
      messages,
      snapshot && snapshot.chat && snapshot.chat.messages ? snapshot.chat.messages : [],
      State.MAX_MESSAGE_ITEMS
    );
    messages = list;
    var container = elements.messages;
    var stick =
      container.scrollHeight - container.scrollTop - container.clientHeight < 90;
    var seen = {};
    var cursor = null;
    for (var i = 0; i < list.length; i += 1) {
      var item = list[i];
      var key = item.id || item.client_id;
      if (!key) {
        continue;
      }
      seen[key] = true;
      var node = messageNodes[key];
      if (!node) {
        node = document.createElement("li");
        node.className = "message";
        messageNodes[key] = node;
      }
      updateMessageNode(node, item);
      var before = cursor ? cursor.nextSibling : container.firstChild;
      if (before !== node) {
        container.insertBefore(node, before);
      }
      cursor = node;
    }
    Object.keys(messageNodes).forEach(function (key) {
      if (!seen[key]) {
        var node = messageNodes[key];
        if (node.parentNode) {
          node.parentNode.removeChild(node);
        }
        delete messageNodes[key];
      }
    });
    while (container.childNodes.length > State.MAX_MESSAGE_ITEMS) {
      container.removeChild(container.firstChild);
    }
    if (stick) {
      container.scrollTop = container.scrollHeight;
    }
  }

  function updateMessageNode(node, item) {
    var parts = node._parts;
    if (!parts) {
      parts = {};
      parts.who = document.createElement("span");
      parts.who.className = "who";
      parts.text = document.createElement("span");
      parts.text.className = "text";
      parts.when = document.createElement("span");
      parts.when.className = "when";
      parts.status = document.createElement("span");
      parts.status.className = "status-note";
      node.appendChild(parts.who);
      node.appendChild(parts.text);
      node.appendChild(parts.when);
      node.appendChild(parts.status);
      node._parts = parts;
    }
    var className = "message";
    if (item.delivery === State.DELIVERY_PENDING) {
      className += " pending";
    } else if (item.delivery === State.DELIVERY_FAILED) {
      className += " failed";
    }
    node.className = className;
    parts.who.textContent =
      item.role === "lacrimosa" ? "Lacrimosa" : shortViewer(item.viewer_id);
    parts.who.className = item.role === "lacrimosa" ? "who lacrimosa" : "who";
    parts.text.textContent = item.text;
    parts.when.textContent = clock(item.at);
    var note = State.deliveryLabel(item);
    parts.status.textContent = note;
    if (item.delivery === State.DELIVERY_FAILED) {
      parts.status.className = "status-note failed";
      if (item.client_id) {
        node.dataset.retry = item.client_id;
        node.title = "Click to retry";
      }
    } else if (item.delivery === State.DELIVERY_DELIVERED) {
      parts.status.className = "status-note delivered";
      delete node.dataset.retry;
      node.removeAttribute("title");
    } else {
      parts.status.className = "status-note";
      delete node.dataset.retry;
      node.removeAttribute("title");
    }
  }

  // -------------------------------------------------------------- playback

  var player = {
    mode: "idle",
    hls: null,
    retry: 0,
    retryTimer: null,
    unsupported: false
  };

  function overlay(text) {
    if (!elements.overlay) {
      return;
    }
    elements.overlay.classList.remove("hidden");
    if (elements.overlayText) {
      elements.overlayText.textContent = text;
    }
  }

  function hideOverlay() {
    if (elements.overlay) {
      elements.overlay.classList.add("hidden");
    }
  }

  function isPlaying() {
    var video = elements.video;
    return Boolean(video) && !video.paused && video.readyState >= 3;
  }

  function offlineText(stream) {
    var connectionName = stream ? stream.connection : "";
    if (connectionName === "CONNECTING") {
      return "Lacrimosa is starting the stream…";
    }
    if (connectionName === "RECONNECTING") {
      return "The stream is reconnecting…";
    }
    if (connectionName === "DEGRADED") {
      return "The stream is temporarily degraded…";
    }
    if (connectionName === "LIVE") {
      return "Stream is live — starting playback…";
    }
    return "Lacrimosa is offline.";
  }

  function destroyPlayer() {
    if (player.hls) {
      try {
        player.hls.destroy();
      } catch (error) {
        /* ignore */
      }
      player.hls = null;
    }
    player.mode = "idle";
  }

  function schedulePlaybackRetry() {
    if (player.unsupported) {
      return;
    }
    if (player.retryTimer) {
      clearTimeout(player.retryTimer);
    }
    var delay = Math.min(15000, 1000 * Math.pow(2, Math.max(0, player.retry - 1)));
    player.retryTimer = setTimeout(function () {
      if (!document.hidden) {
        startPlayback();
      }
    }, delay);
  }

  function startPlayback() {
    if (player.mode !== "idle" || player.unsupported) {
      return;
    }
    var video = elements.video;
    if (!video) {
      return;
    }
    if (window.Hls && window.Hls.isSupported && window.Hls.isSupported()) {
      player.mode = "hls";
      var hls = new window.Hls({
        liveDurationInfinity: true,
        maxBufferLength: 6,
        backBufferLength: 12,
        enableWorker: true
      });
      player.hls = hls;
      hls.on(window.Hls.Events.ERROR, function (event, data) {
        if (!data || !data.fatal) {
          return;
        }
        destroyPlayer();
        player.retry += 1;
        overlay(
          connection === State.STATUS_OFFLINE
            ? State.connectionNotice(connection)
            : "Stream paused — reconnecting…"
        );
        schedulePlaybackRetry();
      });
      hls.loadSource("/hls/live.m3u8");
      hls.attachMedia(video);
    } else if (video.canPlayType && video.canPlayType("application/vnd.apple.mpegurl")) {
      player.mode = "native";
      video.src = "/hls/live.m3u8";
    } else {
      player.unsupported = true;
      overlay("This browser cannot play the livestream.");
      return;
    }
    var attempt = video.play();
    if (attempt && typeof attempt.catch === "function") {
      attempt.catch(function () {
        /* muted autoplay is normally allowed; overlay stays until 'playing' */
      });
    }
  }

  if (elements.video) {
    elements.video.addEventListener("playing", function () {
      player.retry = 0;
      hideOverlay();
    });
    elements.video.addEventListener("click", function () {
      elements.video.muted = !elements.video.muted;
    });
  }

  function renderPlayback(stream) {
    if (connection === State.STATUS_OFFLINE) {
      overlay(State.connectionNotice(connection));
      return;
    }
    var ready = Boolean(
      stream && stream.live === true && stream.playback && stream.playback.ready === true
    );
    if (ready) {
      if (player.mode === "idle" && !player.unsupported) {
        startPlayback();
      }
      if (isPlaying()) {
        hideOverlay();
      } else {
        overlay(
          player.mode === "idle"
            ? "Stream is live — starting playback…"
            : "Buffering the livestream…"
        );
      }
      return;
    }
    overlay(offlineText(stream));
  }

  // --------------------------------------------------------------- render

  function render() {
    var snap = snapshot;
    setBadge(connection);
    if (elements.viewerCount) {
      var count = snap && snap.viewers ? snap.viewers.count : 0;
      elements.viewerCount.textContent = String(count) + " watching";
    }
    var activity = snap && snap.activity ? snap.activity.label : "Idle";
    if (elements.activityLabel) {
      elements.activityLabel.textContent = activity;
    }
    if (elements.activityDot) {
      elements.activityDot.className = "activity-dot " + State.activityDotClass(activity);
    }
    var chatOk = failures === 0;
    if (elements.chatConnection) {
      elements.chatConnection.textContent = chatOk
        ? "connected"
        : connection === State.STATUS_OFFLINE
          ? "offline"
          : "reconnecting…";
      elements.chatConnection.className =
        "chat-state " +
        (chatOk ? "ok" : connection === State.STATUS_OFFLINE ? "bad" : "warn");
    }
    if (elements.thinking) {
      var busy = Boolean(snap && snap.chat && snap.chat.busy);
      elements.thinking.classList.toggle("hidden", !busy);
    }
    if (elements.streamDetail) {
      var stream = failures >= State.OFFLINE_AFTER_FAILURES ? null : snap && snap.stream;
      if (stream) {
        var detail = stream.transport || "";
        if (stream.resolution) {
          detail += " · " + stream.resolution;
        }
        if (stream.fps) {
          detail += " @ " + stream.fps + " fps";
        }
        elements.streamDetail.textContent = detail;
      } else {
        elements.streamDetail.textContent =
          failures >= State.OFFLINE_AFTER_FAILURES
            ? "waiting for the runtime…"
            : "waiting for the stream";
      }
    }
    if (elements.nowLine) {
      elements.nowLine.textContent =
        failures > 0 ? "Reconnecting to the runtime…" : nowSentence(activity);
    }
    renderPlayback(snap && snap.stream);
    renderMessages();
    updateComposer();
  }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) {
      schedule();
    } else {
      refreshNow();
    }
  });

  render();
  fetchState();
})();
