/* Viewer page state logic: DOM-free, dependency-free, unit-testable.
 *
 * Loaded as a plain script in the browser (window.ViewerAppState) and
 * require()d by the Node tests. It holds no relationship, memory or Brain
 * logic: only presentation state derived from the public /viewer/api/state
 * document and the bounded local chat transcript.
 */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ViewerAppState = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var VIEWER_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
  var SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{6,64}$/;
  var CLIENT_ID_PATTERN = /^[A-Za-z0-9_-]{6,64}$/;

  var STATUS_LIVE = "LIVE";
  var STATUS_CONNECTING = "CONNECTING";
  var STATUS_OFFLINE = "OFFLINE";
  var STATUS_RECONNECTING = "RECONNECTING";
  var STATUS_DEGRADED = "DEGRADED";

  var DELIVERY_PENDING = "pending";
  var DELIVERY_ACCEPTED = "accepted";
  var DELIVERY_DELIVERED = "delivered";
  var DELIVERY_FAILED = "failed";

  var MAX_MESSAGE_ITEMS = 200;
  var MAX_OWN_MESSAGES = 40;
  var POLL_BASE_MS = 1500;
  var POLL_HIDDEN_MS = 6000;
  var POLL_MAX_MS = 15000;
  var OFFLINE_AFTER_FAILURES = 4;

  function isValidViewerId(value) {
    return typeof value === "string" && VIEWER_ID_PATTERN.test(value.trim());
  }

  function isValidSessionId(value) {
    return typeof value === "string" && SESSION_ID_PATTERN.test(value.trim());
  }

  function randomHex(random, length) {
    var source = typeof random === "function" ? random : Math.random;
    var alphabet = "0123456789abcdef";
    var out = "";
    for (var i = 0; i < length; i += 1) {
      out += alphabet[Math.floor(source() * 16) % 16];
    }
    return out;
  }

  function createViewerId(random) {
    return "viewer_" + randomHex(random, 8);
  }

  function createSessionId(random) {
    return "sess_" + randomHex(random, 16);
  }

  function createClientId(random) {
    return "c_" + randomHex(random, 16);
  }

  function resolveViewerId(search, storedId) {
    var match = /[?&]viewer=([^&]+)/.exec(typeof search === "string" ? search : "");
    if (match) {
      var fromUrl = "";
      try {
        fromUrl = decodeURIComponent(match[1]).trim();
      } catch (error) {
        fromUrl = match[1].trim();
      }
      if (isValidViewerId(fromUrl)) {
        return fromUrl;
      }
    }
    if (isValidViewerId(storedId)) {
      return storedId.trim();
    }
    return null;
  }

  function nextPollDelay(options) {
    var opts = options || {};
    var failures = Math.max(0, opts.failures || 0);
    var base = opts.hidden ? POLL_HIDDEN_MS : POLL_BASE_MS;
    if (failures === 0) {
      return base;
    }
    return Math.min(POLL_MAX_MS, base * Math.pow(2, failures));
  }

  function deriveConnection(previous, input) {
    var data = input || {};
    var failures = Math.max(0, data.failures || 0);
    if (!data.ok) {
      return failures >= OFFLINE_AFTER_FAILURES ? STATUS_OFFLINE : STATUS_RECONNECTING;
    }
    var stream = String(data.streamConnection || "").toUpperCase();
    if (stream === STATUS_LIVE) return STATUS_LIVE;
    if (stream === STATUS_CONNECTING) return STATUS_CONNECTING;
    if (stream === STATUS_RECONNECTING) return STATUS_RECONNECTING;
    if (stream === STATUS_OFFLINE) return STATUS_OFFLINE;
    if (stream === STATUS_DEGRADED) return STATUS_DEGRADED;
    // Backend answered but the stream field is unknown: never claim LIVE.
    return STATUS_DEGRADED;
  }

  function localMessage(input) {
    var data = input || {};
    return {
      id: null,
      client_id: data.clientId,
      role: "viewer",
      viewer_id: data.viewerId,
      text: data.text,
      at: data.at,
      status: "RECEIVED",
      delivery: DELIVERY_PENDING,
      reason: null,
      local: true
    };
  }

  function mergeServerItem(local, server) {
    var merged = {};
    var key;
    for (key in local) {
      if (Object.prototype.hasOwnProperty.call(local, key)) merged[key] = local[key];
    }
    for (key in server) {
      if (Object.prototype.hasOwnProperty.call(server, key)) merged[key] = server[key];
    }
    merged.client_id = server.client_id || local.client_id;
    merged.local = false;
    return merged;
  }

  function applySendResult(message, response) {
    var next = {};
    var key;
    for (key in message) {
      if (Object.prototype.hasOwnProperty.call(message, key)) next[key] = message[key];
    }
    if (!response || typeof response !== "object") {
      next.delivery = DELIVERY_FAILED;
      next.reason = "the message could not be sent";
      return next;
    }
    if (response.message && typeof response.message === "object" && response.message.id) {
      next = mergeServerItem(next, response.message);
    }
    if (typeof response.delivery === "string" && response.delivery) {
      next.delivery = response.delivery;
    } else if (response.accepted === true) {
      next.delivery = DELIVERY_ACCEPTED;
    } else {
      next.delivery = DELIVERY_FAILED;
    }
    if (response.accepted !== true && next.delivery === DELIVERY_ACCEPTED) {
      next.delivery = DELIVERY_FAILED;
    }
    if (typeof response.reason === "string" && response.reason) {
      next.reason = response.reason;
    } else if (next.delivery !== DELIVERY_FAILED) {
      next.reason = null;
    }
    next.local = false;
    return next;
  }

  function reconcileMessages(localMessages, serverMessages, cap) {
    var limit = Math.max(1, cap || MAX_MESSAGE_ITEMS);
    var locals = Array.isArray(localMessages) ? localMessages : [];
    var server = Array.isArray(serverMessages) ? serverMessages : [];
    var byClient = {};
    var mine = [];
    var result = [];
    var i;
    for (i = 0; i < locals.length; i += 1) {
      var item = locals[i];
      if (!item) continue;
      // Only the viewer's own bubbles are carried across polls. Rows that
      // merely came from an earlier server page must be re-confirmed by the
      // current page or they are dropped: otherwise the carried duplicates
      // fill the cap and the newest messages get sliced away forever.
      if (typeof item.client_id === "string" && item.client_id) {
        byClient[item.client_id] = item;
        mine.push(item);
      } else if (item.local === true) {
        mine.push(item);
      }
    }
    var matched = {};
    for (i = 0; i < server.length; i += 1) {
      var entry = server[i];
      if (!entry || typeof entry.id !== "string" || !entry.id) continue;
      if (entry.client_id && byClient[entry.client_id]) {
        matched[entry.client_id] = true;
        result.push(mergeServerItem(byClient[entry.client_id], entry));
      } else {
        result.push(entry);
      }
    }
    var kept = [];
    for (i = 0; i < mine.length; i += 1) {
      var own = mine[i];
      if (own.client_id && matched[own.client_id]) continue;
      kept.push(own);
    }
    if (kept.length > MAX_OWN_MESSAGES) {
      kept = kept.slice(-MAX_OWN_MESSAGES);
    }
    return result.concat(kept).slice(-limit);
  }

  function isTerminalDelivery(delivery) {
    return delivery === DELIVERY_DELIVERED || delivery === DELIVERY_FAILED;
  }

  function deliveryLabel(message) {
    var delivery = message && message.delivery;
    if (delivery === DELIVERY_PENDING) return "sending…";
    if (delivery === DELIVERY_ACCEPTED) return "sent";
    if (delivery === DELIVERY_DELIVERED) return "delivered";
    if (delivery === DELIVERY_FAILED) {
      return (message && message.reason) || "not delivered";
    }
    return "";
  }

  function activityDotClass(label) {
    var value = String(label || "").toLowerCase();
    if (value === "speaking") return "speaking";
    if (value === "moving" || value === "looking" || value === "waving") return "moving";
    if (value === "dancing") return "dancing";
    if (value === "live") return "live";
    return "";
  }

  function connectionNotice(status) {
    if (status === STATUS_OFFLINE) return "Lacrimosa is offline.";
    if (status === STATUS_RECONNECTING) return "Reconnecting to the runtime…";
    if (status === STATUS_CONNECTING) return "Connecting to the runtime…";
    return "";
  }

  return {
    VIEWER_ID_PATTERN: VIEWER_ID_PATTERN,
    SESSION_ID_PATTERN: SESSION_ID_PATTERN,
    CLIENT_ID_PATTERN: CLIENT_ID_PATTERN,
    MAX_MESSAGE_ITEMS: MAX_MESSAGE_ITEMS,
    MAX_OWN_MESSAGES: MAX_OWN_MESSAGES,
    POLL_BASE_MS: POLL_BASE_MS,
    POLL_HIDDEN_MS: POLL_HIDDEN_MS,
    POLL_MAX_MS: POLL_MAX_MS,
    OFFLINE_AFTER_FAILURES: OFFLINE_AFTER_FAILURES,
    STATUS_LIVE: STATUS_LIVE,
    STATUS_CONNECTING: STATUS_CONNECTING,
    STATUS_OFFLINE: STATUS_OFFLINE,
    STATUS_RECONNECTING: STATUS_RECONNECTING,
    STATUS_DEGRADED: STATUS_DEGRADED,
    DELIVERY_PENDING: DELIVERY_PENDING,
    DELIVERY_ACCEPTED: DELIVERY_ACCEPTED,
    DELIVERY_DELIVERED: DELIVERY_DELIVERED,
    DELIVERY_FAILED: DELIVERY_FAILED,
    isValidViewerId: isValidViewerId,
    isValidSessionId: isValidSessionId,
    createViewerId: createViewerId,
    createSessionId: createSessionId,
    createClientId: createClientId,
    resolveViewerId: resolveViewerId,
    nextPollDelay: nextPollDelay,
    deriveConnection: deriveConnection,
    localMessage: localMessage,
    applySendResult: applySendResult,
    reconcileMessages: reconcileMessages,
    isTerminalDelivery: isTerminalDelivery,
    deliveryLabel: deliveryLabel,
    activityDotClass: activityDotClass,
    connectionNotice: connectionNotice
  };
});
