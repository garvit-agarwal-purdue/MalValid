/* malvalid local web UI: progressive enhancement only.

   Every page works without this script: forms post normally and the run page has a refresh
   link. The script adds the light/dark toggle, live run progress (polls /api/runs/<id> every
   2 s while a run is queued or running), dashboard auto-refresh while runs are active, compare
   selection, the upload/path toggle, the "Validate adapter" button, confirm dialogs for
   cancel/delete/no-sandbox, copy buttons and a live CLI preview on the new-run form.

   It is loaded from <head> (not deferred) so the theme applies before first paint; DOM work
   waits for DOMContentLoaded. Server and user strings are only ever inserted with textContent,
   and the page needs no inline script (CSP script-src 'self'). */
(function () {
  "use strict";
  var doc = document, win = window, root = doc.documentElement;
  root.classList.add("js");

  /* ---------------------------------------------------------------- storage (may throw) */
  function sget(area, key) {
    try { return win[area].getItem(key); } catch (e) { return null; }
  }
  function sset(area, key, value) {
    try { win[area].setItem(key, value); } catch (e) { /* private mode: not remembered */ }
  }

  /* ---------------------------------------------------------------- theme (before paint) */
  var THEME_KEY = "malvalid-ui-theme";
  var savedTheme = sget("localStorage", THEME_KEY);
  if (savedTheme === "dark" || savedTheme === "light") { root.setAttribute("data-theme", savedTheme); }

  /* ---------------------------------------------------------------- helpers */
  function $(sel, ctx) { return (ctx || doc).querySelector(sel); }
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || doc).querySelectorAll(sel)); }
  function el(tag, cls, text) {
    var e = doc.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined && text !== null) { e.textContent = String(text); }
    return e;
  }
  var SVGNS = "http://www.w3.org/2000/svg";
  function icon(name, extra) {
    var svg = doc.createElementNS(SVGNS, "svg");
    svg.setAttribute("class", "ic" + (extra ? " " + extra : "") + (name === "i-running" ? " spin" : ""));
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    var use = doc.createElementNS(SVGNS, "use");
    use.setAttribute("href", "#" + name);
    svg.appendChild(use);
    return svg;
  }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function csrfToken() {
    var m = $('meta[name="csrf-token"]');
    return m ? (m.getAttribute("content") || "") : "";
  }
  /* the --root-path URL prefix (behind a reverse proxy such as Open OnDemand), "" otherwise */
  function rootPath() {
    var m = $('meta[name="root-path"]');
    var r = m ? (m.getAttribute("content") || "") : "";
    return localPath(r) ? r.replace(/\/+$/, "") : "";
  }
  function appUrl(p) { return rootPath() + p; }
  function localPath(p) { return typeof p === "string" && p.charAt(0) === "/" && p.charAt(1) !== "/" && p.charAt(1) !== "\\"; }

  /* fetch JSON with the session cookie and the CSRF header; resolves {ok, status, data} */
  function api(url, opts) {
    opts = opts || {};
    var headers = { "Accept": "application/json", "X-CSRF-Token": csrfToken() };
    return win.fetch(url, {
      method: opts.method || "GET", body: opts.body, headers: headers,
      credentials: "same-origin", cache: "no-store", redirect: "follow"
    }).then(function (res) {
      return res.text().then(function (text) {
        var data = null;
        try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
        return { ok: res.ok, status: res.status, data: data, text: text };
      });
    });
  }
  function messageFrom(res) {
    var d = res && res.data;
    if (d && typeof d === "object") {
      var m = d.error || d.detail || d.message;
      if (typeof m === "string" && m) { return m; }
      if (Array.isArray(d.errors) && d.errors.length) { return d.errors.join(" "); }
    }
    return "";
  }

  /* ---------------------------------------------------------------- vocabulary (mirrors web/filters.py) */
  var STATUS = {
    pass: ["good", "Pass", "i-pass"], warn: ["warning", "Warn", "i-warn"], fail: ["critical", "Fail", "i-fail"],
    error: ["critical", "Error", "i-error"], skipped: ["neutral", "Skipped", "i-skip"],
    pending: ["neutral", "Pending", "i-pending"], running: ["info", "Running", "i-running"],
    queued: ["neutral", "Queued", "i-clock"], finished: ["neutral", "Finished", "i-check"],
    failed: ["critical", "Failed", "i-error"], cancelled: ["neutral", "Cancelled", "i-stop"],
    interrupted: ["warning", "Interrupted", "i-warn"], done: ["good", "Done", "i-check"]
  };
  var ALIASES = { passed: "pass", warning: "warn", failure: "fail", errored: "error", skip: "skipped",
                  canceled: "cancelled", complete: "finished", completed: "finished" };
  function statusKey(s) {
    var k = String(s === null || s === undefined ? "" : s).trim().toLowerCase().replace(/[\s-]+/g, "_");
    k = ALIASES[k] || k;
    return STATUS.hasOwnProperty(k) ? k : "unknown";
  }
  function statusMeta(s) {
    var k = statusKey(s);
    return STATUS[k] || ["neutral", "Unknown", "i-unknown"];
  }
  var ACTIVE = ["queued", "running"];
  var DONE = ["pass", "warn", "fail", "skipped", "error"];
  var STAGES = [
    ["inspecting", "Inspect adapter", "Inspecting the adapter and its declarations"],
    ["scanning", "Scan files", "Scanning the model files before anything is loaded (M0)"],
    ["loading", "Load model", "Loading the model in the sandbox"],
    ["modules", "Run tests", "Running the test modules"],
    ["verdict", "Verdict", "Computing the verdict and score"],
    ["writing", "Write report", "Writing report.json and report.html"]
  ];
  var STAGE_TEXT = { starting: "Starting the run", done: "Finished", failed: "The run stopped with an error",
                     queued: "Waiting for a free worker" };
  STAGES.forEach(function (s) { STAGE_TEXT[s[0]] = s[2]; });
  function stageKey(s) { return String(s === null || s === undefined ? "" : s).trim().toLowerCase(); }

  function fmtDuration(x) {
    if (!isNum(x) || x < 0) { return "—"; }
    if (x < 10) { return x.toFixed(1) + "s"; }
    var t = Math.round(x);
    if (t < 60) { return t + "s"; }
    if (t < 3600) { return Math.floor(t / 60) + "m " + (t % 60) + "s"; }
    if (t < 86400) { return Math.floor(t / 3600) + "h " + Math.floor((t % 3600) / 60) + "m"; }
    return Math.floor(t / 86400) + "d " + Math.floor((t % 86400) / 3600) + "h";
  }
  function chip(status) {
    var m = statusMeta(status);
    var c = el("span", "chip tone-" + m[0]);
    c.appendChild(icon(m[2]));
    c.appendChild(el("span", null, m[1]));
    return c;
  }
  function setToneClass(node, tone) {
    if (!node) { return; }
    node.className = node.className.replace(/\btone-[a-z]+\b/g, "").replace(/\s+/g, " ").trim() + " tone-" + tone;
  }

  function onReady(fn) {
    if (doc.readyState === "loading") { doc.addEventListener("DOMContentLoaded", fn); } else { fn(); }
  }

  /* ---------------------------------------------------------------- theme toggle */
  function initTheme() {
    var toggle = doc.getElementById("theme-toggle");
    if (!toggle) { return; }
    function effective() {
      var t = root.getAttribute("data-theme");
      if (t === "dark" || t === "light") { return t; }
      return win.matchMedia && win.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    function sync() {
      var dark = effective() === "dark";
      toggle.setAttribute("aria-pressed", dark ? "true" : "false");
      toggle.title = dark ? "Switch to the light theme" : "Switch to the dark theme";
    }
    toggle.addEventListener("click", function () {
      var next = effective() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      sset("localStorage", THEME_KEY, next);
      sync();
    });
    sync();
  }

  /* ---------------------------------------------------------------- copy buttons */
  function copyText(text) {
    function fallback() {
      var ta = el("textarea", "visually-hidden");
      ta.value = text;
      ta.setAttribute("readonly", "");
      doc.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = doc.execCommand("copy"); } catch (e) { ok = false; }
      doc.body.removeChild(ta);
      return ok;
    }
    if (win.navigator.clipboard && win.isSecureContext) {
      return win.navigator.clipboard.writeText(text).then(function () { return true; }, fallback);
    }
    return Promise.resolve(fallback());
  }
  function initCopy() {
    doc.addEventListener("click", function (e) {
      var b = e.target && e.target.closest ? e.target.closest("[data-copy]") : null;
      if (!b) { return; }
      var target = doc.getElementById(b.getAttribute("data-copy"));
      if (!target) { return; }
      var label = b.querySelector("span");
      var old = label ? label.textContent : "";
      copyText(target.textContent).then(function (ok) {
        if (label) {
          label.textContent = ok ? "Copied" : "Select and copy";
          win.setTimeout(function () { label.textContent = old; }, 1600);
        }
      });
    });
  }

  /* ---------------------------------------------------------------- confirm + API forms (cancel / delete) */
  function formError(form, text) {
    var p = form.nextElementSibling;
    if (!p || !p.hasAttribute("data-form-error")) {
      p = el("p", "inline-warn");
      p.setAttribute("data-form-error", "");
      p.setAttribute("role", "alert");
      form.parentNode.insertBefore(p, form.nextSibling);
    }
    p.textContent = "";
    p.appendChild(icon("i-warn"));
    p.appendChild(el("span", null, text));
  }
  function initApiForms() {
    $all("form[data-api-form]").forEach(function (form) {
      form.addEventListener("submit", function (e) {
        e.preventDefault();
        var msg = form.getAttribute("data-confirm");
        if (msg && !win.confirm(msg)) { return; }
        var btn = $('button[type="submit"]', form);
        if (btn) { btn.disabled = true; }
        api(form.getAttribute("action"), { method: "POST", body: new FormData(form) }).then(function (res) {
          if (res.ok) {
            var next = form.getAttribute("data-next");
            if (localPath(next)) { win.location.assign(next); } else { win.location.reload(); }
            return;
          }
          formError(form, messageFrom(res) || ("The server refused the request (HTTP " + res.status + ")."));
          if (btn) { btn.disabled = false; }
        }, function () {
          formError(form, "Could not reach the malvalid server. Is malvalid serve still running?");
          if (btn) { btn.disabled = false; }
        });
      });
    });
  }

  /* ---------------------------------------------------------------- run page: live progress */
  function stepperItems(stage) {
    var k = stageKey(stage), order = STAGES.map(function (s) { return s[0]; });
    var cur = k === "done" ? order.length : order.indexOf(k);
    return STAGES.map(function (s, i) {
      var state = k === "failed" ? "todo" : (i < cur ? "done" : (i === cur ? "current" : "todo"));
      return { key: s[0], label: s[1], state: state };
    });
  }
  function renderSteps(ol, stage) {
    if (!ol) { return; }
    ol.textContent = "";
    stepperItems(stage).forEach(function (s) {
      var li = el("li", s.state);
      li.setAttribute("data-step", s.key);
      if (s.state === "current") { li.setAttribute("aria-current", "step"); }
      li.appendChild(icon(s.state === "done" ? "i-check" : (s.state === "current" ? "i-running" : "i-pending")));
      var span = el("span", null, s.label);
      if (s.state !== "todo") { span.appendChild(el("span", "visually-hidden", s.state === "done" ? " (done)" : " (in progress)")); }
      li.appendChild(span);
      ol.appendChild(li);
    });
  }
  function moduleRow(m) {
    var st = statusKey(m.status || "pending");
    var li = el("li", "status-" + st);
    li.setAttribute("data-module", m.id || "");
    var c = el("span");
    c.appendChild(chip(st));
    li.appendChild(c);
    li.appendChild(el("span", "mod-code", m.code || "?"));
    li.appendChild(el("span", "mod-title", m.title || m.id || "Module"));
    li.appendChild(el("span", "mod-dur", isNum(m.duration_s) && ["pass", "warn", "fail", "error"].indexOf(st) >= 0 ? fmtDuration(m.duration_s) : ""));
    return li;
  }
  function parseTime(iso) {
    if (!iso) { return NaN; }
    var t = Date.parse(iso);
    return isNaN(t) ? NaN : t;
  }

  function initRunPage() {
    var page = $("[data-run-page]");
    if (!page) { return; }
    var status = statusKey(page.getAttribute("data-status"));
    var runId = page.getAttribute("data-run-id") || "";
    if (ACTIVE.indexOf(status) < 0 || !runId) { return; }

    var elapsed = $("[data-live-elapsed]", page);
    var elapsedLabel = $("[data-live-elapsed-label]", page);
    var started = parseTime(page.getAttribute("data-started"));
    var queuedSince = parseTime(page.getAttribute("data-queued-since"));
    var curStatus = status;
    function tick() {
      // A queued run shows how long it has waited; once it runs, the clock restarts at its real start.
      var t0 = curStatus === "queued" ? queuedSince : started;
      if (elapsed && !isNaN(t0)) { elapsed.textContent = fmtDuration(Math.max(0, (Date.now() - t0) / 1000)); }
    }
    tick();
    win.setInterval(tick, 1000);

    var url = appUrl("/api/runs/" + encodeURIComponent(runId));
    var failures = 0, last = {}, lastStage = "", first = true;
    var announce = $("[data-live-announce]", page), pollNote = $("[data-live-poll]", page);
    function note(text) { if (pollNote) { pollNote.textContent = text; } }
    function say(text) { if (announce && text) { announce.textContent = text; } }

    function update(st, prog, job) {
      var meta = statusMeta(st);
      var hero = $(".status-hero", page);
      setToneClass(hero, meta[0]);
      var h = $("[data-live-status]", page);
      if (h) { h.textContent = meta[1]; }
      var eyebrow = $("[data-live-status-text]", page);
      if (eyebrow) { eyebrow.textContent = meta[1].toLowerCase(); }
      var ic = $("[data-live-icon]", page);
      if (ic) { ic.textContent = ""; ic.appendChild(icon(meta[2], "ic-xl")); }
      prog = prog && typeof prog === "object" ? prog : {};
      job = job && typeof job === "object" ? job : {};
      if (st !== "queued" && (curStatus === "queued" || isNaN(started))) {
        var t = parseTime(job.started_at || prog.started_at);
        if (!isNaN(t)) { started = t; }
      }
      if (elapsedLabel) { elapsedLabel.textContent = st === "queued" ? "Queued for" : "Elapsed"; }
      curStatus = st;
      var stage = stageKey(prog.stage);
      var stageText = st === "queued" ? STAGE_TEXT.queued : (STAGE_TEXT[stage] || (stage ? String(prog.stage) : STAGE_TEXT.starting));
      var stageNode = $("[data-live-stage]", page);
      if (stageNode) { stageNode.textContent = stageText; }
      var mod = prog.module && typeof prog.module === "object" ? prog.module : null;
      var modNode = $("[data-live-module]", page);
      if (modNode) { modNode.textContent = mod ? ": " + [mod.code, mod.title].filter(Boolean).join(" ") : ""; }
      var msg = $("[data-live-message]", page);
      if (msg && (prog.message || st !== "queued")) { msg.textContent = prog.message ? String(prog.message) : ""; }
      renderSteps($("[data-live-steps]", page), st === "queued" ? "" : stage);

      var mods = Array.isArray(prog.modules) ? prog.modules.filter(function (m) { return m && typeof m === "object"; }) : [];
      var list = $("[data-live-modules]", page);
      if (list && mods.length) {
        list.textContent = "";
        mods.forEach(function (m) { list.appendChild(moduleRow(m)); });
      }
      var done = mods.filter(function (m) { return DONE.indexOf(statusKey(m.status)) >= 0; }).length;
      var count = $("[data-live-count]", page);
      if (count) { count.textContent = mods.length ? done + " of " + mods.length + " tests finished" : "Test list not known yet"; }
      var bar = $("[data-live-bar]", page);
      if (bar) { bar.className = "fill w-" + (mods.length ? Math.round(100 * done / mods.length) : 0); }

      // Screen-reader announcements: finished tests and stage changes (not on the first poll).
      var news = [];
      mods.forEach(function (m) {
        var k = statusKey(m.status), id = m.id || m.code;
        if (!first && last[id] !== k && DONE.indexOf(k) >= 0) {
          news.push([m.code, m.title].filter(Boolean).join(" ") + ": " + statusMeta(k)[1].toLowerCase());
        }
        last[id] = k;
      });
      if (!first && stage && stage !== lastStage && !news.length) { news.push(stageText); }
      lastStage = stage;
      first = false;
      if (news.length) { say(news.join(". ")); }
    }

    var consoleNode = $("[data-live-console]", page);
    function showConsole(text) {
      if (!consoleNode) { return; }
      var lines = typeof text === "string" ? text.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").split("\n") : [];
      lines = lines.map(function (l) { return l.replace(/\s+$/, ""); });
      while (lines.length && !lines[lines.length - 1]) { lines.pop(); }
      var s = lines.slice(-40).join("\n") || "No output yet.";
      if (consoleNode.textContent === s) { return; }
      var atEnd = consoleNode.scrollHeight - consoleNode.scrollTop - consoleNode.clientHeight < 8;
      consoleNode.textContent = s;
      if (atEnd) { consoleNode.scrollTop = consoleNode.scrollHeight; }
    }

    function schedule(ms) { win.setTimeout(poll, ms); }
    function poll() {
      if (doc.hidden) { schedule(4000); return; }
      api(url).then(function (res) {
        if (res.status === 404) {
          note("This run no longer exists.");
          say("This run no longer exists.");
          return;
        }
        if (!res.ok || !res.data || typeof res.data !== "object") { throw new Error("HTTP " + res.status); }
        failures = 0;
        var d = res.data, sum = d.summary || {}, job = d.job || {};
        var st = statusKey(sum.status || job.status || status);
        if (ACTIVE.indexOf(st) < 0) {
          note("The run has ended; loading the result…");
          say("The run has ended. Loading the result.");
          win.setTimeout(function () { win.location.reload(); }, 400);
          return;
        }
        note("Updating every 2 seconds.");
        update(st, d.progress, job);
        showConsole(d.console_tail);
        schedule(2000);
      }).catch(function () {
        failures += 1;
        note("Cannot reach the malvalid server; retrying. Is malvalid serve still running?");
        schedule(Math.min(15000, 2000 * Math.pow(2, Math.min(failures, 3))));
      });
    }
    schedule(first ? 1200 : 2000);
  }

  /* ---------------------------------------------------------------- dashboard: refresh while runs are active */
  var SEL_KEY = "malvalid-compare-selection";
  function initDashboard() {
    if (!$("[data-dashboard-active]")) { return; }
    var rendered = {};
    $all("[data-run-id][data-status]").forEach(function (n) {
      var id = n.getAttribute("data-run-id");
      if (id) { rendered[id] = n.getAttribute("data-status"); }
    });
    var noteNode = $("[data-refresh-note]");
    function poll() {
      if (doc.hidden) { win.setTimeout(poll, 5000); return; }
      // Only the active runs: cheap with many runs, and the page reloads when that set changes
      // (a new submission appears, or a rendered active run ends).
      api(appUrl("/api/runs?active=1")).then(function (res) {
        if (!res.ok || !Array.isArray(res.data)) { throw new Error("HTTP " + res.status); }
        var changed = false, seen = {};
        res.data.forEach(function (r) {
          var id = r && r.run_id;
          if (!id) { return; }
          seen[id] = true;
          if (!rendered.hasOwnProperty(id) || rendered[id] !== statusKey(r.status)) { changed = true; }
        });
        Object.keys(rendered).forEach(function (id) {
          if (ACTIVE.indexOf(rendered[id]) >= 0 && !seen[id]) { changed = true; }
        });
        if (changed) { win.location.reload(); return; }
        if (noteNode) { noteNode.textContent = "This page updates by itself while runs are in progress."; }
        win.setTimeout(poll, 4000);
      }).catch(function () {
        if (noteNode) { noteNode.textContent = "Cannot reach the malvalid server; retrying."; }
        win.setTimeout(poll, 10000);
      });
    }
    win.setTimeout(poll, 4000);
  }

  /* ---------------------------------------------------------------- dashboard: compare selection */
  function initCompare() {
    var form = $("[data-compare-form]");
    if (!form) { return; }
    var boxes = $all('input[type="checkbox"][name="ids"]', form);
    var btn = $("[data-compare-submit]", form), hint = $("[data-compare-hint]", form);
    var label = btn ? btn.querySelector("span") : null;
    boxes.forEach(function (b) { if (b.disabled) { b.setAttribute("data-locked", ""); } });
    var saved = [];
    try { saved = JSON.parse(sget("sessionStorage", SEL_KEY) || "[]") || []; } catch (e) { saved = []; }
    boxes.forEach(function (b) { if (!b.hasAttribute("data-locked") && saved.indexOf(b.value) >= 0) { b.checked = true; } });
    function selected() { return boxes.filter(function (b) { return b.checked && !b.disabled; }); }
    function sync() {
      var sel = selected(), n = sel.length;
      boxes.forEach(function (b) {
        var tr = b.closest ? b.closest("tr") : null;
        if (tr) { tr.classList.toggle("is-selected", b.checked); }
        if (!b.hasAttribute("data-locked")) { b.disabled = n >= 6 && !b.checked; }
      });
      if (btn) { btn.setAttribute("aria-disabled", n < 2 ? "true" : "false"); }
      if (label) { label.textContent = n >= 2 ? "Compare " + n + " runs" : "Compare selected"; }
      if (hint) {
        hint.textContent = n === 0 ? "Tick 2–6 runs that have a verdict to compare them."
          : n === 1 ? "Tick at least one more run."
          : n >= 6 ? "6 runs selected, the most that fit side by side." : n + " runs selected.";
      }
      sset("sessionStorage", SEL_KEY, JSON.stringify(sel.map(function (b) { return b.value; })));
    }
    form.addEventListener("change", sync);
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var ids = selected().map(function (b) { return b.value; });
      if (ids.length < 2) {
        if (hint) { hint.textContent = "Tick at least two runs with a verdict first."; }
        return;
      }
      win.location.assign(appUrl("/compare?ids=" + ids.map(encodeURIComponent).join(",")));
    });
    sync();
  }

  /* ---------------------------------------------------------------- new-run form */
  var PICKLE_RE = /\.(pkl|pickle|joblib|jbl)$/i;
  function shq(s) {
    s = String(s);
    return /^[A-Za-z0-9_\/.,:=@%+-]+$/.test(s) ? s : "'" + s.replace(/'/g, "'\\''") + "'";
  }
  function initRunForm() {
    var form = $("[data-run-form]");
    if (!form) { return; }
    var radios = $all("[data-mode-radio]", form);
    function mode() {
      var r = radios.filter(function (x) { return x.checked; })[0];
      return r ? r.value : "model";
    }
    function field(id) { return doc.getElementById(id); }
    function val(id) { var f = field(id); return f && !f.disabled ? String(f.value || "").trim() : ""; }
    function files(id) { var f = field(id); return f && !f.disabled && f.files ? Array.prototype.slice.call(f.files) : []; }
    var maxMb = parseFloat(form.getAttribute("data-max-mb"));
    var subInput = $("[data-model-submission]", form);
    var storedLine = $("[data-stored-model]", form);
    function storedModel() {
      if (!subInput || subInput.disabled || !subInput.value || !storedLine || storedLine.hidden) { return ""; }
      return storedLine.getAttribute("data-stored-model") || "";
    }
    function thresholdMode() {
      var r = $all("[data-threshold-mode]", form).filter(function (x) { return x.checked && !x.disabled; })[0];
      return r ? r.value : "calibrate";
    }
    function isModelPath(p) { return !!p && !/\.py$/i.test(p); }

    function modelFlags(p) {
      if (val("model_kind")) { p.push("--model-kind", shq(val("model_kind"))); }
      if (val("feature_version")) { p.push("--feature-version", shq(val("feature_version"))); }
      if (thresholdMode() === "declared") { p.push("--threshold", shq(val("threshold") || "T")); }
      else { p.push("--calibrate-fpr", shq(val("calibrate_fpr") || "0.005")); }
      if (val("training_cutoff")) { p.push("--training-cutoff", shq(val("training_cutoff"))); }
    }
    function cli() {
      var out = $("[data-cli-preview]");
      if (!out) { return; }
      var p = ["malvalid", "run"], m = mode(), model = false;
      if (m === "model") {
        var mf = files("model_file")[0];
        p.push("--model", shq(mf ? mf.name : (storedModel() || "my_model.txt")));
        modelFlags(p);
        model = true;
        var h = files("manifest_file")[0];
        if (h) { p.push("--training-hashes", shq(h.name)); }
        var cf = files("config_file")[0];
        if (cf) { p.push("--config", shq(cf.name)); }
      } else if (m === "upload") {
        var a = files("adapter_file")[0];
        p.push("--adapter", shq(a ? a.name : "my_adapter.py"));
        files("model_files").forEach(function (f) { p.push("--model", shq(f.name)); });
        var c = files("config_file")[0];
        if (c) { p.push("--config", shq(c.name)); }
      } else {
        var ap = val("adapter_path");
        if (isModelPath(ap)) {
          model = true;
          p.push("--model", shq(ap));
          modelFlags(p);
          if (val("training_hashes_path")) { p.push("--training-hashes", shq(val("training_hashes_path"))); }
        } else {
          p.push("--adapter", shq(ap || "my_adapter.py"));
        }
        if (val("config_path")) { p.push("--config", shq(val("config_path"))); }
      }
      if (!model && val("class_name")) { p.push("--class", shq(val("class_name"))); }
      if (val("corpus")) { p.push("--corpus", shq(val("corpus"))); }
      if (field("corpus_dir") && val("corpus_dir")) { p.push("--corpus-dir", shq(val("corpus_dir"))); }
      var seed = field("seed");
      if (seed && val("seed") !== "") { p.push("--seed", shq(val("seed"))); }  // the server passes any entered seed
      if (val("only")) { p.push("--only", shq(val("only").replace(/\s+/g, ""))); }
      if (val("skip")) { p.push("--skip", shq(val("skip").replace(/\s+/g, ""))); }
      if (val("fail_on") && val("fail_on") !== "blocked") { p.push("--fail-on", shq(val("fail_on"))); }
      if (field("allow_pickle") && field("allow_pickle").checked) { p.push("--allow-pickle"); }
      if (field("no_sandbox") && field("no_sandbox").checked) { p.push("--no-sandbox"); }
      out.textContent = p.join(" ");
    }

    var corpusAuto = $("[data-corpus-auto]", form);
    function applyMode() {
      var m = mode();
      $all("[data-mode-panel]", form).forEach(function (panel) {
        var on = (panel.getAttribute("data-mode-panel") || "").split(/\s+/).indexOf(m) >= 0;
        panel.hidden = !on;
        $all("input, select, textarea", panel).forEach(function (i) { i.disabled = !on; });
      });
      $all("[data-adapter-only]", form).forEach(function (b) { b.hidden = m === "model"; });
      if (corpusAuto) {
        var label = corpusAuto.getAttribute("data-label-" + m);
        if (label) { corpusAuto.textContent = label; }
      }
      cli();
    }
    radios.forEach(function (r) { r.addEventListener("change", applyMode); });

    var pickleWarns = $all("[data-pickle-warning]", form);
    var allow = field("allow_pickle");
    function checkPickle() {
      pickleWarns.forEach(function (w) {
        var inp = w.parentNode ? $("[data-pickle-check]", w.parentNode) : null;
        var pk = inp && inp.files ? Array.prototype.some.call(inp.files, function (f) { return PICKLE_RE.test(f.name); }) : false;
        w.hidden = !(pk && !(allow && allow.checked));
      });
    }
    $all("[data-pickle-check]", form).forEach(function (i) { i.addEventListener("change", checkPickle); });
    if (allow) { allow.addEventListener("change", checkPickle); }

    $all("[data-size-check]", form).forEach(function (inp) {
      inp.addEventListener("change", function () {
        var msg = "";
        if (isNum(maxMb) && maxMb > 0 && inp.files) {
          Array.prototype.forEach.call(inp.files, function (f) {
            if (!msg && f.size > maxMb * 1024 * 1024) { msg = f.name + " is larger than the " + maxMb + " MB upload limit. Use “Use files on this machine” instead."; }
          });
        }
        inp.setCustomValidity(msg);
        if (msg) { inp.reportValidity(); }
      });
    });

    var ns = $("[data-no-sandbox]", form), nsWrap = $("[data-no-sandbox-confirm]", form), nsConfirm = field("confirm_no_sandbox");
    function syncNs() {
      if (!ns) { return; }
      if (nsWrap) { nsWrap.hidden = !ns.checked; }
      if (!ns.checked && nsConfirm) { nsConfirm.checked = false; }
    }
    if (ns) {
      ns.addEventListener("change", function () {
        if (ns.checked && !win.confirm("Run this model WITHOUT the sandbox?\n\nIt will load inside the malvalid process with no network or file-system isolation, so a hostile model file could read your files or reach the network. Only do this for a model you trust.")) {
          ns.checked = false;
        }
        syncNs();
      });
    }

    // Clear custom validity messages as soon as the user edits a field.
    form.addEventListener("input", function (e) { if (e.target && e.target.setCustomValidity && e.target.type !== "file") { e.target.setCustomValidity(""); } cli(); });
    form.addEventListener("change", function (e) {
      if (e.target && e.target.setCustomValidity && !e.target.hasAttribute("data-size-check")) { e.target.setCustomValidity(""); }
      cli();
    });

    function requireModel() {
      var mf = field("model_file");
      if (mf && !(mf.files && mf.files.length) && !storedModel()) {
        mf.setCustomValidity("Choose your model file.");
        mf.reportValidity();
        return false;
      }
      return true;
    }
    function requireAdapter() {
      var m = mode();
      if (m === "model") { return requireModel(); }
      if (m === "upload") {
        var a = field("adapter_file");
        if (a && !(a.files && a.files.length)) { a.setCustomValidity("Choose your adapter .py file."); a.reportValidity(); return false; }
      } else {
        var ap = field("adapter_path");
        if (ap && !String(ap.value || "").trim()) { ap.setCustomValidity("Enter the path to your model file or adapter .py file."); ap.reportValidity(); return false; }
      }
      return true;
    }
    function requireThreshold() {
      var t = field("threshold");
      if (t && !t.disabled && thresholdMode() === "declared" && !String(t.value || "").trim()) {
        t.setCustomValidity("Enter the threshold you ship (0 to 1), or choose “Calibrate it for me”.");
        t.reportValidity();
        return false;
      }
      return true;
    }

    // Upload mode without model files: the adapter's model_path will not exist in the submission folder
    // (unless it is an absolute path on this machine), so the run would fail within seconds.
    var noModelWarn = $("[data-no-model-warning]", form);
    function missingModels() { return mode() === "upload" && files("adapter_file").length > 0 && files("model_files").length === 0; }
    function syncModelWarn() { if (noModelWarn) { noModelWarn.hidden = !missingModels(); } }
    ["adapter_file", "model_files"].forEach(function (id) { var f = field(id); if (f) { f.addEventListener("change", syncModelWarn); } });
    radios.forEach(function (r) { r.addEventListener("change", syncModelWarn); });

    var submitBtn = $('button[type="submit"]:not([data-inspect])', form);
    form.addEventListener("submit", function (e) {
      var by = e.submitter;
      if (by && by.hasAttribute("data-inspect")) { return; }  // "Inspect model" without JS help (path mode)
      if (!requireAdapter() || !requireThreshold()) { e.preventDefault(); return; }
      if (missingModels()) {
        syncModelWarn();
        if (!win.confirm("No model files are chosen. Your adapter's model_path must be one of the uploaded files (unless it is an absolute path on this machine), otherwise the run fails. Start the run anyway?")) {
          e.preventDefault();
          var mf = field("model_files");
          if (mf && mf.focus) { mf.focus(); }
          return;
        }
      }
      if (ns && ns.checked && nsConfirm && !nsConfirm.checked) {
        e.preventDefault();
        nsConfirm.setCustomValidity("Confirm that you understand the model runs without isolation, or untick “Run without the sandbox”.");
        nsConfirm.reportValidity();
        return;
      }
      if (submitBtn) {
        win.setTimeout(function () {
          submitBtn.disabled = true;
          var s = submitBtn.querySelector("span");
          if (s) { s.textContent = mode() === "path" ? "Starting…" : "Uploading and starting…"; }
        }, 0);
      }
    });
    if (nsConfirm) { nsConfirm.addEventListener("change", function () { nsConfirm.setCustomValidity(""); }); }

    // "Inspect model": upload only the model file to /runs/inspect and fill in what was detected.
    // Without JS the button posts the whole form there and the page comes back filled in.
    var ibtn = $("[data-inspect]", form), ibox = $("[data-inspect-result]", form);
    if (ibtn && ibox && subInput) {
      ibtn.addEventListener("click", function (e) {
        if (mode() !== "model") { return; }
        e.preventDefault();
        if (!requireModel()) { return; }
        var fd = new FormData(form);
        ["adapter_file", "model_files", "manifest_file", "config_file"].forEach(function (k) { fd.delete(k); });
        ibox.hidden = false;
        ibox.textContent = "";
        var wait = el("p", "small muted");
        wait.appendChild(icon("i-running"));
        wait.appendChild(el("span", null, " Reading the model file…"));
        ibox.appendChild(wait);
        ibox.setAttribute("aria-busy", "true");
        ibtn.disabled = true;
        api(appUrl("/runs/inspect"), { method: "POST", body: fd }).then(function (res) {
          applyInspection(res);
        }, function () {
          applyInspection({ ok: false, status: 0, data: { error: "Could not reach the malvalid server. Is malvalid serve still running?" } });
        }).then(function () {
          ibtn.disabled = false;
          ibox.removeAttribute("aria-busy");
          cli();
        });
      });
    }
    function setDetected(key, text, good) {
      var n = $('[data-detected="' + key + '"]', form);
      if (!n) { return; }
      n.textContent = "";
      n.hidden = !text;
      if (text) { n.appendChild(icon(good ? "i-pass" : "i-warn")); n.appendChild(el("span", null, text)); }
    }
    function applyInspection(res) {
      var d = res && res.data && typeof res.data === "object" ? res.data : null;
      ibox.textContent = "";
      if (!res || !res.ok || !d || d.error) {
        var errs = d && Array.isArray(d.errors) ? d.errors.filter(function (x) { return typeof x === "string" && x; }) : [];
        var bad = el("div", "callout tone-critical");
        bad.appendChild(icon("i-error"));
        bad.appendChild(el("p", null, (d && d.error) || errs.join("; ") || messageFrom(res) || ("Could not inspect the model (HTTP " + (res ? res.status : "?") + ").")));
        ibox.appendChild(bad);
        return;
      }
      if (d.model_submission) {
        subInput.value = d.model_submission;
        var mf = field("model_file");
        if (mf) { mf.value = ""; mf.setCustomValidity(""); }
        if (storedLine) {
          storedLine.setAttribute("data-stored-model", d.file_name || "");
          storedLine.hidden = false;
          var t = $("[data-stored-model-text]", storedLine);
          if (t) { t.textContent = "Using uploaded " + d.file_name + (isNum(d.size) ? " (" + fmtSize(d.size) + ")" : "") + " — choose a file to replace it."; }
        }
      }
      var kind = field("model_kind"), fv = field("feature_version");
      if (kind && !kind.value && d.model_kind) { kind.value = d.model_kind; }
      if (fv && !fv.value && d.feature_version && !d.is_pickle) { fv.value = d.feature_version; }
      setDetected("model_kind", d.model_kind ? "detected from model: " + d.model_kind + (d.is_pickle ? " (a guess from a static scan of the pickle)" : "") : "", true);
      setDetected("feature_version", d.feature_version ? "detected from model: " + d.feature_version + " (" + d.n_features + " features)"
        : (isNum(d.n_features) ? "the model expects " + d.n_features + " features, which matches neither" : ""), !!d.feature_version);
      var errs2 = Array.isArray(d.errors) ? d.errors : [];
      var tone = errs2.length ? "tone-critical" : (d.ok ? "tone-good" : "tone-warning");
      var box = el("div", "callout " + tone);
      box.appendChild(icon(errs2.length ? "i-error" : (d.ok ? "i-pass" : "i-warn")));
      var inner = el("div");
      inner.appendChild(el("p", null, errs2.length ? d.file_name + " cannot be evaluated as it is." :
        (d.ok ? "Detected from " + d.file_name + "." : "Detected from " + d.file_name + "; choose what is missing below.")));
      var dl = el("dl", "facts compact");
      [["Format", d.format || "unknown"], ["Model kind", d.model_kind || "not detected"],
       ["Features", isNum(d.n_features) ? String(d.n_features) : "unknown"],
       ["Feature version", d.feature_version || (isNum(d.n_features) ? "no match" : "not detected")]].forEach(function (r) {
        dl.appendChild(el("dt", null, r[0]));
        dl.appendChild(el("dd", null, r[1]));
      });
      if (d.default_corpus) { dl.appendChild(el("dt", null, "Default corpus")); dl.appendChild(el("dd", null, d.default_corpus + " (used when Corpus is Auto)")); }
      inner.appendChild(dl);
      [[errs2, null], [Array.isArray(d.notes) ? d.notes : [], "small"]].forEach(function (pair) {
        if (!pair[0].length) { return; }
        var ul = el("ul", pair[1]);
        pair[0].forEach(function (x) { ul.appendChild(el("li", null, String(x))); });
        inner.appendChild(ul);
      });
      if (d.is_pickle) {
        inner.appendChild(el("p", "inline-warn", "A pickle-based file: malvalid never unpickles it to look inside, so choose its model kind and feature version yourself, and tick “Allow pickle-based model files” only for a model you trust."));
      }
      box.appendChild(inner);
      ibox.appendChild(box);
    }
    function fmtSize(n) {
      var u = ["B", "KiB", "MiB", "GiB"], i = 0;
      while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
      return (i ? n.toFixed(1) : String(n)) + " " + u[i];
    }

    // "Validate adapter": the same fields, posted to /api/validate.
    var vbtn = $("[data-validate]", form), vbox = $("[data-validation-result]", form), vbody = $("[data-validation-body]", form);
    if (vbtn && vbox && vbody) {
      vbtn.addEventListener("click", function () {
        if (!requireAdapter()) { return; }
        vbox.hidden = false;
        vbody.textContent = "";
        var wait = el("p", "small muted");
        wait.appendChild(icon("i-running"));
        wait.appendChild(el("span", null, " Validating: malvalid scans the files, loads the model in the sandbox and probes it. Large models can take a minute."));
        vbody.appendChild(wait);
        vbox.setAttribute("aria-busy", "true");
        vbtn.disabled = true;
        if (vbox.scrollIntoView) { vbox.scrollIntoView({ block: "nearest" }); }
        api(appUrl("/api/validate"), { method: "POST", body: new FormData(form) }).then(function (res) {
          renderValidation(vbody, res);
        }, function () {
          renderValidation(vbody, { ok: false, status: 0, data: { error: "Could not reach the malvalid server. Is malvalid serve still running?" } });
        }).then(function () {
          vbtn.disabled = false;
          vbox.removeAttribute("aria-busy");
        });
      });
    }

    applyMode();
    checkPickle();
    syncNs();
    cli();
  }

  function renderValidation(body, res) {
    body.textContent = "";
    var d = res && res.data && typeof res.data === "object" ? res.data : null;
    // Every failure the server reports (form errors: 400 + errors[]; validator crash, timeout or
    // unreadable output: error) comes with an empty checks list, so no checks = could not validate.
    if (!d || !Array.isArray(d.checks) || !d.checks.length) {
      var box = el("div", "callout tone-critical");
      box.appendChild(icon("i-error"));
      var inner = el("div");
      var errs = d && Array.isArray(d.errors) ? d.errors.filter(function (x) { return typeof x === "string" && x; }) : [];
      if (errs.length > 1) {
        inner.appendChild(el("p", null, "Fix these " + errs.length + " problems, then validate again:"));
        var ul = el("ul");
        errs.forEach(function (x) { ul.appendChild(el("li", null, x)); });
        inner.appendChild(ul);
      } else {
        inner.appendChild(el("p", null, messageFrom(res) || ("Validation could not run (HTTP " + (res ? res.status : "?") + ").")));
      }
      box.appendChild(inner);
      body.appendChild(box);
      return;
    }
    var nFail = d.checks.filter(function (c) { return statusKey(c && c.status) === "fail"; }).length;
    var nWarn = d.checks.filter(function (c) { return statusKey(c && c.status) === "warn"; }).length;
    var ok = d.ok === true && nFail === 0;
    var head = el("p", "callout " + (ok ? (nWarn ? "tone-warning" : "tone-good") : "tone-critical"));
    head.appendChild(icon(ok ? (nWarn ? "i-warn" : "i-pass") : "i-fail"));
    var ht = el("span");
    var strong = el("strong", null, ok ? "The adapter meets the submission contract." : nFail + " check" + (nFail === 1 ? "" : "s") + " failed. Fix " + (nFail === 1 ? "it" : "them") + " before starting the run.");
    ht.appendChild(strong);
    var extra = [];
    if (ok && nWarn) { extra.push(nWarn + " warning" + (nWarn === 1 ? "" : "s") + " to review"); }
    if (d.class_name) { extra.push("class " + d.class_name); }
    if (isNum(d.duration_s)) { extra.push("took " + fmtDuration(d.duration_s)); }
    if (extra.length) { ht.appendChild(doc.createTextNode(" " + extra.join(" · ") + ".")); }
    head.appendChild(ht);
    body.appendChild(head);

    var list = el("ul", "vcheck-list");
    d.checks.forEach(function (c) {
      if (!c || typeof c !== "object") { return; }
      var li = el("li", "vcheck");
      var left = el("span");
      left.appendChild(chip(c.status));
      li.appendChild(left);
      var right = el("div");
      right.appendChild(el("span", "vname", String(c.name || "check").replace(/_/g, " ")));
      if (c.detail) { right.appendChild(el("p", "vdetail", c.detail)); }
      li.appendChild(right);
      list.appendChild(li);
    });
    body.appendChild(list);

    var decl = d.declarations && typeof d.declarations === "object" ? d.declarations : null;
    if (decl) {
      var h = el("h3", "small muted", "What the adapter declares");
      body.appendChild(h);
      var dl = el("dl", "facts");
      [["feature_version", "Feature version"], ["model_kind", "Model kind"], ["operating_threshold", "Operating threshold"],
       ["training_cutoff", "Training cutoff"], ["training_hashes_path", "Training manifest"]].forEach(function (k) {
        var v = decl[k[0]];
        dl.appendChild(el("dt", null, k[1]));
        var dd = el("dd", "wrap-any", v === null || v === undefined || v === "" ? "not declared" : v);
        if (v === null || v === undefined || v === "") { dd.className += " muted"; }
        dl.appendChild(dd);
      });
      if (Array.isArray(decl.model_paths) && decl.model_paths.length) {
        dl.appendChild(el("dt", null, decl.model_paths.length > 1 ? "Model files" : "Model file"));
        dl.appendChild(el("dd", "wrap-any", decl.model_paths.join(", ")));
      }
      body.appendChild(dl);
    }
  }

  /* ---------------------------------------------------------------- misc */
  function initFocus() {
    var t = $("[data-focus-on-load]");
    if (t && t.focus) { t.focus(); }
  }
  function initPrint() {
    var reopened = [];
    win.addEventListener("beforeprint", function () {
      reopened = [];
      $all("details:not([open])").forEach(function (d) { d.setAttribute("open", ""); reopened.push(d); });
    });
    win.addEventListener("afterprint", function () {
      reopened.forEach(function (d) { d.removeAttribute("open"); });
      reopened = [];
    });
  }

  onReady(function () {
    var steps = [initTheme, initCopy, initApiForms, initRunPage, initDashboard, initCompare, initRunForm, initFocus, initPrint];
    steps.forEach(function (fn) {
      try { fn(); } catch (e) { if (win.console && win.console.error) { win.console.error("malvalid ui:", e); } }
    });
  });
})();
