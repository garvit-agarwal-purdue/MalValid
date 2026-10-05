/* malvalid report: progressive enhancement only. The report is complete without this script
   (charts are static SVG); it adds hover/keyboard readouts, a data-table view, CSV export,
   a light/dark toggle and print expansion. It never touches the network and never inserts
   report strings as HTML (textContent only). */
(function () {
  "use strict";
  var doc = document, root = doc.documentElement;
  root.classList.add("js");

  /* ---------------------------------------------------------------- theme */
  var THEME_KEY = "malvalid-report-theme";
  function storedTheme() {
    try { return window.localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function storeTheme(t) {
    try { window.localStorage.setItem(THEME_KEY, t); } catch (e) { /* private mode: not remembered */ }
  }
  function systemDark() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    return t === "dark" || t === "light" ? t : (systemDark() ? "dark" : "light");
  }
  var saved = storedTheme();
  if (saved === "dark" || saved === "light") { root.setAttribute("data-theme", saved); }
  var toggle = doc.getElementById("theme-toggle");
  function syncToggle() {
    if (!toggle) { return; }
    var dark = effectiveTheme() === "dark";
    toggle.setAttribute("aria-pressed", dark ? "true" : "false");
    toggle.title = dark ? "Switch to the light theme" : "Switch to the dark theme";
  }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      storeTheme(next);
      syncToggle();
    });
    syncToggle();
  }

  /* ---------------------------------------------------------------- print: expand collapsed details */
  var reopened = [];
  window.addEventListener("beforeprint", function () {
    reopened = [];
    Array.prototype.forEach.call(doc.querySelectorAll("details:not([open])"), function (d) {
      d.setAttribute("open", ""); reopened.push(d);
    });
  });
  window.addEventListener("afterprint", function () {
    reopened.forEach(function (d) { d.removeAttribute("open"); });
    reopened = [];
  });

  /* ---------------------------------------------------------------- helpers */
  function fmt(v) {
    if (v === null || v === undefined || typeof v !== "number" || !isFinite(v)) { return "—"; }
    var a = Math.abs(v);
    if (a === 0) { return "0"; }
    if (a >= 1e6 || a < 1e-3) { return v.toExponential(2).replace("e+", "e"); }
    if (Math.round(v) === v) { return v.toLocaleString("en-US"); }
    if (a >= 1000) { return v.toLocaleString("en-US", { maximumFractionDigits: 1 }); }
    return String(parseFloat(v.toPrecision(4)));
  }
  function el(tag, cls, text) {
    var e = doc.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined && text !== null) { e.textContent = String(text); }
    return e;
  }
  function lg(v) { return Math.log(v) / Math.LN10; }
  function mapper(a) {
    var la = a.log ? lg(a.lo) : a.lo, lb = a.log ? lg(a.hi) : a.hi;
    return function (v) { return a.p0 + ((a.log ? lg(v) : v) - la) / (lb - la) * (a.p1 - a.p0); };
  }
  function okv(a, v) { return typeof v === "number" && isFinite(v) && (!a.log || v > 0); }
  function lowerBound(arr, x, key) {  /* first index with arr[i][key] >= x */
    var lo = 0, hi = arr.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (arr[mid][key] < x) { lo = mid + 1; } else { hi = mid; } }
    return lo;
  }
  function nearestIdx(arr, x, key) {
    if (!arr.length) { return -1; }
    var i = lowerBound(arr, x, key);
    if (i >= arr.length) { return arr.length - 1; }
    if (i > 0 && Math.abs(arr[i - 1][key] - x) <= Math.abs(arr[i][key] - x)) { return i - 1; }
    return i;
  }
  function xname(spec, x) {  /* period label for an index-valued x (M2 decay charts) */
    var n = spec.xnames && typeof x === "number" ? spec.xnames[String(x)] : null;
    return n ? " (" + n + ")" : "";
  }
  function seriesName(s, j) { return s.label || (j === 0 ? "value" : "series " + (j + 1)); }

  var specs = {};
  try {
    var blob = doc.getElementById("mg-chart-data");
    specs = blob ? JSON.parse(blob.textContent || "{}") : {};
  } catch (e) { specs = {}; }

  /* ---------------------------------------------------------------- tabular view of a chart */
  function tabulate(spec) {
    var cols = [], rows = [], series = spec.series || [];
    if (spec.kind === "bar") {
      cols.push(spec.xlabel || "category");
      series.forEach(function (s, j) { cols.push(seriesName(s, j)); });
      (spec.cats || []).forEach(function (c, i) {
        var r = [c];
        series.forEach(function (s) { r.push(s.y ? s.y[i] : null); });
        rows.push(r);
      });
      return { cols: cols, rows: rows };
    }
    var shared = series.length > 0 && series.every(function (s) {
      var a = s.x || [], b = series[0].x || [];
      if (a.length !== b.length) { return false; }
      for (var i = 0; i < a.length; i++) { if (a[i] !== b[i]) { return false; } }
      return true;
    });
    if (shared) {
      cols.push(spec.xlabel || "x");
      if (spec.xnames) { cols.push("period"); }
      series.forEach(function (s, j) { cols.push(seriesName(s, j)); });
      (series[0].x || []).forEach(function (x, i) {
        var r = [x];
        if (spec.xnames) { r.push(xname(spec, x).replace(/^ \((.*)\)$/, "$1")); }
        series.forEach(function (s) { r.push(s.y ? s.y[i] : null); });
        rows.push(r);
      });
    } else {
      cols = ["series", spec.xlabel || "x", spec.ylabel || "y"];
      series.forEach(function (s, j) {
        var xs = s.x || [], ys = s.y || [];
        for (var i = 0; i < Math.min(xs.length, ys.length); i++) { rows.push([seriesName(s, j), xs[i], ys[i]]); }
      });
    }
    return { cols: cols, rows: rows };
  }
  function csvCell(v) {
    if (v === null || v === undefined) { return ""; }
    if (typeof v === "number") { return isFinite(v) ? String(v) : ""; }
    var s = String(v);
    if (/^[=+@\t\r]/.test(s)) { s = "'" + s; }  /* neutralise spreadsheet formulas */
    return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function download(name, text) {
    var a = el("a");
    a.download = name;
    var url = null;
    try {
      url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
      a.href = url;
    } catch (e) {
      a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(text);
    }
    a.style.display = "none";
    doc.body.appendChild(a);
    a.click();
    setTimeout(function () { doc.body.removeChild(a); if (url) { URL.revokeObjectURL(url); } }, 0);
  }

  /* ---------------------------------------------------------------- interactive chart */
  function Chart(fig, spec) {
    this.fig = fig; this.spec = spec;
    this.svg = fig.querySelector("svg.plot");
    this.body = fig.querySelector(".chart-body");
    this.tip = fig.querySelector(".tip");
    this.layer = this.svg ? this.svg.querySelector("g.hover") : null;
    this.cursor = -1;
    if (!this.svg || !this.tip || !this.layer) { return; }
    var P = spec.plot || [0, 0, spec.w, spec.h];
    this.P = { x0: P[0], y0: P[1], x1: P[2], y1: P[3] };
    if (spec.kind === "bar") { this.initBar(); } else { this.initXY(); }
    var self = this;
    this.svg.addEventListener("pointermove", function (e) { self.onPointer(e); });
    this.svg.addEventListener("pointerdown", function (e) { self.onPointer(e); });
    this.svg.addEventListener("pointerleave", function () { self.hide(); });
    this.svg.addEventListener("blur", function () { self.hide(); });
    this.svg.addEventListener("focus", function () { if (self.cursor < 0) { self.cursor = 0; } self.showIndex(self.cursor); });
    this.svg.addEventListener("keydown", function (e) { self.onKey(e); });
  }
  Chart.prototype.toSvg = function (e) {
    var ctm = this.svg.getScreenCTM();
    if (!ctm) { return null; }
    var pt = this.svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    return pt.matrixTransform(ctm.inverse());
  };
  Chart.prototype.initXY = function () {
    var spec = this.spec, P = this.P, X = mapper(spec.x), Y = mapper(spec.y), all = [], self = this;
    this.series = (spec.series || []).map(function (s, j) {
      var pts = [], xs = s.x || [], ys = s.y || [];
      for (var i = 0; i < Math.min(xs.length, ys.length); i++) {
        if (!okv(spec.x, xs[i]) || !okv(spec.y, ys[i])) { continue; }
        var px = X(xs[i]), py = Y(ys[i]);
        if (px < P.x0 - 0.5 || px > P.x1 + 0.5 || py < P.y0 - 0.5 || py > P.y1 + 0.5) { continue; }
        var p = { j: j, i: i, x: xs[i], y: ys[i], px: px, py: py };
        pts.push(p); all.push(p);
      }
      var sorted = pts.slice().sort(function (a, b) { return a.px - b.px || a.i - b.i; });
      return { label: seriesName(s, j), pts: sorted, raw: pts,
               min: sorted.length ? sorted[0].px : 0, max: sorted.length ? sorted[sorted.length - 1].px : 0 };
    });
    all.sort(function (a, b) { return a.px - b.px || a.j - b.j; });
    this.all = all;
    var seen = {}, stops = [];
    all.forEach(function (p) { var k = p.px.toFixed(1); if (!seen[k]) { seen[k] = 1; stops.push({ px: p.px, x: p.x }); } });
    this.stops = stops;
    this.dots = Array.prototype.slice.call(this.layer.querySelectorAll(".hdot"));
    this.xhair = this.layer.querySelector(".xhair");
    this.count = spec.kind === "scatter" ? all.length : stops.length;
    self.X = X;
  };
  Chart.prototype.initBar = function () {
    var spec = this.spec;
    this.count = (spec.cats || []).length;
    this.hband = this.layer.querySelector(".hband");
  };
  Chart.prototype.onPointer = function (e) {
    var p = this.toSvg(e);
    if (!p || !this.count) { return; }
    var P = this.P, spec = this.spec;
    if (p.x < P.x0 - 24 || p.x > P.x1 + 24 || p.y < P.y0 - 24 || p.y > P.y1 + 24) { this.hide(); return; }
    if (spec.kind === "bar") {
      var coord = spec.orient === "h" ? p.y : p.x;
      var i = Math.floor((coord - spec.b0) / spec.band);
      if (i < 0 || i >= this.count) { this.hide(); return; }
      this.cursor = i; this.showIndex(i);
    } else if (spec.kind === "scatter") {
      var best = -1, bd = 26 * 26;
      for (var k = 0; k < this.all.length; k++) {
        var q = this.all[k], d = (q.px - p.x) * (q.px - p.x) + (q.py - p.y) * (q.py - p.y);
        if (d < bd) { bd = d; best = k; }
      }
      if (best < 0) { this.hide(); return; }
      this.cursor = best; this.showIndex(best);
    } else {
      var n = nearestIdx(this.stops, p.x, "px");
      if (n < 0) { return; }
      this.cursor = n; this.showIndex(n);
    }
  };
  Chart.prototype.onKey = function (e) {
    if (!this.count) { return; }
    var step = e.shiftKey ? 10 : 1, c = this.cursor < 0 ? 0 : this.cursor;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") { c = Math.min(this.count - 1, c + step); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { c = Math.max(0, c - step); }
    else if (e.key === "Home") { c = 0; }
    else if (e.key === "End") { c = this.count - 1; }
    else if (e.key === "Escape") { this.hide(); return; }
    else { return; }
    e.preventDefault();
    this.cursor = c; this.showIndex(c);
  };
  Chart.prototype.hide = function () {
    if (this.layer) { this.layer.setAttribute("visibility", "hidden"); }
    if (this.tip) { this.tip.hidden = true; }
  };
  Chart.prototype.showIndex = function (idx) {
    if (!this.count) { return; }
    var spec = this.spec, rows = [], head = "", ax = 0, ay = 0, self = this;
    if (spec.kind === "bar") {
      var P = this.P, cat = (spec.cats || [])[idx], x0 = spec.b0 + idx * spec.band;
      if (spec.orient === "h") {
        this.setRect(P.x0, x0, P.x1 - P.x0, spec.band);
        ax = P.x1; ay = x0 + spec.band / 2;
      } else {
        this.setRect(x0, P.y0, spec.band, P.y1 - P.y0);
        ax = x0 + spec.band / 2; ay = P.y0 + (P.y1 - P.y0) / 3;
      }
      head = cat;
      (spec.series || []).forEach(function (s, j) { rows.push([seriesName(s, j), fmt(s.y ? s.y[idx] : null), j]); });
    } else if (spec.kind === "scatter") {
      var q = this.all[idx];
      this.xhair.setAttribute("visibility", "hidden");
      this.dots.forEach(function (d, j) { self.place(d, j === q.j ? q.px : null, q.py); });
      head = this.series[q.j].label;
      rows.push([spec.xlabel || "x", fmt(q.x) + xname(spec, q.x), -1]);
      rows.push([spec.ylabel || "y", fmt(q.y), -1]);
      ax = q.px; ay = q.py;
    } else {
      var stop = this.stops[idx], tx = stop.px, tol = 10;
      this.xhair.removeAttribute("visibility");
      this.xhair.setAttribute("x1", tx); this.xhair.setAttribute("x2", tx);
      head = (spec.xlabel || "x") + " = " + fmt(stop.x) + xname(spec, stop.x);
      var ys = [];
      this.series.forEach(function (s, j) {
        var dot = self.dots[j], p = null;
        if (s.pts.length && tx >= s.min - tol && tx <= s.max + tol) {
          if (spec.kind === "step") {
            var k = lowerBound(s.pts, tx + 0.51, "px") - 1;
            p = k >= 0 ? { px: tx, py: s.pts[k].py, y: s.pts[k].y } : null;
          } else {
            var n = nearestIdx(s.pts, tx, "px");
            p = n >= 0 ? s.pts[n] : null;
          }
        }
        if (dot) { self.place(dot, p ? p.px : null, p ? p.py : null); }
        if (p) { rows.push([s.label, fmt(p.y), j]); ys.push(p.py); }
      });
      ax = tx;
      ay = ys.length ? ys.reduce(function (a, b) { return a + b; }, 0) / ys.length : (this.P.y0 + this.P.y1) / 2;
    }
    this.layer.setAttribute("visibility", "visible");
    this.fillTip(head, rows);
    this.moveTip(ax, ay);
  };
  Chart.prototype.setRect = function (x, y, w, h) {
    var r = this.hband;
    if (!r) { return; }
    r.setAttribute("x", x); r.setAttribute("y", y); r.setAttribute("width", Math.max(0, w)); r.setAttribute("height", Math.max(0, h));
  };
  Chart.prototype.place = function (dot, x, y) {
    if (x === null) { dot.setAttribute("visibility", "hidden"); return; }
    dot.removeAttribute("visibility");
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
  };
  Chart.prototype.fillTip = function (head, rows) {
    var tip = this.tip;
    while (tip.firstChild) { tip.removeChild(tip.firstChild); }
    if (head !== "") { tip.appendChild(el("div", "tip-h", head)); }
    rows.forEach(function (r) {
      var row = el("div", "tip-r"), name = el("span");
      if (r[2] >= 0) {
        var sw = el("i", "sw");
        sw.style.background = "var(--s" + (r[2] % 8) + ")";
        name.appendChild(sw);
      }
      name.appendChild(doc.createTextNode(r[0]));
      row.appendChild(name);
      row.appendChild(el("b", "", r[1]));
      tip.appendChild(row);
    });
    tip.hidden = false;
  };
  Chart.prototype.moveTip = function (x, y) {
    var tip = this.tip, sr = this.svg.getBoundingClientRect(), br = this.body.getBoundingClientRect();
    var k = sr.width / (this.spec.w || sr.width || 1);
    var ox = sr.left - br.left, oy = sr.top - br.top;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = ox + x * k + 14;
    if (left + w > br.width) { left = ox + x * k - 14 - w; }
    left = Math.max(0, Math.min(left, br.width - w));
    var top = oy + y * k - h / 2;
    top = Math.max(0, Math.min(top, br.height - h));
    tip.style.left = left + "px"; tip.style.top = top + "px";
  };

  function wireTools(fig, spec) {
    var tableBox = fig.querySelector(".chart-table");
    Array.prototype.forEach.call(fig.querySelectorAll("[data-act]"), function (btn) {
      btn.addEventListener("click", function () {
        var t = tabulate(spec);
        if (btn.getAttribute("data-act") === "csv") {
          var lines = [t.cols.map(csvCell).join(",")].concat(t.rows.map(function (r) { return r.map(csvCell).join(","); }));
          download(String(spec.key || "chart").replace(/[^A-Za-z0-9._-]+/g, "_") + ".csv", lines.join("\r\n") + "\r\n");
          return;
        }
        if (!tableBox) { return; }
        var open = !tableBox.hidden;
        if (!open && !tableBox.firstChild) {
          var table = el("table", "data"), thead = el("thead"), tr = el("tr"), tbody = el("tbody");
          t.cols.forEach(function (c, ci) {
            var numeric = t.rows.length > 0 && t.rows.every(function (r) { return r[ci] === null || typeof r[ci] === "number"; });
            var th = el("th", numeric ? "num" : "", c); th.scope = "col"; tr.appendChild(th);
          });
          thead.appendChild(tr);
          t.rows.forEach(function (r) {
            var row = el("tr");
            r.forEach(function (v) {
              var td = el("td", typeof v === "number" ? "num" : "", typeof v === "number" || v === null ? fmt(v) : v);
              row.appendChild(td);
            });
            tbody.appendChild(row);
          });
          table.appendChild(thead); table.appendChild(tbody); tableBox.appendChild(table);
        }
        tableBox.hidden = open;
        btn.setAttribute("aria-expanded", open ? "false" : "true");
        btn.textContent = open ? "Show data" : "Hide data";
      });
    });
  }

  Array.prototype.forEach.call(doc.querySelectorAll("figure[data-chart]"), function (fig) {
    var spec = specs[fig.getAttribute("data-chart")];
    if (!spec) {
      Array.prototype.forEach.call(fig.querySelectorAll(".chart-tools"), function (t) { t.hidden = true; });
      return;
    }
    try { new Chart(fig, spec); } catch (e) { /* static SVG remains */ }
    try { wireTools(fig, spec); } catch (e) { /* tools unavailable */ }
  });
})();
