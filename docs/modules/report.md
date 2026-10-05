# HTML report (`malvalid.report.html`)

`report.html` is the human-readable face of a MalValid run: one self-contained file a researcher can
open offline, attach to a paper or a model-review ticket, or print. `report.json` next to it stays the
source of truth; the HTML is a pure function of it.

```python
from malvalid.report.html import render_html, write_html

html: str = render_html(report_dict)            # malvalid-report/1 dict (build contract §4.4)
path = write_html(report_dict, "run/report.html")  # atomic write, parents created
```

The runner calls `write_html` after writing `report.json`; `malvalid report render REPORT_JSON [-o OUT]`
re-renders an existing report (e.g. after upgrading MalValid).

## What the page shows (in this order)

1. **Verdict banner** – Ready / Conditional / Not ready / Blocked with icon + word + color, the
   0–100 production-readiness score on a 0–60–80–100 band scale (bands come from `verdict.bands`), the
   blocked-score cap marker and the uncapped score when capped, coverage (with the Ready minimum),
   the summary, blocking issues and other reasons.
2. **Hard-gate strip** – one chip per hard gate (links to its section), the CI exit code and its
   meaning, the `--fail-on` level, an M0 abort or model-load error, and modules disabled in config.
3. **Per-axis scorecard** – one row per module in run order: code, title, status, gate mode and
   outcome, axis score meter (the tick marks the pass line, 75 = exactly on threshold), weight, the key
   metric vs its threshold (the failing / weakest check), a *Screening* badge, and the skip reason or
   crash message. Rows become stacked cards on phones.
4. **One section per module** – description, finding, notes, screening caveat, checks table (rule,
   value, pass/fail, score and the floor→threshold→ideal anchors), metrics grid (gated metrics marked),
   charts, tables, other artifacts, collapsible details and parameters, seed and duration. Skipped
   modules show the reason plus an actionable "to enable it" hint; errored modules show the crash
   message and a collapsed traceback (partial charts are kept and labelled as not counted).
5. **Appendix** – model declarations, tree access, capabilities, artifacts with sha256, training
   manifest, feature schema, corpus identity + content hash + splits/roles, sandbox, environment and
   library versions (missing ones called out), run info and per-module seeds, the resolved gate config
   (collapsible) and the disclaimers.

Artifacts that no module claims are rendered in an "Other artifacts" section, so nothing in
`report.json` silently disappears.

## Charts

Charts are drawn server-side as static SVG by a small built-in renderer (no chart library, no network),
so they display without JavaScript and print as vectors. They follow
`ArtifactStore.add_chart(...)` exactly:

| field | support |
|---|---|
| `kind` | `line`, `step` (steps-post: `y[i]` holds on `[x[i], x[i+1])`), `scatter` (distinct marker shapes per series), `bar` (x = category labels; grouped for ≥2 series; turns horizontal when a label is longer than 10 chars or there are more than 40 categories, so short ordered labels such as M2's `YYYY-MM` windows stay left-to-right; single-series bars get value labels) |
| `xscale` / `yscale` | `linear` or `log` (non-positive values are dropped on log axes) |
| `xlim` / `ylim` | fixed domain; invalid limits fall back to the data range |
| `reference_lines` | `{"axis": "x" or "y", "value", "label"}` – dashed line, halo label placed to avoid other labels |
| `note` | shown as a caption; a note starting with `diagonal` also draws the y = x reference (the rest of the note, if any, is the caption); a note that is *only* `k=label` pairs separated by `;` (M2's decay chart: `1=2024-01; 2=2024-02; …`) is read as an x-value → label map: x ticks show the labels, hover readouts and the data table add them, and the caption becomes a one-line summary |
| series | ≤ 8 distinguishable colors in a fixed, color-blind-validated order; a legend for ≥ 2 series, the single series is named in the caption |

The inline script (pinned by a CSP hash) adds a crosshair + tooltip readout (nearest x for lines/steps,
nearest point for scatter, whole category for bars), keyboard access (focus a chart, ←/→, Shift for
×10, Home/End, Esc), a "Show data" table and "Download CSV" per chart, a light/dark toggle (remembered
per browser when storage is available) and expansion of collapsed sections when printing.

Guidance for module authors: keep `max_points` modest (the store downsamples to 400 by default), give
axes labels with units, use `reference_lines` for thresholds and operating points, prefer a table
artifact for anything that needs exact values, and never put anything sensitive in a chart or table
(use `save_private`).

## Safety and robustness

* **No network**: all CSS/JS is inline; a Content-Security-Policy (`default-src 'none'`, the one
  script pinned by sha256) blocks any request. Tests assert there is no external `src`/`href`/`@import`
  /`url(`.
* **Escaping**: Jinja2 autoescape for every report string; SVG built in Python goes through
  `markupsafe.escape`; the chart data block is JSON with `<`, `>`, `&`, U+2028/2029 escaped; the script
  only ever writes report text with `textContent`. CSV export neutralises spreadsheet formulas.
* **Missing / malformed keys**: every section is built defensively; a section that still fails is
  replaced by a visible "could not be rendered" notice instead of breaking the page, and a malformed
  chart shows a notice in its place.
* **Accessibility**: status is always icon + text + color; charts have `<title>`/`<desc>`, keyboard
  focus and a table view; light and dark palettes are separately chosen (`prefers-color-scheme` and a
  toggle); `forced-colors` and `print` styles are included.
* Output is deterministic for a given report and typically 0.1–0.35 MB (the real-data runs produced 220–322 KB, a one-module run about 90 KB; tests
  cap it at 1.5 MB).

## Tests and fixtures

`tests/unit/test_report_html.py` renders five fixture reports:
`tests/fixtures/sample_report.json` (the *blocked* kitchen sink: failed hard gate, errored module with
traceback and partial chart, skipped third-party module, screening badge, every §5 chart, long strings
and HTML/script-injection strings), plus `sample_report_{ready,conditional,not_ready,aborted}.json`.
They are generated from MalValid's real result types (`ModuleResult`, `GateCheck.evaluate`,
`ArtifactStore`, `compute_verdict`) by `tests/fixtures/make_sample_report.py`; a test fails if the
committed JSON drifts from the generator. Regenerate with:

```bash
.venv/bin/python tests/fixtures/make_sample_report.py
```

The fixture artifacts mirror what the finished modules emit (chart/table keys, labels, reference
lines, M2's window-number note, M0's artifact table, M1's operating-points table). In addition,
`test_renders_real_module_output` runs the modules present in the tree (M1, M2, M4, M6) on the toy
corpus from `malvalid.testing` and renders their real `ArtifactStore` output; a module that is missing
or crashes there is skipped, so that test only fails on renderer problems.
