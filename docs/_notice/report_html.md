# Third-party notice entries — agent report_html (self-contained HTML report)

No third-party source code was copied, ported or vendored into:

* `src/malvalid/report/html.py`
* `src/malvalid/report/templates/report.html.j2`, `src/malvalid/report/templates/_macros.html.j2`
* `src/malvalid/report/static/report.css`, `src/malvalid/report/static/report.js`
* `tests/unit/test_report_html.py`, `tests/fixtures/make_sample_report.py`, `tests/fixtures/sample_report*.json`

All of it is original code written for MalValid under Apache-2.0. In particular the charts are drawn
by a small hand-written SVG renderer (`_ChartBuilder` in `html.py`) plus an inline hover script; **no
chart library (uPlot or otherwise) is vendored**, so `src/malvalid/report/static/` contains only our
own `report.css` and `report.js`. The status/series color values are plain hex choices (no licensed
asset). The icons are hand-drawn inline SVG symbols.

Already-declared core dependencies that are **imported** at runtime (not copied):

| name | license | URL | used in |
|---|---|---|---|
| Jinja2 | BSD-3-Clause | https://github.com/pallets/jinja | `html.py` (templating, autoescape) |
| MarkupSafe (Jinja2 dependency) | BSD-3-Clause | https://github.com/pallets/markupsafe | `html.py` (`escape`, `Markup` for SVG built in Python) |
| PyYAML | MIT | https://github.com/yaml/pyyaml | `html.py` (renders the resolved gate config in the appendix) |
| NumPy / scikit-learn | BSD-3-Clause | https://numpy.org / https://scikit-learn.org | `tests/fixtures/make_sample_report.py` only (synthetic fixture curves) |

The fixtures contain only synthetic numbers, fake paths and fake hashes — no malware, no real sample
data.
