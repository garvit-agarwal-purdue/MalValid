"""``malvalid`` command-line interface (typer).

malvalid is for malware-detection researchers: submit your trained detector through a small
adapter and get a production-readiness verdict (READY / CONDITIONAL / NOT_READY / BLOCKED) backed
by a 0-100 score and per-axis evidence.

Exit codes: ``run`` follows :func:`malvalid.verdict.exit_code_for` (0 passed, 1 gate failed /
blocked / at-or-below ``--fail-on``, 2 a module errored or the model could not be loaded). Usage,
config and adapter errors exit 2 with a one-line message (traceback only with ``-v``).
"""

import contextlib
import datetime as dt
import json
import logging
import re
import shlex
import traceback
from pathlib import Path
from typing import Any, Iterator, List, NoReturn, Optional

import click
import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from malvalid import __version__

log = logging.getLogger("malvalid.cli")

_HELP = (
    "MalValid — a pre-deployment reliability & security gate for malware-detection models.\n\n"
    "Submit your trained model file (or, for custom models, a small adapter) and get a production-readiness verdict "
    "(READY / CONDITIONAL / NOT_READY / BLOCKED) backed by a 0-100 score, plus per-axis evidence "
    "(file safety, detection/false-positive rate, drift, privacy leakage, backdoor screening, "
    "extraction risk, spurious-feature reliance) for a promotion decision under your gate policy. "
    "It is evidence, not a certificate.\n\n"
    "Typical flow: `malvalid inspect-model model.txt` → `malvalid run --model model.txt --threshold 0.83`. "
    "Custom models: `malvalid validate-adapter --adapter my_adapter.py` → "
    "`malvalid run --adapter my_adapter.py --config gate.yaml`."
)

app = typer.Typer(
    name="malvalid",
    help=_HELP,
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
corpus_app = typer.Typer(
    help="Manage canonical evaluation corpora (feature vectors + hashes + labels + timestamps; never samples).",
    no_args_is_help=True,
)
report_app = typer.Typer(help="Work with report.json files from earlier runs.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(report_app, name="report")

_STATE: dict[str, Any] = {"verbose": False}
_LOG_HANDLER: Optional[logging.Handler] = None

_VERDICT_STYLE = {
    "ready": "bold green",
    "conditional": "bold yellow",
    "not_ready": "bold red",
    "blocked": "bold white on red",
}
_STATUS_STYLE = {"pass": "green", "warn": "yellow", "fail": "bold red", "skipped": "dim", "error": "magenta"}
_OUTCOME_MARK = {"passed": "✓", "failed": "✗", "not_evaluated": "–"}


# --------------------------------------------------------------------------------------------------
# Plumbing: consoles, logging, error handling
# --------------------------------------------------------------------------------------------------


def _out() -> Console:
    return Console(highlight=False, soft_wrap=False)


def _err() -> Console:
    return Console(stderr=True, highlight=False)


def _setup_logging(verbose: bool, level: int = logging.WARNING) -> None:
    """Route ``malvalid.*`` log records to stderr (idempotent across invocations)."""
    global _LOG_HANDLER
    from rich.logging import RichHandler

    lg = logging.getLogger("malvalid")
    if _LOG_HANDLER is not None:
        lg.removeHandler(_LOG_HANDLER)
    handler = RichHandler(
        console=_err(), show_time=False, show_path=False, markup=False, rich_tracebacks=False,
        show_level=True,
    )
    eff = logging.DEBUG if verbose else level
    handler.setLevel(eff)
    lg.addHandler(handler)
    lg.setLevel(eff)
    _LOG_HANDLER = handler


def _verbose(flag: bool = False) -> bool:
    return bool(flag or _STATE.get("verbose"))


# Same rule as the web UI's run ids (malvalid.web.settings): safe as a directory name.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _fail(message: str, *, verbose: bool = False, code: int = 2) -> NoReturn:
    msg = " ".join(str(message).split()) or "unknown error"
    if verbose:
        _err().print(Text(traceback.format_exc().rstrip(), style="dim"))
    # soft_wrap: the message stays on one line whatever the terminal width (easy to grep in CI logs).
    _err().print(Text.assemble(("error: ", "bold red"), msg), soft_wrap=True)
    raise typer.Exit(code)


@contextlib.contextmanager
def _guard(verbose: bool = False) -> Iterator[None]:
    """Turn expected failures into a one-line message + exit 2."""
    from malvalid.core import MalValidError

    try:
        yield
    except (typer.Exit, click.exceptions.ClickException, click.exceptions.Abort):
        raise
    except KeyboardInterrupt:
        _err().print(Text("interrupted", style="bold red"))
        raise typer.Exit(130)
    except MalValidError as e:
        _fail(str(e), verbose=verbose)
    except KeyError as e:
        _fail(str(e.args[0]) if e.args else repr(e), verbose=verbose)
    except (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError) as e:
        _fail(f"{e.strerror or type(e).__name__}: {e.filename or e}", verbose=verbose)
    except ValueError as e:
        _fail(str(e), verbose=verbose)
    except Exception as e:
        hint = "" if verbose else " (re-run with -v for the traceback)"
        _fail(f"internal error: {type(e).__name__}: {e}{hint}", verbose=verbose)


def _version_cb(value: bool) -> None:
    if value:
        typer.echo(f"malvalid {__version__}")
        raise typer.Exit(0)


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_cb, is_eager=True, help="Print the MalValid version and exit."
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging and full tracebacks on errors."),
) -> None:
    _STATE["verbose"] = verbose


def _split_ids(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        out.extend(x.strip() for x in v.split(",") if x.strip())
    return out


def _config_overrides(
    *,
    seed: Optional[int] = None,
    no_sandbox: bool = False,
    allow_pickle: bool = False,
    corpus: Optional[str] = None,
    verify_corpus: bool = False,
    calibration_period: Optional[str] = None,
    allow_reduced_isolation: bool = False,
) -> dict:
    ov: dict = {}
    rt: dict = {}
    if calibration_period is not None:
        rt["calibration_period"] = calibration_period
    if seed is not None:
        rt["seed"] = int(seed)
    if verify_corpus:
        rt["corpus_verification"] = "full"
    if no_sandbox:
        rt["sandbox"] = False
    if allow_pickle:
        rt["allow_pickle"] = True
    if allow_reduced_isolation:
        rt["allow_reduced_isolation"] = True
    if rt:
        ov["runtime"] = rt
    if corpus:
        ov["corpus"] = corpus
    return ov


def _resolve_corpus_dir(d: Path, corpus_name: str) -> Path:
    """``--corpus-dir`` may point at the corpus itself or at a root that contains ``<name>/``
    (the same rule the corpus providers apply to ``corpus_dir`` in gate.yaml)."""
    from malvalid.corpora.base import resolve_corpus_dir

    return resolve_corpus_dir(d.expanduser().resolve(), corpus_name)


def _load_cfg(config: Optional[Path], overrides: dict, corpus_dir: Optional[Path] = None) -> Any:
    from malvalid.config import load_config

    cfg = load_config(config, overrides=overrides or None)
    if corpus_dir is not None:
        cfg = cfg.model_copy(update={"corpus_dir": str(_resolve_corpus_dir(corpus_dir, cfg.corpus))})
    return cfg


def _corpus_for_model(cfg: Any, feature_version: str) -> Optional[str]:
    """Model-file mode without --corpus: the default canonical corpus of the model's feature version
    when the policy's corpus is in another feature space (None = keep the policy's corpus)."""
    from malvalid import registry
    from malvalid.inspect_model import default_corpus_for

    try:
        current_fv = registry.get_corpus_provider(cfg.corpus).info().get("feature_version")
    except Exception:  # noqa: BLE001 - unknown corpus: let the runner report it
        current_fv = None
    if current_fv == feature_version:
        return None
    target = default_corpus_for(feature_version)
    return target if target and target != cfg.corpus else None


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _command_line(argv: List[str]) -> str:
    return shlex.join(["malvalid", *argv])


# --------------------------------------------------------------------------------------------------
# Terminal summary of a report
# --------------------------------------------------------------------------------------------------


def _key_evidence(m: dict) -> str:
    status = m.get("status")
    if status == "skipped":
        return f"not run — {m.get('skip_reason') or 'requirements not met'}"
    if status == "error":
        return m.get("finding") or "module errored"
    checks = m.get("checks") or []
    if checks:
        parts = []
        for c in checks[:2]:
            mark = {True: "✓", False: "✗", None: "?"}.get(c.get("passed"))
            parts.append(f"{mark} {c.get('metric') or c.get('name')} {_fmt(c.get('value'))} "
                         f"(gate {c.get('op')} {_fmt(c.get('threshold'))})")
        return "; ".join(parts)
    return (m.get("finding") or "").strip()


def _bullets(con: Console, title: str, items: List[str], style: str) -> None:
    """A titled bullet list with hanging indents (wrapped lines align under the text)."""
    if not items:
        return
    con.print(Text(title, style=style))
    grid = Table.grid(padding=(0, 1))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    for it in items:
        grid.add_row("  •", Text(str(it)))
    con.print(grid)


def render_summary(report: dict, console: Optional[Console] = None) -> None:
    """Print the verdict banner, per-axis scorecard, blockers/reasons and report paths."""
    con = console or _out()
    v = report.get("verdict") or {}
    gate = report.get("gate") or {}
    run = report.get("run") or {}
    verdict = str(v.get("verdict", "?"))
    style = _VERDICT_STYLE.get(verdict, "bold")
    score = v.get("score")
    score_txt = "n/a" if score is None else f"{float(score):.1f}/100"
    head = Text.assemble((f" {verdict.upper().replace('_', ' ')} ", style), "  ", (v.get("label") or "", "bold"))
    lines = [head, Text.assemble(("Production-readiness score: ", "bold"), score_txt)]
    if v.get("capped"):
        lines.append(Text(f"  capped at {_fmt((v.get('bands') or {}).get('blocked_score_cap'))} "
                          f"(uncapped {_fmt(v.get('raw_score'))}) because the run is blocked", style="dim"))
    cov = v.get("coverage")
    lines.append(Text.assemble(("Coverage: ", "bold"),
                               "n/a" if cov is None else f"{100 * float(cov):.0f}% of the weighted test battery evaluated"))
    border = style.split()[-1] if verdict != "blocked" else "red"
    con.print(Panel(Group(*lines), title="MalValid production-readiness verdict", border_style=border, expand=False))

    modules = report.get("modules") or []
    axes = {a.get("module_id"): a for a in v.get("axes") or []}
    t = Table(
        title="Per-axis scorecard",
        caption="Gate: ✓ passed  ✗ failed  – not evaluated.  * = scored but not counted (weight 0).  "
        "— = precondition, not scored.",
        caption_justify="left",
        expand=False,
    )
    t.add_column("Code", no_wrap=True)
    t.add_column("Module", overflow="fold", min_width=12, ratio=2)
    t.add_column("Status", no_wrap=True)
    t.add_column("Gate", no_wrap=True)
    t.add_column("Score", justify="right", no_wrap=True)
    t.add_column("Key evidence", overflow="fold", min_width=20, ratio=4)
    for m in modules:
        st = str(m.get("status", ""))
        ax = axes.get(m.get("module_id"))
        if ax is None:
            sc = "—"  # unscored precondition (M0)
        elif ax.get("score") is None:
            sc = "n/a"
        else:
            sc = f"{float(ax['score']):.1f}" + ("" if ax.get("counted") else "*")
        title = str(m.get("title", ""))
        if m.get("screening"):
            title += " (screening)"
        mark = _OUTCOME_MARK.get(str(m.get("gate_outcome", "")), "?")
        t.add_row(
            str(m.get("code", "")),
            title,
            Text(st.upper(), style=_STATUS_STYLE.get(st, "")),
            f"{m.get('gate', '')} {mark}",
            sc,
            _key_evidence(m),
        )
    con.print(t)

    blockers = list(v.get("blockers") or [])
    reasons = [r for r in v.get("reasons") or [] if r not in blockers]
    _bullets(con, "Blockers", blockers, "bold red")
    _bullets(con, "Why not READY" if verdict != "ready" else "Notes", reasons, "bold")
    _bullets(con, "Warnings", report.get("warnings") or [], "bold yellow")
    disabled = list(gate.get("disabled") or [])
    if disabled:
        con.print(Text.assemble(("Not run (disabled in the config): ", "bold"), ", ".join(disabled),
                                ("  — enable with modules.<id>.enabled: true", "dim")))
    deselected = list(gate.get("deselected") or [])
    if deselected:
        con.print(Text.assemble(("Partial run — not run (deselected with --only/--skip): ", "bold yellow"),
                                ", ".join(deselected),
                                ("  — they count as skipped: coverage is lower and a deselected hard gate blocks", "dim")))
    con.print(Text("Reports", style="bold"))
    # soft_wrap keeps each path on one logical line so it can be copied from the terminal.
    con.print(Text(f"  report.json  {run.get('report_json') or '(not written)'}"), soft_wrap=True)
    con.print(Text(f"  report.html  {run.get('report_html') or '(not written)'}"), soft_wrap=True)
    if run.get("run_log"):
        con.print(Text(f"  run.log      {run['run_log']}"), soft_wrap=True)
    code = gate.get("exit_code")
    if code is not None:
        con.print(Text.assemble(("Exit code ", "bold"), f"{code}: {gate.get('exit_meaning', '')}"))


# --------------------------------------------------------------------------------------------------
# malvalid run
# --------------------------------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    adapter: Optional[Path] = typer.Option(
        None, "--adapter", "-a",
        help="Advanced: your own adapter .py file (a class that wraps your detector and declares its "
        "feature_version, model_kind, operating_threshold and training manifest). Not needed when you "
        "submit just the model file with --model.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Gate policy YAML (create one with `malvalid init-config`). Default: the packaged policy."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Run directory for report.json / report.html / run.log. Default: ./malvalid-runs/<UTC time>."
    ),
    allow_pickle: bool = typer.Option(
        False, "--allow-pickle",
        help="Accept pickle-based model artifacts. Pickles can execute code when loaded; prefer LightGBM .txt, XGBoost .json/.ubj or ONNX.",
    ),
    model: Optional[List[Path]] = typer.Option(
        None, "--model", "-m",
        help="Your model file (LightGBM .txt/.model, XGBoost .json/.ubj, ONNX; pickles need --allow-pickle). "
        "Without --adapter, malvalid detects the model kind and feature version from the file. With --adapter, "
        "overrides the adapter's model_path (repeatable).",
    ),
    feature_version: Optional[str] = typer.Option(
        None, "--feature-version", help="Model-file mode: feature schema (ember_v2 | ember_v3); default: detected from the model."
    ),
    model_kind: Optional[str] = typer.Option(
        None, "--model-kind", help="Model-file mode: lightgbm | xgboost | sklearn_gbdt | onnx; default: detected from the file."
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Model-file mode: the operating threshold (score cut-off in [0, 1]) you ship."
    ),
    calibrate_fpr: Optional[float] = typer.Option(
        None, "--calibrate-fpr",
        help="Model-file mode: instead of --threshold, calibrate the threshold to this false-positive rate "
        "(e.g. 0.005 = 0.5%) on a held-out slice of the corpus that no test uses. Default when neither is given: 0.005.",
    ),
    calibration_period: Optional[str] = typer.Option(
        None, "--calibration-period",
        help="Model-file mode with a calibrated threshold: which benign rows it is fit on. earliest (default): the "
        "earliest 10% of the corpus's evaluation benign rows by date, so it is tested only on later data, as when "
        "deployed; uniform: a 10% hash sample across the whole period (optimistic under drift). "
        "Overrides runtime.calibration_period.",
    ),
    training_cutoff: Optional[str] = typer.Option(
        None, "--training-cutoff", help="Model-file mode: date of the newest training sample (YYYY-MM or YYYY-MM-DD); unlocks M2 drift."
    ),
    training_hashes: Optional[Path] = typer.Option(
        None, "--training-hashes",
        help="Model-file mode: file with one training-sample sha256 per line (or a CSV/TSV with a sha256 column); unlocks M4.",
    ),
    class_name: Optional[str] = typer.Option(
        None, "--class", help="Adapter class to use when the file defines more than one detector class."
    ),
    only: Optional[List[str]] = typer.Option(
        None, "--only", help="Run only these modules (ids or codes, comma-separated, e.g. M1,drift). M0 always runs."
    ),
    skip: Optional[List[str]] = typer.Option(None, "--skip", help="Do not run these modules (ids or codes, comma-separated)."),
    corpus: Optional[str] = typer.Option(None, "--corpus", help="Canonical corpus to evaluate on (see `malvalid corpus list`)."),
    corpus_dir: Optional[Path] = typer.Option(
        None, "--corpus-dir", help="Directory of the canonical corpus (or a root containing <corpus>/)."
    ),
    verify_corpus: bool = typer.Option(
        False, "--verify-corpus",
        help="Re-hash every canonical-corpus file instead of trusting the verification cache "
        "(runtime.corpus_verification: full; slower, for CI).",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Base random seed (overrides runtime.seed); recorded in the report."),
    no_sandbox: bool = typer.Option(
        False, "--no-sandbox", help="Load the model in-process without isolation. Debugging trusted models only."
    ),
    allow_reduced_isolation: bool = typer.Option(
        False, "--allow-reduced-isolation",
        help="If this machine has no OS sandbox (Windows, macOS, Linux without bubblewrap/user namespaces), run "
        "the model in a plain worker process instead of failing: pickle refusal and resource limits, but NO "
        "network or file-system isolation. The report records isolation: process_only. Never downgrades a "
        "machine where the sandbox works.",
    ),
    no_html: bool = typer.Option(False, "--no-html", help="Write report.json only."),
    fail_on: str = typer.Option(
        "blocked", "--fail-on",
        help="Exit 1 when the verdict is at or below this level: blocked | not_ready | conditional. "
        "(BLOCKED always exits non-zero.)",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", hidden=True,
        help="Run id to record in the report (used by `malvalid serve`; default: a new timestamped id).",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", hidden=True,
        help="Report title, overriding report.title in the gate policy (used by `malvalid serve`).",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging and full tracebacks."),
) -> None:
    """Evaluate your detector and print its production-readiness verdict and 0-100 score.

    Scans the model artifact (M0) before anything is loaded, loads the model in a no-network sandbox, runs the enabled test modules on the canonical corpus, and writes report.json, report.html and run.log.

    Exit code: 0 passed; 1 blocked or verdict at/below --fail-on; 2 a module errored, the model could not be loaded, or a usage/config/adapter error.
    """
    verbose = _verbose(verbose)
    _setup_logging(verbose, logging.INFO)
    with _guard(verbose):
        from malvalid.runner import RunOptions, run_gate, validate_fail_on

        validate_fail_on(fail_on)
        if run_id is not None and not _RUN_ID_RE.match(run_id):
            _fail("--run-id must be 1-128 characters [A-Za-z0-9._-], starting with a letter or digit")
        spec_flags = {"--feature-version": feature_version, "--model-kind": model_kind, "--threshold": threshold,
                      "--calibrate-fpr": calibrate_fpr, "--calibration-period": calibration_period,
                      "--training-cutoff": training_cutoff, "--training-hashes": training_hashes}
        if calibration_period is not None:
            from malvalid.submission import CALIBRATION_PERIODS

            calibration_period = calibration_period.strip().lower()
            spec_flags["--calibration-period"] = calibration_period
            if calibration_period not in CALIBRATION_PERIODS:
                _fail(f"--calibration-period must be one of {' | '.join(CALIBRATION_PERIODS)} "
                      f"(got {calibration_period!r})")
            if threshold is not None:
                _fail("--calibration-period only applies to a calibrated threshold (--calibrate-fpr), not --threshold")
        model_only = adapter is None
        if model_only:
            if not model:
                _fail("give your model file with --model PATH (or an adapter with --adapter for custom models)")
            if len(model) != 1:
                _fail("submit exactly one --model file without --adapter (a custom adapter can load several files)")
            if class_name:
                _fail("--class only applies to --adapter submissions")
        else:
            used = [k for k, v in spec_flags.items() if v is not None]
            if used:
                _fail(f"{', '.join(used)} only apply when you submit a model file without --adapter "
                      "(an adapter declares these itself)")
        # the report's reproduction command (no --run-id)
        argv = ["run", "--adapter", str(adapter)] if adapter is not None else ["run"]
        for flag, val in (("--config", config), ("--out", out), ("--class", class_name), ("--corpus", corpus),
                          ("--corpus-dir", corpus_dir), ("--seed", seed)):
            if val is not None:
                argv += [flag, str(val)]
        for p in model or []:
            argv += ["--model", str(p)]
        for flag, val in spec_flags.items():
            if val is not None:
                argv += [flag, str(val)]
        for flag, vals in (("--only", only), ("--skip", skip)):
            for x in vals or []:
                argv += [flag, x]
        for flag, on in (("--allow-pickle", allow_pickle), ("--no-sandbox", no_sandbox), ("--no-html", no_html),
                         ("--verify-corpus", verify_corpus), ("--allow-reduced-isolation", allow_reduced_isolation)):
            if on:
                argv.append(flag)
        if fail_on != "blocked":
            argv += ["--fail-on", fail_on]

        overrides = _config_overrides(seed=seed, no_sandbox=no_sandbox, allow_pickle=allow_pickle, corpus=corpus,
                                      verify_corpus=verify_corpus, calibration_period=calibration_period,
                                      allow_reduced_isolation=allow_reduced_isolation)
        if title is not None and title.strip():
            overrides["report"] = {"title": " ".join(title.split())}
        cfg = _load_cfg(config, overrides, corpus_dir)
        out_dir = out or Path("malvalid-runs") / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        only_ids = _split_ids(only)
        notes: List[str] = []
        model_paths = list(model or [])
        if model_only:
            from malvalid.submission import prepare_model_submission

            prepared = prepare_model_submission(
                model_paths[0], Path(out_dir).expanduser().resolve() / "submission",
                model_kind=model_kind, feature_version=feature_version, threshold=threshold,
                calibrate_fpr=calibrate_fpr, training_cutoff=training_cutoff, training_hashes=training_hashes,
                allow_pickle=bool(allow_pickle or cfg.runtime.allow_pickle),
            )
            adapter = prepared.spec_path
            model_paths = []
            notes = list(prepared.notes)
            target = None if corpus is not None else _corpus_for_model(cfg, prepared.spec.feature_version)
            if target is not None:
                notes.append(f"the model uses {prepared.spec.feature_version} features, so it is evaluated on the "
                             f"{target} corpus (the gate policy names {cfg.corpus}; pass --corpus to choose another)")
                overrides["corpus"] = target
                cfg = _load_cfg(config, overrides, corpus_dir)
            _err().print(Text(
                f"Model file {Path(model_paths[0] if model_paths else model[0]).name}: "
                f"{prepared.spec.model_kind}, {prepared.spec.feature_version}"
                + (f", threshold {prepared.spec.operating_threshold:g}" if prepared.spec.operating_threshold is not None
                   else f", threshold auto-calibrated to {prepared.spec.calibrate_fpr:.2%} FPR")
                + f", corpus {cfg.corpus}", style="dim"))
            for n in notes:
                _err().print(Text(f"note: {n}", style="yellow"))
        assert adapter is not None
        opts = RunOptions(
            adapter=adapter,
            config=cfg,
            out_dir=out_dir,
            allow_pickle=allow_pickle,
            class_name=class_name,
            notes=notes,
            model_paths=model_paths,
            only=only_ids or None,
            skip=_split_ids(skip),
            fail_on=fail_on,
            write_html=not no_html,
            command=_command_line(argv),
            progress_path=Path(out_dir) / "progress.json",
            run_id=run_id,
        )
        outcome = run_gate(opts)
    render_summary(outcome.report)
    raise typer.Exit(outcome.exit_code)


# --------------------------------------------------------------------------------------------------
# list-modules / init-config / validate-adapter
# --------------------------------------------------------------------------------------------------


def _needs(info: dict) -> str:
    parts = [" + ".join(info.get("requires") or [])] if info.get("requires") else []
    if info.get("requires_any"):
        parts.append("(" + " or ".join(info["requires_any"]) + ")")
    return " + ".join(p for p in parts if p) or "—"


@app.command("list-modules")
def list_modules_cmd(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List the test modules that make up the production-readiness battery."""
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        from malvalid import registry
        from malvalid.config import load_config

        defaults = load_config()
        mods = registry.modules()
        rows = []
        for mid, cls in mods.items():
            info = cls.info()
            mc = defaults.modules.get(mid)
            info["enabled_by_default"] = bool(mc is not None and mc.enabled)
            info["weight"] = float(defaults.verdict.weights.get(mid, defaults.verdict.default_weight))
            rows.append(info)
        unavailable = registry.unavailable("modules")
    if as_json:
        typer.echo(json.dumps({"modules": rows, "unavailable": unavailable}, indent=2, default=str))
        return
    con = _out()
    t = Table(title="MalValid test modules (scorecard order)")
    t.add_column("Code", no_wrap=True)
    t.add_column("ID", overflow="fold")
    t.add_column("Title", overflow="fold")
    t.add_column("Gate", no_wrap=True)
    t.add_column("On", no_wrap=True)
    t.add_column("Needs", overflow="fold")
    for r in rows:
        title = r["title"] + (" (screening)" if r.get("screening") else "")
        t.add_row(r["code"], r["id"], title, r["default_gate"], "yes" if r["enabled_by_default"] else "no", _needs(r))
    con.print(t)
    for r in rows:
        if r.get("description"):
            con.print(Text.assemble((f"{r['code']} {r['id']}: ", "bold"), r["description"]))
    _bullets(con, "Unavailable (failed to import):", [f"{k}: {why}" for k, why in unavailable.items()], "bold yellow")


@app.command("init-config")
def init_config_cmd(
    path: Path = typer.Argument(Path("gate.yaml"), help="Where to write the gate policy."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
) -> None:
    """Write the default gate policy (thresholds, hard/warn gates, verdict weights) to edit."""
    verbose = _verbose()
    with _guard(verbose):
        from malvalid.config import default_config_text

        if path.exists() and not force:
            _fail(f"{path} already exists; pass --force to overwrite it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default_config_text())
    con = _out()
    con.print(Text.assemble(("Wrote the default gate policy to ", ""), (str(path), "bold")))
    con.print(Text(
        "Thresholds and weights are your organization's policy — review them, then check your adapter with "
        f"`malvalid validate-adapter --adapter my_adapter.py --config {path}` and evaluate it with "
        f"`malvalid run --adapter my_adapter.py --config {path}`."
    ))


@app.command("validate-adapter")
def validate_adapter_cmd(
    adapter: Path = typer.Option(..., "--adapter", "-a", help="Your adapter .py file."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Gate policy YAML (for sandbox settings and the corpus)."),
    allow_pickle: bool = typer.Option(False, "--allow-pickle", help="Accept pickle-based model artifacts (they can execute code on load)."),
    model: Optional[List[Path]] = typer.Option(None, "--model", "-m", help="Model artifact(s), overriding the adapter's model_path."),
    class_name: Optional[str] = typer.Option(None, "--class", help="Adapter class to use if the file defines several."),
    allow_reduced_isolation: bool = typer.Option(
        False, "--allow-reduced-isolation",
        help="If this machine has no OS sandbox (Windows, macOS, Linux without bubblewrap/user namespaces), run "
        "the model in a plain worker process instead of failing: pickle refusal and resource limits, but NO "
        "network or file-system isolation. The report records isolation: process_only. Never downgrades a "
        "machine where the sandbox works.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging and full tracebacks."),
) -> None:
    """Check that your adapter meets the submission contract before a full run.

    Inspects declarations, scans the artifact, loads the model in the sandbox and probes predict_proba / predict / featurize / tree access.

    Exit code: 0 every check passed; 1 a check failed; 2 validation could not run.
    """
    verbose = _verbose(verbose)
    _setup_logging(verbose)
    with _guard(verbose):
        try:
            from malvalid.adapters.validate import validate_adapter
        except ImportError as e:
            _fail(f"adapter validation is not available in this installation ({e})", verbose=verbose)
        if not adapter.exists():
            _fail(f"adapter not found: {adapter}")
        cfg = _load_cfg(config, _config_overrides(allow_pickle=allow_pickle,
                                                  allow_reduced_isolation=allow_reduced_isolation))
        rep = validate_adapter(
            adapter.expanduser().resolve(), cfg, allow_pickle=allow_pickle, class_name=class_name,
            model_paths=[Path(p).expanduser().resolve() for p in (model or [])],
        )
        if as_json:
            from malvalid.core import to_jsonable

            typer.echo(json.dumps(to_jsonable(rep.to_dict()), indent=2, allow_nan=False))
        else:
            typer.echo(rep.render_text())
        ok = bool(getattr(rep, "ok", False))
    raise typer.Exit(0 if ok else 1)


@app.command("inspect-model")
def inspect_model_cmd(
    path: Path = typer.Argument(..., help="The model file to inspect."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show what malvalid detects from a model file: kind, number of features, feature version, corpus.

    Reads the file as data only (LightGBM text header, XGBoost JSON/UBJSON, ONNX graph); nothing is executed and pickles are never unpickled.

    Exit code: 0 everything needed for `malvalid run --model FILE` was detected; 1 something must be chosen by hand (see the output); 2 the file could not be read.
    """
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        from malvalid.inspect_model import inspect_model

        if not path.expanduser().is_file():
            _fail(f"model file not found: {path}")
        info = inspect_model(path.expanduser())
    d = info.to_dict()
    if as_json:
        typer.echo(json.dumps(d, indent=2, default=str))
        raise typer.Exit(0 if info.ok else 1)
    con = _out()
    t = Table(title=f"malvalid inspect-model: {info.file_name}", show_header=False)
    t.add_column("", style="bold", no_wrap=True)
    t.add_column("", overflow="fold")
    t.add_row("format", info.format)
    t.add_row("model kind", info.model_kind or "not detected (choose with --model-kind)")
    t.add_row("features", str(info.n_features) if info.n_features is not None else "unknown")
    t.add_row("feature version", info.feature_version or ("no match" if info.n_features is not None else "not detected (choose with --feature-version)"))
    t.add_row("default corpus", info.default_corpus or "n/a")
    t.add_row("pickle", "yes (needs --allow-pickle)" if info.is_pickle else "no")
    for k, v in info.details.items():
        if v is not None:
            t.add_row(k, str(v))
    con.print(t)
    _bullets(con, "Notes:", info.notes, "yellow")
    _bullets(con, "Problems:", info.errors, "bold red")
    if info.ok:
        con.print(Text(f"Next: malvalid run --model {path} --threshold <your cut-off>   "
                       "(or --calibrate-fpr 0.005 to calibrate it on held-out data)"))
    raise typer.Exit(0 if info.ok else 1)


# --------------------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------------------


def _provider(name: str) -> Any:
    from malvalid import registry

    try:
        return registry.get_corpus_provider(name)
    except KeyError as e:
        raise KeyError(
            (e.args[0] if e.args else f"unknown corpus {name!r}") + " — see `malvalid corpus list`"
        ) from e


def _corpus_cfg(name: str, corpus_dir: Optional[Path]) -> Any:
    from malvalid.config import GateConfig

    cd = str(_resolve_corpus_dir(corpus_dir, name)) if corpus_dir is not None else None
    return GateConfig(corpus=name, corpus_dir=cd)


@corpus_app.command("list")
def corpus_list_cmd(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """List registered canonical corpora and whether each is present locally."""
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        from malvalid import registry

        rows = []
        for name, cls in registry.corpora().items():
            try:
                prov = cls()
                info = prov.info()
                loc = prov.locate(None)
                avail = bool(prov.is_available(None))
                info.update(location=str(loc), available=avail)
            except Exception as e:  # a broken provider should not hide the others
                info = {"name": name, "error": f"{type(e).__name__}: {e}", "available": False}
            rows.append(info)
        unavailable = registry.unavailable("corpora")
    if as_json:
        typer.echo(json.dumps({"corpora": rows, "unavailable": unavailable}, indent=2, default=str))
        return
    con = _out()
    t = Table(title="Canonical corpora")
    for col in ("Name", "Feature version", "Version", "Present"):
        t.add_column(col, no_wrap=col != "Name", overflow="fold")
    t.add_column("Location", overflow="fold", min_width=20)
    for r in rows:
        name = str(r.get("name")) + (" (synthetic)" if r.get("synthetic") else "")
        t.add_row(name, str(r.get("feature_version", "?")), str(r.get("version", "?")),
                  Text("yes", style="green") if r.get("available") else Text("no", style="dim"),
                  str(r.get("location") or r.get("error") or ""))
    con.print(t)
    _bullets(con, "Unavailable (failed to import):", [f"{k}: {why}" for k, why in unavailable.items()], "bold yellow")


@corpus_app.command("info")
def corpus_info_cmd(
    name: str = typer.Argument(..., help="Corpus name (see `malvalid corpus list`)."),
    corpus_dir: Optional[Path] = typer.Option(None, "--corpus-dir", help="Corpus directory override."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show a corpus's identity, location, content hash and split/label counts."""
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        prov = _provider(name)
        cfg = _corpus_cfg(name, corpus_dir)
        info: dict = dict(prov.info())
        loc = prov.locate(cfg)
        info["location"] = str(loc)
        info["available"] = bool(prov.is_available(cfg))
        if info["available"]:
            info["summary"] = prov.load(cfg, verify=False).summary()
        else:
            info["hint"] = prov.unavailable_hint(loc)
    if as_json:
        from malvalid.core import to_jsonable

        typer.echo(json.dumps(to_jsonable(info), indent=2))
        return
    con = _out()
    con.print(Text.assemble(("Corpus ", "bold"), (str(info.get("name")), "bold"), f"  ({info.get('feature_version')}, version {info.get('version')})"))
    if info.get("description"):
        con.print(Text(str(info["description"])))
    con.print(Text(f"Location: {info['location']}"))
    con.print(Text(f"Pinned content hash: {info.get('expected_content_hash') or 'not pinned'}"))
    if not info["available"]:
        con.print(Text(f"Not present: {info.get('hint')}", style="yellow"))
        con.print(Text(f"Build it with `malvalid corpus build {name} --source DIR`.", style="yellow"))
        return
    s = info["summary"]
    con.print(Text(f"Content hash: {s.get('content_hash')}"))
    con.print(Text(f"Rows: {s.get('n')} × {s.get('dim')} features; time range: {s.get('time_range') or 'n/a'}"))
    t = Table(title="Splits")
    for col in ("Split", "Rows", "Malicious", "Benign", "Unlabeled"):
        t.add_column(col, justify="right" if col != "Split" else "left")
    for sp, c in (s.get("splits") or {}).items():
        t.add_row(sp, str(c.get("n")), str(c.get("malicious")), str(c.get("benign")), str(c.get("unlabeled")))
    con.print(t)
    roles = s.get("roles") or {}
    if roles:
        con.print(Text("Roles: " + "; ".join(f"{k} = {', '.join(v) or '(none)'}" for k, v in roles.items())))


@corpus_app.command("verify")
def corpus_verify_cmd(
    name: str = typer.Argument(..., help="Corpus name."),
    corpus_dir: Optional[Path] = typer.Option(None, "--corpus-dir", help="Corpus directory override."),
) -> None:
    """Re-hash every corpus file against its manifest (exit 1 if anything does not match)."""
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        from malvalid.core import CorpusUnavailable
        from malvalid.corpora.base import verify_corpus_dir

        prov = _provider(name)
        cfg = _corpus_cfg(name, corpus_dir)
        d = prov.locate(cfg)
        try:
            if (d / "manifest.json").exists():
                manifest = json.loads((d / "manifest.json").read_text())
                verify_corpus_dir(d, manifest, force=True)
                corpus = prov.load(cfg, verify=False)  # provider checks: pinned hash, feature_version
                n_files = len(manifest.get("files", {}))
            elif prov.is_available(cfg):
                corpus = prov.load(cfg, verify=True)
                n_files = 0
            else:
                _fail(f"{prov.unavailable_hint(d)}; build it with `malvalid corpus build {name} --source DIR`")
        except CorpusUnavailable as e:
            _err().print(Text.assemble(("verification FAILED: ", "bold red"), str(e)))
            raise typer.Exit(1)
    what = f"{n_files} file(s) re-hashed" if n_files else "generated in memory"
    _out().print(Text.assemble(("OK ", "bold green"), f"{corpus.name} {corpus.version}: content hash {corpus.content_hash}, "
                               f"{corpus.n} rows × {corpus.dim} ({what})"), soft_wrap=True)


@corpus_app.command("build")
def corpus_build_cmd(
    name: str = typer.Argument(..., help="Corpus name (see `malvalid corpus list`)."),
    source: Optional[Path] = typer.Option(None, "--source", "-s", help="Raw source directory (e.g. EMBER feature JSONL files)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory. Default: $MALVALID_CORPUS_DIR/<name>."),
    workers: Optional[int] = typer.Option(None, "--workers", "-j", min=1, help="Parallel worker processes."),
) -> None:
    """Build a canonical corpus directory from its raw source (feature vectors only, never samples)."""
    verbose = _verbose()
    _setup_logging(verbose, logging.INFO)
    with _guard(verbose):
        prov = _provider(name)
        if source is not None and not source.exists():
            _fail(f"--source {source}: directory not found")
        out_dir = (out or prov.locate(None)).expanduser().resolve()
        kwargs = {"workers": int(workers)} if workers else {}
        try:
            manifest = prov.build(source.expanduser().resolve() if source else None, out_dir, **kwargs)
        except NotImplementedError as e:
            _fail(f"corpus {name!r} cannot be built locally: {e}")
    m = manifest if isinstance(manifest, dict) else {}
    _out().print(Text.assemble(("Built ", "bold green"), f"{m.get('name', name)} {m.get('version', '')} at {out_dir}: "
                               f"{m.get('n', '?')} rows × {m.get('dim', '?')}, content hash {m.get('content_hash', '?')}"),
                 soft_wrap=True)


# --------------------------------------------------------------------------------------------------
# report / sandbox-check
# --------------------------------------------------------------------------------------------------


@report_app.command("render")
def report_render_cmd(
    report_json: Path = typer.Argument(..., help="A report.json written by `malvalid run`."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output HTML path. Default: report.html next to the JSON."),
) -> None:
    """Re-render the self-contained HTML report from a report.json."""
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        from malvalid.report.json_writer import load_report_json

        data = load_report_json(report_json)
        target = out or (report_json.with_name("report.html") if report_json.name == "report.json"
                         else report_json.with_suffix(".html"))
        try:
            from malvalid.report.html import write_html
        except ImportError as e:
            _fail(f"the HTML report renderer is not available in this installation ({e}); report.json is unaffected")
        try:
            written = write_html(data, target)
        except Exception as e:
            _fail(f"could not render {report_json} to HTML: {type(e).__name__}: {e}", verbose=verbose)
    _out().print(Text.assemble(("Wrote ", "bold green"), str(written or target)), soft_wrap=True)


@app.command("sandbox-check")
def sandbox_check_cmd(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Show which isolation backends can run your model with no network access on this machine.

    Exit code: 0 if at least one network-isolating backend works, 1 otherwise.
    """
    verbose = _verbose()
    _setup_logging(verbose)
    with _guard(verbose):
        try:
            from malvalid.sandbox.host import probe_backends
        except ImportError as e:
            _fail(f"the malvalid sandbox is not available in this installation ({e})")
        res = probe_backends()
    isolated = [k for k, v in res.items() if v.get("available") and v.get("network_isolated")]
    if as_json:
        typer.echo(json.dumps(res, indent=2, default=str))
        raise typer.Exit(0 if isolated else 1)
    con = _out()
    t = Table(title="Sandbox backends")
    for col in ("Backend", "Available", "Network isolated", "Detail"):
        t.add_column(col)
    for k, v in res.items():
        t.add_row(k, _fmt(bool(v.get("available"))), _fmt(bool(v.get("network_isolated"))), str(v.get("detail") or ""))
    con.print(t)
    if isolated:
        con.print(Text.assemble(("OK ", "bold green"), f"`sandbox_backend: auto` will use {isolated[0]} (network isolated)."))
        raise typer.Exit(0)
    con.print(Text(
        "No network-isolating backend (OS sandbox) is available: `malvalid run` refuses to load a model here "
        "unless you pass --allow-reduced-isolation, which runs it in a plain worker process where the network and "
        "file system are NOT isolated (pickle refusal and resource limits only; the report records "
        "isolation: process_only). "
        "For full isolation install bubblewrap (bwrap) or enable unprivileged user namespaces (Linux, WSL2).",
        style="bold yellow",
    ))
    raise typer.Exit(1)


# --------------------------------------------------------------------------------------------------
# serve (local web UI)
# --------------------------------------------------------------------------------------------------


@app.command("serve")
def serve_cmd(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Interface to listen on. Only loopback addresses unless --allow-remote."
    ),
    port: int = typer.Option(8765, "--port", min=0, max=65535, help="TCP port (0 = pick a free port)."),
    find_free_port: bool = typer.Option(
        False, "--find-free-port",
        help="If --port is in use, listen on the next free port above it (up to 100 tried) instead of failing.",
    ),
    runs_dir: Path = typer.Option(
        Path("malvalid-runs"), "--runs-dir",
        help="Where web runs are stored; `malvalid run` results written here are listed too.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Default gate policy for new runs (a run's form can bring its own)."
    ),
    max_concurrent: int = typer.Option(1, "--max-concurrent", min=1, help="Runs executed at the same time."),
    max_upload_mb: int = typer.Option(4096, "--max-upload-mb", min=1, help="Per-file upload limit in MB."),
    allow_remote: bool = typer.Option(
        False, "--allow-remote",
        help="Allow a non-loopback --host. There is no TLS, the token is the only protection, and the UI runs "
        "submitted models: use it only on a trusted network.",
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a web browser."),
    root_path: Optional[str] = typer.Option(
        None, "--root-path",
        help="Public URL path prefix when the UI is reached through a reverse proxy, e.g. "
        "/node/<host>/<port> for Open OnDemand. Links, redirects and the cookie path use it; requests are "
        "accepted with or without it (prefix-keeping and prefix-stripping proxies). See docs/web.md.",
    ),
    trusted_host: Optional[List[str]] = typer.Option(
        None, "--trusted-host",
        help="Public host name of the reverse proxy (repeatable), e.g. ondemand.example.edu: accepted in the "
        "Host and Origin/Referer checks. The access token is still required. The first one is used in the "
        "printed proxy link (https).",
    ),
    allow_client: Optional[List[str]] = typer.Option(
        None, "--allow-client",
        help="Only serve connections from this IP address or CIDR network (repeatable; loopback is always "
        "allowed), e.g. the reverse proxy's address when --allow-remote binds a cluster-internal interface.",
    ),
    token: Optional[str] = typer.Option(
        None, "--token", envvar="MALVALID_SERVE_TOKEN", show_envvar=True,
        help="Access token to use instead of a random per-launch secret (scripts and tests). Prefer the "
        "environment variable or --token-file: other users of this machine can see command lines.",
    ),
    token_file: Optional[Path] = typer.Option(
        None, "--token-file",
        help="Keep the access token in this file so the login link survives restarts: read it (first line, "
        "16-512 URL-safe characters) if the file exists, else generate one and create the file with mode 0600.",
    ),
    allow_path_mode: bool = typer.Option(
        False, "--allow-path-mode",
        help="Enable “Use files on this machine” (path mode): the server then reads model, adapter, hash-list "
        "and policy files from its own file system by path. Off by default.",
    ),
    allow_no_sandbox: bool = typer.Option(
        False, "--allow-no-sandbox",
        help="Enable “Run without the sandbox” in the web UI: submitted models then load with no isolation. "
        "Off by default (`malvalid run --no-sandbox` is not affected).",
    ),
    allow_reduced_isolation: bool = typer.Option(
        False, "--allow-reduced-isolation",
        help="On a machine with no OS sandbox (Windows, macOS, Linux without bubblewrap/user namespaces), run "
        "models in a plain worker process (pickle refusal and resource limits, but NO network or file-system "
        "isolation) instead of failing. Every report and the UI flag it (isolation: process_only). Has no "
        "effect where the sandbox works. Used by the double-click launchers.",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging and full tracebacks."),
) -> None:
    """Start the local web UI: submit your detector, watch the run module by module, read and compare verdicts.

    Listens on 127.0.0.1 and prints a private link with a per-launch access token. Every run is the same `malvalid run`, executed in a subprocess with the same sandbox; the UI itself never imports your adapter or loads your model.

    Exit code: 0 after a normal shutdown (Ctrl+C); 2 for a usage error, a port already in use, or missing web dependencies (install malvalid[web]).
    """
    verbose = _verbose(verbose)
    _setup_logging(verbose, logging.WARNING)
    from malvalid.web import missing_web_dependencies

    missing = missing_web_dependencies()
    if missing:
        _fail(f"the web UI needs extra packages ({', '.join(missing)}): install the [web] extra: "
              "from the MalValid source folder, pip install -e '.[web]'")
    with _guard(verbose):
        from malvalid.web import serve as web_serve
        from malvalid.web.settings import (
            TOKEN_RE,
            WebSettings,
            is_loopback_host,
            is_wildcard_host,
            normalize_client_network,
            normalize_root_path,
            normalize_trusted_host,
        )

        if not is_loopback_host(host) and not allow_remote:
            _fail(f"refusing to listen on {host}: the web UI binds loopback addresses only unless you pass "
                  "--allow-remote (no TLS, token-only auth, and it runs submitted models)")
        token_source = "flag" if token is not None else "generated"
        token_created = token_loose = False
        if token_file is not None:
            from malvalid.web.settings import TokenFileError, load_or_create_token_file

            try:
                token, token_created, token_loose = load_or_create_token_file(token_file)
            except TokenFileError as e:
                _fail(f"--token-file {e}")
            token_source = "token-file"
        if token is not None and not TOKEN_RE.match(token):
            _fail("--token must be 1-512 URL-safe characters (A-Z a-z 0-9 . _ ~ -)")
        try:
            root_path = normalize_root_path(root_path)
            trusted = tuple(normalize_trusted_host(h) for h in trusted_host or [])
            clients = tuple(normalize_client_network(c) for c in allow_client or [])
        except ValueError as e:
            _fail(f"--root-path/--trusted-host/--allow-client: {e}")
        if config is not None:
            from malvalid.config import load_config, validate_against_registry

            if not config.is_file():
                _fail(f"--config {config}: file not found")
            validate_against_registry(load_config(config))
        rd = runs_dir.expanduser()
        if rd.exists() and not rd.is_dir():
            _fail(f"--runs-dir {runs_dir} exists and is not a directory")
        rd.mkdir(parents=True, exist_ok=True)
        try:
            if find_free_port and port:
                sock = web_serve.bind_first_free(host, port)
            else:
                sock = web_serve.bind_socket(host, port)
        except OSError as e:
            _fail(f"cannot listen on {host}:{port}: {e.strerror or e}"
                  + ("; pass --port 0 to pick a free port" if port else ""))
        actual_port = int(sock.getsockname()[1])
        kw: dict[str, Any] = {}
        if token is not None:
            kw["token"] = token
        settings = WebSettings(host=host, port=actual_port, runs_dir=rd, config_path=config,
                               max_concurrent=max_concurrent, max_upload_mb=max_upload_mb,
                               allow_remote=allow_remote, root_path=root_path, trusted_hosts=trusted,
                               allowed_clients=clients, allow_path_mode=allow_path_mode,
                               allow_no_sandbox=allow_no_sandbox, allow_reduced_isolation=allow_reduced_isolation,
                               token_source=token_source, **kw)
    err = _err()
    if not is_loopback_host(host):
        err.print(Panel(Text.assemble(
            ("The MalValid web UI is reachable from other machines ", "bold"), f"({settings.bind}).\n",
            "There is no TLS: the access link, your uploads and your results travel in clear text, and anyone "
            "holding the link can run models on this machine. Use --allow-remote only on a trusted network "
            "(or keep the default 127.0.0.1 and use an SSH tunnel).",
        ), title="WARNING: remote access enabled", border_style="bold red"))
    if token_file is not None and token_created:
        err.print(Text(f"Created the token file {token_file} (mode 0600): the same link works after a restart.",
                       style="dim"), soft_wrap=True)
    if token_loose:
        err.print(Text(f"warning: the token file {token_file} is readable by other users; run chmod 600 on it",
                       style="bold yellow"), soft_wrap=True)
    if allow_path_mode:
        err.print(Text("warning: --allow-path-mode: anyone holding the link can make the server read files on "
                       "this machine by path", style="bold yellow"))
    if allow_no_sandbox:
        err.print(Text("warning: --allow-no-sandbox: anyone holding the link can run a model without the sandbox",
                       style="bold yellow"))
    if allow_reduced_isolation:
        from malvalid.sandbox.host import os_sandbox_available

        if not os_sandbox_available():
            err.print(Panel(Text.assemble(
                ("Reduced isolation: this platform has no OS sandbox.\n", "bold"),
                "Models run in a separate worker process with pickle refusal and resource limits, but with NO "
                "network or file-system isolation. Every report is marked isolation: process_only. Only evaluate "
                "models whose origin you trust; use Linux with bubblewrap (or WSL2) for untrusted ones.",
            ), title="WARNING: reduced isolation", border_style="bold yellow"))
    if settings.weak_token:
        err.print(Text("warning: --token is shorter than 16 characters; use a long random token outside tests",
                       style="bold yellow"))
    con = _out()
    proxy_url = settings.proxy_login_url()
    if proxy_url:
        con.print(Text.assemble(("MalValid web UI", "bold"),
                                f" {__version__} — open this private link in your browser (through the proxy):"))
        con.print(Text(f"  {proxy_url}", style="bold"), soft_wrap=True)
        con.print(Text("Direct link on this machine:", style="dim"))
    else:
        con.print(Text.assemble(("MalValid web UI", "bold"),
                                f" {__version__} — open this private link in your browser:"))
    con.print(Text(f"  {settings.login_url()}", style="bold"), soft_wrap=True)
    fallback = settings.fallback_login_url()
    if fallback:
        con.print(Text("Or, if your browser cannot open *.localhost addresses (weaker cookie scoping):",
                       style="dim"))
        con.print(Text(f"  {fallback}"), soft_wrap=True)
    for url in settings.remote_urls():
        con.print(Text(f"  {url}"), soft_wrap=True)
    con.print(Text(f"Runs directory: {settings.runs_dir}"), soft_wrap=True)
    if settings.config_path:
        con.print(Text(f"Default gate policy: {settings.config_path}"), soft_wrap=True)
    con.print(Text("Keep the link private: it signs in whoever opens it. Press Ctrl+C to stop "
                   "(queued and running runs are stopped too).", style="dim"))
    open_browser = not no_browser and web_serve.has_display()
    try:
        web_serve.run_server(settings, sock, open_browser=open_browser,
                             log_level="info" if verbose else "warning")
    except KeyboardInterrupt:  # pragma: no cover - uvicorn handles SIGINT itself
        pass
    finally:
        with contextlib.suppress(OSError):
            sock.close()


def main() -> None:
    """Console-script entry point (``malvalid``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
