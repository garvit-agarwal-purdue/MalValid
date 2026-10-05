"""``malvalid validate-adapter``: check a submission against the adapter contract before a full run.

:func:`validate_adapter` walks the same path a gate run takes, stopping at the first step that makes
the rest meaningless, and records every step as a check:

0. ``config`` — the gate config validated exactly as ``malvalid run`` validates it (module ids,
   parameter names and values; :func:`malvalid.config.validate_against_registry`), so a config the
   run would refuse never passes pre-flight;
1. ``declarations`` — import the adapter in the sandbox *without* calling ``load()`` and validate the
   declared attributes (:func:`malvalid.sandbox.host.inspect_adapter`);
2. ``feature_schema`` — ``feature_version`` names a registered malvalid feature schema;
3. ``training_manifest`` — the declared training hashes / cutoff parse;
4. ``file_safety`` — M0's static scan of every artifact (:func:`malvalid.modules.file_safety.scan_artifacts`);
   a finding that would abort a run stops validation before anything is deserialized;
5. ``load`` — ``load()`` inside the sandbox (:func:`malvalid.sandbox.host.open_model`);
6. probes on a few rows of the configured canonical corpus (or, if it is unavailable, schema-shaped
   random vectors): ``predict_proba`` shape/dtype/range, batch-size independence, ``predict`` ==
   ``predict_proba >= operating_threshold``, determinism across two calls, ``featurize`` on a benign
   Windows executable shipped with setuptools, and tree access + fidelity.

Failures never raise: they become failed checks with an actionable message (only a broken
``malvalid`` installation or an invalid config raises).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from malvalid.core import MalValidError, to_jsonable

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.config import GateConfig
    from malvalid.context import ModelDeclarations

log = logging.getLogger("malvalid.adapters")

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
_STATUS_LABEL = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}

#: Probe sizes: small, so validation takes seconds even on 2568-dimensional EMBER vectors.
N_PROBE_ROWS = 256
#: Tree-dump fidelity tolerance (the runner disables tree access above it).
TREE_FIDELITY_TOL = 1e-4
#: Two identical calls must agree to within this (float noise from multithreading is tolerated).
DETERMINISM_TOL = 1e-9
#: Scoring a row alone vs inside a batch must agree to within this.
BATCH_TOL = 1e-6


@dataclass
class Check:
    """One validation step. ``ok`` is False only for :data:`FAIL`."""

    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != FAIL

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "status": self.status, "detail": self.detail}


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_adapter`: ``ok`` iff no check failed."""

    adapter_path: str
    checks: list[Check] = field(default_factory=list)
    declarations: "ModelDeclarations | None" = None
    sandbox: dict[str, Any] | None = None
    probe_source: str | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def check(self, name: str) -> Check | None:
        """The check called ``name`` (or None if validation stopped before it)."""
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def add(self, name: str, status: str, detail: str) -> Check:
        c = Check(name, status, detail)
        self.checks.append(c)
        log.info("validate-adapter: %s %s: %s", status.upper(), name, detail.splitlines()[0] if detail else "")
        return c

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "ok": self.ok,
            "adapter_path": self.adapter_path,
            "class_name": self.declarations.class_name if self.declarations else None,
            "n_failed": len(self.failed),
            "checks": [c.to_dict() for c in self.checks],
            "declarations": self.declarations.to_dict() if self.declarations else None,
            "sandbox": self.sandbox,
            "probe_source": self.probe_source,
            "duration_s": round(self.duration_s, 3),
        })

    def render_text(self) -> str:
        """Plain-text summary for the terminal."""
        name = Path(self.adapter_path).name
        cls = f" ({self.declarations.class_name})" if self.declarations and self.declarations.class_name else ""
        lines = [f"malvalid adapter validation: {name}{cls}"]
        width = max((len(c.name) for c in self.checks), default=10)
        for c in self.checks:
            detail = c.detail.splitlines() or [""]
            lines.append(f"  {_STATUS_LABEL.get(c.status, c.status.upper()):4}  {c.name:<{width}}  {detail[0]}")
            for extra in detail[1:12]:
                lines.append(f"  {'':4}  {'':<{width}}  {extra}")
            if len(detail) > 12:
                lines.append(f"  {'':4}  {'':<{width}}  ... ({len(detail) - 12} more lines)")
        if self.probe_source:
            lines.append(f"  probe inputs: {self.probe_source}")
        n_fail, n_warn = len(self.failed), sum(c.status == WARN for c in self.checks)
        if n_fail:
            lines.append(f"Result: {n_fail} check(s) FAILED — fix them before `malvalid run`.")
        elif n_warn:
            lines.append(f"Result: OK with {n_warn} warning(s) — the adapter meets the submission contract.")
        else:
            lines.append("Result: OK — the adapter meets the submission contract.")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _msg(e: BaseException, limit: int = 4000) -> str:
    s = str(e).strip() or type(e).__name__
    return s if len(s) <= limit else s[:limit] + " ..."


def benign_test_executable() -> Path | None:
    """A benign Windows PE shipped with setuptools (used to exercise ``featurize``), if installed."""
    try:
        import setuptools
    except Exception:  # noqa: BLE001 - optional
        return None
    base = Path(setuptools.__file__).resolve().parent
    for name in ("cli-64.exe", "cli-32.exe", "cli.exe", "gui-64.exe", "gui-32.exe", "gui.exe"):
        p = base / name
        if p.is_file():
            return p
    return None


def _probe_rows(cfg: "GateConfig", decl: "ModelDeclarations", dim: int, seed: int) -> tuple[np.ndarray, str]:
    """Up to :data:`N_PROBE_ROWS` corpus rows (both classes when possible), else random vectors."""
    reason = ""
    name = getattr(cfg, "corpus", None)
    if name:
        try:
            from malvalid import registry

            prov = registry.get_corpus_provider(str(name))
            if getattr(prov, "feature_version", None) != decl.feature_version:
                reason = (f"corpus {name!r} is in feature space {prov.feature_version!r}, the model uses "
                          f"{decl.feature_version!r}")
            elif not prov.is_available(cfg):
                reason = f"corpus {name!r} is not available ({prov.unavailable_hint(prov.locate(cfg))})"
            else:
                corpus = prov.load(cfg, verify=False)
                rng = np.random.default_rng(seed)
                picks: list[np.ndarray] = []
                for lab in (0, 1):
                    idx = np.flatnonzero(np.asarray(corpus.label) == lab)
                    if idx.size:
                        picks.append(np.sort(rng.choice(idx, size=min(N_PROBE_ROWS // 2, idx.size), replace=False)))
                if picks:
                    idx = np.sort(np.concatenate(picks))
                    X = np.ascontiguousarray(corpus.take(idx), dtype=np.float32)
                    if X.ndim == 2 and X.shape[1] == dim:
                        return X, f"{X.shape[0]} rows of corpus {name!r}"
                    reason = f"corpus {name!r} rows have shape {X.shape}, schema dim is {dim}"
                else:
                    reason = f"corpus {name!r} has no labelled rows"
        except Exception as e:  # noqa: BLE001 - fall back to random vectors
            reason = f"corpus {name!r} could not be read ({type(e).__name__}: {_msg(e, 300)})"
    rng = np.random.default_rng(seed)
    X = (rng.random((64, dim)) * rng.choice([1.0, 10.0, 1000.0], size=(1, dim))).astype(np.float32)
    return X, f"64 random schema-shaped vectors ({reason or 'no corpus configured'})"


def _file_safety_params(cfg: "GateConfig") -> dict[str, Any]:
    from malvalid.config import module_params
    from malvalid.modules.file_safety import FileSafetyModule

    try:
        return module_params(cfg, FileSafetyModule)
    except Exception:  # noqa: BLE001 - a partial config object: fall back to the defaults
        return dict(FileSafetyModule.default_params)


# --------------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------------


def _check_config(rep: ValidationReport, cfg: "GateConfig") -> None:
    from malvalid.config import validate_against_registry
    from malvalid.core import ConfigError

    src = getattr(cfg, "source_path", None)
    where = Path(src).name if src else "the packaged default gate config"
    try:
        warnings = validate_against_registry(cfg)
    except ConfigError as e:
        rep.add("config", FAIL, f"{where}: {e} — `malvalid run` refuses this config")
        return
    if warnings:
        rep.add("config", WARN, f"{where} is valid, with warning(s): " + "; ".join(warnings))
    else:
        rep.add("config", PASS, f"{where} is valid for `malvalid run` (module ids, parameter names and values)")


def _check_schema(rep: ValidationReport, decl: "ModelDeclarations") -> Any:
    from malvalid import registry

    try:
        schema = registry.get_schema(decl.feature_version)
    except KeyError as e:
        rep.add("feature_schema", FAIL, f"{e.args[0] if e.args else e}. Set feature_version to one of malvalid's "
                                        "feature schemas (see `malvalid list-modules --json`).")
        return None
    feat = "a raw-PE featurizer is available" if schema.featurize_available() else "no raw-PE featurizer installed"
    rep.add("feature_schema", PASS, f"{decl.feature_version}: {schema.dim} features; {feat}")
    return schema


def _check_manifest(rep: ValidationReport, decl: "ModelDeclarations") -> None:
    try:
        from malvalid.manifest import load_training_manifest
    except ImportError:  # pragma: no cover - runner component missing
        return
    try:
        tm = load_training_manifest(decl)
    except Exception as e:  # noqa: BLE001
        rep.add("training_manifest", FAIL, f"could not parse the training manifest: {_msg(e)}")
        return
    parts, status = [], PASS
    if tm.path:
        parts.append(f"{tm.n_hashes} training sha256 hashes in {Path(tm.path).name}")
    else:
        parts.append("training_hashes_path is None (training-member exclusion and membership inference will skip)")
        status = WARN
    if tm.cutoff_parsed:
        parts.append(f"training cutoff {tm.cutoff_parsed.isoformat()}")
    else:
        parts.append("training_cutoff is None (temporal drift tests will skip)")
        status = WARN
    for w in tm.warnings:
        parts.append(w)
        status = WARN
    rep.add("training_manifest", status, "; ".join(parts))


def _check_file_safety(rep: ValidationReport, decl: "ModelDeclarations", cfg: "GateConfig",
                       allow_pickle: bool) -> bool:
    """Returns True when loading may proceed."""
    from malvalid.modules.file_safety import SEVERITY_RANK, discover_model_files, scan_artifacts

    params = _file_safety_params(cfg)
    fail_on = str(params.get("fail_on", "HIGH")).upper()
    declared = [Path(p) for p in decl.model_paths]
    adapter_dir = Path(decl.adapter_path).parent
    stray: list[Path] = []
    if params.get("scan_adapter_dir", True) and adapter_dir.is_dir():
        found = discover_model_files(adapter_dir)
        dset = {p.resolve() for p in declared if p.exists()}
        stray = [p for p in found if p.resolve() not in dset]
    primary = declared or stray
    if not declared:
        stray = []
    try:
        report = scan_artifacts(primary, allow_pickle=allow_pickle, undeclared=stray)
    except Exception as e:  # noqa: BLE001
        rep.add("file_safety", FAIL, f"the artifact scan crashed: {type(e).__name__}: {_msg(e)}")
        return False
    counts = report.counts()
    names = ", ".join(a.name for a in report.artifacts) or "none"
    lines = [f"{len(report.artifacts)} artifact(s) scanned ({names}); findings: "
             + ", ".join(f"{k} {v}" for k, v in counts.items())]
    for a in report.artifacts:
        if not a.scanned and a.scan_reason:
            lines.append(f"{a.name}: modelscan: {a.scan_reason}")
    if report.abort:
        lines += report.abort_reasons()
        lines.append("A gate run would abort before loading the model; validation stops here.")
        rep.add("file_safety", FAIL, "\n".join(lines))
        return False
    worst = [f"{a.name}: {f.severity} {f.description}" for a, f in report.findings()
             if SEVERITY_RANK.get(f.severity, 0) >= SEVERITY_RANK.get(fail_on, 3)]
    if worst:
        lines += worst
        lines.append(f"Findings at or above file_safety.fail_on={fail_on} fail the M0 hard gate.")
        rep.add("file_safety", FAIL, "\n".join(lines))
        return True  # the run would not abort, so the remaining probes are still informative
    status = PASS
    if not declared:
        status = WARN
        lines.append("The adapter declares no model_path/model_paths; declare the file load() reads so M0 "
                     "scans exactly that artifact.")
    if report.n_pickle and allow_pickle:
        status = WARN
        lines.append(f"{report.n_pickle} pickle-based artifact(s) accepted because of --allow-pickle; M0 will "
                     "report status warn. Prefer a non-pickle format for production models.")
    if any(f.severity in ("MEDIUM", "LOW") for _, f in report.findings()):
        status = WARN
    rep.add("file_safety", status, "\n".join(lines))
    return True


def _check_predict_proba(rep: ValidationReport, model: Any, X: np.ndarray) -> np.ndarray | None:
    from malvalid.core import AdapterError

    try:
        p = model.predict_proba(X)
    except AdapterError as e:
        rep.add("predict_proba", FAIL, _msg(e))
        return None
    except MalValidError as e:
        rep.add("predict_proba", FAIL, f"{type(e).__name__}: {_msg(e)}")
        return None
    thr = float(model.declarations.operating_threshold)
    detail = (f"({X.shape[0]},) float64 in [0, 1]: min {p.min():.4g}, mean {p.mean():.4g}, max {p.max():.4g}; "
              f"{(p >= thr).mean():.1%} of probe rows >= operating_threshold {thr:g}")
    if X.shape[0] > 1 and float(np.ptp(p)) == 0.0:
        rep.add("predict_proba", WARN, detail + ". Every probe row got the same score — is the model reading "
                                                  "the feature vector?")
    else:
        rep.add("predict_proba", PASS, detail)
    return p


def _check_batching(rep: ValidationReport, model: Any, X: np.ndarray, p: np.ndarray) -> None:
    try:
        k = min(3, X.shape[0])
        single = np.concatenate([model.predict_proba(X[i : i + 1]) for i in range(k)])
    except MalValidError as e:
        rep.add("batch_independence", FAIL, f"scoring a single row failed: {_msg(e)}")
        return
    diff = float(np.max(np.abs(single - p[:k]))) if k else 0.0
    if diff > BATCH_TOL:
        rep.add("batch_independence", FAIL,
                f"scoring rows one at a time differs from scoring them in a batch (max |Δ| = {diff:.3g}); "
                "predict_proba must score each row independently")
    else:
        rep.add("batch_independence", PASS, f"single-row and batched scores agree (max |Δ| = {diff:.2g})")


def _check_predict(rep: ValidationReport, model: Any, X: np.ndarray, p: np.ndarray) -> None:
    try:
        y = model.predict(X)
    except MalValidError as e:
        rep.add("predict", FAIL, _msg(e))
        return
    thr = float(model.declarations.operating_threshold)
    want = (p >= thr).astype(np.int8)
    bad = np.flatnonzero(y != want)
    if bad.size:
        i = int(bad[0])
        rep.add("predict", FAIL,
                f"predict disagrees with predict_proba >= operating_threshold ({thr:g}) on {bad.size} of "
                f"{y.size} rows (e.g. row {i}: predict={int(y[i])}, predict_proba={p[i]:.6g}). "
                "malvalid reports detection rates at the declared operating_threshold, so the two must agree.")
    else:
        rep.add("predict", PASS, f"(n,) in {{0, 1}}, consistent with predict_proba >= {thr:g}")


def _check_determinism(rep: ValidationReport, model: Any, X: np.ndarray, p: np.ndarray) -> None:
    try:
        p2 = model.predict_proba(X)
    except MalValidError as e:
        rep.add("determinism", FAIL, f"the second predict_proba call failed: {_msg(e)}")
        return
    diff = float(np.max(np.abs(p2 - p))) if p.size else 0.0
    if diff > DETERMINISM_TOL:
        rep.add("determinism", FAIL,
                f"two identical predict_proba calls differ (max |Δ| = {diff:.3g}); malvalid's tests assume a "
                "deterministic model (fix random seeds / disable dropout at inference)")
    else:
        rep.add("determinism", PASS, "two identical calls returned identical scores")


def _check_featurize(rep: ValidationReport, model: Any, dim: int | None) -> None:
    from malvalid.core import FeaturizeUnavailable

    if not model.has_featurize():
        rep.add("featurize", SKIP, "no raw-PE featurizer (the adapter has no featurize() and the schema's "
                                   "extractor is not installed); tests that need raw executables will skip")
        return
    exe = benign_test_executable()
    if exe is None:
        rep.add("featurize", SKIP, "no benign test executable found (setuptools is not installed)")
        return
    src = model.featurize_source() if hasattr(model, "featurize_source") else None
    try:
        v = model.featurize(exe.read_bytes())
        s = model.predict_proba(v[None, :])
    except FeaturizeUnavailable as e:
        rep.add("featurize", SKIP, _msg(e))
        return
    except MalValidError as e:
        rep.add("featurize", FAIL, f"featurize({exe.name}) failed: {_msg(e)}")
        return
    except ValueError as e:
        rep.add("featurize", FAIL, f"featurize({exe.name}) failed: {_msg(e)}")
        return
    rep.add("featurize", PASS, f"featurize ({src or 'adapter'}) on the benign {exe.name} -> ({v.shape[0]},) "
                               f"finite vector{'' if dim is None else f' (schema dim {dim})'}; score {float(s[0]):.4g}")


def _check_trees(rep: ValidationReport, model: Any, X: np.ndarray, p: np.ndarray, dim: int | None) -> None:
    from malvalid.core import ModuleTimeout

    try:
        te = model.tree_ensemble()
    except ModuleTimeout as e:
        rep.add("tree_access", WARN, f"tree extraction timed out: {_msg(e)}")
        return
    except Exception as e:  # noqa: BLE001 - optional capability
        rep.add("tree_access", WARN, f"tree extraction failed ({type(e).__name__}: {_msg(e, 500)}); tree-based "
                                     "tests will use black-box queries only")
        return
    note = getattr(model, "tree_access_note", None)
    if te is None:
        rep.add("tree_access", SKIP, f"no tree access ({note or 'not a supported tree ensemble'}); set "
                                     "native_model or implement tree_ensemble() for faster, exact tree-based tests")
        return
    if dim is not None and te.n_features > dim:
        rep.add("tree_access", WARN, f"the tree ensemble uses {te.n_features} features but the schema has {dim}")
        return
    try:
        fid = te.fidelity(X, p)
    except Exception as e:  # noqa: BLE001
        rep.add("tree_access", WARN, f"the tree dump could not score the probe rows ({type(e).__name__}: {_msg(e, 300)})")
        return
    if not np.isfinite(fid) or fid > TREE_FIDELITY_TOL:
        rep.add("tree_access", WARN,
                f"{te.n_trees} trees extracted, but they do not reproduce predict_proba (max |Δ| = {fid:.3g} > "
                f"{TREE_FIDELITY_TOL:g}); a run will disable tree access. This usually means predict_proba "
                "post-processes the model's output (calibration, ensembling).")
        return
    rep.add("tree_access", PASS, f"{te.n_trees} trees ({note or 'from the adapter'}); tree dump reproduces "
                                 f"predict_proba to max |Δ| = {fid:.2g}")


def _check_sandbox(rep: ValidationReport, info: dict[str, Any]) -> None:
    rep.sandbox = info
    if not info.get("enabled", True):
        rep.add("sandbox", WARN, "SANDBOX DISABLED (runtime.sandbox: false): the adapter ran in-process with no "
                                 "isolation")
        return
    backend = info.get("backend")
    if not info.get("network_isolated"):
        rep.add("sandbox", WARN, f"backend {backend}: the worker's network is NOT isolated on this host")
        return
    extra = "; ".join(info.get("warnings") or [])
    rep.add("sandbox", WARN if extra else PASS,
            f"backend {backend}: network isolated; file system: {info.get('filesystem')}" + (f"; {extra}" if extra else ""))


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------


def validate_adapter(
    adapter_path: Path,
    cfg: "GateConfig",
    *,
    allow_pickle: bool = False,
    class_name: str | None = None,
    model_paths: Sequence[Path] = (),
    work_dir: Path | None = None,
) -> ValidationReport:
    """Check ``adapter_path`` against the submission contract (see the module docstring).

    ``work_dir`` keeps the sandbox scratch directory (with ``worker.log``) for inspection; by default
    a temporary directory is used and removed afterwards.
    """
    from malvalid.core import AdapterError, SandboxError
    from malvalid.sandbox.host import SandboxPolicy, inspect_adapter, open_model

    t0 = time.monotonic()
    adapter = Path(adapter_path).expanduser().resolve()
    rep = ValidationReport(adapter_path=str(adapter))
    own = work_dir is None
    run_dir = Path(tempfile.mkdtemp(prefix="malvalid-validate-")) if own else Path(work_dir)
    model: Any = None
    try:
        overrides = [Path(p).expanduser().resolve() for p in model_paths]
        policy = SandboxPolicy.from_config(cfg, run_dir=run_dir, allow_pickle=allow_pickle,
                                           extra_ro=[p.parent for p in overrides])
        if not policy.enabled:
            log.warning("validate-adapter: the sandbox is disabled; the adapter runs in-process")

        # 0. the gate config, validated as `malvalid run` does
        _check_config(rep, cfg)

        # 1. declarations
        try:
            decl = inspect_adapter(adapter, policy, class_name=class_name, model_paths_override=overrides)
        except (AdapterError, SandboxError) as e:
            rep.add("declarations", FAIL, _msg(e))
            return rep
        rep.declarations = decl
        rep.add("declarations", PASS,
                f"class {decl.class_name}: feature_version={decl.feature_version}, model_kind={decl.model_kind}, "
                f"operating_threshold={decl.operating_threshold:g}, "
                f"{len(decl.model_paths)} model artifact(s)"
                + (f" ({decl.extras.get('model_paths_source')})" if decl.extras.get("model_paths_source") else ""))

        # 2-3. schema, training manifest
        schema = _check_schema(rep, decl)
        _check_manifest(rep, decl)

        # 4. file safety (static, before load)
        if not _check_file_safety(rep, decl, cfg, allow_pickle):
            return rep

        # 5. load
        try:
            model = open_model(adapter, policy, declarations=decl, class_name=class_name)
        except (AdapterError, SandboxError, MalValidError) as e:
            rep.add("load", FAIL, f"{decl.class_name}.load() failed in the sandbox: {_msg(e)}")
            return rep
        info = model.sandbox_info() if hasattr(model, "sandbox_info") else {}
        load_s, startup_s = info.get("load_s"), info.get("startup_s")
        timing = ""
        if isinstance(load_s, (int, float)):
            timing = f" ({load_s:.2f} s" + (f"; worker start-up {startup_s:.1f} s)" if isinstance(startup_s, (int, float)) else ")")
        rep.add("load", PASS, f"{decl.class_name}.load() succeeded in the sandbox{timing}")
        _check_sandbox(rep, info)

        # 6. probes
        if schema is None:
            rep.add("predict_proba", SKIP, "no feature schema, so no probe vectors can be built")
            return rep
        X, rep.probe_source = _probe_rows(cfg, decl, int(schema.dim), int(getattr(cfg.runtime, "seed", 0)))
        p = _check_predict_proba(rep, model, X)
        if p is not None:
            _check_batching(rep, model, X, p)
            _check_predict(rep, model, X, p)
            _check_determinism(rep, model, X, p)
        _check_featurize(rep, model, int(schema.dim))
        if p is not None:
            _check_trees(rep, model, X, p, int(schema.dim))
        if hasattr(model, "sandbox_info"):
            rep.sandbox = model.sandbox_info()
        return rep
    finally:
        if model is not None:
            try:
                model.close()
            except Exception:  # noqa: BLE001 - best effort
                log.debug("closing the validated model failed", exc_info=True)
        rep.duration_s = time.monotonic() - t0
        if own:
            shutil.rmtree(run_dir, ignore_errors=True)


__all__ = ["Check", "ValidationReport", "benign_test_executable", "validate_adapter"]
