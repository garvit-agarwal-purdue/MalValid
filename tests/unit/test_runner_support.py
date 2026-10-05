"""Shared fakes for the runner and CLI tests (plus one self-check of the fakes).

The sandbox (``malvalid.sandbox.host``) and M0 are other agents' components; the runner reaches
them only through small seams (``runner._make_policy``, ``_inspect_adapter``, ``_open_model``,
``_file_safety_cls``, ``_write_html``). :func:`install_fake_sandbox` monkeypatches those seams with
:class:`malvalid.testing.InProcessModel`-based fakes over a LightGBM model trained on the ``toy_v1``
corpus, so a full gate run takes well under a second.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import pytest

from malvalid import registry, runner
from malvalid.config import GateConfig, ModuleConfig, RuntimeConfig, VerdictConfig
from malvalid.context import ModelDeclarations, RunContext
from malvalid.core import GateCheck, GateMode, Module, ModuleResult, Requirement, Status
from malvalid.corpora.base import Corpus, CorpusProvider
from malvalid.testing import InProcessModel, ToySchema, make_toy_corpus, train_toy_lgbm

TOY = "toy_v1"
TOY_CORPUS = "toy_v1_corpus"
THRESHOLD = 0.5


# --------------------------------------------------------------------------------------------------
# Toy data + model (cached: identical to what ToyCorpusProvider.load() returns)
# --------------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def toy_corpus() -> Corpus:
    return make_toy_corpus()


@lru_cache(maxsize=1)
def toy_booster() -> Any:
    return train_toy_lgbm(toy_corpus(), rounds=20)


def train_hashes(extra_eval_members: int = 0) -> list[str]:
    """Hashes of the toy ``train`` split, plus the first ``extra_eval_members`` eval-role rows."""
    c = toy_corpus()
    hs = c.sha256[c.indices(splits=("train",))].tolist()
    if extra_eval_members:
        hs += c.sha256[c.indices(splits=("test",))[:extra_eval_members]].tolist()
    return hs


class _Detector:
    """A trusted in-process detector over the toy booster; ``offset`` breaks tree fidelity."""

    def __init__(self, booster: Any, offset: float = 0.0):
        self.native_model = booster
        self.offset = offset

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.native_model.predict(X)
        return np.clip(p * (1 - 2 * self.offset) + self.offset, 0.0, 1.0) if self.offset else p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= THRESHOLD).astype(np.int8)


class RecordingModel(InProcessModel):
    """InProcessModel + the SandboxedModel extras the runner uses (deadline, restart, info)."""

    def __init__(self, detector: Any, declarations: ModelDeclarations, *, trees: bool = True):
        super().__init__(detector, declarations)
        self.deadlines: list[float | None] = []
        self.restarts = 0
        self.closed = False
        self._expose_trees = trees

    def set_deadline(self, deadline: float | None) -> None:
        self.deadlines.append(deadline)

    def restart(self) -> None:
        self.restarts += 1

    def tree_ensemble(self):  # type: ignore[override]
        return super().tree_ensemble() if self._expose_trees else None

    scratch: Path | None = None

    def sandbox_info(self) -> dict[str, Any]:
        # Like the real SandboxedModel, this mentions the worker scratch dir under private/.
        return {"backend": "in-process-fake", "network_isolated": True, "enabled": True,
                "scratch_dir": str(self.scratch) if self.scratch else None}

    def featurize_source(self) -> str | None:
        return None

    def close(self) -> None:
        self.closed = True


def make_decl(
    tmp_path: Path,
    *,
    feature_version: str = TOY,
    hashes: list[str] | None = None,
    cutoff: str | None = "2017-12",
    model_paths: tuple[str, ...] | None = None,
) -> ModelDeclarations:
    adapter = tmp_path / "adapter" / "my_adapter.py"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    if not adapter.exists():
        adapter.write_text("# fake adapter (the sandbox seams are monkeypatched in these tests)\n")
    model_file = adapter.parent / "model.txt"
    if not model_file.exists():
        model_file.write_text("tree\nversion=v4\n")
    hp = None
    if hashes is not None:
        hp = adapter.parent / "train_hashes.txt"
        hp.write_text("# training manifest\n" + "\n".join(hashes) + "\n")
    return ModelDeclarations(
        feature_version=feature_version,
        model_kind="lightgbm",
        operating_threshold=THRESHOLD,
        training_hashes_path=str(hp) if hp else None,
        training_cutoff=cutoff,
        model_paths=model_paths if model_paths is not None else (str(model_file),),
        adapter_path=str(adapter),
        class_name="ToyDetector",
    )


# --------------------------------------------------------------------------------------------------
# Fake M0 (file safety) modules — injected through runner._file_safety_cls
# --------------------------------------------------------------------------------------------------


class _M0Base(Module):
    id = "file_safety"
    code = "M0"
    title = "Model file safety"
    requires = ()
    default_gate = GateMode.HARD
    default_params = {"fail_on": "HIGH", "scan_adapter_dir": True}
    seen: ClassVar[list[RunContext]] = []


class FakeM0Pass(_M0Base):
    def run(self, ctx: RunContext) -> ModuleResult:
        type(self).seen.append(ctx)
        return self.result(
            ctx, finding="No unsafe content found in 1 artifact.",
            checks=[GateCheck.evaluate("n_blocking_findings", 0, "<=", 0)],
            metrics={"n_artifacts": 1, "n_pickle": 0}, score=1.0,
        )


class FakeM0Abort(_M0Base):
    def run(self, ctx: RunContext) -> ModuleResult:
        type(self).seen.append(ctx)
        res = self.result(
            ctx, finding="CRITICAL: model.pkl imports os.system when unpickled.",
            checks=[GateCheck.evaluate("n_critical", 1, "<=", 0)], score=0.0,
        )
        res.details["abort"] = True
        return res


class FakeM0Crash(_M0Base):
    def run(self, ctx: RunContext) -> ModuleResult:
        raise RuntimeError("scanner exploded")


# --------------------------------------------------------------------------------------------------
# Fake test modules (registered via registry.register for the duration of a test)
# --------------------------------------------------------------------------------------------------


class _Fake(Module):
    seen: ClassVar[list[RunContext]] = []

    def _note(self, ctx: RunContext) -> None:
        type(self).seen.append(ctx)


class PassMod(_Fake):
    id = "t_pass"
    code = "T1"
    title = "Fake passing axis"
    description = "Always passes (scores a few eval rows)."
    requires = (Requirement.FEATURE_SPACE,)
    default_params = {"min_rate": 0.5}

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        assert ctx.corpus is not None
        idx = ctx.corpus.eval_indices(1, exclude_hashes=ctx.training_hashes)[:64]
        p = ctx.score(ctx.corpus.take(idx))
        c = GateCheck.evaluate("rate", 0.9, ">=", ctx.params["min_rate"], ideal=0.95, floor=0.2)
        return self.result(ctx, finding="Rate 0.90 vs 0.50.", checks=[c],
                           metrics={"rate": 0.9, "mean_score": float(p.mean()), "n_scored": int(p.size)})


class WarnMod(_Fake):
    id = "t_warn"
    code = "T2"
    title = "Fake warn-gate axis"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        c = GateCheck.evaluate("leak", 0.2, "<=", 0.1, ideal=0.01, floor=0.3)
        return self.result(ctx, finding="Leak 0.20 exceeds 0.10.", checks=[c])


class HardFailMod(_Fake):
    id = "t_hard"
    code = "T3"
    title = "Fake hard-gate axis"
    requires = (Requirement.QUERY_ONLY,)
    default_gate = GateMode.HARD

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        c = GateCheck.evaluate("fpr", 0.2, "<=", 0.01, ideal=0.001, floor=0.05, scale="log")
        return self.result(ctx, finding="FPR 0.2 vs 0.01.", checks=[c])


class NeedsHashesMod(_Fake):
    id = "t_hashes"
    code = "T4"
    title = "Needs the training manifest"
    requires = (Requirement.FEATURE_SPACE, Requirement.TRAINING_HASHES)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        return self.result(ctx, finding="ok", checks=[GateCheck.evaluate("x", 1.0, ">=", 0.5, ideal=1.0, floor=0.0)])


class NeedsAnyMod(_Fake):
    id = "t_any"
    code = "T5"
    title = "Needs trees or the feature space"
    requires_any = (Requirement.TREE_ACCESS, Requirement.FEATURE_SPACE)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        return self.result(ctx, finding="ok", checks=[GateCheck.evaluate("x", 1.0, ">=", 0.5, ideal=1.0, floor=0.0)])


class NeedsTreesMod(_Fake):
    id = "t_trees"
    code = "T6"
    title = "Needs tree access"
    requires = (Requirement.TREE_ACCESS,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        trees = ctx.model.tree_ensemble()
        assert trees is not None
        return self.result(ctx, finding=f"{trees.n_trees} trees", metrics={"n_trees": trees.n_trees}, score=0.9)


class NeedsSampleMod(_Fake):
    id = "t_sample"
    code = "T7"
    title = "Needs a sample directory"
    requires = (Requirement.SAMPLE_DIR,)

    def run(self, ctx: RunContext) -> ModuleResult:  # pragma: no cover - always skipped
        self._note(ctx)
        return self.result(ctx, finding="ok")


class CrashMod(_Fake):
    id = "t_crash"
    code = "T8"
    title = "Crashes"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        raise ValueError("boom: something went wrong\nsecond line")


class SlowMod(_Fake):
    """Never checks its deadline: only the SIGALRM timer can stop it."""

    id = "t_slow"
    code = "T9"
    title = "Hangs"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            time.sleep(0.01)
        return self.result(ctx, finding="finished (the timer did not fire!)")


class SwallowMod(_Fake):
    """Swallows every exception, including the timeout — must still be reported as an error."""

    id = "t_swallow"
    code = "TA"
    title = "Swallows the timeout"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            try:
                time.sleep(0.01)
            except Exception:  # noqa: BLE001 - deliberately bad module
                break
        return self.result(ctx, finding="pretends to have finished", score=1.0)


class DirtyMod(_Fake):
    """Non-finite numbers everywhere + a private artifact path leaked into the result."""

    id = "t_dirty"
    code = "TB"
    title = "Dirty result"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        p = ctx.artifacts.save_private(self.id, "vectors", X=np.ones((3, 2), dtype=np.float32))
        ctx.artifacts.add_curve(self.id, "curve", [0.0, 1.0, 2.0], [float("nan"), 1.0, float("inf")], title="c")
        checks = [GateCheck.evaluate("ratio", float("inf"), "<=", 1.0, ideal=0.1, floor=5.0),
                  GateCheck.evaluate("nan_metric", float("nan"), ">=", 0.5, ideal=1.0, floor=0.0)]
        res = self.result(
            ctx, finding=f"saved vectors to {p}", checks=checks,
            metrics={"nan": float("nan"), "inf": float("-inf"), "arr": np.array([1.0, np.nan]),
                     "np_scalar": np.float32(0.25), "date": dt.date(2020, 1, 2)},
            details={"private_file": str(p), "nested": {"paths": [str(p), "ok"]}},
            notes=[f"see {p.parent} for raw vectors"],
        )
        return res


class BadReturnMod(_Fake):
    id = "t_badret"
    code = "TC"
    title = "Returns garbage"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> Any:
        self._note(ctx)
        return {"status": "pass"}


class WrongGateMod(_Fake):
    """Returns a result claiming gate=warn although the config made it hard."""

    id = "t_wronggate"
    code = "TD"
    title = "Mislabels its gate"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        c = GateCheck.evaluate("x", 0.1, ">=", 0.5, ideal=1.0, floor=0.0)
        res = self.result(ctx, finding="fails", checks=[c])
        res.gate = GateMode.WARN
        res.status = Status.WARN
        res.score = float("nan")
        return res


class DeadlineMod(_Fake):
    """Honours ctx.check_deadline() (the cooperative timeout path)."""

    id = "t_deadline"
    code = "TE"
    title = "Cooperative timeout"
    requires = (Requirement.QUERY_ONLY,)

    def run(self, ctx: RunContext) -> ModuleResult:
        self._note(ctx)
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            ctx.check_deadline()
            time.sleep(0.01)
        return self.result(ctx, finding="finished (deadline ignored!)")


FAKE_MODULES: tuple[type[_Fake], ...] = (
    PassMod, WarnMod, HardFailMod, NeedsHashesMod, NeedsAnyMod, NeedsTreesMod, NeedsSampleMod,
    CrashMod, SlowMod, SwallowMod, DirtyMod, BadReturnMod, WrongGateMod, DeadlineMod,
)


# --------------------------------------------------------------------------------------------------
# Fake corpora / schemas
# --------------------------------------------------------------------------------------------------


class MissingCorpusProvider(CorpusProvider):
    name: ClassVar[str] = "t_missing_corpus"
    feature_version: ClassVar[str] = TOY
    version: ClassVar[str] = "v0"

    def is_available(self, config=None) -> bool:
        return False

    def unavailable_hint(self, d: Path) -> str:
        return f"canonical corpus 't_missing_corpus' not found at {d}; build it with `malvalid corpus build`"


class BrokenCorpusProvider(CorpusProvider):
    name: ClassVar[str] = "t_broken_corpus"
    feature_version: ClassVar[str] = TOY
    version: ClassVar[str] = "v0"

    def is_available(self, config=None) -> bool:
        return True

    def load(self, config=None, *, verify: bool = True) -> Corpus:
        raise OSError("disk on fire")


class ToyV2Schema(ToySchema):
    name = "toy_v2"


# --------------------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------------------


@dataclass
class FakeSandbox:
    """What the monkeypatched seams return; tests tweak fields before calling run_gate."""

    decl: ModelDeclarations
    offset: float = 0.0
    trees: bool = True
    m0: type[Module] = FakeM0Pass
    inspect_error: Exception | None = None
    open_error: Exception | None = None
    calls: list[str] = field(default_factory=list)
    models: list[RecordingModel] = field(default_factory=list)
    policies: list[dict[str, Any]] = field(default_factory=list)

    def model_factory(self) -> RecordingModel:
        m = RecordingModel(_Detector(toy_booster(), self.offset), self.decl, trees=self.trees)
        if self.policies:
            m.scratch = Path(self.policies[-1]["run_dir"]) / "private" / "sandbox"
        self.models.append(m)
        return m


def install_fake_sandbox(monkeypatch: pytest.MonkeyPatch, fs: FakeSandbox) -> FakeSandbox:
    def _policy(cfg: GateConfig, *, run_dir: Path, allow_pickle: bool, extra_ro: Any) -> dict[str, Any]:
        fs.calls.append("policy")
        pol = {"run_dir": run_dir, "allow_pickle": allow_pickle, "extra_ro": list(extra_ro)}
        fs.policies.append(pol)
        return pol

    def _inspect(adapter: Path, policy: Any, *, class_name: str | None, model_paths_override: Any) -> ModelDeclarations:
        fs.calls.append("inspect")
        if fs.inspect_error is not None:
            raise fs.inspect_error
        if model_paths_override:
            return ModelDeclarations(**{**fs.decl.__dict__, "model_paths": tuple(str(p) for p in model_paths_override)})
        return fs.decl

    def _open(adapter: Path, policy: Any, *, declarations: ModelDeclarations, class_name: str | None) -> RecordingModel:
        fs.calls.append("open")
        if fs.open_error is not None:
            raise fs.open_error
        return fs.model_factory()

    monkeypatch.setattr(runner, "_make_policy", _policy)
    monkeypatch.setattr(runner, "_inspect_adapter", _inspect)
    monkeypatch.setattr(runner, "_open_model", _open)
    monkeypatch.setattr(runner, "_file_safety_cls", lambda: fs.m0)
    return fs


@pytest.fixture
def fake_plugins() -> Any:
    """Register the fake modules / corpora / schema; unregister afterwards."""
    for cls in FAKE_MODULES:
        registry.register("modules", cls.id, cls)
        cls.seen = []
    for m0 in (FakeM0Pass, FakeM0Abort, FakeM0Crash):
        m0.seen = []
    registry.register("corpora", MissingCorpusProvider.name, MissingCorpusProvider)
    registry.register("corpora", BrokenCorpusProvider.name, BrokenCorpusProvider)
    registry.register("feature_schemas", "toy_v2", ToyV2Schema)
    yield
    for cls in FAKE_MODULES:
        registry.unregister("modules", cls.id)
    registry.unregister("corpora", MissingCorpusProvider.name)
    registry.unregister("corpora", BrokenCorpusProvider.name)
    registry.unregister("feature_schemas", "toy_v2")


def make_cfg(
    modules: dict[str, str | None] | list[str] = (),  # type: ignore[assignment]
    *,
    corpus: str = TOY_CORPUS,
    m0: dict[str, Any] | None = None,
    verdict: dict[str, Any] | None = None,
    **runtime: Any,
) -> GateConfig:
    """A config enabling exactly ``modules`` ({id: gate or None for the module default})."""
    if not isinstance(modules, dict):
        modules = {m: None for m in modules}
    mods: dict[str, ModuleConfig] = {"file_safety": ModuleConfig(**({"enabled": True, "gate": "hard"} | (m0 or {})))}
    for mid, gate in modules.items():
        mods[mid] = ModuleConfig(enabled=True, **({"gate": gate} if gate else {}))
    rt = {"sandbox": False, **runtime}
    vd = {"required_for_ready": [], **(verdict or {})}
    return GateConfig(corpus=corpus, modules=mods, runtime=RuntimeConfig(**rt), verdict=VerdictConfig(**vd))


def module(report: dict[str, Any], module_id: str) -> dict[str, Any]:
    for m in report["modules"]:
        if m["module_id"] == module_id:
            return m
    raise AssertionError(f"{module_id} not in report modules {[m['module_id'] for m in report['modules']]}")


def run_opts(tmp_path: Path, cfg: GateConfig, **kw: Any) -> runner.RunOptions:
    kw.setdefault("write_html", False)
    return runner.RunOptions(adapter=tmp_path / "adapter" / "my_adapter.py", config=cfg,
                             out_dir=tmp_path / "run", **kw)


Factory = Callable[..., FakeSandbox]


def test_fakes_are_well_formed() -> None:
    """Self-check of the fakes (also keeps this file a valid pytest target on its own)."""
    ids = [c.id for c in FAKE_MODULES]
    codes = [c.code for c in FAKE_MODULES]
    assert len(set(ids)) == len(ids) and len(set(codes)) == len(codes)
    assert all(not c.code.startswith("M") for c in FAKE_MODULES)  # never collide with real modules
    model = FakeSandbox(decl=ModelDeclarations("toy_v1", "lightgbm", THRESHOLD, None, None)).model_factory()
    X = toy_corpus().take(np.arange(8))
    assert model.predict_proba(X).shape == (8,) and model.tree_ensemble() is not None
