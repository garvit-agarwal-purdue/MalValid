"""Test helpers: an in-process model handle, a tiny toy feature schema/corpus, and a context
builder. Used by malvalid's own test suite and handy for plugin authors.

Nothing here is used by a production run; ``InProcessModel`` bypasses the sandbox and must only
wrap trusted objects.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from malvalid import registry
from malvalid.context import (
    ArtifactStore,
    ModelDeclarations,
    RunContext,
    compute_capabilities,
    module_seed,
)
from malvalid.core import AdapterContractError, FeaturizeUnavailable, GateMode, Module
from malvalid.corpora.base import Corpus, CorpusProvider, compute_content_hash
from malvalid.loaders.trees import TreeEnsemble, from_lightgbm_dump
from malvalid.schemas.base import Controllability, FeatureGroup, FeatureSchema

# --------------------------------------------------------------------------------------------------
# Output validation shared by model handles
# --------------------------------------------------------------------------------------------------


def validate_proba(p: Any, n: int) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        raise AdapterContractError(
            "predict_proba returned (n, 2); the contract is (n,) malicious-class scores"
        )
    arr = arr.reshape(-1) if arr.ndim == 2 and arr.shape[1] == 1 else arr
    if arr.shape != (n,):
        raise AdapterContractError(f"predict_proba returned shape {arr.shape}, expected ({n},)")
    if not np.all(np.isfinite(arr)):
        raise AdapterContractError("predict_proba returned non-finite values")
    if arr.size and (arr.min() < -1e-9 or arr.max() > 1 + 1e-9):
        raise AdapterContractError(
            f"predict_proba returned values outside [0, 1] (min={arr.min()}, max={arr.max()})"
        )
    return np.clip(arr, 0.0, 1.0)


def validate_pred(p: Any, n: int) -> np.ndarray:
    arr = np.asarray(p)
    arr = arr.reshape(-1) if arr.ndim == 2 and arr.shape[1] == 1 else arr
    if arr.shape != (n,):
        raise AdapterContractError(f"predict returned shape {arr.shape}, expected ({n},)")
    if not np.all(np.isin(arr, (0, 1))):
        raise AdapterContractError("predict returned values outside {0, 1}")
    return arr.astype(np.int8)


# --------------------------------------------------------------------------------------------------
# In-process model handle
# --------------------------------------------------------------------------------------------------


class InProcessModel:
    """A :class:`~malvalid.context.ModelHandle` over a trusted in-process detector object.

    ``detector`` must provide ``predict_proba(X)`` and ``predict(X)``; optional ``featurize``,
    ``tree_ensemble()`` or ``native_model``.
    """

    def __init__(self, detector: Any, declarations: ModelDeclarations):
        self.detector = detector
        self.declarations = declarations
        self._queries = 0
        self._trees: TreeEnsemble | None | bool = False  # False = not computed yet

    @classmethod
    def from_lightgbm(
        cls,
        booster: Any,
        *,
        threshold: float,
        feature_version: str,
        training_hashes_path: str | None = None,
        training_cutoff: str | None = None,
        model_paths: tuple[str, ...] = (),
    ) -> "InProcessModel":
        class _Det:
            native_model = booster

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                return booster.predict(X)

            def predict(self, X: np.ndarray) -> np.ndarray:
                return (booster.predict(X) >= threshold).astype(np.int8)

        decl = ModelDeclarations(
            feature_version=feature_version,
            model_kind="lightgbm",
            operating_threshold=float(threshold),
            training_hashes_path=training_hashes_path,
            training_cutoff=training_cutoff,
            model_paths=model_paths,
            class_name="InProcessLightGBM",
        )
        return cls(_Det(), decl)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._queries += X.shape[0]
        return validate_proba(self.detector.predict_proba(X), X.shape[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._queries += X.shape[0]
        return validate_pred(self.detector.predict(X), X.shape[0])

    def _schema(self) -> FeatureSchema | None:
        try:
            return registry.get_schema(self.declarations.feature_version)
        except KeyError:
            return None

    def has_featurize(self) -> bool:
        if callable(getattr(self.detector, "featurize", None)):
            return True
        sch = self._schema()
        return bool(sch is not None and sch.featurize_available())

    def featurize(self, raw: bytes) -> np.ndarray:
        if callable(getattr(self.detector, "featurize", None)):
            return np.asarray(self.detector.featurize(raw), dtype=np.float32).reshape(-1)
        sch = self._schema()
        if sch is None or not sch.featurize_available():
            raise FeaturizeUnavailable("detector has no featurize() and the schema has no extractor")
        return np.asarray(sch.featurize(raw), dtype=np.float32).reshape(-1)

    def tree_ensemble(self) -> TreeEnsemble | None:
        if self._trees is False:
            self._trees = None
            fn = getattr(self.detector, "tree_ensemble", None)
            native = getattr(self.detector, "native_model", None)
            try:
                if callable(fn):
                    self._trees = fn()
                elif native is not None and type(native).__module__.startswith("lightgbm"):
                    booster = getattr(native, "booster_", native)
                    self._trees = from_lightgbm_dump(booster.dump_model())
                elif native is not None:
                    from malvalid.loaders.base import find_loader_for_native

                    ld = find_loader_for_native(native)
                    self._trees = ld.tree_ensemble(native) if ld else None
            except Exception as e:  # tree access is optional; degrade
                logging.getLogger("malvalid").info("tree access unavailable: %s", e)
                self._trees = None
        return self._trees  # type: ignore[return-value]

    @property
    def query_count(self) -> int:
        return self._queries

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------------------------------
# Toy feature schema (32 dims) with every controllability level
# --------------------------------------------------------------------------------------------------

TOY_DIM = 32
_HIST = slice(1, 8)
_STR = slice(8, 16)
_HDR = slice(16, 24)
_CODE = slice(24, 32)


class ToySchema(FeatureSchema):
    name = "toy_v1"
    dim = TOY_DIM
    description = "32-feature toy schema for tests (not a real PE feature space)."

    def groups(self) -> list[FeatureGroup]:
        C = Controllability
        return [
            FeatureGroup("general.size", 0, 1, C.APPEND_ONLY, "file size in bytes"),
            FeatureGroup("histogram", 1, 8, C.DERIVED, "normalized byte histogram"),
            FeatureGroup("strings", 8, 16, C.APPEND_ONLY, "string counts"),
            FeatureGroup("header", 16, 24, C.CONTROLLABLE, "free header fields"),
            FeatureGroup("code", 24, 32, C.FIXED, "code/semantics features"),
        ]


def _month_add(d: dt.date, k: int) -> dt.date:
    m = d.month - 1 + k
    return dt.date(d.year + m // 12, m % 12 + 1, 1)


def make_toy_corpus(
    n: int = 4000,
    *,
    seed: int = 0,
    start: dt.date = dt.date(2017, 1, 1),
    months: int = 24,
    train_months: int = 12,
    drift: float = 0.6,
    malicious_fraction: float = 0.5,
    challenge_fraction: float = 0.05,
    string_signal: float = 1.0,
) -> Corpus:
    """A deterministic toy corpus in ``toy_v1`` space.

    Signal lives in FIXED code features (decaying over time by ``drift``), CONTROLLABLE header
    features and APPEND_ONLY strings (scaled by ``string_signal``). Splits: ``train`` (first
    ``train_months``), ``test`` (later), ``challenge`` (benign-looking malicious from test time).
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < malicious_fraction).astype(np.int8)
    month = rng.integers(0, months, size=n)
    ts = np.array(
        [np.datetime64(_month_add(start, int(m)), "D") + np.timedelta64(int(rng.integers(0, 28)), "D") for m in month]
    )
    tfrac = month / max(months - 1, 1)
    X = np.zeros((n, TOY_DIM), dtype=np.float64)
    X[:, 0] = np.exp(rng.normal(11.5, 1.0, n))  # ~100KB
    hist = rng.dirichlet(np.ones(7) * 2.0, size=n)
    hist[y == 1, 6] += 0.15
    X[:, _HIST] = hist / hist.sum(1, keepdims=True)
    X[:, _STR] = rng.poisson(20, size=(n, 8)).astype(np.float64)
    X[y == 1, 8] += rng.poisson(15 * string_signal, size=int(y.sum()))
    X[:, _HDR] = rng.normal(0, 1, size=(n, 8))
    X[y == 1, 16] += 1.0
    X[:, _CODE] = rng.normal(0, 1, size=(n, 8))
    strength = 1.8 * (1 - drift * tfrac)
    X[:, 24] += np.where(y == 1, strength, 0.0)
    X[:, 25] += np.where(y == 1, 0.8 * strength, 0.0)
    split = np.where(month < train_months, "train", "test").astype("<U16")
    mal_test = np.flatnonzero((y == 1) & (split == "test"))
    k = int(round(challenge_fraction * n))
    if k and mal_test.size:
        ch = rng.choice(mal_test, size=min(k, mal_test.size), replace=False)
        X[ch, 24:26] -= 1.2
        X[ch, 16] -= 1.0
        split[ch] = "challenge"
    X = X.astype(np.float32)
    sha = np.array([hashlib.sha256(f"toy-{seed}-{i}".encode()).hexdigest() for i in range(n)], dtype="<U64")
    manifest: dict[str, Any] = {
        "format": "malvalid-corpus/1",
        "name": "toy_v1_corpus",
        "version": f"seed{seed}",
        "feature_version": "toy_v1",
        "dim": TOY_DIM,
        "n": n,
        "synthetic": True,
        "roles": {"eval": ["test"], "temporal": ["train", "test"], "challenge": ["challenge"]},
        "files": {"X": hashlib.sha256(X.tobytes()).hexdigest()},
        "source": {"generator": "malvalid.testing.make_toy_corpus", "seed": seed},
    }
    manifest["content_hash"] = compute_content_hash(manifest)
    return Corpus(
        name="toy_v1_corpus",
        version=manifest["version"],
        feature_version="toy_v1",
        content_hash=manifest["content_hash"],
        manifest=manifest,
        X=X,
        sha256=sha,
        label=y,
        timestamp=ts.astype("datetime64[D]"),
        split=split,
        synthetic=True,
    )


class ToyCorpusProvider(CorpusProvider):
    """Registered as ``toy_v1_corpus``; generates the toy corpus in memory (no files)."""

    name: ClassVar[str] = "toy_v1_corpus"
    feature_version: ClassVar[str] = "toy_v1"
    version: ClassVar[str] = "seed0"
    description: ClassVar[str] = "In-memory toy corpus for tests."
    synthetic: ClassVar[bool] = True

    def is_available(self, config=None) -> bool:
        return True

    def load(self, config=None, *, verify: bool = True) -> Corpus:
        return make_toy_corpus()


def register_toy_plugins() -> None:
    registry.register("feature_schemas", "toy_v1", ToySchema)
    registry.register("corpora", "toy_v1_corpus", ToyCorpusProvider)


register_toy_plugins()


def train_toy_lgbm(
    corpus: Corpus,
    *,
    splits: tuple[str, ...] = ("train",),
    rounds: int = 60,
    params: dict[str, Any] | None = None,
    row_filter: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> Any:
    import lightgbm as lgb

    idx = corpus.indices(splits=splits)
    if row_filter is not None:
        idx = idx[row_filter[idx]]
    X = corpus.take(idx)
    y = corpus.label[idx] if labels is None else labels[idx]
    p = {"objective": "binary", "verbose": -1, "num_leaves": 15, "min_data_in_leaf": 10, "seed": 0,
         "deterministic": True, "num_threads": 2}
    p.update(params or {})
    return lgb.train(p, lgb.Dataset(X, y), rounds)


def training_hashes_for(corpus: Corpus, splits: tuple[str, ...] = ("train",)) -> frozenset[str]:
    return frozenset(corpus.sha256[corpus.indices(splits=splits)].tolist())


def make_context(
    module_cls: type[Module],
    *,
    model: Any,
    corpus: Corpus | None,
    schema: FeatureSchema | None = None,
    params: dict[str, Any] | None = None,
    gate: GateMode | None = None,
    training_hashes: frozenset[str] | None = None,
    training_cutoff: dt.date | None = None,
    sample_dir: Path | None = None,
    run_dir: Path | None = None,
    seed: int = 0,
    deadline: float | None = None,
) -> RunContext:
    """Build a RunContext for unit-testing one module (mirrors what the runner does)."""
    import tempfile

    from malvalid.config import GateConfig

    schema = schema or registry.get_schema(model.declarations.feature_version)
    run_dir = Path(run_dir or tempfile.mkdtemp(prefix="malvalid-test-"))
    private = run_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    trees = model.tree_ensemble() if model is not None else None
    caps = compute_capabilities(
        model=model,
        schema=schema,
        corpus=corpus,
        training_hashes=training_hashes,
        training_cutoff=training_cutoff,
        sample_dir=sample_dir,
        tree_access=trees is not None,
    )
    merged = dict(module_cls.default_params)
    merged.update(params or {})
    s = module_seed(seed, module_cls.id)
    return RunContext(
        config=GateConfig(),
        module_id=module_cls.id,
        params=merged,
        gate=gate or module_cls.default_gate,
        model=model,
        schema=schema,
        corpus=corpus,
        training_hashes=training_hashes,
        training_cutoff=training_cutoff,
        sample_dir=sample_dir,
        run_dir=run_dir,
        private_dir=private,
        artifacts=ArtifactStore(private),
        seed=s,
        rng=np.random.default_rng(s),
        capabilities=caps,
        deadline=deadline,
    )
