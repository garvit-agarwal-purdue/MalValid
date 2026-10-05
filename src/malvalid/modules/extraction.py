"""M6 — model-extraction (model-stealing) susceptibility of the researcher's own detector.

Question answered: *if this model is exposed as a query service that returns verdicts, how many
queries does it take to clone it?* An attacker draws files from a query distribution (canonical
corpus pool rows that are **not** eval rows), submits them, records the model's hard verdicts at its
operating threshold, and trains a surrogate on those labels. For each query budget we report the
surrogate's **fidelity** (agreement with the model's verdicts) and **accuracy** (agreement with the
true labels) on held-out eval rows. The gate is fidelity at ``fidelity_budget`` queries. When a corpus defines
no pool rows outside its eval role, the held-out rows are reserved first and the queries come from
the rest (see :func:`select_query_and_eval_rows`).

ART in 1.20 offers three extraction attacks. ``KnockoffNets`` with ``sampling_strategy="random"``
is exactly the query-and-fit procedure above and accepts any ART ``ClassifierMixin`` as the thieved
classifier, so we run it with a thin shim around the (non-neural) surrogate. ART's own estimator
wrappers cannot be the thief: ``LightGBMClassifier.fit`` raises ``NotImplementedError`` (and the
wrapper rejects binary boosters), and ``SklearnClassifier.fit`` forwards the neural-network-only
``batch_size`` / ``nb_epochs`` / ``verbose`` keywords to the sklearn estimator. The other attacks do
not apply to tree/tabular surrogates — see :data:`ART_APPLICABILITY`. If ART is unavailable the
module falls back to the equivalent direct query-and-fit procedure and records that.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from malvalid.core import ConfigError, GateCheck, GateMode, Module, ModuleResult, Requirement
from malvalid.corpora.base import LABEL_UNLABELED, ROLE_EVAL, ROLE_POOL

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext
    from malvalid.corpora.base import Corpus

log = logging.getLogger("malvalid.extraction")

SURROGATES: tuple[str, ...] = ("lightgbm", "random_forest", "logistic_regression")
_SCORE_CHUNK = 20000

METHOD_ART = "art.KnockoffNets(sampling_strategy='random')"
METHOD_DIRECT = "direct query-and-fit"

ART_APPLICABILITY: dict[str, str] = {
    "KnockoffNets(random)": (
        "used: random query selection + fit of a thieved classifier on the victim's hard labels; "
        "the surrogate is wrapped in a minimal ART ClassifierMixin shim because ART's "
        "LightGBMClassifier.fit raises NotImplementedError (and rejects binary boosters) and "
        "SklearnClassifier.fit forwards neural-network-only fit kwargs (batch_size, nb_epochs, "
        "verbose) to the sklearn estimator"
    ),
    "KnockoffNets(adaptive)": (
        "not applicable: it refits the thieved classifier after every single query with a "
        "reward computed from its probability outputs (designed for neural networks); refitting "
        "a tree surrogate per query is infeasible at these budgets"
    ),
    "CopycatCNN": (
        "not applicable beyond KnockoffNets(random): the same random query-and-fit procedure, "
        "specialised for CNN image classifiers trained with epochs/batches"
    ),
    "FunctionallyEquivalentExtraction": (
        "not applicable: requires the victim to be a two-layer ReLU neural network with logit "
        "outputs (ART NeuralNetworkMixin); a black-box detector does not satisfy that"
    ),
}


# --------------------------------------------------------------------------------------------------
# Surrogates
# --------------------------------------------------------------------------------------------------


class _ConstantModel:
    """Surrogate when every stolen label is the same class."""

    def __init__(self, label: int):
        self.label = int(label)

    def predict_hard(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.label, dtype=np.int8)


class _FittedSurrogate:
    def __init__(self, predict_hard: Callable[[np.ndarray], np.ndarray]):
        self.predict_hard = predict_hard


def fit_surrogate(kind: str, X: np.ndarray, y: np.ndarray, *, seed: int, threads: int) -> Any:
    """Fit a surrogate of type ``kind`` on ``(X, y)``; returns an object with ``predict_hard``."""
    y = np.asarray(y).astype(np.int64).reshape(-1)
    classes = np.unique(y)
    if classes.size < 2:
        return _ConstantModel(int(classes[0]) if classes.size else 0)
    n = int(X.shape[0])
    if kind == "lightgbm":
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": max(1, min(20, n // 50)),
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": int(seed),
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": int(threads),
        }
        booster = lgb.train(params, lgb.Dataset(X, y, params={"verbose": -1}), num_boost_round=200)
        return _FittedSurrogate(lambda A: (booster.predict(A) >= 0.5).astype(np.int8))
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=200, random_state=int(seed), n_jobs=int(threads))
        rf.fit(X, y)
        return _FittedSurrogate(lambda A: rf.predict(A).astype(np.int8))
    if kind == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=int(seed)))
        lr.fit(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y)
        return _FittedSurrogate(
            lambda A: lr.predict(np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.int8)
        )
    raise ValueError(f"extraction.surrogate must be one of {list(SURROGATES)} (got {kind!r})")


# --------------------------------------------------------------------------------------------------
# ART plumbing
# --------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _seeded_global_numpy(seed: int) -> Iterator[None]:
    """ART's KnockoffNets draws its query order from the global NumPy RNG; seed it, then restore."""
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def _import_art() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ART warns about optional DL frameworks at import
        from art.attacks.extraction import KnockoffNets
        from art.estimators.classification import BlackBoxClassifier
        from art.estimators.classification.classifier import ClassifierMixin
        from art.estimators.estimator import BaseEstimator
    return {"KnockoffNets": KnockoffNets, "BlackBoxClassifier": BlackBoxClassifier,
            "ClassifierMixin": ClassifierMixin, "BaseEstimator": BaseEstimator}


def _make_thief_class(art: dict[str, Any]) -> type:
    """An ART thieved-classifier shim that fits a non-neural surrogate on one-hot stolen labels."""

    class SurrogateThief(art["ClassifierMixin"], art["BaseEstimator"]):  # type: ignore[misc, valid-type]
        def __init__(self, fit_fn: Callable[[np.ndarray, np.ndarray], Any], input_shape: tuple[int, ...]):
            super().__init__(model=None, clip_values=None)
            self._fit_fn = fit_fn
            self._shape = input_shape
            self.nb_classes = 2
            self.surrogate: Any = None
            self.n_fit = 0

        @property
        def input_shape(self) -> tuple[int, ...]:
            return self._shape

        def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:  # NN kwargs ignored
            labels = np.argmax(np.asarray(y), axis=1) if np.ndim(y) == 2 else np.asarray(y)
            self.surrogate = self._fit_fn(np.asarray(x, dtype=np.float32), labels)
            self.n_fit = int(np.asarray(x).shape[0])

        def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
            hard = self.surrogate.predict_hard(np.asarray(x, dtype=np.float32)).astype(np.int64)
            return np.eye(2, dtype=np.float32)[hard]

        def save(self, filename: str, path: str | None = None) -> None:
            raise NotImplementedError

    return SurrogateThief


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _verdicts(ctx: "RunContext", X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=np.int8)
    for s in range(0, X.shape[0], _SCORE_CHUNK):
        ctx.check_deadline()
        out[s : s + _SCORE_CHUNK] = (ctx.score(X[s : s + _SCORE_CHUNK]) >= ctx.threshold).astype(np.int8)
    return out


def _fidelity_anchors(t: float) -> tuple[float, float]:
    """``ideal = 0.80``, ``floor = 0.995`` (kept strictly on either side of ``t``)."""
    ideal = 0.80 if t > 0.80 else t - 0.05
    floor = 0.995 if t < 0.995 else t + max((1.0 - t) / 2.0, 1e-4)
    return ideal, floor


def _budgets(p: dict[str, Any]) -> tuple[list[int], int]:
    raw = p.get("query_budgets") or []
    if isinstance(raw, (int, float)):
        raw = [raw]
    budgets = sorted({int(b) for b in raw} | {int(p["fidelity_budget"])})
    if any(b < 1 for b in budgets):
        raise ValueError(f"extraction.query_budgets / fidelity_budget must be positive (got {budgets})")
    return budgets, int(p["fidelity_budget"])


def _threads(ctx: "RunContext") -> int:
    n = int(getattr(ctx.config.runtime, "threads", 1) or 1)
    return max(1, min(n, os.cpu_count() or 1))


QUERY_POOL_NON_EVAL = "pool_non_eval"
QUERY_POOL_EVAL_REMAINDER = "eval_remainder"


def select_query_and_eval_rows(
    corpus: "Corpus",
    *,
    training_hashes: frozenset[str] | None,
    n_eval: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | str:
    """Pick held-out eval rows (fidelity/accuracy) and a disjoint attacker query pool.

    Queries come from pool-role rows outside the eval role. When the corpus's pool role has no
    rows outside its eval role (e.g. ``ember_v3_2024``, whose pool and eval are both the EMBER2024
    test set), the held-out sample is drawn first and the queries come from the remaining pool
    rows. In both cases query rows never share a row or a sha256 with the held-out rows.

    Returns ``(eval_idx, eval_pool, query_pool, info)`` or a skip reason.
    """
    eval_all = corpus.indices(role=ROLE_EVAL, labeled_only=False)
    eval_pool = eval_all
    if training_hashes:
        eval_pool = corpus.indices(role=ROLE_EVAL, labeled_only=False, exclude_hashes=training_hashes)
    pool_all = corpus.indices(role=ROLE_POOL, labeled_only=False)
    query_pool = np.setdiff1d(pool_all, eval_all, assume_unique=True)
    notes: list[str] = []
    source = QUERY_POOL_NON_EVAL
    n_hold = n_eval
    if query_pool.size == 0:
        source = QUERY_POOL_EVAL_REMAINDER
        half = int(eval_pool.size // 2)
        if n_hold is None or n_hold > half:
            n_hold = half
    eval_idx = corpus.subsample(eval_pool, n_hold, rng)
    if eval_idx.size == 0:
        return "no held-out eval rows in the canonical corpus to measure fidelity on"
    if source == QUERY_POOL_EVAL_REMAINDER:
        query_pool = np.setdiff1d(pool_all, eval_idx, assume_unique=True)
        notes.append(
            f"The corpus's pool role ({corpus.role_splits(ROLE_POOL)}) has no rows outside its eval "
            f"role ({corpus.role_splits(ROLE_EVAL)}), so {eval_idx.size} held-out eval rows were "
            f"reserved first and the attacker's queries were drawn from the remaining pool rows "
            f"(same distribution, disjoint rows and sha256s)."
        )
    held_hashes = np.unique(corpus.sha256[eval_idx])
    dup = np.isin(corpus.sha256[query_pool], held_hashes)
    n_dup = int(np.sum(dup))
    if n_dup:
        query_pool = query_pool[~dup]
        notes.append(f"Excluded {n_dup} query-pool rows whose sha256 also appears among the held-out "
                     f"eval rows (duplicates would inflate fidelity).")
    if query_pool.size == 0:
        return ("no query distribution: the canonical corpus has no pool rows left after "
                "reserving the held-out eval rows")
    info: dict[str, Any] = {
        "source": source,
        "pool_splits": corpus.role_splits(ROLE_POOL),
        "eval_splits": corpus.role_splits(ROLE_EVAL),
        "n_query_pool": int(query_pool.size),
        "n_eval_candidates": int(eval_pool.size),
        "excluded_duplicate_sha256": n_dup,
        "notes": notes,
    }
    return eval_idx, eval_pool, query_pool, info


# --------------------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------------------


class ExtractionModule(Module):
    id = "extraction"
    code = "M6"
    title = "Model-extraction susceptibility"
    description = (
        "How cheaply the model can be cloned through its verdicts: queries corpus files disjoint "
        "from the held-out eval rows, trains a surrogate on the model's hard labels at each query "
        "budget, and reports the surrogate's fidelity (agreement with your model) and accuracy on "
        "held-out eval rows. Only relevant if the model will be exposed as a queryable service."
    )
    requires = (Requirement.QUERY_ONLY,)
    default_gate = GateMode.WARN
    default_params = {
        "query_budgets": [100, 1000, 10000, 50000],
        "fidelity_budget": 10000,
        "max_fidelity": 0.95,
        "surrogate": "lightgbm",
        "n_eval": 5000,
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        kind = str(params.get("surrogate"))
        if kind not in SURROGATES:
            raise ConfigError(f"extraction.surrogate must be one of {list(SURROGATES)} (got {kind!r})")
        try:
            float(params["max_fidelity"])
            _budgets(params)
            if params.get("n_eval") is not None:
                int(params["n_eval"])
        except (TypeError, ValueError, KeyError) as e:
            raise ConfigError(f"{cls.id}: invalid parameter value: {e}") from e

    def run(self, ctx: "RunContext") -> ModuleResult:
        t_start = time.monotonic()
        q_start = int(getattr(ctx.model, "query_count", 0) or 0)
        corpus = ctx.corpus
        if corpus is None:
            return self.skip(ctx, f"no query distribution: {ctx.missing_reason(Requirement.FEATURE_SPACE)}")
        if corpus.feature_version != ctx.model.declarations.feature_version:
            return self.skip(
                ctx,
                f"no query distribution: the canonical corpus is in feature_version "
                f"{corpus.feature_version!r} but the model expects "
                f"{ctx.model.declarations.feature_version!r}",
            )
        p = ctx.params
        t = float(p["max_fidelity"])
        kind = str(p["surrogate"])
        if kind not in SURROGATES:
            raise ValueError(f"extraction.surrogate must be one of {list(SURROGATES)} (got {kind!r})")
        budgets, fid_budget = _budgets(p)
        n_eval = p.get("n_eval")
        threads = _threads(ctx)

        notes: list[str] = []
        sel = select_query_and_eval_rows(
            corpus, training_hashes=ctx.training_hashes,
            n_eval=None if n_eval is None else int(n_eval), rng=ctx.rng,
        )
        if isinstance(sel, str):
            return self.skip(ctx, sel)
        eval_idx, eval_pool, query_pool, sel_info = sel
        notes.extend(sel_info.pop("notes"))

        eff = sorted({min(b, int(query_pool.size)) for b in budgets})
        capped = [b for b in budgets if b > query_pool.size]
        fid_eff = min(fid_budget, int(query_pool.size))
        order = ctx.rng.permutation(query_pool)[: max(eff)]
        X_query = corpus.take(order)
        X_eval = corpus.take(eval_idx)
        y_eval = corpus.label[eval_idx]
        labeled = y_eval != LABEL_UNLABELED

        v_eval = _verdicts(ctx, X_eval)
        model_acc = float(np.mean(v_eval[labeled] == y_eval[labeled])) if labeled.any() else None
        majority = float(max(np.mean(v_eval), 1.0 - np.mean(v_eval)))

        method, art_error = METHOD_DIRECT, None
        art: dict[str, Any] | None = None
        try:
            art = _import_art()
            method = METHOD_ART
        except Exception as e:  # pragma: no cover - ART is a core dependency
            art_error = f"{type(e).__name__}: {e}"
            log.warning("ART unavailable for M6, using the direct procedure: %s", e)

        rows: list[dict[str, Any]] = []
        direct_labels: np.ndarray | None = None
        for b in eff:
            ctx.check_deadline()
            t_b = time.monotonic()
            seed = int(ctx.rng.integers(0, 2**31 - 1))
            fit_fn = (lambda X, y, _s=seed: fit_surrogate(kind, X, y, seed=_s, threads=threads))
            surrogate = None
            if art is not None:
                try:
                    surrogate = self._extract_with_art(ctx, art, X_query[:b], fit_fn, seed)
                except Exception as e:
                    art_error = f"{type(e).__name__}: {e}"
                    log.warning("ART KnockoffNets failed (%s); falling back to direct query-and-fit", e)
                    art, method = None, METHOD_DIRECT
            if surrogate is None:
                if direct_labels is None:
                    direct_labels = _verdicts(ctx, X_query)
                surrogate = fit_fn(X_query[:b], direct_labels[:b])
            s_eval = surrogate.predict_hard(X_eval)
            fid = float(np.mean(s_eval == v_eval))
            acc = float(np.mean(s_eval[labeled] == y_eval[labeled])) if labeled.any() else None
            rows.append({"budget": b, "fidelity": fid, "accuracy": acc,
                         "seconds": round(time.monotonic() - t_b, 3)})
            log.info("M6: budget %d -> fidelity %.4f", b, fid)

        by_budget = {r["budget"]: r for r in rows}
        fid_at = by_budget[fid_eff]["fidelity"]
        if fid_eff < fid_budget:
            # Fidelity grows with the budget, so a failure at fewer queries is a failure at more;
            # a pass at fewer queries proves nothing about fidelity_budget queries.
            value: float | None = fid_at if fid_at > t else None
            notes.append(
                f"The query pool has only {query_pool.size} non-eval rows, fewer than "
                f"fidelity_budget={fid_budget}; fidelity at {fid_eff} queries is {fid_at:.4f}, a lower "
                f"bound for {fid_budget} queries"
                + (", and already above the limit." if value is not None
                   else ", so the gate cannot be evaluated.")
            )
        else:
            value = fid_at
        ideal, floor = _fidelity_anchors(t)
        check = GateCheck.evaluate(
            "surrogate_fidelity", value, "<=", t, metric="fidelity",
            description=f"agreement between a surrogate trained on {fid_budget} hard-label queries "
                        f"and the model's verdicts on held-out eval rows",
            ideal=ideal, floor=floor,
        )
        reach = next((r["budget"] for r in rows if r["fidelity"] > t), None)
        metrics: dict[str, Any] = {
            "fidelity": fid_at,
            "fidelity_budget": fid_budget,
            "fidelity_budget_effective": fid_eff,
            "accuracy_at_fidelity_budget": by_budget[fid_eff]["accuracy"],
            "model_accuracy": model_acc,
            "majority_baseline_fidelity": majority,
            "max_fidelity_observed": max(r["fidelity"] for r in rows),
            "queries_to_exceed_max_fidelity": reach,
            "budgets": [r["budget"] for r in rows],
            "fidelities": [r["fidelity"] for r in rows],
            "accuracies": [r["accuracy"] for r in rows],
            "surrogate": kind,
            "method": method,
            "n_eval": int(eval_idx.size),
            "n_query_pool": int(query_pool.size),
            "query_pool_source": sel_info["source"],
        }

        series = [{"label": "fidelity (agreement with your model)",
                   "x": [r["budget"] for r in rows], "y": [r["fidelity"] for r in rows]}]
        if labeled.any():
            series.append({"label": "surrogate accuracy (vs true labels)",
                           "x": [r["budget"] for r in rows], "y": [r["accuracy"] for r in rows]})
        refs: list[dict[str, Any]] = [
            {"axis": "y", "value": t, "label": f"max_fidelity {t:g}"},
            {"axis": "x", "value": float(fid_budget), "label": f"fidelity_budget {fid_budget}"},
        ]
        if model_acc is not None:
            refs.append({"axis": "y", "value": model_acc, "label": "your model's accuracy"})
        ctx.artifacts.add_chart(
            self.id, "fidelity_vs_queries", title="Surrogate fidelity vs number of queries",
            series=series, kind="line", xlabel="queries to the model (hard verdicts)",
            ylabel="agreement on held-out eval rows", xscale="log", ylim=(0.0, 1.0),
            reference_lines=refs,
        )

        if capped:
            notes.append(
                f"Budgets {capped} exceed the {query_pool.size}-row query pool and were capped at "
                f"{query_pool.size} queries."
            )
        if eval_pool.size > eval_idx.size:
            notes.append(f"Fidelity measured on {eval_idx.size} of {eval_pool.size} eval rows (n_eval).")
        if method == METHOD_DIRECT:
            notes.append(
                "ART's extraction attack could not be used"
                + (f" ({art_error})" if art_error else "")
                + "; the equivalent direct query-and-fit procedure was used instead."
            )
        notes.append(
            "Threat model: the attacker sees only the model's hard verdicts (malicious/benign at the "
            "operating threshold) for files drawn from the canonical corpus's query pool "
            "(rows disjoint from the held-out eval rows). Returning raw scores, or an attacker with "
            "better query selection, would make extraction cheaper."
        )
        notes.append(
            "Only relevant if the model will be exposed as a queryable service (e.g. a cloud "
            "lookup or scanning API). A cloned surrogate lets an attacker probe for evasions "
            "offline and free-rides on your training data and labeling effort. Rate limits, query "
            "auditing and coarse outputs are the usual mitigations."
        )

        details = {
            "budgets": rows,
            "method": method,
            "art_error": art_error,
            "art_applicability": ART_APPLICABILITY,
            "surrogate": kind,
            "budgets_capped": capped,
            "query_pool": sel_info,
            "total_model_queries": int(getattr(ctx.model, "query_count", 0) or 0) - q_start,
            "duration_s": round(time.monotonic() - t_start, 3),
        }
        finding = (
            f"A {kind} surrogate trained on {fid_eff:,} hard-label queries agrees with your model on "
            f"{fid_at:.1%} of held-out samples (limit {t:.1%})"
        )
        if value is None:
            finding += (f"; the query pool is smaller than fidelity_budget={fid_budget:,}, so the "
                        f"gate could not be evaluated.")
        elif check.passed:
            finding += (f": at this budget a clone built from its verdicts still disagrees with it on "
                        f"{1.0 - fid_at:.1%} of files.")
        else:
            finding += (": the model can be cloned cheaply if it is exposed as a query service"
                        + (f" (fidelity first exceeds the limit at {reach:,} queries)." if reach else "."))
        return self.result(ctx, finding=finding, checks=[check], metrics=metrics, details=details,
                           notes=notes)

    @staticmethod
    def _extract_with_art(ctx: "RunContext", art: dict[str, Any], X: np.ndarray,
                          fit_fn: Callable[[np.ndarray, np.ndarray], Any], seed: int) -> Any:
        """One budget through ART KnockoffNets (random strategy); returns the fitted surrogate."""
        b = int(X.shape[0])
        threshold = ctx.threshold

        def victim_fn(A: np.ndarray) -> np.ndarray:  # hard verdicts only, as a service would return
            v = (ctx.score(np.asarray(A, dtype=np.float32)) >= threshold).astype(np.int64)
            return np.eye(2, dtype=np.float32)[v]

        victim = art["BlackBoxClassifier"](predict_fn=victim_fn, input_shape=(X.shape[1],), nb_classes=2)
        thief = _make_thief_class(art)(fit_fn, (X.shape[1],))
        attack = art["KnockoffNets"](
            classifier=victim, batch_size_fit=b, batch_size_query=b, nb_epochs=1, nb_stolen=b,
            sampling_strategy="random", verbose=False, use_probability=False,
        )
        with _seeded_global_numpy(seed):
            stolen = attack.extract(X, thieved_classifier=thief)
        if stolen.surrogate is None or stolen.n_fit != b:
            raise RuntimeError(f"KnockoffNets fitted {stolen.n_fit} rows, expected {b}")
        return stolen.surrogate
