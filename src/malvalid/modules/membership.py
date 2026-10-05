"""M4 — black-box membership inference against the researcher's own detector.

Question answered: *can someone who can only query the model tell which files were in its
training set?* Members are canonical-corpus rows whose sha256 appears in the adapter's training
manifest; non-members are labeled corpus rows that are not in it, sampled to match the members'
months and labels so that the attack measures memorization rather than distribution shift.

Three attacks run through the model's ``predict_proba`` interface only (via ``ctx.score``):

* ``loss_threshold`` — the loss/confidence-threshold attack (Yeom et al., 2018): a sample is called
  a member when the model's cross-entropy loss on its true label is below a threshold chosen on the
  attack's training half;
* ``art_rf`` / ``art_gb`` — ART's :class:`MembershipInferenceBlackBox` with random-forest and
  gradient-boosting attack models, fed the model's two-class probabilities plus the true label
  through an ART :class:`BlackBoxClassifier` wrapper around ``ctx.score``.

Every attack is fit on ``1 - test_fraction`` of the member/non-member pairs and evaluated on the
rest. The gate uses the worst case (largest) membership advantage = TPR − FPR at each attack's own
decision threshold.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from malvalid.core import ConfigError, GateCheck, GateMode, Module, ModuleResult, Requirement
from malvalid.corpora.base import LABEL_UNLABELED, ROLE_CHALLENGE, ROLE_POOL

CALIBRATION_SPLIT = "calibration"  # == malvalid.submission.CALIBRATION_SPLIT (not imported: cycle)

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext
    from malvalid.corpora.base import Corpus

log = logging.getLogger("malvalid.membership")

ATTACKS: tuple[str, ...] = ("loss_threshold", "art_rf", "art_gb")
ATTACK_LABELS: dict[str, str] = {
    "loss_threshold": "loss-threshold attack",
    "art_rf": "ART black-box attack (random forest)",
    "art_gb": "ART black-box attack (gradient boosting)",
}
_SCORE_CHUNK = 20000
_HIST_EDGES = np.linspace(-8.0, 2.0, 41)  # log10(cross-entropy loss) bins for member_score_hist

DESIGN_TIME_MATCHED = "time_matched"
DESIGN_NEAREST_TIME = "nearest_time"
DESIGN_LABEL_MATCHED = "label_matched"


# --------------------------------------------------------------------------------------------------
# Member / non-member selection
# --------------------------------------------------------------------------------------------------


@dataclass
class MembershipSample:
    """Paired member / non-member corpus rows (``member_idx[i]`` is matched to ``nonmember_idx[i]``)."""

    member_idx: np.ndarray
    nonmember_idx: np.ndarray
    design: str  # time_matched | nearest_time | label_matched
    upper_bound: bool  # True => distribution shift may inflate the advantage (report as upper bound)
    n_members_in_corpus: int
    n_nonmember_candidates: int
    subsampled_from: int | None = None  # number of matched pairs before the max_per_side cap
    notes: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.member_idx.size)


class TooFewSamples(Exception):
    """Raised by :func:`select_membership_sample` with a researcher-facing skip reason."""


def _months(ts: np.ndarray) -> np.ndarray:
    return ts.astype("datetime64[M]").astype(np.int64)


def _time_range(ts: np.ndarray) -> list[str] | None:
    ts = ts[~np.isnat(ts)]
    if ts.size == 0:
        return None
    return [str(ts.min()), str(ts.max())]


def _nearest_to(targets: np.ndarray, query: np.ndarray) -> np.ndarray:
    """For each ``query`` time (int days), the absolute distance to the nearest ``targets`` time."""
    t = np.sort(targets)
    pos = np.searchsorted(t, query)
    left = t[np.clip(pos - 1, 0, t.size - 1)]
    right = t[np.clip(pos, 0, t.size - 1)]
    return np.minimum(np.abs(query - left), np.abs(query - right))


def _k_nearest(rows: np.ndarray, rows_days: np.ndarray, other_days: np.ndarray, k: int,
               rng: np.random.Generator) -> np.ndarray:
    """The ``k`` rows whose timestamps are closest to any of ``other_days`` (random tie-break)."""
    if k >= rows.size:
        return rows.copy()
    dist = _nearest_to(other_days, rows_days)
    order = np.lexsort((rng.random(rows.size), dist))
    return rows[order[:k]]


def _pairs_time_matched(corpus: "Corpus", mem: np.ndarray, cand: np.ndarray,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Stratify by (month, label): in each stratum draw as many members as non-members."""
    m_key = _months(corpus.timestamp[mem]) * 4 + corpus.label[mem].astype(np.int64)
    c_key = _months(corpus.timestamp[cand]) * 4 + corpus.label[cand].astype(np.int64)
    out_m: list[np.ndarray] = []
    out_c: list[np.ndarray] = []
    c_sorted = np.argsort(c_key, kind="stable")
    c_keys_sorted = c_key[c_sorted]
    n_strata = 0
    for key in np.unique(m_key):
        m_rows = mem[m_key == key]
        lo, hi = np.searchsorted(c_keys_sorted, [key, key + 1])
        c_rows = cand[c_sorted[lo:hi]]
        k = min(m_rows.size, c_rows.size)
        if k == 0:
            continue
        n_strata += 1
        out_m.append(rng.permutation(m_rows)[:k])
        out_c.append(rng.permutation(c_rows)[:k])
    if not out_m:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0
    return np.concatenate(out_m), np.concatenate(out_c), n_strata


def _pairs_nearest_time(corpus: "Corpus", mem: np.ndarray, cand: np.ndarray, cap: int | None,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Per label, the members closest in time to the non-members and vice versa.

    Returns the paired rows and the number of pairs available before the ``cap``.
    """
    days_all = corpus.timestamp.astype("datetime64[D]").astype(np.int64)
    out_m: list[np.ndarray] = []
    out_c: list[np.ndarray] = []
    available = 0
    for lab in (0, 1):
        m_rows = mem[corpus.label[mem] == lab]
        c_rows = cand[corpus.label[cand] == lab]
        k = min(m_rows.size, c_rows.size)
        available += k
        if cap is not None and mem.size:
            k = min(k, max(1, int(round(cap * m_rows.size / mem.size))))
        if k == 0:
            continue
        m_days, c_days = days_all[m_rows], days_all[c_rows]
        pick_m = _k_nearest(m_rows, m_days, c_days, k, rng)
        pick_c = _k_nearest(c_rows, c_days, days_all[pick_m], k, rng)
        out_m.append(rng.permutation(pick_m))
        out_c.append(rng.permutation(pick_c))
    if not out_m:
        return np.empty(0, np.int64), np.empty(0, np.int64), available
    return np.concatenate(out_m), np.concatenate(out_c), available


def _pairs_label_matched(corpus: "Corpus", mem: np.ndarray, cand: np.ndarray,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    out_m: list[np.ndarray] = []
    out_c: list[np.ndarray] = []
    for lab in (0, 1):
        m_rows = mem[corpus.label[mem] == lab]
        c_rows = cand[corpus.label[cand] == lab]
        k = min(m_rows.size, c_rows.size)
        if k == 0:
            continue
        out_m.append(rng.permutation(m_rows)[:k])
        out_c.append(rng.permutation(c_rows)[:k])
    if not out_m:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(out_m), np.concatenate(out_c)


def select_membership_sample(
    corpus: "Corpus",
    training_hashes: frozenset[str] | set[str],
    *,
    rng: np.random.Generator,
    min_members: int,
    max_per_side: int | None,
    time_matched: bool = True,
    extra_nonmember_splits: tuple[str, ...] = (),
) -> MembershipSample:
    """Build the paired member / non-member sets for the attack.

    Raises :class:`TooFewSamples` (with the skip reason) when fewer than ``min_members`` labeled
    members are in the corpus, or fewer than ``min_members`` non-members can be paired with them.
    """
    hashes = frozenset(h.lower() for h in training_hashes)
    members = corpus.indices(include_hashes=hashes)  # labeled rows only
    members = members[corpus.label[members] != LABEL_UNLABELED]
    n_mem = int(members.size)
    if n_mem < min_members:
        raise TooFewSamples(
            f"training manifest has {n_mem} members in the canonical corpus "
            f"(need at least min_members={min_members}; {len(hashes)} hashes in the manifest)"
        )
    cand = corpus.indices(role=ROLE_POOL, exclude_hashes=hashes)
    if extra_nonmember_splits:
        # Held-out threshold-calibration rows (``calibration`` split, in no role): benign, not in the
        # training manifest, and used by no attack to set anything, so valid time-near non-members.
        extra = corpus.indices(splits=extra_nonmember_splits, exclude_hashes=hashes)
        cand = np.union1d(cand, extra)
    cand = cand[corpus.label[cand] != LABEL_UNLABELED]
    notes: list[str] = []
    info: dict[str, Any] = {}
    # Challenge rows were selected for being atypical (hard to detect); as non-members they would
    # make "unseen" look different from "trained on" for reasons other than memorization.
    challenge = corpus.role_splits(ROLE_CHALLENGE)
    if challenge:
        keep = ~np.isin(corpus.split[cand], np.array(challenge, dtype=corpus.split.dtype))
        info["excluded_challenge_rows"] = int(np.sum(~keep))
        cand = cand[keep]

    has_ts_m = ~np.isnat(corpus.timestamp[members])
    has_ts_c = ~np.isnat(corpus.timestamp[cand])
    mem_ts, cand_ts = members[has_ts_m], cand[has_ts_c]
    timestamps_ok = mem_ts.size > 0 and cand_ts.size > 0
    design = DESIGN_LABEL_MATCHED
    upper_bound = False
    pm = pc = np.empty(0, np.int64)
    subsampled_from: int | None = None

    if time_matched and timestamps_ok:
        pm, pc, n_strata = _pairs_time_matched(corpus, mem_ts, cand_ts, rng)
        info["n_strata"] = n_strata
        info["members_without_timestamp"] = int(members.size - mem_ts.size)
        if pm.size >= min_members:
            design = DESIGN_TIME_MATCHED
            unmatched = int(mem_ts.size - pm.size)
            if unmatched:
                notes.append(
                    f"{unmatched} of {mem_ts.size} timestamped members had no same-month, same-label "
                    f"non-member partner and were left out, so both sets share one time/label profile."
                )
        else:
            overlap = "no" if pm.size == 0 else f"only {pm.size} member(s) with"
            notes.append(
                f"UPPER BOUND — {overlap} time overlap between training members and non-members in "
                f"the canonical corpus, so non-members were taken from the nearest available time "
                f"period. Differences between time periods (concept drift) make members and "
                f"non-members easier to tell apart, which inflates the measured advantage: treat it "
                f"as an upper bound on the true membership leakage."
            )
            pm, pc, available = _pairs_nearest_time(corpus, mem_ts, cand_ts, max_per_side, rng)
            if available > pm.size:
                subsampled_from = available  # the cap kept the pairs closest in time
            design = DESIGN_NEAREST_TIME
            upper_bound = True
    else:
        why = ("time matching is disabled (time_matched: false)" if not time_matched
               else "the corpus has no timestamps for the members or the non-members")
        notes.append(
            f"Members and non-members are matched by label only because {why}. If they come from "
            f"different time periods or sources, the advantage may be overstated."
        )
        pm, pc = _pairs_label_matched(corpus, members, cand, rng)

    if pm.size < min_members:
        raise TooFewSamples(
            f"only {pm.size} labeled non-member rows could be paired with the {n_mem} training "
            f"members in the canonical corpus (need at least min_members={min_members})"
        )
    if max_per_side is not None and pm.size > max_per_side:
        keep = np.sort(rng.choice(pm.size, size=int(max_per_side), replace=False))
        subsampled_from = max(subsampled_from or 0, int(pm.size))
        pm, pc = pm[keep], pc[keep]
    info.update(
        member_time_range=_time_range(corpus.timestamp[pm]),
        nonmember_time_range=_time_range(corpus.timestamp[pc]),
        member_label_counts={"benign": int(np.sum(corpus.label[pm] == 0)),
                             "malicious": int(np.sum(corpus.label[pm] == 1))},
        nonmember_label_counts={"benign": int(np.sum(corpus.label[pc] == 0)),
                                "malicious": int(np.sum(corpus.label[pc] == 1))},
    )
    if design == DESIGN_NEAREST_TIME:
        d = np.abs(corpus.timestamp[pm].astype(np.int64) - corpus.timestamp[pc].astype(np.int64))
        info["median_pair_time_gap_days"] = float(np.median(d)) if d.size else None
    return MembershipSample(
        member_idx=np.asarray(pm, dtype=np.int64),
        nonmember_idx=np.asarray(pc, dtype=np.int64),
        design=design,
        upper_bound=upper_bound,
        n_members_in_corpus=n_mem,
        n_nonmember_candidates=int(cand.size),
        subsampled_from=subsampled_from,
        notes=notes,
        info=info,
    )


# --------------------------------------------------------------------------------------------------
# Attack evaluation helpers
# --------------------------------------------------------------------------------------------------


def log_confidence(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log P_model(true label) — the negative cross-entropy loss (higher => more member-like)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    pos = np.log(np.maximum(p, 1e-300))
    neg = np.log1p(-np.minimum(p, 1.0 - 2.0**-53))
    return np.where(np.asarray(y) == 1, pos, neg)


def _roc(scores_m: np.ndarray, scores_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve

    y = np.r_[np.ones(scores_m.size), np.zeros(scores_n.size)]
    fpr, tpr, thr = roc_curve(y, np.r_[scores_m, scores_n], drop_intermediate=False)
    return fpr, tpr, thr


def attack_metrics(scores_m: np.ndarray, scores_n: np.ndarray, *, decision: Callable[[np.ndarray], np.ndarray]
                   ) -> dict[str, Any]:
    """Advantage at the attack's decision rule, best-threshold advantage, AUROC and TPR@1%FPR.

    ``scores_*`` are the attack's membership scores on held-out members / non-members (higher =>
    member); ``decision`` maps scores to the attack's hard member/non-member call.
    """
    from sklearn.metrics import roc_auc_score

    tpr_d = float(np.mean(decision(scores_m)))
    fpr_d = float(np.mean(decision(scores_n)))
    fpr, tpr, _ = _roc(scores_m, scores_n)
    y = np.r_[np.ones(scores_m.size), np.zeros(scores_n.size)]
    auroc = float(roc_auc_score(y, np.r_[scores_m, scores_n]))
    low = fpr <= 0.01 + 1e-12
    return {
        "advantage": tpr_d - fpr_d,
        "tpr": tpr_d,
        "fpr": fpr_d,
        "best_advantage": float(np.max(tpr - fpr)),
        "auroc": auroc,
        "tpr_at_fpr_0.01": float(np.max(tpr[low])) if np.any(low) else 0.0,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
    }


def _best_threshold(scores_m: np.ndarray, scores_n: np.ndarray) -> float:
    """Threshold (member iff score >= t) maximizing TPR − FPR on the attack's training half."""
    fpr, tpr, thr = _roc(scores_m, scores_n)
    j = int(np.argmax(tpr - fpr))
    t = float(thr[j])
    return t if math.isfinite(t) else float(np.max(np.r_[scores_m, scores_n])) + 1.0


def _two_column(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return np.stack([1.0 - p, p], axis=1)


def _import_art() -> tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ART warns about optional DL frameworks at import
        from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
        from art.estimators.classification import BlackBoxClassifier
    return MembershipInferenceBlackBox, BlackBoxClassifier


def _score_rows(ctx: "RunContext", corpus: "Corpus", idx: np.ndarray) -> np.ndarray:
    out = np.empty(idx.size, dtype=np.float64)
    for s in range(0, idx.size, _SCORE_CHUNK):
        ctx.check_deadline()
        sl = idx[s : s + _SCORE_CHUNK]
        out[s : s + sl.size] = ctx.score(corpus.take(sl))
    return out


def _advantage_anchors(t: float) -> tuple[float, float]:
    """``ideal = t/10``, ``floor = min(1, 3t)`` (kept strictly on either side of ``t``)."""
    ideal = t / 10.0 if t > 0 else t - 0.01
    floor = min(1.0, 3.0 * t)
    if floor <= t:
        floor = t + max(abs(t), 0.01)
    return ideal, floor


def _check_params(p: dict[str, Any]) -> tuple[float, int | None, int, float, bool]:
    t = float(p["max_advantage"])
    mps = p.get("max_per_side")
    mps = None if mps is None else int(mps)
    mm = int(p["min_members"])
    tf = float(p["test_fraction"])
    if mps is not None and mps < 2:
        raise ValueError(f"membership_inf.max_per_side must be >= 2 or null (got {mps})")
    if mm < 2:
        raise ValueError(f"membership_inf.min_members must be >= 2 (got {mm})")
    if not 0.0 < tf < 1.0:
        raise ValueError(f"membership_inf.test_fraction must be strictly between 0 and 1 (got {tf})")
    return t, mps, mm, tf, bool(p["time_matched"])


def _interpretation(worst: float, upper_bound: bool) -> str:
    level = (
        "a small advantage" if worst <= 0.05 else "a moderate advantage" if worst <= 0.15 else
        "a large advantage"
    )
    ub = " (an upper bound here, see the sampling caveat)" if upper_bound else ""
    return (
        "What this leakage would expose: anyone who can query the deployed model — for example "
        "through a cloud file-reputation or scanning API — could test whether a specific file was "
        "in your training set. For a detection vendor that reveals which samples you had and when "
        "(useful to a malware author checking whether their sample was collected, and to "
        "competitors), and if the training data included customer telemetry or customer-submitted "
        "files, it can confirm that a particular customer's file was in it — a privacy and "
        f"contractual exposure. The measured worst-case advantage of {worst:.3f}{ub} is {level}: "
        "the best attack flags true training members that much more often than it falsely flags "
        "unseen files (0 = chance, 1 = perfect). Regularization (fewer/shallower trees, larger "
        "min_data_in_leaf, subsampling) and returning coarse verdicts instead of raw scores reduce it."
    )


# --------------------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------------------


class MembershipInferenceModule(Module):
    id = "membership_inf"
    code = "M4"
    title = "Membership inference (training-data leakage)"
    description = (
        "Black-box membership inference through predict_proba: how well an attacker who can only "
        "query the model can tell files that were in its training set (from the training manifest) "
        "from comparable unseen files, using time- and label-matched non-members. Runs a "
        "loss-threshold attack and ART's learned black-box attacks (random forest, gradient "
        "boosting) and gates on the worst-case membership advantage (TPR − FPR)."
    )
    requires = (Requirement.FEATURE_SPACE, Requirement.TRAINING_HASHES)
    default_gate = GateMode.WARN
    default_params = {
        "max_advantage": 0.10,
        "max_per_side": 5000,
        "min_members": 200,
        "test_fraction": 0.5,
        "time_matched": True,
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        try:
            _check_params(params)
        except (TypeError, ValueError, KeyError) as e:
            raise ConfigError(f"{cls.id}: invalid parameter value: {e}") from e

    def run(self, ctx: "RunContext") -> ModuleResult:
        t0 = time.monotonic()
        if not ctx.training_hashes:
            return self.skip(ctx, ctx.missing_reason(Requirement.TRAINING_HASHES))
        if ctx.corpus is None or not ctx.has(Requirement.FEATURE_SPACE):
            return self.skip(ctx, ctx.missing_reason(Requirement.FEATURE_SPACE))
        t, max_per_side, min_members, test_fraction, time_matched = _check_params(ctx.params)
        corpus = ctx.corpus

        try:
            sample = select_membership_sample(
                corpus, ctx.training_hashes, rng=ctx.rng, min_members=min_members,
                max_per_side=max_per_side, time_matched=time_matched,
                # Calibration rows stay out of M4's non-member pool (docs/modules/model_submission.md:
                # "excluded from M4's non-member pool too"); pass extra_nonmember_splits to opt in.
            )
        except TooFewSamples as e:
            return self.skip(ctx, str(e))
        ctx.check_deadline()
        log.info("M4: %d member/non-member pairs (%s design)", sample.n_pairs, sample.design)

        # Split pairs into attack-train / attack-test (each half keeps the matched structure).
        P = sample.n_pairs
        n_test = min(max(int(round(P * test_fraction)), 1), P - 1)
        perm = ctx.rng.permutation(P)
        test_pairs, train_pairs = perm[:n_test], perm[n_test:]

        rows = np.concatenate([sample.member_idx, sample.nonmember_idx])
        p_all = _score_rows(ctx, corpus, rows)
        y_all = corpus.label[rows].astype(np.int64)
        p_m, p_n = p_all[:P], p_all[P:]
        y_m, y_n = y_all[:P], y_all[P:]

        results: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}

        # (1) loss / confidence-threshold attack.
        lc_m, lc_n = log_confidence(p_m, y_m), log_confidence(p_n, y_n)
        thr = _best_threshold(lc_m[train_pairs], lc_n[train_pairs])
        res = attack_metrics(lc_m[test_pairs], lc_n[test_pairs], decision=lambda s: s >= thr)
        res["decision_threshold_loss"] = float(-thr)
        results["loss_threshold"] = res

        # (2) ART MembershipInferenceBlackBox with rf / gb attack models.
        threads = max(1, int(getattr(ctx.config.runtime, "threads", 1) or 1))
        try:
            MIA, BlackBoxClassifier = _import_art()
            estimator = BlackBoxClassifier(
                predict_fn=lambda X: _two_column(ctx.score(np.asarray(X, dtype=np.float32))),
                input_shape=(corpus.dim,),
                nb_classes=2,
            )
        except Exception as e:  # pragma: no cover - ART is a core dependency
            log.warning("ART unavailable for M4: %s", e)
            errors["art"] = f"{type(e).__name__}: {e}"
            MIA = estimator = None
        if MIA is not None:
            pred_m, pred_n = _two_column(p_m), _two_column(p_n)
            for name, kind in (("art_rf", "rf"), ("art_gb", "gb")):
                ctx.check_deadline()
                try:
                    attack = MIA(estimator, input_type="prediction", attack_model_type=kind)
                    seed = int(ctx.rng.integers(0, 2**31 - 1))
                    attack.attack_model.set_params(random_state=seed)
                    if kind == "rf":
                        attack.attack_model.set_params(n_jobs=threads)
                    # Predictions are passed precomputed (``pred=``) so the model is queried once,
                    # in large batches, instead of in ART's default 128-row batches.
                    attack.fit(
                        x=None, y=y_m[train_pairs], pred=pred_m[train_pairs],
                        test_x=None, test_y=y_n[train_pairs], test_pred=pred_n[train_pairs],
                    )
                    if kind == "rf":  # parallel predict_proba sums trees in nondeterministic order
                        attack.attack_model.set_params(n_jobs=1)
                    s_m = np.asarray(attack.infer(None, y_m[test_pairs], pred=pred_m[test_pairs],
                                                  probabilities=True), dtype=np.float64).reshape(-1)
                    s_n = np.asarray(attack.infer(None, y_n[test_pairs], pred=pred_n[test_pairs],
                                                  probabilities=True), dtype=np.float64).reshape(-1)
                    # ART's hard decision rounds the member probability (0.5 rounds to non-member).
                    results[name] = attack_metrics(s_m, s_n, decision=lambda s: s > 0.5)
                except Exception as e:  # keep the other attacks' evidence
                    log.warning("M4 attack %s failed: %s", name, e)
                    errors[name] = f"{type(e).__name__}: {e}"

        # Worst case across attacks.
        worst_name = max(results, key=lambda k: results[k]["advantage"])
        worst = float(results[worst_name]["advantage"])
        ideal, floor = _advantage_anchors(t)
        check = GateCheck.evaluate(
            "membership_advantage", worst, "<=", t, metric="advantage",
            description="worst-case membership advantage (TPR − FPR at the attack's decision "
                        "threshold) over all attacks, on held-out member/non-member pairs",
            ideal=ideal, floor=floor,
        )

        # Generalization gap at the operating threshold (context for the advantage).
        acc_m = float(np.mean((p_m >= ctx.threshold) == (y_m == 1)))
        # Calibration rows fixed the threshold, so their accuracy at it is true by construction:
        # leave them out of this context metric (they stay in the attacks, which never use the threshold).
        is_cal = corpus.split[sample.nonmember_idx] == CALIBRATION_SPLIT
        n_cal = int(is_cal.sum())
        keep_n = ~is_cal if n_cal < P else np.ones(P, dtype=bool)
        acc_n = float(np.mean((p_n[keep_n] >= ctx.threshold) == (y_n[keep_n] == 1)))

        metrics: dict[str, Any] = {
            "advantage": worst,
            "worst_attack": worst_name,
            "advantage_is_upper_bound": sample.upper_bound,
            "sampling_design": sample.design,
            "time_matched": sample.design == DESIGN_TIME_MATCHED,
            "n_manifest_hashes": len(ctx.training_hashes),
            "n_members_in_corpus": sample.n_members_in_corpus,
            "n_members_used": P,
            "n_nonmembers_used": P,
            "n_calibration_nonmembers_used": n_cal,
            "n_attack_train_pairs": int(train_pairs.size),
            "n_attack_test_pairs": int(test_pairs.size),
            "member_accuracy": acc_m,
            "nonmember_accuracy": acc_n,
            "accuracy_gap": acc_m - acc_n,
            "max_auroc": max(r["auroc"] for r in results.values()),
            "max_best_advantage": max(r["best_advantage"] for r in results.values()),
            "max_tpr_at_fpr_0.01": max(r["tpr_at_fpr_0.01"] for r in results.values()),
        }
        for name, r in results.items():
            for k in ("advantage", "best_advantage", "auroc", "tpr_at_fpr_0.01"):
                metrics[f"{name}_{k}"] = float(r[k])

        # Charts.
        ctx.artifacts.add_chart(
            self.id, "attack_roc", title="Membership-inference attack ROC (held-out pairs)",
            series=[{"label": ATTACK_LABELS[k], "x": r["roc_fpr"], "y": r["roc_tpr"]}
                    for k, r in results.items()],
            kind="line", xlabel="false-positive rate (non-members called members)",
            ylabel="true-positive rate (members detected)", xlim=(0.0, 1.0), ylim=(0.0, 1.0),
            reference_lines=[{"axis": "x", "value": 0.01, "label": "1% FPR"}], note="diagonal",
        )
        loss_m = np.clip(-log_confidence(p_m, y_m), 1e-8, 1e2)
        loss_n = np.clip(-log_confidence(p_n, y_n), 1e-8, 1e2)
        centers = 0.5 * (_HIST_EDGES[:-1] + _HIST_EDGES[1:])
        h_m = np.histogram(np.log10(loss_m), bins=_HIST_EDGES)[0] / max(P, 1)
        h_n = np.histogram(np.log10(loss_n), bins=_HIST_EDGES)[0] / max(P, 1)
        ctx.artifacts.add_chart(
            self.id, "member_score_hist",
            title="Model loss on the true label: training members vs matched non-members",
            series=[{"label": "training members", "x": centers, "y": h_m},
                    {"label": "non-members", "x": centers, "y": h_n}],
            kind="step", xlabel="log10 cross-entropy loss on the true label (lower = more confident)",
            ylabel="fraction of samples",
            reference_lines=[{"axis": "x", "value": float(np.log10(max(results["loss_threshold"]
                              ["decision_threshold_loss"], 1e-8))), "label": "loss-attack threshold"}],
        )

        details: dict[str, Any] = {
            "attacks": {
                k: {kk: vv for kk, vv in r.items() if kk not in ("roc_fpr", "roc_tpr")}
                | {"label": ATTACK_LABELS[k]}
                for k, r in results.items()
            },
            "attack_errors": errors,
            "sampling": {
                "design": sample.design,
                "upper_bound": sample.upper_bound,
                "n_nonmember_candidates": sample.n_nonmember_candidates,
                "pairs_before_cap": sample.subsampled_from,
                "max_per_side": max_per_side,
                "test_fraction": test_fraction,
                **sample.info,
            },
            "duration_s": round(time.monotonic() - t0, 3),
        }

        notes = list(sample.notes)
        if sample.subsampled_from is not None:
            how = ("keeping the pairs closest in time" if sample.design == DESIGN_NEAREST_TIME
                   else "uniformly at random")
            notes.append(
                f"Subsampled {P} of {sample.subsampled_from} available member/non-member pairs "
                f"{how} (max_per_side={max_per_side}); {sample.n_members_in_corpus} training "
                f"members are in the corpus."
            )
        for k, e in errors.items():
            notes.append(f"Attack {k} could not run and is excluded from the worst case: {e}")
        notes.append(
            "Advantage = TPR − FPR of the attack at its own decision threshold (chosen on the "
            "attack-training half for the loss attack; 0.5 member probability for ART's attacks), "
            f"measured on {test_pairs.size} held-out pairs. best_advantage picks the threshold on the "
            "held-out pairs themselves and is optimistic for the attacker."
        )
        notes.append(_interpretation(worst, sample.upper_bound))

        ub = " (upper bound: members and non-members come from different time periods)" \
            if sample.upper_bound else ""
        if check.passed:
            finding = (
                f"Worst-case membership advantage is {worst:.3f} ({ATTACK_LABELS[worst_name]}){ub}, "
                f"within the {t:.2f} limit: querying the model reveals little about which files were "
                f"in its training set (tested on {P} training members vs {P} matched non-members)."
            )
        else:
            others = ("unseen files of the nearest time period" if sample.upper_bound
                      else "comparable unseen files")
            finding = (
                f"Worst-case membership advantage is {worst:.3f} ({ATTACK_LABELS[worst_name]}){ub}, "
                f"above the {t:.2f} limit: an attacker who can query the model can tell its training "
                f"files from {others} well above chance (attack AUROC up to "
                f"{metrics['max_auroc']:.3f})."
            )
        return self.result(ctx, finding=finding, checks=[check], metrics=metrics, details=details,
                           notes=notes)
