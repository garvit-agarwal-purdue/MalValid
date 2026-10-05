"""Train an XGBoost detector on EMBER2018 and write everything a malvalid submission needs.

Outputs (in this directory):
  model.json         XGBoost model in its JSON format (safe: no pickle)
  train_sha256.txt   sha256 of every training sample (for M4 membership inference)
  threshold.txt      operating threshold chosen for 1% FPR on held-out *training-period* data

The threshold is chosen on a validation month carved out of the training period (October 2018),
never on the canonical eval set: tuning on the harness's eval data would inflate the gate result.

    python train.py [--rounds 600] [--threads 8]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb

from malvalid import registry

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    args = ap.parse_args()

    corpus = registry.get_corpus_provider("ember_v2_2018").load(verify=False)
    train = corpus.indices(splits=["train"], label=[0, 1])
    month = corpus.timestamp[train].astype("datetime64[M]")
    is_val = month == np.datetime64("2018-10", "M")
    fit_idx, val_idx = train[~is_val], train[is_val]
    print(f"fit rows {fit_idx.size:,}  validation rows (2018-10) {val_idx.size:,}")

    dfit = xgb.DMatrix(corpus.take(fit_idx), label=corpus.label[fit_idx])
    dval = xgb.DMatrix(corpus.take(val_idx), label=corpus.label[val_idx])
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 10,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "min_child_weight": 5,
        "nthread": args.threads,
        "seed": 0,
    }
    booster = xgb.train(params, dfit, args.rounds, evals=[(dval, "val")], verbose_eval=100)

    p_val = booster.predict(dval)
    benign = np.sort(p_val[corpus.label[val_idx] == 0])
    k = int(np.ceil((1 - args.target_fpr) * benign.size)) - 1
    threshold = float(np.nextafter(benign[min(max(k, 0), benign.size - 1)], np.float32(1)))
    dr = float(np.mean(p_val[corpus.label[val_idx] == 1] >= threshold))
    print(f"threshold {threshold:.6f}: validation FPR <= {args.target_fpr:.2%}, detection {dr:.2%}")

    booster.save_model(HERE / "model.json")
    (HERE / "threshold.txt").write_text(f"{threshold:.6f}\n")
    with open(HERE / "train_sha256.txt", "w") as f:
        f.write("# sha256 of every sample used to fit or validate model.json (EMBER2018 train split)\n")
        f.writelines(h + "\n" for h in corpus.sha256[train].tolist())
    print(f"wrote model.json, threshold.txt, train_sha256.txt ({train.size:,} hashes)")


if __name__ == "__main__":
    main()
