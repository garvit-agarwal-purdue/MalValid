"""Train a demo LightGBM detector on the SYNTHETIC synthetic_v2 corpus and write a malvalid submission.

Synthetic data is for demos and CI only. The verdict you get from this example says nothing about
real-world readiness; see README.md.

Outputs (in this directory):
  model.txt          LightGBM text format (a safe, non-pickle format)
  threshold.txt      operating threshold for 1% FPR on the `holdout` split (validation data)
  train_sha256.txt   sha256 of every `train` row (the training members, for M4 membership inference)

The threshold is picked on `holdout`, never on the `test` split that M1 evaluates on: tuning on
the gate's own evaluation data would inflate the result. The last line printed is the
`training_cutoff` that adapter.py declares.

    python train.py [--threads 8]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np

from malvalid import registry

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    args = ap.parse_args()

    corpus = registry.get_corpus_provider("synthetic_v2").load()  # generates it on first use (~3 s)
    train = corpus.indices(splits=["train"], label=[0, 1])
    hold = corpus.indices(splits=["holdout"], label=[0, 1])
    print(f"train rows {train.size:,}  holdout rows {hold.size:,}")

    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "num_threads": args.threads,
        "seed": 0,
        "verbose": -1,
    }
    booster = lgb.train(params, lgb.Dataset(corpus.take(train), label=corpus.label[train]), num_boost_round=200)

    p_hold = booster.predict(corpus.take(hold))
    y_hold = corpus.label[hold]
    benign = np.sort(p_hold[y_hold == 0])
    k = int(np.ceil((1 - args.target_fpr) * benign.size)) - 1
    threshold = float(np.nextafter(benign[min(max(k, 0), benign.size - 1)], 1.0))
    dr = float(np.mean(p_hold[y_hold == 1] >= threshold))
    print(f"threshold {threshold:.6f}: holdout FPR <= {args.target_fpr:.2%}, detection {dr:.2%}")

    booster.save_model(str(HERE / "model.txt"))
    (HERE / "threshold.txt").write_text(f"{threshold:.6f}\n")
    with open(HERE / "train_sha256.txt", "w") as f:
        f.write("# sha256 of every synthetic_v2 `train` row (training members of model.txt)\n")
        f.writelines(h + "\n" for h in corpus.sha256[train].tolist())
    print(f"wrote model.txt, threshold.txt, train_sha256.txt ({train.size:,} hashes)")

    cutoff = corpus.timestamp[train].max()
    print(f'training_cutoff = "{cutoff}"   # newest train row; adapter.py declares this value')


if __name__ == "__main__":
    main()
