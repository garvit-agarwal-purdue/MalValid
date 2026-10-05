"""Write train_sha256.txt: the sha256 of every labeled EMBER2018 training sample.

The reference EMBER2018 model was trained on the labeled rows of the EMBER2018 train split, so
those are its training members (used by M4 membership inference, and excluded from M1/M2 eval
sets). This reads them from the canonical ember_v2_2018 corpus, so build that first:

    malvalid corpus build ember_v2_2018 --source /path/to/ember2018
"""

from pathlib import Path

from malvalid import registry


def main() -> None:
    corpus = registry.get_corpus_provider("ember_v2_2018").load(verify=False)
    idx = corpus.indices(splits=["train"], label=[0, 1])
    out = Path(__file__).resolve().parent / "train_sha256.txt"
    with open(out, "w") as f:
        f.write("# sha256 of the labeled EMBER2018 train split (training members of ember_model_2018.txt)\n")
        f.writelines(h + "\n" for h in corpus.sha256[idx].tolist())
    print(f"wrote {idx.size:,} hashes to {out}")


if __name__ == "__main__":
    main()
