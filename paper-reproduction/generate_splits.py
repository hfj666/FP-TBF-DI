"""Recreate the ordered study splits from the original score table."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/online_study/splits.json"))
    args = parser.parse_args()
    data = np.loadtxt(args.data)
    if (data.shape != (1062, 19) or not np.isfinite(data).all()
            or set(data[:, 0]) != {-1, 1}
            or not np.equal(data[:, 1:], np.rint(data[:, 1:])).all()):
        parser.error("Expected 1062 rows with a -1/+1 label and 18 integer scores.")
    splits = []
    for seed in (17, 29, 43):
        stream, holdout = train_test_split(
            np.arange(len(data)), test_size=0.2, stratify=data[:, 0], random_state=seed
        )
        np.random.default_rng(seed).shuffle(stream)
        splits.append(dict(seed=seed, train_stream=stream.tolist(), holdout=holdout.tolist()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)


if __name__ == "__main__":
    main()
