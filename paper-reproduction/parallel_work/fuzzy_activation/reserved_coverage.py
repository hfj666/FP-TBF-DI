"""Reserve original rules, then fill with harmonic coverage."""
import numpy as np


def select(pool, original):
    selected = []
    for label in (-1, 1):
        candidates = pool.pools[label]
        masks = [item[2] for item in candidates]
        own = pool.own[label]
        own_rows = [i for i in range(own.bit_length()) if own >> i & 1]
        covers = np.array([[bool(mask >> i & 1) for i in own_rows] for mask in masks], dtype=float)
        quality = np.array([item[3] for item in candidates])
        quality /= quality.max()
        overlap = np.zeros(len(masks))
        count = np.zeros(len(own_rows))
        keys = {tuple(zip(rule.columns, rule.values)): i for i, (_, rule, _, _) in enumerate(candidates)}
        reserved = [r for r in original if r.label == label][:16]
        reserve_indices = [keys[tuple(zip(r.columns, r.values))] for r in reserved]
        assert len(reserve_indices) == 16 and len(set(reserve_indices)) == 16
        for step in range(32):
            if step < len(reserve_indices):
                best = reserve_indices[step]
            else:
                gain = (covers @ (1 / (1 + count))) / len(own_rows)
                best = int(np.argmax(.5 * quality + .5 * (gain - overlap)))
            selected.append(candidates[best][1])
            count += covers[best]
            quality[best] = -np.inf
            mask = masks[best]
            overlap = np.maximum(overlap, [(mask & other).bit_count() / (mask | other).bit_count()
                                            for other in masks])
    return selected

