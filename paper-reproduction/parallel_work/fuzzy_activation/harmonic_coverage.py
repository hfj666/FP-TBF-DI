"""CoveragePool with harmonic marginal credit for same-class training rows."""
import numpy as np


def select(pool, budget=64):
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
        for _ in range(budget // 2):
            gain = (covers @ (1 / (1 + count))) / len(own_rows)
            best = int(np.argmax(.5 * quality + .5 * (gain - overlap)))
            selected.append(candidates[best][1])
            count += covers[best]
            quality[best] = -np.inf
            mask = masks[best]
            overlap = np.maximum(overlap, [(mask & other).bit_count() / (mask | other).bit_count()
                                            for other in masks])
    return selected
