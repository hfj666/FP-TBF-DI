"""Greedy training-only complementary coverage selection from unchanged FP patterns."""
import numpy as np
from .models import Rule


def bitmask(values):
    return int.from_bytes(np.packbits(values, bitorder='little').tobytes(), 'little')


class CoveragePool:
    def __init__(self, patterns, x, y, confidence=.65, min_rows=5):
        self.patterns, self.pools = patterns, {-1: [], 1: []}
        y = np.asarray(y)
        self.own = {label: bitmask(y == label) for label in (-1, 1)}
        self.class_n = {label: int((y == label).sum()) for label in (-1, 1)}
        if min(self.class_n.values()) == 0:
            raise ValueError('Both classes required for coverage selection.')
        for items, (count, positive) in sorted(patterns.items()):
            purity = max(positive, count-positive)/count
            if count < min_rows or purity <= confidence:
                continue
            columns = tuple(j for j, _ in items)
            values = np.array([v for _, v in items], dtype=float)
            mask = np.all(x[:, list(columns)] == values, axis=1)
            if int(mask.sum()) != count or int(np.sum(mask & (y == 1))) != positive:
                raise ValueError('Patterns must be mined on exactly the supplied training rows.')
            label = 1 if positive > count/2 else -1
            quality = (purity-.5)*np.sqrt(count)*np.sqrt(len(items))
            rule = Rule(columns, values, label, purity, count)
            self.pools[label].append((items, rule, bitmask(mask), float(quality)))

    def select(self, alpha, max_rules=64):
        if not 0 <= alpha <= 1 or max_rules < 2 or max_rules % 2:
            raise ValueError('alpha in [0,1] and even rule budget >=2 required.')
        selected = []
        for label in (-1, 1):
            pool = self.pools[label]
            if not pool:
                continue
            scale = max(r[3] for r in pool)
            covered, remaining = 0, list(range(len(pool)))
            max_overlap = [0.] * len(pool)
            for _ in range(min(max_rules//2, len(pool))):
                best, best_score = None, -np.inf
                for i in remaining:
                    _, rule, mask, quality = pool[i]
                    novelty = (mask & self.own[label] & ~covered).bit_count()/self.class_n[label]
                    score = alpha*quality/scale+(1-alpha)*(novelty-max_overlap[i])
                    # Stable item order breaks exact score ties without test data.
                    if score > best_score:
                        best, best_score = i, score
                _, rule, mask, _ = pool[best]
                selected.append(rule)
                covered |= mask & self.own[label]
                remaining.remove(best)
                # Older pairwise overlaps stay unchanged after adding a rule.
                for i in remaining:
                    other = pool[i][2]
                    overlap = (other & mask).bit_count()/(other | mask).bit_count()
                    max_overlap[i] = max(max_overlap[i], overlap)
        return selected
