"""Insert-only, class-counted FP tree with immutable lexicographic item order.

All observed items are retained, even when currently infrequent. Mining uses
conditional FP trees; no old transactions or old-pattern-only summary is needed.
The tree is exact for a cumulative (no deletions/drift window) database.
"""
from collections import defaultdict
from math import ceil
import numpy as np
from .models import Rule


class Node:
    __slots__ = ('item', 'parent', 'children', 'count', 'positive')
    def __init__(self, item=None, parent=None):
        self.item, self.parent = item, parent
        self.children = {}
        self.count = self.positive = 0


class FPTree:
    def __init__(self):
        self.root = Node()
        self.header = defaultdict(list)
        self.n = 0

    def insert(self, items, count=1, positive=0):
        node = self.root
        self.n += count
        for item in sorted(set(items)):
            if item not in node.children:
                child = Node(item, node)
                node.children[item] = child
                self.header[item].append(child)
            node = node.children[item]
            node.count += count
            node.positive += positive

    def update(self, x, y):
        for row, label in zip(x, y, strict=True):
            self.insert([(j, int(value)) for j, value in enumerate(row)], positive=int(label == 1))
        return self

    def mine(self, min_support=.1, max_length=3):
        if not 0 < min_support <= 1 or max_length < 1:
            raise ValueError('Require support in (0,1] and positive max_length.')
        threshold = ceil(min_support*self.n)
        patterns = {}
        def visit(tree, suffix):
            for item in sorted(tree.header, reverse=True):
                nodes = tree.header[item]
                count = sum(n.count for n in nodes)
                if count < threshold:
                    continue
                positive = sum(n.positive for n in nodes)
                pattern = tuple(sorted((item,)+suffix))
                patterns[pattern] = (count, positive)
                if len(pattern) >= max_length:
                    continue
                paths, totals = [], defaultdict(int)
                for node in nodes:
                    path, parent = [], node.parent
                    while parent.item is not None:
                        path.append(parent.item)
                        totals[parent.item] += node.count
                        parent = parent.parent
                    paths.append((path, node.count, node.positive))
                conditional = FPTree()
                for path, num, pos in paths:
                    path = [i for i in path if totals[i] >= threshold]
                    if path:
                        conditional.insert(path, num, pos)
                if conditional.header:
                    visit(conditional, pattern)
        try:
            if self.n:
                visit(self, ())
        finally:
            # Release the recursive closure (and its captured pattern dictionary).
            # Conditional nodes can still form parent/child cycles handled by GC.
            visit = None
        return patterns

    @property
    def nodes(self):
        return 1+sum(len(nodes) for nodes in self.header.values())


def pattern_rules(patterns, max_rules=64, confidence=.65, min_rows=5):
    """Class-balanced deterministic ranking by purity, length, and coverage.

    A modeling choice, not a claim that these are the paper's heuristic rules.
    """
    pools = {-1: [], 1: []}
    for items, (count, positive) in patterns.items():
        purity = max(positive, count-positive)/count
        if count < min_rows or purity <= confidence:
            continue
        label = 1 if positive > count/2 else -1
        rule = Rule(tuple(j for j, _ in items), np.array([v for _, v in items], dtype=float),
                    label, purity, count)
        quality = (purity-.5)*np.sqrt(count)*np.sqrt(len(items))
        pools[label].append((quality, items, rule))
    selected = []
    for label in (-1, 1):
        ranked = sorted(pools[label], key=lambda r: (-r[0], r[1]))
        selected.extend(r[2] for r in ranked[:max_rules//2])
    return selected
