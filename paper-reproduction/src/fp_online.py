"""Matched cumulative FP backends with identical batch-only Sugeno learning."""
import copy
from time import perf_counter
import numpy as np
from .fptree import FPTree, pattern_rules
from .models import SparseSugeno
from .online import LocalSugeno


def rule_records(rules):
    return [dict(columns=list(r.columns), values=r.values.tolist(), label=int(r.label),
                 confidence=float(r.confidence), count=int(r.count)) for r in rules]


def select_rules(patterns):
    # Existing ranking; 32 slots per class, unfilled slots are not reassigned.
    return pattern_rules(patterns, max_rules=64, confidence=.65, min_rows=5)


def initialize(x, y):
    t = perf_counter(); tree = FPTree().update(x, y); insert = perf_counter()-t
    t = perf_counter(); patterns = tree.mine(.1, 3); query = perf_counter()-t
    t = perf_counter(); rules = select_rules(patterns); selection = perf_counter()-t
    scaling = (x.min(0), np.maximum(np.ptp(x, axis=0), 1.))
    t = perf_counter(); model = SparseSugeno(rules, maxiter=500).fit(x, y, scaling=scaling)
    fit = perf_counter()-t
    return tree, model, dict(insert_seconds=insert, query_seconds=query,
        selection_seconds=selection, fit_seconds=fit, total_seconds=insert+query+selection+fit,
        rules=len(rules), patterns=len(patterns), **model.optimization)


class FPOnline:
    def __init__(self, tree, base, lr, active, rebuild=False, initial_x=None, initial_y=None):
        self.tree = copy.deepcopy(tree)
        self.model = LocalSugeno(base, lr, active)
        self.rebuild = rebuild
        if rebuild:
            self.history_x = initial_x.copy(); self.history_y = initial_y.copy()

    def decision_function(self, x):
        return self.model.decision_function(x)

    def update(self, x, y):
        begin = perf_counter(); old_n = self.tree.n
        t = perf_counter()
        if self.rebuild:
            self.history_x = np.vstack([self.history_x, x]); self.history_y = np.r_[self.history_y, y]
            self.tree = FPTree().update(self.history_x, self.history_y)
        else:
            self.tree.update(x, y)
        insert = perf_counter()-t
        t = perf_counter(); patterns = self.tree.mine(.1, 3); query = perf_counter()-t
        t = perf_counter(); rules = select_rules(patterns); selection = perf_counter()-t
        if not rules:
            raise ValueError('No eligible rule; this fixed protocol has no fallback.')
        t = perf_counter(); change = self.model.replace_rules(rules); sync = perf_counter()-t
        before = self.model.parameter_rule_updates
        t = perf_counter(); self.model.partial_fit(x, y); learning = perf_counter()-t
        total = perf_counter()-begin
        stats = dict(insert_seconds=insert, query_seconds=query, selection_seconds=selection,
            sync_seconds=sync, learning_seconds=learning, total_seconds=total,
            history_raw_rows_read=old_n if self.rebuild else 0,
            parameter_history_rows=0, retained_raw_rows=self.tree.n if self.rebuild else 0,
            inserted_rows=self.tree.n if self.rebuild else len(y), learned_rows=len(y),
            patterns=len(patterns), rules=len(rules), tree_nodes=self.tree.nodes,
            parameter_rule_updates=self.model.parameter_rule_updates-before, **change)
        return stats, patterns, rule_records(rules)
