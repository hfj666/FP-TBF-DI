"""Default and fixed control for the internal cumulative FP study."""
from time import perf_counter

from .models import SparseSugeno
from .projected_selection import select
from .selection import CoveragePool


DEFAULT_ARM = 'equal_score_mean'
PROJECTED_ARM = 'soft_projected'
CONTROL_ARM = 'coverage_saved'
MODEL_ARMS = (PROJECTED_ARM, CONTROL_ARM)
SCORE_ARMS = (DEFAULT_ARM, *MODEL_ARMS)
FIT_MAXITER = 2000
SUPPORT = .1
MAX_LENGTH = 3
PURITY = .65
MIN_ROWS = 5
RULES_PER_CLASS = 32
COVERAGE_ALPHA = .5


def fit_pair(patterns, x, y, scaling):
    """Select and cold fit both arms on the same revealed prefix."""
    pool = CoveragePool(patterns, x, y, confidence=PURITY, min_rows=MIN_ROWS)
    if any(len(entries) < RULES_PER_CLASS for entries in pool.pools.values()):
        raise ValueError('Internal research protocol requires 32 eligible rules per class.')

    models = {}
    records = {}
    for arm in MODEL_ARMS:
        started = perf_counter()
        if arm == PROJECTED_ARM:
            rules, selection = select(pool, x, y, scaling, joint=True)
        else:
            rules = pool.select(COVERAGE_ALPHA, 2*RULES_PER_CLASS)
            selection = {}
        selection_seconds = perf_counter()-started
        assert len(rules) == 2*RULES_PER_CLASS
        assert sum(r.label == 1 for r in rules) == RULES_PER_CLASS

        started = perf_counter()
        model = SparseSugeno(rules, maxiter=FIT_MAXITER).fit(x, y, scaling=scaling)
        fit_seconds = perf_counter()-started
        models[arm] = model
        records[arm] = dict(selection=selection, selection_seconds=selection_seconds,
                            fit_seconds=fit_seconds, **model.optimization)
    return models, records


def score_arms(models, x):
    """Return the default equal-score fusion and both standalone controls."""
    projected = models[PROJECTED_ARM].decision_function(x)
    coverage = models[CONTROL_ARM].decision_function(x)
    return {DEFAULT_ARM: (projected+coverage)/2,
            PROJECTED_ARM: projected, CONTROL_ARM: coverage}
