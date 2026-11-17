"""Smoke test FP-TBF-DI on generated scores without a cohort file."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from src.fp_tbf_simplified import SimplifiedFPTBF
from src.fptree import FPTree

rng = np.random.default_rng(123)
y = np.repeat([-1, 1], 160)
x = rng.integers(0, 2, size=(320, 18))
x[:, :8] = (y[:, None] == 1).astype(int)
order = rng.permutation(len(y))
x, y = x[order], y[order]
tree = FPTree().update(x[:300], y[:300])
model = SimplifiedFPTBF().fit(x[:300], y[:300], tree)
scores = model.decision_scores(x[300:])
assert scores.shape == (20, 6) and np.isfinite(scores).all()
A, B, C = scores[:, 2:5].T
np.testing.assert_allclose(scores[:, -1], (
    0.38633621372064075 * A + 0.3109653252978891 * B + 0.30269846098146996 * C
), atol=1e-12, rtol=0)
print('FP-TBF-DI smoke test passed: 20 queries, finite branch and fusion scores.')
