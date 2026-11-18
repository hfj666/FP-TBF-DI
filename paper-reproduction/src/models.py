"""Paper-derived static baseline and trainable sparse Sugeno inference."""
from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


@dataclass
class Rule:
    columns: tuple
    values: np.ndarray
    label: int
    confidence: float
    count: int


def msr(a):
    residue = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    return float(np.mean(residue ** 2))


def deletion_scores(a):
    """MSR for every single row/column deletion, from sufficient sums."""
    n, m = a.shape
    row, col, total = a.sum(1), a.sum(0), a.sum()
    square = np.square(a).sum()
    def score(sq, rsq, csq, total_, nr, nc):
        return sq/(nr*nc) - rsq/(nr*nc**2) - csq/(nc*nr**2) + total_**2/(nr**2*nc**2)
    rows = score(square-np.square(a).sum(1), np.square(row).sum()-row**2,
                 np.square(col[None, :]-a).sum(1), total-row, n-1, m) if n > 1 else np.full(n, np.inf)
    cols = score(square-np.square(a).sum(0), np.square(row[:, None]-a).sum(0),
                 np.square(col).sum()-col**2, total-col, n, m-1) if m > 1 else np.full(m, np.inf)
    return rows, cols


class StaticBiclusterBoost:
    """Section 3: hierarchical seeds, greedy MSR deletion, FSDND pairs, AdaBoost.

    Unreported settings are explicit constructor arguments; defaults are an
    exact-value seed / zero-residue instantiation, not original author settings.
    """
    def __init__(self, seed_distance=0., residue_threshold=1e-12, confidence=.65,
                 min_rows=5, rounds=8):
        self.seed_distance = seed_distance
        self.residue_threshold = residue_threshold
        self.confidence = confidence
        self.min_rows = min_rows
        self.rounds = rounds

    def mine_rules(self, x, y):
        self.minimum = x.min(0)
        self.training_max = x.max(0)
        self.span = np.maximum(x.max(0)-self.minimum, 1.)
        z = (x-self.minimum)/self.span
        candidates = {}
        for j in range(z.shape[1]):
            groups = fcluster(linkage(z[:, j, None], method='complete'),
                              self.seed_distance, criterion='distance')
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                cols = np.arange(z.shape[1])
                if len(rows) < self.min_rows:
                    continue
                while msr(z[np.ix_(rows, cols)]) > self.residue_threshold:
                    rs, cs = deletion_scores(z[np.ix_(rows, cols)])
                    if len(rows) <= self.min_rows:
                        rs[:] = np.inf
                    if rs.min() < cs.min():
                        rows = np.delete(rows, rs.argmin())
                    else:
                        cols = np.delete(cols, cs.argmin())
                candidates[(tuple(rows), tuple(cols))] = (rows, cols)
        # Drop exact duplicate and contained biclusters (both axes).
        blocks = sorted(candidates.values(), key=lambda p: -len(p[0])*len(p[1]))
        retained = []
        for rows, cols in blocks:
            sr, sc = set(rows), set(cols)
            if not any(sr <= ar and sc <= ac for ar, ac in retained):
                retained.append((sr, sc))
        self.rules = []
        for rows, cols in retained:
            rows, cols = sorted(rows), tuple(sorted(cols))
            positive = np.mean(y[rows] == 1)
            conf = max(positive, 1-positive)
            if conf > self.confidence:
                self.rules.append(Rule(cols, z[np.ix_(rows, cols)].mean(0),
                                       1 if positive > .5 else -1, conf, len(rows)))
        return self

    def raw_rules(self):
        """Copy mined centers back to original score units for shared classifiers."""
        return [Rule(r.columns, r.values*self.span[list(r.columns)]+self.minimum[list(r.columns)],
                     r.label, r.confidence, r.count) for r in self.rules]

    def fit(self, x, y):
        self.mine_rules(x, y)
        return self._fit_boost(x, y)

    def fit_rules(self, x, y, rules):
        """Train the identical pairwise boosting classifier on supplied RAW rules."""
        self.minimum, self.training_max = x.min(0), x.max(0)
        self.span = np.maximum(self.training_max-self.minimum, 1.)
        self.rules = [Rule(r.columns, (r.values-self.minimum[list(r.columns)])/self.span[list(r.columns)],
                           r.label, r.confidence, r.count) for r in rules]
        return self._fit_boost(x, y)

    def _fit_boost(self, x, y):
        benign = [i for i, r in enumerate(self.rules) if r.label == -1]
        malignant = [i for i, r in enumerate(self.rules) if r.label == 1]
        self.pairs = [(b, m) for b in benign for m in malignant]
        if not self.pairs:
            raise ValueError('No opposite-class rule pairs; inspect mining settings.')
        pred = self._pair_predictions(x)
        weights = np.ones(len(y))/len(y)
        self.selected = []
        for _ in range(self.rounds):
            errors = (pred != y[:, None]).T @ weights
            best = int(errors.argmin())
            error = errors[best]
            if error >= .5:
                break
            alpha = .5*np.log((1-np.clip(error, 1e-12, 1-1e-12))/max(error, 1e-12))
            self.selected.append((best, float(alpha)))
            weights *= np.exp(-alpha*y*pred[:, best])
            weights /= weights.sum()
            if error <= 1e-12:
                break
        if not self.selected:
            raise ValueError('No weak learner beats chance.')
        return self

    def _pair_predictions(self, x):
        z = (x-self.minimum)/self.span
        # FSDND denominator is sqrt(k) after training min-max scaling;
        # constant training columns have range zero and contribute no distance.
        distances = []
        for rule in self.rules:
            cols = list(rule.columns)
            ranges = ((self.training_max-self.minimum)/self.span)[cols]
            denom = np.linalg.norm(ranges)
            distances.append(np.linalg.norm(z[:, cols]-rule.values, axis=1)/max(denom, 1e-12))
        d = np.array(distances).T
        return np.column_stack([np.where(d[:, b] < d[:, m], -1, 1) for b, m in self.pairs])

    def decision_function(self, x):
        pred = self._pair_predictions(x)
        return sum(alpha*pred[:, index] for index, alpha in self.selected)


class SparseSugeno:
    """ANFIS-inspired zero-order sparse Sugeno rules with learned widths/consequents.

    Antecedent centers/subspaces are mined rules. Joint BCE optimization trains
    Gaussian premise widths and constant consequents. Not the proposal's full
    five-layer ANFIS or hybrid least-squares training algorithm.
    """
    def __init__(self, rules, maxiter=200):
        if not rules:
            raise ValueError('No rules available.')
        self.rules, self.maxiter = rules, maxiter

    def _distances(self, x):
        z = (x-self.minimum)/self.span
        return np.column_stack([np.mean((z[:, list(r.columns)]-r.values)**2, axis=1)
                                for r in self.rules])

    def fit(self, x, y, initial_model=None, scaling=None):
        # Rules enter with RAW categorical score centers; scale on current train only.
        raw_rules = self.rules
        self.minimum = x.min(0) if scaling is None else np.asarray(scaling[0]).copy()
        self.span = np.maximum(x.max(0)-self.minimum, 1.) if scaling is None else np.asarray(scaling[1]).copy()
        self.rules = [Rule(r.columns, (r.values-self.minimum[list(r.columns)])/self.span[list(r.columns)],
                           r.label, r.confidence, r.count) for r in self.rules]
        d2 = self._distances(x)
        target = (y == 1).astype(float)
        k = len(self.rules)
        logprior = np.log([r.confidence for r in self.rules])
        def objective(theta):
            logwidth, consequents = theta[:k], theta[k:]
            a = d2*np.exp(-2*logwidth)
            logfire = -.5*a + logprior
            weights = np.exp(logfire-logsumexp(logfire, axis=1)[:, None])
            score = weights @ consequents
            err = (expit(score)-target)/len(target)
            loss = np.mean(np.logaddexp(0, score)-target*score)+.001*np.mean(consequents**2)
            grad_c = weights.T@err + .002*consequents/k
            grad_w = np.sum(err[:, None]*weights*(consequents-score[:, None])*a, axis=0)
            return loss, np.r_[grad_w, grad_c]
        initial = np.r_[np.full(k, np.log(.25)), [2*r.label for r in self.rules]]
        transferred = 0
        if initial_model is not None:
            old = initial_model
            old_rules = {(r.columns, tuple(np.round(r.values*old.span[list(r.columns)]+old.minimum[list(r.columns)],9))): i
                         for i,r in enumerate(old.rules)}
            for i,r in enumerate(raw_rules):
                previous=old_rules.get((r.columns,tuple(np.round(r.values,9))))
                if previous is not None:
                    initial[k+i]=old.theta[len(old.rules)+previous]
                    if np.array_equal(self.span[list(r.columns)],old.span[list(r.columns)]):
                        initial[i]=old.theta[previous]
                    transferred+=1
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          bounds=[(np.log(.03), np.log(2.))]*k+[(-10., 10.)]*k,
                          options={'maxiter': self.maxiter})
        self.theta = result.x
        self.optimization = {'success': bool(result.success), 'message': str(result.message),
                             'iterations': int(result.nit), 'loss': float(result.fun),
                             'transferred_rules': transferred,
                             'function_evaluations': int(result.nfev)}
        return self

    def firing_weights(self, x):
        k = len(self.rules)
        logfire = -.5*self._distances(x)*np.exp(-2*self.theta[:k])+np.log([r.confidence for r in self.rules])
        return np.exp(logfire-logsumexp(logfire, axis=1)[:, None])

    def decision_function(self, x):
        return self.firing_weights(x)@self.theta[len(self.rules):]
