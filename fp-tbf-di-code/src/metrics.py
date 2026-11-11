"""Metrics reported by the FP-TBF-DI command line entry point."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    pred = np.where(score >= 0, 1, -1)
    pos, neg = y == 1, y == -1
    return dict(n=len(y), positive=int(pos.sum()), negative=int(neg.sum()), correct=int((pred == y).sum()),
                accuracy=float(accuracy_score(y, pred)),
                recall=float((pred[pos] == 1).mean()) if pos.any() else None,
                specificity=float((pred[neg] == -1).mean()) if neg.any() else None,
                f1=float(f1_score(y, pred, pos_label=1)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                auc=float(roc_auc_score(y == 1, score)) if pos.any() and neg.any() else None)
