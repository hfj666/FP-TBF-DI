"""A shared-width HVDM objective, using the original sum normalization."""
import numpy as np
from scipy.special import expit, logsumexp


def objective(theta, distance, prior, labels, target):
    consequents = theta[1:]
    scaled = distance * np.exp(-2 * theta[0])
    logfire = prior - .5 * scaled
    weights = np.exp(logfire - logsumexp(logfire, axis=1, keepdims=True))
    score = weights @ consequents
    loss = np.mean(np.logaddexp(0, score) - target * score) + .001 * np.mean(consequents**2)
    error = (expit(score) - target) / len(target)
    width_gradient = np.sum(error[:, None] * weights * (consequents - score[:, None]) * scaled)
    return loss, np.r_[width_gradient, weights.T @ error + .002 * consequents / len(labels)]
