"""Simple cumulative comparator with frozen initial scaling; no tuning."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class CumulativeLogistic:
    def __init__(self,x,y,scaling):
        self.minimum=np.array(scaling[0],copy=True);self.span=np.array(scaling[1],copy=True)
        self.fit(x,y)

    def fit(self,x,y):
        self.model=LogisticRegression(C=1.,solver='lbfgs',max_iter=500)
        self.model.fit((x-self.minimum)/self.span,y)
        return self

    def decision_function(self,x):
        return self.model.decision_function((x-self.minimum)/self.span)
