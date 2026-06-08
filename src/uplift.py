"""Uplift (CATE) meta-learners built on top of any base classifier.

Implements the three standard meta-learners:
  - S-learner: one model with treatment as a feature; uplift = f(x,1) - f(x,0)
  - T-learner: separate treated/control models; uplift = f_t(x) - f_c(x)
  - X-learner: T-learner + imputed-effect models reweighted by propensity
These mirror EconML/CausalML APIs without the heavy dependency.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

FEATURES = ["recency", "frequency", "monetary", "is_loyal"]


def _base_clf():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      learning_rate=0.05, random_state=0)


class SLearner:
    def __init__(self, base=None):
        self.model = base or _base_clf()

    def fit(self, X, t, y):
        self.model.fit(np.column_stack([X, t]), y)
        return self

    def predict_uplift(self, X):
        ones = np.column_stack([X, np.ones(len(X))])
        zeros = np.column_stack([X, np.zeros(len(X))])
        return self.model.predict_proba(ones)[:, 1] - \
            self.model.predict_proba(zeros)[:, 1]


class TLearner:
