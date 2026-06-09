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
    def __init__(self, base=None):
        base = base or _base_clf()
        self.m_t, self.m_c = clone(base), clone(base)

    def fit(self, X, t, y):
        self.m_t.fit(X[t == 1], y[t == 1])
        self.m_c.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        return self.m_t.predict_proba(X)[:, 1] - self.m_c.predict_proba(X)[:, 1]


class XLearner:
    def __init__(self, base=None):
        base = base or _base_clf()
        self.m_t, self.m_c = clone(base), clone(base)
        self.tau_t = GradientBoostingRegressor(max_depth=3, random_state=0)
        self.tau_c = GradientBoostingRegressor(max_depth=3, random_state=0)
        self.propensity = 0.5

    def fit(self, X, t, y):
        self.m_t.fit(X[t == 1], y[t == 1])
        self.m_c.fit(X[t == 0], y[t == 0])
        # Imputed treatment effects.
        d_t = y[t == 1] - self.m_c.predict_proba(X[t == 1])[:, 1]
        d_c = self.m_t.predict_proba(X[t == 0])[:, 1] - y[t == 0]
        self.tau_t.fit(X[t == 1], d_t)
        self.tau_c.fit(X[t == 0], d_c)
        self.propensity = float(t.mean())
        return self

    def predict_uplift(self, X):
        g = self.propensity
        return g * self.tau_c.predict(X) + (1 - g) * self.tau_t.predict(X)


LEARNERS = {"S-learner": SLearner, "T-learner": TLearner, "X-learner": XLearner}
