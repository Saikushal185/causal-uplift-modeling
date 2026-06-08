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
