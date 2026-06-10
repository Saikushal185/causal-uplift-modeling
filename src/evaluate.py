"""Train and compare uplift meta-learners.

Metrics:
  - Qini coefficient: area between the model's Qini curve and random targeting
    (higher = better customer ranking by incremental effect).
  - True-uplift correlation: because the data is synthetic, we can correlate
    predicted uplift with the ground-truth CATE.
Also writes a Qini-curve comparison plot.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift import FEATURES, LEARNERS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "campaign.csv"
REPORTS = ROOT / "reports"

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def qini_curve(y, t, uplift):
    """Cumulative incremental conversions as we target by descending uplift."""
    order = np.argsort(uplift)[::-1]
    y, t = np.asarray(y)[order], np.asarray(t)[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    yt = np.cumsum(y * t)
    yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = yt - yc * (n_t / np.maximum(n_c, 1))
    return np.nan_to_num(gain)


def qini_coefficient(y, t, uplift) -> float:
    curve = qini_curve(y, t, uplift)
    n = len(curve)
    rand = np.linspace(0, curve[-1], n)
