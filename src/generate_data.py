"""Seeded synthetic campaign dataset with a KNOWN heterogeneous treatment effect.

A promotional email (treatment) is randomized 50/50. Conversion depends on
customer features, and crucially the *uplift* (effect of the email) varies by
segment: it helps mid-value, recently-active customers and actually hurts
"sleeping dogs" (loyal customers annoyed by promos). Because we generate the
true per-customer uplift, we can score any uplift model against ground truth —
something real experiments can never do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

