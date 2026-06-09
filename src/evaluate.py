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
