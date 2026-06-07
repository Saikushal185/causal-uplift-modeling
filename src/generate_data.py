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


def make(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 365, n) / 365.0          # 0..1, lower = recent
    frequency = rng.poisson(5, n)                       # past purchases
    monetary = rng.gamma(2.0, 50, n)                    # avg spend
    is_loyal = (frequency > 8).astype(int)

    # Baseline conversion (no treatment).
    base = _sigmoid(-1.0 - 1.5 * recency + 0.05 * frequency + 0.002 * monetary)

    # True uplift (CATE): helps recent mid-value users, hurts loyal customers.
    tau = (0.18 * (recency < 0.3)
           - 0.12 * is_loyal
           + 0.08 * ((monetary > 60) & (monetary < 160)))
    tau = np.clip(tau, -0.25, 0.35)

    treat = rng.integers(0, 2, n)
    p = np.clip(base + treat * tau, 0.01, 0.99)
    converted = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "recency": recency.round(3), "frequency": frequency,
        "monetary": monetary.round(2), "is_loyal": is_loyal,
        "treatment": treat, "converted": converted,
        "true_uplift": tau.round(4),       # ground truth (not a model input!)
    })


def main() -> None:
    DATA.mkdir(exist_ok=True)
    df = make()
    df.to_csv(DATA / "campaign.csv", index=False)
    rate_t = df.loc[df.treatment == 1, "converted"].mean()
    rate_c = df.loc[df.treatment == 0, "converted"].mean()
    print(f"Wrote {len(df)} rows -> {DATA/'campaign.csv'}")
    print(f"Avg conversion: treated={rate_t:.3f}  control={rate_c:.3f}  "
          f"ATE={rate_t - rate_c:+.3f}")


if __name__ == "__main__":
    main()
