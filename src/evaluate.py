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
    return float(_trapz(curve - rand) / n)


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python3 src/generate_data.py")
    df = pd.read_csv(DATA)
    tr, te = train_test_split(df, test_size=0.3, random_state=42,
                              stratify=df["treatment"])
    Xtr, Xte = tr[FEATURES].values, te[FEATURES].values

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    results = {}
    for name, Cls in LEARNERS.items():
        model = Cls().fit(Xtr, tr["treatment"].values, tr["converted"].values)
        pred = model.predict_uplift(Xte)
        q = qini_coefficient(te["converted"].values, te["treatment"].values, pred)
        corr = float(np.corrcoef(pred, te["true_uplift"].values)[0, 1])
        results[name] = {"qini": round(q, 2), "true_uplift_corr": round(corr, 3)}
        ax.plot(np.linspace(0, 1, len(Xte)),
                qini_curve(te["converted"].values, te["treatment"].values, pred),
                label=f"{name} (Qini={q:.1f})")

    ax.plot([0, 1], [0, ax.get_ylim()[1]], "k--", alpha=0.5, label="random")
    ax.set_xlabel("Fraction of population targeted (by predicted uplift)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Qini curves — uplift meta-learners")
    ax.legend()
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / "qini_curves.png", dpi=110)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))

    print("Uplift model comparison (held-out):")
    for name, r in results.items():
        print(f"  {name:11s} Qini={r['qini']:7.1f}  "
              f"corr_with_true_uplift={r['true_uplift_corr']:.3f}")
    best = max(results, key=lambda k: results[k]["true_uplift_corr"])
    print(f"\nBest ranking model: {best}  -> reports/qini_curves.png")


if __name__ == "__main__":
    main()
