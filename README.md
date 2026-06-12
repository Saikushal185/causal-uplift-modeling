# 🎯 Causal Inference — Uplift Modeling

"Which customers should we target?" — answered with **uplift modeling** (CATE
estimation), not a churn/response score. The project implements the three
standard meta-learners and evaluates them with **Qini curves** against a
**known ground-truth treatment effect**.

## Why uplift, not classification
A response model predicts *who will convert*. An uplift model predicts *who
converts **because** of the treatment* — and it can spot **"sleeping dogs"**
(customers a promo actually drives away). Targeting by uplift beats targeting by
predicted response whenever the effect is heterogeneous.

## What this project demonstrates
- **Meta-learners from scratch** — S-learner, T-learner, X-learner on top of any
  base estimator (mirrors EconML/CausalML APIs, no heavy dependency).
- **Honest evaluation** — Qini coefficient (incremental ranking quality) plus,
  because the data is synthetic, direct **correlation with the true CATE**.
- **Heterogeneous, negative effects** — the generator bakes in segments the
  treatment helps *and* hurts, so the models have something real to discover.

## Demo

```text
$ python3 src/generate_data.py
Avg conversion: treated=0.311  control=0.228  ATE=+0.083

$ python3 src/evaluate.py
Uplift model comparison (held-out):
  S-learner   Qini=   82.3  corr_with_true_uplift=0.895
  T-learner   Qini=   77.3  corr_with_true_uplift=0.829
  X-learner   Qini=   79.2  corr_with_true_uplift=0.873

Best ranking model: S-learner  -> reports/qini_curves.png
```

The average treatment effect is +8.3 points, but the value is in the
*heterogeneity*: the models recover individual uplift at ~0.85–0.90 correlation
with ground truth, so targeting the top deciles captures far more incremental
conversions than blanket sending (the gap from the dashed "random" line in
`reports/qini_curves.png`).

## Methods

| Learner | Idea |
|---|---|
