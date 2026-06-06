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
