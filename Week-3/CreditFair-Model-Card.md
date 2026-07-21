# Model Card — CreditFair

*A responsible credit-scoring model for the German Credit (Statlog) dataset. Prepared for the Responsible-AI & Data-Ethics project (SRH Heidelberg, SS 2026). This card follows the AI Act Annex IV / model-card convention.*

## Model details
- **Name / version:** CreditFair v1 (Week 3).
- **Type:** Two-stage — a **Random-Forest risk scorer** (400 trees, class-weighted) produces P(bad); an **intersectional, cost-aware ExponentiatedGradient** layer (Demographic Parity, ErrorRate 1:5) makes the fair accept/reject decision. Every decision is then routed to a human by priority tier.
- **Interface:** `CreditFairModel.decide(applicant_row, group) → {risk_score, decision, review, reasons}` (`src/creditfair.py`).
- **Reproducibility:** seeded (`RS=42`); identical on rerun on the same machine. Tree-model DIs are hardware-sensitive on a 200-row test (see Limitations).

## Intended use
- **Purpose:** decision *support* for consumer-credit approval — the model **advises**, a human decides.
- **Users:** trained credit officers with the review dashboard/queue.
- **Out of scope:** any fully-automated final decision; use on populations unlike 1990s German applicants without re-validation; any use of the protected attributes as decision inputs.

## Factors (groups audited)
Sex, age band, foreign-worker status — audited and mitigated, **never** used as decision inputs (legal safeguard against direct discrimination, AGG §3(1)). `foreign_worker` is treated as a proxy for ethnic origin.

## Data
- **Source:** UCI Statlog German Credit (Prof. Hans Hofmann, 1994), 1,000 applicants, 20 attributes, 30% "bad".
- **Split:** 600 train / 200 validation / 200 test, stratified; numeric outliers capped at the train 99th pct (no leakage).
- **Cost matrix:** official 5:1 (a bad loan approved costs 5× a good loan refused).

## Metrics (our reference run)
- **Performance (RF scorer):** ROC-AUC ≈ 0.79; at the cost-tuned threshold, FNR (bad approved) ≈ 23%, FPR (good refused) ≈ 32%; expected cost ≈ 115 (95% CI [82, 151]); validation and test AUC are close (generalises).
- **Fairness (baseline → after mitigation):**
  - sex DI 0.84 → **~0.95** (clears 0.80)
  - foreign DI 0.61 → **~0.87** (clears 0.80)
  - age DI 0.56 → **~0.77–0.84 (borderline — clears on some machines, just under on others)**
  - Cost of mitigation: +25 (115 → ~140). Demographic Parity narrows *approval* gaps but slightly widens *equal-opportunity* gaps — reported honestly.
- **Amplification:** the baseline model widens all three gaps beyond what applicants' actual risk warrants (e.g. foreign 0.79 risk-justified vs 0.61 delivered) — the reason mitigation is required, not optional.

## Human oversight (AI Act Art. 14 / GDPR Art. 22)
No decision is auto-finalised. All applications are routed to a human by tier: **P1** every rejection (mandatory adverse review), **P2** borderline approvals, **P3** approvals in disadvantaged subgroups, **P4** clear approvals (sampled). Reason codes + an adverse-action notice support the GDPR right to explanation and contestation.

## Ethical & legal considerations
- Removing protected attributes is a **legal safeguard, not a fairness fix**: proxies (marital status, job, housing) reconstruct the trait; a counterfactual test shows ~6–7% of decisions flip through proxies while 0 flip from the trait itself.
- The residual disparity is addressed with in-processing mitigation that never uses a protected attribute at decision time; group-thresholding (post-processing) is **rejected** as direct discrimination.
- An objective-justification file (AGG §3(2)) is documented: legitimate aim (creditworthiness), evidence the raw gap was *un*justified (amplification + calibration), proportionate remedy (EG at modest cost), plus oversight.

## Caveats & limitations
- **Small test set (200 rows):** wide bootstrap CIs; DI figures are directional. **Age DI is machine-sensitive** around the 0.80 line — we flag it and cover it with human review rather than over-claim "all three pass."
- **Selection bias / reject inference:** labels exist only for applicants a historical bank *approved*; training on a censored "approved-only" subset collapses AUC (~0.79 → ~0.55) and distorts fairness. Proper reject inference is future work.
- **Vintage data:** 1994 German lending; not representative of a current population.
- **Explainability scope:** SHAP explains the RF *risk score*; the EG constraint is a training-time adjustment documented separately, not per-feature.

## Maintenance & monitoring
- **Drift:** Population Stability Index on key inputs; PSI > 0.2 triggers a re-audit before trusting the score.
- **Fairness regression:** an automated test fails the build if mitigation stops improving sex DI or drops below 0.80.
- **Cadence:** re-audit fairness + performance on each new data vintage; keep the environment pinned for reproducibility.

## Testing
24 unit tests, ~99% line coverage (`tests/`), including no-leakage, cost/DI correctness, P1–P4 routing, reproducibility, and the fairness-regression guard.
