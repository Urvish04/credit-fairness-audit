# Robust Fairness Audit of Credit Approval Models

**TL;DR:** Audited Logistic Regression and Random Forest for demographic bias on the German Credit dataset. An initial single-split finding — Random Forest showing a larger disparate impact than Logistic Regression on the "young" subgroup (DIR 0.93 → 0.89) — did not survive statistical stress-testing. Multi-seed resampling and bootstrapped confidence intervals both show the effect is within noise at this sample size (N=47). The result is not a novel discovery; it is a concrete demonstration of a known but frequently ignored risk: **single-split fairness metrics on small subgroups are statistically unstable and require a stability check before being reported as a finding.**

## Motivation

Model risk management guidance — most notably the Federal Reserve/OCC's *Supervisory Guidance on Model Risk Management* (SR 11-7) — requires that model outputs, including fairness and bias metrics, be validated for stability before being used in decisioning. Most introductory fairness projects report a single DIR or demographic-parity number and stop there. This project demonstrates why that single number is not sufficient evidence on its own: **does the metric survive resampling**, or does it evaporate under a stability check? That question — not the audit's headline numbers — is the point of this repository.

## Dataset

[German Credit Data](https://www.openml.org/d/31) (`credit-g`) via OpenML — 1,000 loan applicants, binary target (good/bad credit risk).

Protected attributes extracted from raw fields:
- **Sex** — parsed from `personal_status` (e.g. "male single", "female div/dep/mar")
- **Age group** — binarized at age 25 (young / old), per convention in fairness literature for this dataset
- **Foreign worker status** — excluded from subgroup analysis (37 "no" values across the full dataset — too imbalanced to audit)

All protected attributes were excluded from model training features (blind training); reattached only at evaluation time for the audit.

## Pipeline

```
German Credit Dataset (OpenML)
        │
        ▼
Protected Attribute Extraction        (sex, age_group)
        │
        ▼
Blind Feature Encoding                (protected attributes dropped from X)
        │
        ▼
Baseline Model Training               (Logistic Regression, 70/30 split)
        │
        ▼
Fairness Audit vs. EEOC 4/5ths Rule   (approval rate, FNR, DIR per subgroup)
        │
        ▼
Model Comparison                      (Random Forest, identical split)
        │
        ▼
Multi-Seed Robustness Check           (5 independent splits)
        │
        ▼
Bootstrapped Confidence Intervals     (1,000 resamples, test-set predictions)
        │
        ▼
fairlearn Verification                (cross-check manual DIR calculations)
```

## Method

1. **Baseline model** — Logistic Regression, 70/30 stratified split. Verified no protected/derived attribute leaked into training features via feature inspection (not accuracy alone).
2. **Fairness audit** — Approval rate, false negative rate, and Disparate Impact Ratio (DIR) computed per subgroup on the test set, against the EEOC "four-fifths rule" (DIR < 0.80 = adverse impact).
3. **Model comparison** — Random Forest trained on the identical split to test whether a non-linear model changes fairness characteristics.
4. **Multi-seed robustness check** — Both models re-evaluated across 5 independent splits (`random_state` = 42, 100, 2026, 777, 99). This captures variance from **both** the train/test split *and* re-fitting the model — i.e., "would this conclusion hold if I trained again on different data?"
5. **Bootstrapped confidence intervals** — 1,000 resamples with replacement drawn from the fitted models' test-set predictions, used to compute exact 95% CIs and empirical violation rates for DIR. This isolates **pure test-set sampling variance** — i.e., "given this exact trained model, how much does the DIR estimate wobble just from which 47 people happened to land in the test set?" It's a narrower, complementary question to the multi-seed check, not a repeat of it.
6. **Verification** — Manual calculations cross-checked against `fairlearn.metrics.demographic_parity_difference` — exact agreement confirmed.

## Results

**Single-split baseline (seed 42).** DIR is expressed relative to the majority/reference group, so Male and Old are the reference points (DIR = 1.00 by definition) — they are not missing data.

| Metric | Male | Female | Old | Young |
|---|---|---|---|---|
| Test-set N | 204 | 96 | 253 | 47 |
| Approval rate (LR) | 77.94% | 73.96% | 77.47% | 72.34% |
| DIR — Logistic Regression | 1.00 (ref) | 0.95 | 1.00 (ref) | 0.93 |
| DIR — Random Forest | 1.00 (ref) | 0.96 | 1.00 (ref) | 0.89 |

**Initial (single-split) observation:** RF appeared to increase disparate impact on the young subgroup relative to LR (DIR 0.93 → 0.89) — suggestive of the model amplifying an age-related proxy signal. This observation is addressed, and shown not to hold, below.

![DIR Variance Chart](https://github.com/Urvish04/credit-fairness-audit/blob/main/dir_variance_chart.png?raw=true)

## Statistical Robustness

**1. Multi-seed variance (model + split variance).** Re-running both models across 5 seeds shows the young-subgroup DIR swings by roughly the same magnitude as the effect originally attributed to model choice:

- Logistic Regression: range 0.828–0.944 (std ≈ 0.049)
- Random Forest: range 0.852–0.951 (std ≈ 0.038)

A 0.04-point difference between two models means very little when each model's own DIR estimate moves by ~0.05 just from re-splitting the data.

**2. Bootstrap variance and empirical violation rate (pure sampling variance).** Holding the trained models fixed and resampling only the test-set predictions (N=1,000 iterations) gives the following for the young subgroup:

| Model | 95% CI | Mean | % of iterations with DIR < 0.80 |
|---|---|---|---|
| Logistic Regression | 0.762 – 1.110 | 0.937 | 12.1% |
| Random Forest | 0.744 – 1.056 | 0.899 | 9.5% |

These intervals overlap substantially, and both models cross the EEOC 0.80 threshold in a non-trivial share of resamples — meaning either model could be flagged as "adverse impact" or "compliant" depending on which 47 people happened to be in the test set, independent of any model-training variance at all.

Both bootstrap distributions are **right-skewed rather than normal** — the mean is pulled upward by infrequent, high-DIR outlier resamples. This is a known characteristic of bootstrapped ratio metrics on small subgroups (the denominator can occasionally be very small, producing large ratio spikes) and explains why a normal approximation from the CI alone understates the true violation rate: the empirical 12.1% figure for LR is higher than a Gaussian approximation using the same mean and CI width would predict, because the skew concentrates extra density in the lower (sub-0.80) tail.

![Bootstrap Distribution Chart](https://github.com/Urvish04/credit-fairness-audit/blob/main/bootstrap_distribution_chart.png?raw=true)

**3. Verification.** `fairlearn.metrics.demographic_parity_difference` (Sex = 0.0398, Age = 0.0513) exactly matched manual calculations — ruling out implementation error as a source of the observed instability.

## Discussion

Two independent variance checks — one capturing model re-training + re-splitting, the other capturing pure test-set resampling — arrive at the same conclusion from different angles: the RF-vs-LR fairness gap does not survive resampling. At N=47, the Disparate Impact Ratio is not a stable estimate for either model; a single unlucky (or lucky) split can make a linear model look fairer, or less fair, than a non-linear one purely by chance, and roughly 1 in 10 resamples for either model crosses the regulatory threshold on sampling noise alone. The correct audit conclusion isn't "Random Forest is more biased" — it's that **this subgroup's sample size cannot support that conclusion**, and any fairness metric reported from a single train/test split, without a stability check, is liable to produce a false conclusion in either direction. This is a demonstration of a known model-risk failure mode, consistent with the validation standards set out in SR 11-7, applied concretely to a fairness-metric context.

## Limitations

- **Small-N bootstrap constraint.** Bootstrapping itself is not immune to the small-sample problem it's used to diagnose — resampling from a 47-row subgroup produces a CI, and a violation rate, that are themselves noisy estimates. Using the standard sample-size formula for a proportion (n ≈ z²p(1−p)/E², z=1.96, p≈0.75, E=0.05), a stable estimate with a ±0.05 margin of error at this approval rate would need a minority subgroup of roughly **N ≈ 290–300** — more than 6x the 47 available here. This is a constraint on what conclusions the data can support, not a flaw introduced by the analysis.
- **Generalizability.** German Credit is a widely-used academic benchmark from 1994, not a production lending dataset — the finding describes a methodological failure mode (small-N fairness metrics are unreliable), not a real-world lending outcome. It should not be read as "fairness audits don't work," only as "fairness audits need adequate subgroup N to work."
- Fairness metrics used (DIR, demographic parity, FNR gap) are outcome-based; no causal claims are made about the source of disparity.

## Reproducibility

```bash
pip install -r requirements.txt
```

Open `credit_fairness_audit.ipynb` in Jupyter or Google Colab and run cells top to bottom. Expected runtime: under 2 minutes (no GPU required). Section 1 confirms data load (1,000 rows); Section 3 baseline accuracy should land at ~74–75%; Section 4 reproduces the subgroup tables above; Section 5 reproduces the bootstrap CIs and violation rates.

## Stack

`Python` · `pandas` · `scikit-learn` · `fairlearn` · `matplotlib` · `Google Colab`

## Repository structure

```
.
├── credit_fairness_audit.ipynb
├── bootstrap_distribution_chart.png
├── dir_variance_chart.png
├── requirements.txt
└── README.md
```

## References

- Hofmann, H. German Credit Data. UCI Machine Learning Repository / OpenML (dataset id 31).
- Weerts, H. et al. Fairlearn: Assessing and Improving Fairness of AI Systems. Microsoft, 2023.
- U.S. Equal Employment Opportunity Commission. Uniform Guidelines on Employee Selection Procedures — the "four-fifths rule."
- Board of Governors of the Federal Reserve System / Office of the Comptroller of the Currency. *Supervisory Guidance on Model Risk Management* (SR 11-7), 2011.
