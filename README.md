# Robust Fairness Audit of Credit Approval Models

**TL;DR:** Audited Logistic Regression and Random Forest for demographic bias on the German Credit dataset. An apparent fairness difference between models disappeared after evaluation across five train/test splits — the real result is that single-split fairness metrics on small subgroups (N=47) are statistically unstable and shouldn't be trusted without a stability check.

## Motivation

Algorithmic lending bias is an active regulatory concern — the Reserve Bank of India has flagged disparate treatment risk in automated credit decisioning as institutions increasingly rely on ML-based approval systems. Unlike many introductory ML projects that focus primarily on predictive accuracy, this repository evaluates whether fairness *conclusions themselves* remain reproducible under repeated sampling.

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
Protected Attribute Extraction  (sex, age_group)
        │
        ▼
Blind Feature Encoding  (protected attributes dropped from X)
        │
        ▼
Baseline Model Training  (Logistic Regression, 70/30 split)
        │
        ▼
Per-Subgroup Fairness Evaluation  (approval rate, FNR, DIR)
        │
        ▼
Model Comparison  (Random Forest, same split)
        │
        ▼
Multi-Seed Robustness Check  (5 splits, stability test)
        │
        ▼
Verification  (fairlearn cross-check)
```

## Method

1. **Baseline model** — Logistic Regression, 70/30 stratified split. Verified no protected/derived attribute leaked into training features via feature inspection (not accuracy alone).
2. **Fairness audit** — Approval rate, false negative rate, and Disparate Impact Ratio (DIR) computed per subgroup on the test set, against the EEOC "4/5ths rule" (DIR < 0.8 = adverse impact).
3. **Model comparison** — Random Forest trained on the identical split to test whether a non-linear model changes fairness characteristics.
4. **Robustness check** — Both models re-evaluated across 5 independent splits (`random_state` = 42, 100, 2026, 777, 99) to test whether the single-split DIR result holds.
5. **Verification** — Manual calculations cross-checked against `fairlearn.metrics.demographic_parity_difference` — exact agreement confirmed.

## Results

| Metric | Male | Female | Old | Young |
|---|---|---|---|---|
| Test-set N | 204 | 96 | 253 | 47 |
| Approval rate (LR) | 77.94% | 73.96% | 77.47% | 72.34% |
| DIR (LR baseline) | — | 0.95 | — | 0.93 |

![Approval rate by subgroup](images/approval_rate_by_group.png)

**Single-split result (seed 42):** RF appeared to increase disparate impact on the young subgroup vs. LR (DIR 0.93 → 0.89) — suggestive of the model amplifying an age-related proxy.

**Multi-seed result:** DIR estimates for *both* models swing by ~0.10–0.12 across seeds (LR: 0.828–0.944, std ≈ 0.049 · RF: 0.852–0.951, std ≈ 0.038) — variance of the same magnitude as the effect initially attributed to model choice.

![DIR Variance Chart](https://github.com/Urvish04/credit-fairness-audit/blob/main/dir_variance_chart.png?raw=true)

`fairlearn.metrics.demographic_parity_difference`: Sex = 0.0398, Age = 0.0513 — matches manual calculations exactly.

## Discussion

The apparent RF-vs-LR fairness gap does not survive re-sampling. At N=47, the Disparate Impact Ratio is not a stable estimate for either model — a single unlucky (or lucky) split can make a linear model look fairer or less fair than a non-linear one purely by chance. The correct audit conclusion isn't "Random Forest is more biased," it's that **this dataset's subgroup sample size cannot support that conclusion**, and that any fairness metric reported from a single train/test split — without a stability check — is liable to produce false conclusions in either direction.

## Limitations

- Test-set subgroup sizes (female N=96, young N=47) limit statistical power, especially for age-group analysis.
- German Credit is a widely-used academic benchmark, not a production lending dataset — findings describe methodology, not real-world lending outcomes.
- Fairness metrics used (DIR, demographic parity, FNR gap) are outcome-based; no causal claims are made about the source of disparity.

## Reproducibility

```bash
pip install -r requirements.txt
```

Open `credit_fairness_audit.ipynb` in Jupyter or Google Colab and run cells top to bottom. Expected runtime: under 2 minutes (no GPU required). Section 1 confirms data load (1000 rows); Section 3 baseline accuracy should land at ~74–75%; Section 4 reproduces the subgroup tables above.

## Stack

`Python` · `pandas` · `scikit-learn` · `fairlearn` · `matplotlib` · `Google Colab`

## Repository structure

```
.
├── credit_fairness_audit.ipynb
├── images/
│   ├── approval_rate_by_group.png
│   └── dir_across_seeds.png
├── requirements.txt
└── README.md
```

## References

- Hofmann, H. German Credit Data. UCI Machine Learning Repository / OpenML (dataset id 31).
- Weerts, H. et al. Fairlearn: Assessing and Improving Fairness of AI Systems. Microsoft, 2023.
- U.S. Equal Employment Opportunity Commission. Uniform Guidelines on Employee Selection Procedures — the "four-fifths rule."
