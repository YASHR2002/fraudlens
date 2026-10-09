# Model card: FraudLens fraud detector

> Status: sections marked *(Phase 13)* are completed at the end of the project. The model,
> metrics, explainability and fairness sections reflect the current champion.

## Model details

- **Name:** `fraudlens-fraud-detector`, version 2, alias `@champion` (MLflow Model Registry).
- **Type:** LightGBM gradient-boosted trees (450 trees, 26 leaves, L1/L2 regularisation),
  hyperparameters tuned with Optuna on Kaggle; class-weighted for the 0.6% fraud rate.
- **Input:** 18 behavioural features per transaction, computed in SQL for training and in
  Python for serving (parity-tested). **Output:** a fraud score in [0, 1].
- **Decision rule:** flag when score >= 0.4326, the threshold that minimised total cost (missed
  fraud amount + $5 per false alarm) on validation.

## Intended use *(Phase 13: expand)*

Ranking card transactions for review by fraud analysts, with an explanation of each decision.
Not intended to block cards automatically without human review, and not validated on real data.

## Data *(Phase 13: expand)*

Sparkov synthetic transactions (Kaggle, CC0): 1.85M transactions, 2019-2020, ~1,000 customers.
Time-based split: train to 2020-04-21, validation to 2020-06-21, test = all of fraudTest.

## Performance (test set, used once)

| Metric | Value |
|---|---|
| PR-AUC | 0.9745 |
| Recall / precision at threshold | 95.2% / 86.1% |
| Total cost | $25,150, vs $1,133,325 flagging nothing and $91,335 for the best amount rule |

## Explainability

- **SHAP (TreeSHAP, exact):** every score is decomposed into feature contributions in log-odds;
  they add up exactly to the model's raw score. Globally, merchant category, 24-hour card spend,
  amount and hour of day carry the most weight; distance to the merchant almost none.
- **Analyst notes:** a Gemini model (`gemini-3.5-flash-lite`, configurable) turns the top five
  SHAP factors into a short note with a recommended action. It receives only system-computed
  facts, never names, card numbers, gender or age. Replies are validated (the recommended action
  must agree with the model's decision; no protected terms; no card numbers) and, after one
  retry, replaced by a deterministic template note if the LLM fails. The API therefore never
  depends on the LLM being available.

## Fairness

Measured on the test set at the production threshold (`fraudlens fairness-audit`, full tables in
`reports/fairness.md`). Gender is **not** a model input; age **is** (`age_at_txn`).

| Group | Recall | Precision | False positive rate |
|---|---|---|---|
| Female | 93.3% | 84.4% | 0.066% |
| Male | 97.5% | 88.0% | 0.052% |
| Under 30 | 98.7% | 82.2% | 0.073% |
| 30-49 | 92.0% | 82.0% | 0.067% |
| 50-69 | 96.6% | 90.2% | 0.051% |
| 70+ | 97.3% | 93.8% | 0.031% |

**Gaps that exceed their 95% confidence intervals:**

1. **Recall by gender:** fraud against women's cards is caught less often (93.3% vs 97.5%), so
   women bear more of the missed-fraud burden, even though gender is not an input. The model can
   still pick up gender indirectly through correlated features (spending categories, amounts,
   times).
2. **Recall and precision by age:** the 30-49 band has the lowest recall (92.0%) and precision
   (82.0%).
3. **False positive rate by age:** legitimate transactions of customers under 30 are wrongly
   flagged 2.4 times as often as those of customers 70+ (0.073% vs 0.031%). In absolute terms that
   is about 7 in 10,000 transactions, i.e. roughly 69 of the under-30 group's 94,000.

**Discussion.** Sparkov generates customers from demographic profiles (age band, gender,
urban/rural), so spending and fraud behaviour genuinely differ by group in this data; part of the
gaps may reflect that simulation design rather than model bias, and on synthetic data the two
cannot be separated cleanly. The audit deliberately measures and reports without "fixing":
possible responses in a real deployment include group-aware threshold review, dropping or
monotone-constraining age, or reweighting, each with a trade-off against overall cost that the
business and compliance would need to decide. Age remains an input because it is a strong,
legitimate risk signal here; that choice should be revisited if the gaps persist on real data.

## Limitations *(Phase 13: expand)*

- Synthetic data: scores are higher than real fraud detection would achieve.
- No fraud above $1,500 exists in the data, so very large purchases are treated as low-risk
  (`docs/decisions.md` D3.3).
- The decision threshold drifted: on the test period a much lower threshold would have halved
  the cost (`docs/decisions.md` D6.4); it must be monitored.

## Ethical considerations *(Phase 13)*
