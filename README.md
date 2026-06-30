# Loan Default Risk Prediction

End-to-end credit risk analysis and machine learning project predicting loan default risk using the Lending Club loans dataset. Covers data cleaning, statistical hypothesis testing, feature engineering, model comparison, threshold diagnosis, and business recommendations.

## Business Problem

A lending institution needs to assess borrower creditworthiness **before loan approval**. The goal is to identify likely defaulters using only information available at application time — with no reliance on post-approval data that would leak the outcome.

Since a missed defaulter (approving a bad loan) costs the bank more than a false alarm (rejecting a good customer), **recall** — not accuracy — is treated as the priority metric throughout this project.

## Dataset

- Source: Lending Club loans dataset (`loans_full_schema.csv`)
- 10,000 raw loan applications → 8,505 after cleaning (individual applications only)
- Severe class imbalance: 141 defaulters (1.7%) vs 8,364 non-defaulters

## Project Structure

The analysis is organized into 12 phases inside a single Jupyter notebook:

1. **Business Understanding** — problem framing, cost asymmetry (FN vs FP)
2. **Data Understanding** — column inventory, target variable construction
3. **Data Cleaning** — leakage removal, structural null handling, sentinel values
4. **Exploratory Data Analysis** — 10 visualizations using default rates (not counts) given class imbalance
5. **Statistical Hypothesis Testing** — 5 tests at α = 0.01 (Mann-Whitney U, Chi-Square), with Q-Q plot–based assumption checking
6. **Feature Engineering** — 3 engineered features: `loan_to_income_ratio`, `credit_stress_score` (dimensionally normalized), `delinquency_risk_score`
7. **Preprocessing** — encoding, scaling (fit on train only), 80/20 stratified split
8. **Model Building** — Logistic Regression, Decision Tree, Random Forest, XGBoost
9. **Model Evaluation** — classification reports, ROC-AUC, RF/XGBoost threshold diagnosis, precision-recall tradeoff, 5-fold cross-validation, SMOTE comparison, hyperparameter tuning
10. **Feature Importance** — Random Forest importance extraction and interpretation
11. **Business Recommendations** — quantified, evidence-based recommendations
12. **Conclusion** — summary of findings and model performance

## Key Findings

- Of 5 hypothesis tests, only **home ownership** (p = 0.0089) and **loan purpose** (p = 0.0001) were statistically significant in isolation. Income, debt-to-income ratio, and employment length were not — though DTI later proved highly predictive in combination with other variables (top feature in Random Forest importance).
- **Random Forest and XGBoost both achieved 0.00 recall** despite class-imbalance handling (`class_weight='balanced'`, `scale_pos_weight`). This was diagnosed, not just reported: both models' predicted probabilities for actual defaulters topped out at 0.12 and 0.28 respectively — far below the 0.5 decision threshold — a known limitation of tree ensemble averaging on severely imbalanced data.
- **SMOTE was tested against `class_weight='balanced'`** and underperformed it on both recall and ROC-AUC for Logistic Regression.
- **5-fold cross-validation** showed the single train/test split recall (64%) sits at the high end of a wider, less certain range (50.4% ± 11.5%), given only 28 defaulters in the test set.
- **Hyperparameter tuning** (GridSearchCV on `C`) confirmed the default regularization was already optimal — no performance gain, indicating the model's ceiling is set by data signal and class imbalance, not tuning.

## Model Results

| Model               | Accuracy | Precision | Recall | ROC-AUC |
|---------------------|----------|-----------|--------|---------|
| **Logistic Regression** | 0.71 | 0.04 | **0.64** | **0.69** |
| Decision Tree        | 0.97     | 0.09      | 0.07   | 0.53    |
| Random Forest         | 0.98     | 0.00      | 0.00   | 0.55    |
| XGBoost                | 0.98     | 0.00      | 0.00   | 0.60    |

**Selected model: Logistic Regression**, deployed as a pre-screening flag for manual underwriter review rather than an automated approval/rejection system, given its low precision (4%).

## Business Recommendations

1. Incorporate debt-to-income ratio into a **combined risk score** rather than a standalone cutoff — DTI alone is not statistically significant, but a hard threshold would still flag ~2,126 applicants to catch only 42 of 141 historical defaulters
2. Prioritize **debt consolidation loans** for underwriting review — this single category accounts for 47% of all historical defaults due to high loan volume, despite a modest 2% default rate
3. Deploy Logistic Regression as a **pre-screening tool**, not a final decision-maker, given its precision-recall tradeoff

## Tech Stack

`pandas` · `numpy` · `scipy` · `scikit-learn` · `xgboost` · `imbalanced-learn` · `matplotlib` · `seaborn`

## How to Run

```bash
pip install pandas numpy scipy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

1. Clone this repo
2. Ensure `loans_full_schema.csv` is in the same directory as the notebook
3. Run `Loan_default_prediction_project.ipynb` top to bottom

## Limitations

- Test set contains only 28 defaulters, leading to high metric variance (confirmed via cross-validation)
- Precision remains low (4%) at the selected model's default threshold — not suitable as a standalone automated decision system
- Hyperparameter tuning was limited to Logistic Regression's `C` parameter only
