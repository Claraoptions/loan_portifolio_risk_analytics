# Loan Portfolio Risk Analytics
**Clare Mujuni | Banking Data Analyst Portfolio**

---

## Business Problem

Credit default is the primary source of capital loss for commercial banks. 
In Uganda, the Bank of Uganda sets a Non-Performing Assets (NPA) threshold of 4% — 
breaching it triggers regulatory intervention and reputational damage. 

This project builds a loan default prediction system using real applicant data, 
replicating the kind of risk analysis performed in a bank's credit department. 
The goal: identify high-risk applicants before loan approval, not after default.

---

## Dataset

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)  
**Size:** 307,511 loan applications | 122 features  
**Target variable:** `TARGET` — 1 = defaulted, 0 = repaid  
**Class distribution:** 8.1% defaulters (highly imbalanced)

Secondary file: `bureau.csv` — 1.7M credit bureau records (external credit history)

*Note: Raw data files are excluded from this repository due to file size. 
Download from Kaggle and place in the project root to reproduce results.*

---

## Project Structure

loan-portfolio-risk-analytics/
-- 01_eda_cleaning.ipynb    # Full analysis notebook
--.gitignore
-- README.md

---

## Methodology

### 1. Exploratory Data Analysis & Cleaning
- Audited 122 columns for missing values, outliers, and data quality issues
- Decoded sentinel values: `DAYS_EMPLOYED = 365,243` flags non-employed applicants
- Converted negative day columns (`DAYS_BIRTH`, `DAYS_EMPLOYED`) to interpretable years
- Feature engineering: `AGE_YEARS`, `EMPLOYMENT_YEARS`, `DEBT_TO_INCOME`, `IS_NOT_EMPLOYED`

### 2. Risk Segmentation Analysis (SQL)
Key findings from segmentation:
- Applicants **under 25** default at **10%** — 2× the portfolio average
- **Maternity leave** income type: **40% default rate** — highest risk segment
- **Non-employed** applicants approved and tracked: **11.6% default rate**
- High loan-to-income ratios correlate with elevated default probability

### 3. Class Imbalance Problem
With 91.9% good loans and 8.1% defaulters, a naive model predicts "good loan" 
for everyone — achieving 92% accuracy while catching **zero defaulters**. 
Addressed using `class_weight='balanced'` in all models.

### 4. Models Built

| Model | AUC Score | Notes |
|---|---|---|
| Logistic Regression (baseline) | 0.642 | With StandardScaler + class balancing |
| Random Forest (application data) | 0.656 | 100 trees, max_depth=10 |
| Random Forest (+ bureau features) | 0.665 | Enriched with credit history |

**Primary metric: AUC-ROC** — measures ability to rank defaulters above good payers, 
unaffected by class imbalance. Accuracy is misleading on imbalanced datasets.

### 5. Bureau Data Enrichment
Aggregated 1.7M bureau records to applicant level, engineering 6 credit history features:
- Total previous credits, cumulative overdue days, maximum overdue episode
- Total outstanding debt, overdue amounts, loan restructurings (prolongations)

**Finding:** Applicants with any overdue history default at **15.9%** — 
nearly double the portfolio average. Validates standard credit policy requiring 
enhanced review for applicants with previous overdue payments.

### 6. Feature Importance (Random Forest)
Top predictors of default:
1. **Age** — younger applicants carry significantly higher risk
2. **Loan amount** — higher credit exposure increases default probability  
3. **Annuity / repayment amount** — affordability constraint
4. **Debt-to-income ratio** — engineered feature, strong signal
5. **Employment years** — stability indicator

---

## Key Business Findings

- The portfolio's **8.1% default rate** would breach BOU's 4% NPA threshold — 
  indicating the need for tighter credit scoring at origination
- **Age and loan size** are the two strongest predictors — a finding consistent 
  with credit policy at DFCU and standard banking practice
- Bureau enrichment added **+0.9pp AUC** — limited by only 1.1% of applicants 
  having overdue history in this dataset. With richer bureau coverage, 
  the improvement would be substantially larger
- **False Negatives (approved bad loans) are the most dangerous model error** — 
  representing principal loss, recovery costs, and regulatory risk

---

## Tools & Libraries

- Python 3 | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn
- SQLite (in-memory) for SQL-based segmentation analysis
- Jupyter Notebook

---

## Author

**Clare Mujuni**  
20 years banking experience across Branch Management, Credit Risk, and Internal Audit  
(Commercial Microfinance, DFCU Bank, Global Trust Bank)  
[GitHub](https://github.com/Claraoptions) | mujuniclare@gmail.com