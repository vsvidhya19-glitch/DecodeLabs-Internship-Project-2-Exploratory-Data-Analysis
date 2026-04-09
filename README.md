# 🏥 Indian Health Insurance Claims — EDA Project

An end-to-end Exploratory Data Analysis (EDA) on an Indian health insurance claims dataset containing **8,000 records across 38 columns**. The project covers data cleaning, feature engineering, statistical analysis, and visual insights — a strong portfolio piece for data analyst and data science roles.

---

## 📁 Project Structure

```
healthcare-claims-eda/
│
├── indian_health_insurance_claims_dataset.csv   # Raw dataset
├── healthcare_claims_eda.ipynb                  # Main Jupyter notebook
├── outputs/                                     # Saved plots
│   ├── output_40_0.png                          # Hospital bill box plot
│   ├── output_40_1.png                          # ICU days vs claim scatter
│   ├── output_43_0.png                          # Univariate histograms
│   ├── output_43_2.png                          # Diabetes / Hypertension counts
│   ├── output_46_0.png                          # Correlation heatmap
│   ├── output_48_0.png                          # Claims vs diabetes box plot
│   └── output_48_1.png                          # Claims vs hypertension box plot
└── README.md                                    # This file
```

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Rows | 8,000 |
| Columns (raw) | 38 |
| Columns (after feature engineering) | 42 |
| Date range | 2018-03-02 → 2029-05-24 |
| States covered | 6 (Tamil Nadu, Karnataka, West Bengal, Delhi, Uttar Pradesh, Maharashtra) |
| Unique hospitals | 7,188 |
| Policy types | Individual, Family Floater, Senior Citizen |
| Mean claim amount | ₹1,21,636 |
| Median claim amount | ₹1,11,817 |
| Max claim amount | ₹13,29,766 |
| Cashless claims | 69.6% |

---

## 🧹 Data Cleaning Steps

### 1. Currency Columns — Mixed Formats
`total_claim_amount` and `hospital_bill_amount` contained mixed formats:
```
162494.46       ← plain float
"₹151,678"      ← rupee symbol + commas
"₹1,46,969"     ← lakh-style Indian comma formatting
```
**Fix:** Strip `₹`, remove all commas, convert to `float`

### 2. Annual Income — "LPA" Suffix
```
"7.5 LPA"  →  750000.0
```
**Fix:** Strip `LPA` suffix and whitespace, handle `NaN` strings, convert to float

### 3. Date Columns
- `policy_start_date` and `claim_date` stored as raw strings
- **Fix:** Parsed with `pd.to_datetime(errors="coerce")`

### 4. Gender — 6 Variants → 2 Clean Values
```
"M", "male", "Male"   →  "Male"
"F", "female", "Female"  →  "Female"
```
**Fix:** Dictionary mapping via `.map()`

### 5. Categorical Standardisation
- `marital_status` — inconsistent casing → `.str.title()`
- `cashless_claim` — inconsistent casing → `.str.lower()`
- All string columns — stripped of leading/trailing whitespace

### 6. Binary Health Flags
- `has_diabetes`, `has_hypertension`, `family_history_cardiac` stored as float with NaNs
- **Fix:** Cast to nullable `Int8` to preserve NaN support while reducing memory

### 7. Missing Values

| Column | Missing | % | Strategy |
|---|---|---|---|
| `annual_income_inr` | 560 | 7.0% | Median imputation |
| `bmi` | 560 | 7.0% | Median imputation |
| `stress_level_score` | 560 | 7.0% | Median imputation |
| `has_hypertension` | 391 | 4.89% | Median imputation |
| `has_diabetes` | 383 | 4.79% | Median imputation |

---

## 🔧 Feature Engineering

New columns derived from existing data:

| New Column | Source | Description |
|---|---|---|
| `policy_duration_days` | `claim_date - policy_start_date` | Days between policy start and claim |
| `claim_year` | `claim_date` | Year of the claim |
| `claim_month` | `claim_date` | Month number of the claim |
| `claim_month_name` | `claim_date` | Abbreviated month name (e.g. "Apr") |

---

## 📈 Analysis Performed

### Univariate Analysis
- Histograms for `age`, `annual_income_inr`, `bmi`, `hospital_bill_amount`, `total_claim_amount`
- Count plots for `has_diabetes` and `has_hypertension`

### Bivariate Analysis
- Correlation heatmap across all numeric columns
- Box plots: claim amount vs diabetes status
- Box plots: claim amount vs hypertension status
- Scatter plot: ICU days vs total claim amount

### Outlier Detection
- Box plot on `hospital_bill_amount` to identify high-cost cases

---

## 💡 Key Insights

- **Hospital bills are the strongest predictor of claim amounts** — correlation of 0.84 between `hospital_bill_amount` and `total_claim_amount`
- **Outliers drive skewness** — most bills fall between ₹50,000–₹1,50,000, but extreme outliers above ₹3,00,000 exist and warrant separate investigation
- **ICU days are not a sole cost driver** — claim amounts vary widely even at the same ICU day count, indicating multi-factor cost dynamics (treatment complexity, hospital tier, billing practices)
- **Chronic conditions alone don't determine high claims** — median claim amounts are similar across diabetic/non-diabetic and hypertensive/non-hypertensive groups; multi-factor risk assessment is needed
- **Income inequality is visible in the data** — most incomes are below ₹10,00,000, but a few extend to ₹47,00,000, creating a long-tailed income distribution
- **BMI peaks around 25** — borderline overweight across the cohort, relevant for health risk modelling
- **69.6% of claims are cashless** — majority of policyholders use the cashless facility, important for network hospital strategy

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations, NaN handling |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/healthcare-claims-eda.git
cd healthcare-claims-eda
```

2. Place the dataset in the project folder:
```
indian_health_insurance_claims_dataset.csv
```

3. Launch the notebook:
```bash
jupyter notebook healthcare_claims_eda.ipynb
```

---

## 📋 Final Column Reference

| Column | Type | Description |
|---|---|---|
| `policy_id` | string | Unique policy identifier |
| `customer_name` | string | Policyholder name |
| `gender` | string | Male / Female |
| `age` | int | Age of policyholder |
| `marital_status` | string | Married / Single / etc. |
| `state` | string | Indian state |
| `annual_income_inr` | float | Annual income in INR |
| `bmi` | float | Body Mass Index |
| `has_diabetes` | Int8 | 1 = Yes, 0 = No |
| `has_hypertension` | Int8 | 1 = Yes, 0 = No |
| `family_history_cardiac` | Int8 | 1 = Yes, 0 = No |
| `stress_level_score` | float | Score from 1–10 |
| `policy_type` | string | Individual / Family Floater / Senior Citizen |
| `sum_insured` | int | Policy coverage amount in INR |
| `hospital_bill_amount` | float | Total hospital bill in INR |
| `total_claim_amount` | float | Final settled claim in INR |
| `icu_days` | int | Number of ICU days |
| `length_of_stay` | int | Total hospital stay in days |
| `cashless_claim` | string | yes / no |
| `policy_duration_days` | int | Days from policy start to claim date |
| `claim_year` | int | Year of claim |
| `claim_month` | int | Month of claim (1–12) |

---

## 💼 Skills Demonstrated

- Real-world currency string parsing (₹ symbol, Indian lakh formatting)
- Multi-format date parsing and time-based feature engineering
- Categorical standardisation with mapping dictionaries
- Nullable integer types (`Int8`) for binary health flags
- Systematic missing value handling (median imputation for numeric, mode for categorical)
- Correlation analysis and outlier detection
- Univariate and bivariate visual storytelling with matplotlib and seaborn
- Clean, reproducible notebook structure with inline commentary
