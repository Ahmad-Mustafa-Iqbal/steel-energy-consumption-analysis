# Steel Industry Energy Consumption: EDA & Baseline Regression Modeling

## Project Overview

This project analyzes real energy consumption data from a steel manufacturing plant and builds a baseline regression pipeline to predict `Usage_kWh`. The work is split into two notebooks that together represent the core workflow of a real machine learning project:

1. **`eda.ipynb`** — Deep exploratory data analysis and feature engineering.
2. **`Baseline_models.ipynb`** — Baseline regression modeling, evaluation, and model selection.

## Dataset Information

- **Name:** Steel Industry Energy Consumption Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip)
- **Records:** 35,040 rows (15-minute interval readings across all of 2018)
- **Original columns (11):** `date`, `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor`, `NSM`, `WeekStatus`, `Day_of_week`, `Load_Type`
- **Target variable:** `Usage_kWh` (industrial energy consumption)

The raw data file is provided at `data/Steel_industry_data.csv`. The engineered version produced by Part 1 (`data/steel_industry_data_modified.csv`) is used as the input to Part 2.

## Environment Setup

```bash
# Clone the repository
git clone <this-repo-url>
cd <repo-folder>

# (Optional) create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Then open `eda.ipynb` first (it generates `data/steel_industry_data_modified.csv`), followed by `Baseline_models.ipynb`.

## Feature Engineering Steps

Performed in `eda.ipynb`:

| Feature | Description |
|---|---|
| `Hour`, `DayOfWeekNum`, `Month` | Extracted from the parsed `date` column |
| `Is_Weekend` | Binary flag, 1 if `DayOfWeekNum` is Saturday/Sunday |
| `Power_Factor_Ratio` | `Leading_Current_Power_Factor / Lagging_Current_Power_Factor` (one division-by-zero row fixed by median imputation) |
| `High_Usage_Flag` | Binary flag, 1 if `Usage_kWh` is above its 75th percentile (**dropped before modeling** — see below, this is target leakage) |

## EDA Findings

- **Data quality:** No missing values, no duplicate rows. One row had `Lagging_Current_Power_Factor = 0`, producing an undefined `Power_Factor_Ratio`, resolved via median imputation.
- **Outliers:** IQR method flagged 328 readings (0.94%) in `Usage_kWh` as outliers — these correspond to genuine `Maximum_Load` operating periods, not sensor errors, and were kept.
- **Top correlated features with `Usage_kWh`:** `CO2(tCO2)` (r ≈ 0.99), `Lagging_Current_Reactive.Power_kVarh` (r ≈ 0.90), and the engineered `High_Usage_Flag` (r ≈ 0.87, excluded from modeling as it's derived from the target).
- **Key pattern:** Average usage rises sharply from `Light_Load` (~8.6 kWh) to `Maximum_Load` (~59.3 kWh), and the hourly usage curve shows a clear ramp-up starting around 7–8 AM consistent with a day-shift production schedule.
- **Hypothesis:** Energy spikes are primarily driven by production shift scheduling (weekday daytime `Maximum_Load` operations) rather than random variation.

## Model Training Process

Performed in `Baseline_models.ipynb`:

1. Loaded the engineered dataset from Part 1.
2. Dropped `date`, `High_Usage_Flag` (target leakage — directly derived from `Usage_kWh`), and `WeekStatus` (redundant with `Is_Weekend`).
3. **One-hot encoded** `Load_Type` and `Day_of_week` (both nominal categorical variables with no natural ordering — see notebook for full rationale).
4. Split data 80/20 with `random_state=42`.
5. Trained **Linear Regression, Ridge Regression, Decision Tree Regressor, and Random Forest Regressor**.
6. Evaluated each model on the test set (MAE, RMSE, R²) and via 5-fold cross-validation (mean RMSE).
7. Compared test RMSE across models with a bar chart, and plotted Predicted vs Actual for the best model.

## Results and Conclusions

| Model | Test MAE | Test RMSE | Test R² | 5-Fold CV RMSE (mean ± std) |
|---|---|---|---|---|
| Linear Regression | 2.63 | 4.15 | 0.985 | 4.60 ± 1.39 |
| Ridge Regression | 4.36 | 6.27 | 0.966 | 6.69 ± 1.10 |
| Decision Tree | 0.55 | 1.64 | 0.998 | 2.59 ± 2.14 |
| **Random Forest** | **0.35** | **1.04** | **0.999** | **2.21 ± 2.28** |

**Best model: Random Forest Regressor**, with the lowest test RMSE (1.04 kWh), lowest mean CV RMSE (2.21 kWh), and highest R² (0.999). Tree-based models substantially outperform the linear models, indicating the relationship between the features and `Usage_kWh` is non-linear (e.g. threshold-like jumps between load states). Random Forest is the model carried forward as the baseline for future tuning. Full overfitting analysis and reasoning are documented in the Model Selection section of `Baseline_models.ipynb`.

## Repository Structure

```
.
├── eda.ipynb              # Part 1: EDA & feature engineering
├── Baseline_models.ipynb  # Part 2: Baseline regression modeling
├── data/
│   ├── Steel_industry_data.csv  # Raw dataset
│   └── steel_industry_data_modified.csv     # Output of Part 1, input to Part 2
├── README.md
└── requirements.txt
```
