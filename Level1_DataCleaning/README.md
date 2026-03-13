# 🧹 Level 1 — Task 1: Data Cleaning & Preprocessing

## 📌 Overview
Performed a full data cleaning and preprocessing pipeline on the **Boston Housing Dataset** — a real-world dataset containing housing price data across Boston neighbourhoods. The raw dataset was loaded, realistic data quality issues were introduced for demonstration, and then systematically resolved using best-practice cleaning techniques.

---

## 📁 Files

| File | Description |
|------|-------------|
| `level1_data_cleaning_housing.py` | Main Python script |
| `housing_cleaned.csv` | Final cleaned dataset (output) |
| `plots/clean_plot1_missing_before.png` | Missing values before cleaning |
| `plots/clean_plot2_boxplots_after.png` | Boxplots after cleaning (outlier overview) |
| `plots/clean_plot3_missing_after.png` | Missing values after cleaning (all zero) |
| `plots/clean_plot4_correlation_heatmap.png` | Correlation heatmap on cleaned data |

---

## 📂 Dataset

**Name:** Boston Housing Dataset  
**Source:** `4__house_Prediction_Data_Set.csv`  
**Shape:** 506 rows × 14 columns  
**Features:**

| Column | Description |
|--------|-------------|
| `CRIM` | Per capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres |
| `CHAS` | Charles River dummy variable (1 = bounds river) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of units built before 1940 |
| `DIS` | Weighted distance to employment centres |
| `RAD` | Accessibility index to radial highways |
| `TAX` | Property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | Demographic proportion statistic |
| `LSTAT` | % lower-status population |
| `MEDV` | Median home value in $1000s *(target variable)* |

---

## 🎯 Objectives

- Load a raw dataset using pandas
- Identify and handle missing values (imputation)
- Remove duplicate rows
- Standardise inconsistent data formats

---

## 🛠️ Tools Used

```
Python | pandas | NumPy | matplotlib | seaborn
```

---

## ⚙️ How to Run

```bash
# Place 4__house_Prediction_Data_Set.csv in the same folder as the script
python level1_data_cleaning_housing.py
```

The script will output `housing_cleaned.csv` and save all 4 plots automatically.

---

## 🧽 Cleaning Steps Performed

| Step | Issue | Action Taken |
|------|-------|--------------|
| 1 | Duplicate rows (10) | Removed with `drop_duplicates()` |
| 2 | Missing values (5 columns, ~75 nulls) | Imputed with column **median** |
| 3 | CHAS column — whitespace formatting | Stripped whitespace, cast to `int` |
| 4 | Float precision | Rounded to 4 decimal places |
| 5 | RAD column — wrong dtype | Cast from `float` to `int` |

---

## 📈 Key Findings

| Metric | Before Cleaning | After Cleaning |
|--------|----------------|----------------|
| Shape | (516, 14) | (506, 14) |
| Missing values | 96 | **0** |
| Duplicate rows | 10 | **0** |
| CHAS dtype | string (inconsistent) | int64 |

- **MEDV** (house price) is strongly negatively correlated with `LSTAT` (r = -0.74)
- **RM** (rooms per dwelling) is the strongest positive predictor of price (r = 0.70)
- Several columns (e.g. `CRIM`, `B`) show notable outliers detected via IQR method
