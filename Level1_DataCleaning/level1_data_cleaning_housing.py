# ============================================================
# Codveda Technologies - Data Analysis Internship
# Level 1 - Task 1: Data Cleaning and Preprocessing
# Dataset: Boston Housing Dataset
# Tools: Python, pandas
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
SAVE = True

def savefig(name):
    if SAVE:
        plt.savefig(name, dpi=150, bbox_inches="tight")
        print(f"  [saved] {name}")
    else:
        plt.show()
    plt.close()


# ── 0. Column Descriptions ───────────────────────────────────
COLUMNS = {
    "CRIM"    : "Per capita crime rate by town",
    "ZN"      : "Proportion of residential land zoned for large lots",
    "INDUS"   : "Proportion of non-retail business acres per town",
    "CHAS"    : "Charles River dummy variable (1=tract bounds river, 0=otherwise)",
    "NOX"     : "Nitric oxide concentration (parts per 10 million)",
    "RM"      : "Average number of rooms per dwelling",
    "AGE"     : "Proportion of owner-occupied units built before 1940",
    "DIS"     : "Weighted distances to Boston employment centres",
    "RAD"     : "Index of accessibility to radial highways",
    "TAX"     : "Full-value property tax rate per $10,000",
    "PTRATIO" : "Pupil-teacher ratio by town",
    "B"       : "1000(Bk - 0.63)^2 where Bk is proportion of Black residents",
    "LSTAT"   : "Percentage of lower-status population",
    "MEDV"    : "Median value of owner-occupied homes (in $1000s) — TARGET",
}


# ── 1. Load Raw Dataset ──────────────────────────────────────
print("=" * 60)
print("  BOSTON HOUSING - DATA CLEANING & PREPROCESSING")
print("=" * 60)

# Read the raw whitespace-delimited file and assign column names
df_raw = pd.read_csv(
    "4__house_Prediction_Data_Set.csv",
    sep=r"\s+",
    header=None,
    engine="python"
)
df_raw.columns = list(COLUMNS.keys())

print(f"\n[1] Raw dataset loaded: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
print("\nColumn reference:")
for col, desc in COLUMNS.items():
    print(f"  {col:<10} {desc}")


# ── 2. Introduce Realistic Data Quality Issues ────────────────
# (Simulating a real-world messy dataset for demonstration)
np.random.seed(42)
df = df_raw.copy()

# Missing values in 5 columns
for col in ["CRIM", "RM", "AGE", "TAX", "LSTAT"]:
    idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[idx, col] = np.nan

# Duplicate rows
dup_rows = df.sample(10, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# Inconsistent formatting in CHAS (categorical: should be 0 or 1)
df["CHAS"] = df["CHAS"].astype(str)
df.loc[np.random.choice(df.index, 20, replace=False), "CHAS"] = \
    df.loc[np.random.choice(df.index, 20, replace=False), "CHAS"].apply(
        lambda x: "  " + x.strip() + "  "
    )

print(f"\n[2] After simulating real-world issues:")
print(f"    Shape          : {df.shape}")
print(f"    Duplicates     : {df.duplicated().sum()}")
print(f"    Total nulls    : {df.isnull().sum().sum()}")


# ── 3. Initial Inspection ────────────────────────────────────
print("\n[3] First 5 rows (raw):")
print(df.head())

print("\n[4] Data types:")
print(df.dtypes)

print("\n[5] Missing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])


# ── 4. Visualise Missing Values Before Cleaning ───────────────
fig, ax = plt.subplots(figsize=(10, 4))
missing_cols = missing[missing > 0]
ax.bar(missing_cols.index, missing_cols.values, color="#E74C3C", edgecolor="black")
ax.set_title("Missing Values Before Cleaning", fontsize=14, fontweight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Number of Missing Values")
for i, v in enumerate(missing_cols.values):
    ax.text(i, v + 0.2, str(v), ha="center", fontsize=11)
plt.tight_layout()
savefig("clean_plot1_missing_before.png")


# ── 5. Step 1 — Remove Duplicate Rows ────────────────────────
print("\n" + "-" * 50)
print("STEP 1: Removing Duplicate Rows")
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"  Rows before : {before}")
print(f"  Rows after  : {after}")
print(f"  Removed     : {before - after} duplicate rows")


# ── 6. Step 2 — Standardise CHAS Column ─────────────────────
print("\n" + "-" * 50)
print("STEP 2: Standardising CHAS column (strip whitespace + convert to int)")
print(f"  Sample before: {df['CHAS'].unique()[:6]}")
df["CHAS"] = df["CHAS"].str.strip()
df["CHAS"] = pd.to_numeric(df["CHAS"], errors="coerce")
df["CHAS"] = df["CHAS"].fillna(df["CHAS"].mode()[0]).astype(int)
print(f"  Unique values after: {sorted(df['CHAS'].unique())}")


# ── 7. Step 3 — Handle Missing Values ────────────────────────
print("\n" + "-" * 50)
print("STEP 3: Handling Missing Values")

# Numerical columns — impute with median (robust to outliers)
num_cols_with_nulls = ["CRIM", "RM", "AGE", "TAX", "LSTAT"]

for col in num_cols_with_nulls:
    median_val = df[col].median()
    null_count = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f"  {col:<8} — filled {null_count} nulls with median ({median_val:.4f})")

print(f"\n  Missing values remaining: {df.isnull().sum().sum()}")


# ── 8. Step 4 — Fix Data Types ───────────────────────────────
print("\n" + "-" * 50)
print("STEP 4: Ensuring correct data types")

# RAD should be integer
df["RAD"] = df["RAD"].astype(int)

# Round floats to 4 decimal places for cleanliness
float_cols = df.select_dtypes(include="float64").columns
df[float_cols] = df[float_cols].round(4)

print("  Data types after fixing:")
print(df.dtypes)


# ── 9. Step 5 — Outlier Detection (IQR Method) ───────────────
print("\n" + "-" * 50)
print("STEP 5: Outlier Detection using IQR method")

outlier_summary = {}
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = n_outliers

outlier_df = pd.Series(outlier_summary)
print(outlier_df[outlier_df > 0].to_string())

# Visualise outliers via boxplots
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
fig.suptitle("Boxplots After Cleaning — Outlier Overview", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), df.columns):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor="#3498DB", color="navy"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
for ax in axes.flatten()[len(df.columns):]:
    ax.set_visible(False)
plt.tight_layout()
savefig("clean_plot2_boxplots_after.png")


# ── 10. Final State ──────────────────────────────────────────
print("\n" + "-" * 50)
print("STEP 6: Final Validation")
print(f"  Final shape      : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")
print(f"  Duplicate rows   : {df.duplicated().sum()}")
print(f"\nFirst 5 rows of cleaned dataset:")
print(df.head())

# Visualise missing values after cleaning
fig, ax = plt.subplots(figsize=(10, 4))
missing_after = df.isnull().sum()
ax.bar(df.columns, missing_after.values, color="#2ECC71", edgecolor="black")
ax.set_title("Missing Values After Cleaning", fontsize=14, fontweight="bold")
ax.set_xlabel("Column")
ax.set_ylabel("Number of Missing Values")
ax.set_ylim(0, 5)
plt.tight_layout()
savefig("clean_plot3_missing_after.png")


# ── 11. Correlation Heatmap on Cleaned Data ──────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap — Cleaned Housing Data",
             fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("clean_plot4_correlation_heatmap.png")


# ── 12. Save Cleaned Dataset ─────────────────────────────────
df.to_csv("housing_cleaned.csv", index=False)
print("\n[Saved] housing_cleaned.csv")


# ── 13. Cleaning Summary ─────────────────────────────────────
print("\n" + "=" * 60)
print("  CLEANING SUMMARY")
print("=" * 60)
print("""
Issue                        Action Taken
─────────────────────────────────────────────────────────
Duplicate rows (10)          Removed with drop_duplicates()
Missing values (5 columns)   Imputed with column median
CHAS whitespace formatting   Stripped and cast to integer
Float precision              Rounded to 4 decimal places
Data types (RAD)             Cast from float to integer
─────────────────────────────────────────────────────────
Original shape : (516, 14)
Cleaned shape  : (506, 14)
Missing values : 0
Duplicates     : 0
""")
print("Data cleaning complete.")
