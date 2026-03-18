# ============================================================
# Codveda Technologies - Data Analysis Internship
# Level 1 - Task 2: Exploratory Data Analysis (EDA)
# Dataset: Iris Flower Dataset
# Tools: Python, pandas, matplotlib, seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 0. Setup ────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
SAVE = True   # set False to just plt.show() instead of saving

def savefig(name):
    if SAVE:
        plt.savefig(name, dpi=150, bbox_inches="tight")
        print(f"  [saved] {name}")
    else:
        plt.show()
    plt.close()


# ── 1. Load Dataset ─────────────────────────────────────────
print("=" * 55)
print("  IRIS DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

df = pd.read_csv("1__iris.csv")

print("\n[1] First 5 rows:")
print(df.head())

print("\n[2] Shape:", df.shape)
print("[3] Columns:", df.columns.tolist())
print("[4] Data Types:\n", df.dtypes)


# ── 2. Missing Values ────────────────────────────────────────
print("\n[5] Missing Values:")
print(df.isnull().sum())


# ── 3. Summary Statistics ────────────────────────────────────
print("\n[6] Summary Statistics:")
print(df.describe().round(2))

# Per-species summary
print("\n[7] Mean values per species:")
print(df.groupby("species").mean().round(2))

# Mode
print("\n[8] Mode (numerical columns):")
print(df.select_dtypes(include=np.number).mode().iloc[0])

# Standard deviation
print("\n[9] Standard Deviation:")
print(df.select_dtypes(include=np.number).std().round(4))


# ── 4. Species Distribution ──────────────────────────────────
print("\n[10] Species count:\n", df["species"].value_counts())

fig, ax = plt.subplots(figsize=(6, 4))
df["species"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
ax.set_title("Species Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Species")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
savefig("plot1_species_distribution.png")


# ── 5. Histograms ────────────────────────────────────────────
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("Feature Distributions (Histograms)", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    ax.hist(df[feat], bins=20, edgecolor="black", color="#4CAF50", alpha=0.8)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel("Value (cm)")
    ax.set_ylabel("Frequency")

plt.tight_layout()
savefig("plot2_histograms.png")


# ── 6. Boxplots per species ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distributions by Species (Boxplots)", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=df, x="species", y=feat, ax=ax)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel("Species")
    ax.set_ylabel("Value (cm)")

plt.tight_layout()
savefig("plot3_boxplots_by_species.png")


# ── 7. Scatter Plots ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Scatter Plots", fontsize=15, fontweight="bold")

pairs = [("sepal_length", "sepal_width"), ("petal_length", "petal_width")]
for ax, (x, y) in zip(axes, pairs):
    sns.scatterplot(data=df, x=x, y=y, hue="species", ax=ax, s=70)
    ax.set_title(f"{x.replace('_',' ').title()} vs {y.replace('_',' ').title()}")
    ax.set_xlabel(x.replace("_", " ").title() + " (cm)")
    ax.set_ylabel(y.replace("_", " ").title() + " (cm)")

plt.tight_layout()
savefig("plot4_scatter_plots.png")


# ── 8. Pairplot ──────────────────────────────────────────────
print("\n[11] Generating pairplot (this may take a moment)...")
pair = sns.pairplot(df, hue="species", diag_kind="kde", plot_kws={"alpha": 0.6})
pair.figure.suptitle("Pairplot - All Feature Combinations", y=1.02, fontsize=14, fontweight="bold")
savefig("plot5_pairplot.png")


# ── 9. Correlation Heatmap ───────────────────────────────────
corr = df.select_dtypes(include=np.number).corr()

print("\n[12] Correlation Matrix:")
print(corr.round(2))

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("plot6_correlation_heatmap.png")


# ── 10. Violin Plots ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Violin Plots by Species", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    sns.violinplot(data=df, x="species", y=feat, ax=ax, inner="quartile")
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel("Species")
    ax.set_ylabel("Value (cm)")

plt.tight_layout()
savefig("plot7_violin_plots.png")


# ── 11. Key Insights ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)
print("""
1. The dataset has 150 rows, 5 columns, and NO missing values.
2. Each species (setosa, versicolor, virginica) has exactly
   50 samples — perfectly balanced.
3. Petal length and petal width are highly correlated (r=0.96),
   making them strong features for species classification.
4. Setosa is clearly separable from the other two species
   based on petal dimensions alone.
5. Versicolor and virginica overlap slightly in sepal features
   but are more distinct in petal measurements.
6. Sepal width has the lowest correlation with other features,
   suggesting it is a less discriminative feature.
""")

print("EDA complete. All plots saved successfully.")
