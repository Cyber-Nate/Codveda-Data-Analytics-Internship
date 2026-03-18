# 📊 Level 1 — Task 2: Exploratory Data Analysis (EDA)

## 📌 Overview
Performed a comprehensive Exploratory Data Analysis on the classic **Iris Flower Dataset** to uncover patterns, trends, and statistical relationships across 4 numerical features and 3 species.

---

## 📁 Files

| File | Description |
|------|-------------|
| `level1_eda_iris.py` | Main Python script |
| `plots/plot1_species_distribution.png` | Bar chart of species counts |
| `plots/plot2_histograms.png` | Feature distribution histograms |
| `plots/plot3_boxplots_by_species.png` | Boxplots per species |
| `plots/plot4_scatter_plots.png` | Sepal and petal scatter plots |
| `plots/plot5_pairplot.png` | Full pairplot of all feature combinations |
| `plots/plot6_correlation_heatmap.png` | Pearson correlation heatmap |
| `plots/plot7_violin_plots.png` | Violin plots by species |

---

## 📂 Dataset

**Name:** Iris Flower Dataset  
**Source:** `1__iris.csv`  
**Shape:** 150 rows × 5 columns  
**Features:**

| Column | Description |
|--------|-------------|
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |
| `species` | Flower species (setosa, versicolor, virginica) |

---

## 🎯 Objectives

- Calculate summary statistics (mean, median, mode, standard deviation)
- Visualise data distributions using histograms, boxplots, and scatter plots
- Find correlations between numerical features

---

## 🛠️ Tools Used

```
Python | pandas | matplotlib | seaborn | NumPy
```

---

## ⚙️ How to Run

```bash
# Place 1__iris.csv in the same folder as the script
python level1_eda_iris.py
```

All 7 plots will be saved automatically as PNG files in the current directory.

---

## 📈 Key Findings

| Finding | Detail |
|---------|--------|
| Dataset quality | No missing values, perfectly balanced (50 samples per species) |
| Highest correlation | Petal length ↔ Petal width (r = **0.96**) |
| Most separable species | **Setosa** — completely distinct in petal dimensions |
| Weakest discriminator | Sepal width (lowest correlation with other features) |
| Overlapping species | Versicolor and Virginica show slight overlap in sepal space |

### Summary Statistics

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| Sepal Length | 5.84 cm | 0.83 | 4.30 | 7.90 |
| Sepal Width | 3.05 cm | 0.43 | 2.00 | 4.40 |
| Petal Length | 3.76 cm | 1.76 | 1.00 | 6.90 |
| Petal Width | 1.20 cm | 0.76 | 0.10 | 2.50 |
