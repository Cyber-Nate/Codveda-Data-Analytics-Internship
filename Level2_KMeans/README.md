# 🔵 Level 2 — Task 3: Clustering Analysis (K-Means)

## 📌 Overview
Implemented **K-Means clustering** on the Iris Flower Dataset to group similar data points based on feature similarity — without using the species labels. Used the Elbow Method and Silhouette Score to determine the optimal number of clusters, then visualised results in both raw feature space and 2D PCA-reduced space.

---

## 📁 Files

| File | Description |
|------|-------------|
| `level2_kmeans_clustering.py` | Main Python script |
| `plots/km_plot1_elbow.png` | Elbow method (inertia vs k) |
| `plots/km_plot2_silhouette.png` | Silhouette scores (k = 2 to 8) |
| `plots/km_plot3_pca_clusters.png` | Clusters vs true labels in PCA space |
| `plots/km_plot4_raw_feature_clusters.png` | Clusters in raw feature space |
| `plots/km_plot5_cluster_profiles.png` | Mean feature values per cluster (heatmap) |
| `plots/km_plot6_cluster_vs_species.png` | Cluster vs true species (confusion style) |

---

## 📂 Dataset

**Name:** Iris Flower Dataset  
**Source:** `1__iris.csv`  
**Shape:** 150 rows × 5 columns  
**Features used for clustering:** sepal_length, sepal_width, petal_length, petal_width  
*(Species label excluded — unsupervised learning)*

---

## 🎯 Objectives

- Standardise the dataset using `StandardScaler`
- Apply K-Means clustering
- Determine optimal number of clusters using the Elbow Method
- Visualise clusters using 2D scatter plots

---

## 🛠️ Tools Used

```
Python | scikit-learn | pandas | matplotlib | seaborn | NumPy
```

---

## ⚙️ How to Run

```bash
# Place 1__iris.csv in the same folder as the script
python level2_kmeans_clustering.py
```

All 6 plots will be saved automatically as PNG files.

---

## 📊 Results

### Optimal k Selection

| Method | Optimal k | Reason |
|--------|-----------|--------|
| Elbow Method | **k = 3** | Clear bend in inertia curve at k=3 |
| Silhouette Score | **k = 2–3** | Highest score at k=2 (0.580), good at k=3 (0.459) |
| Domain knowledge | **k = 3** | 3 known species confirms k=3 |

### Final Model (k = 3)

| Metric | Value |
|--------|-------|
| Inertia | 140.97 |
| Silhouette Score | **0.459** |
| PCA Variance Explained | **95.8%** (PC1: 72.8%, PC2: 23.0%) |

### Cluster Sizes

| Cluster | Size |
|---------|------|
| Cluster 0 | 53 |
| Cluster 1 | 50 |
| Cluster 2 | 47 |

### Cluster vs True Species Mapping

| Cluster | Setosa | Versicolor | Virginica | Interpretation |
|---------|--------|------------|-----------|----------------|
| 0 | 0 | 39 | 14 | Mostly Versicolor |
| 1 | **50** | 0 | 0 | Pure Setosa ✅ |
| 2 | 0 | 11 | **36** | Mostly Virginica |

### Key Findings

1. K-Means correctly identified **k=3** as the optimal cluster count — matching the 3 real species
2. **Setosa** is perfectly isolated in its own cluster with zero misassignments
3. Minor overlap between **Versicolor** and **Virginica** mirrors what EDA revealed
4. **Petal dimensions** are the most discriminative features for clustering
5. PCA retains **95.8% of variance** in just 2 components — excellent dimensionality reduction
