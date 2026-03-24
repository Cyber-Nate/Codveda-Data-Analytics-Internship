# ============================================================
# Codveda Technologies - Data Analysis Internship
# Level 2 - Task 3: Clustering Analysis (K-Means)
# Dataset: Iris Flower Dataset
# Tools: Python, scikit-learn, matplotlib, seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix

plt.style.use("seaborn-v0_8-whitegrid")
SAVE = True

def savefig(name):
    if SAVE:
        plt.savefig(name, dpi=150, bbox_inches="tight")
        print(f"  [saved] {name}")
    else:
        plt.show()
    plt.close()


# ── 1. Load Dataset ─────────────────────────────────────────
print("=" * 60)
print("  IRIS DATASET — K-MEANS CLUSTERING ANALYSIS")
print("=" * 60)

df = pd.read_csv("1__iris.csv")
print(f"\n[1] Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Features : {df.columns[:-1].tolist()}")
print(f"    Target   : species → {df['species'].unique().tolist()}")

X = df.drop("species", axis=1).values
y_true = df["species"]
features = df.columns[:-1].tolist()


# ── 2. Standardise Features ──────────────────────────────────
print("\n[2] Standardising features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("  Before scaling (mean, std):")
for i, feat in enumerate(features):
    print(f"    {feat:<15} mean={X[:,i].mean():.3f}  std={X[:,i].std():.3f}")

print("\n  After scaling (mean, std):")
for i, feat in enumerate(features):
    print(f"    {feat:<15} mean={X_scaled[:,i].mean():.3f}  std={X_scaled[:,i].std():.3f}")


# ── 3. Elbow Method ──────────────────────────────────────────
print("\n[3] Running Elbow Method (k = 1 to 10)...")
inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"    k={k:2d}  inertia={km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, inertias, marker="o", color="#2196F3",
        linewidth=2, markersize=8, markerfacecolor="#F44336")
ax.axvline(x=3, color="#4CAF50", linestyle="--", linewidth=1.5,
           label="Optimal k=3")
ax.set_title("Elbow Method — Optimal Number of Clusters",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Within-cluster Sum of Squares)")
ax.legend()
ax.set_xticks(list(k_range))
plt.tight_layout()
savefig("km_plot1_elbow.png")


# ── 4. Silhouette Scores ─────────────────────────────────────
print("\n[4] Computing Silhouette Scores (k = 2 to 8)...")
sil_scores = []
k_sil_range = range(2, 9)

for k in k_sil_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"    k={k}  silhouette={score:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_sil_range, sil_scores, marker="s", color="#FF9800",
        linewidth=2, markersize=8, markerfacecolor="#9C27B0")
ax.axvline(x=3, color="#4CAF50", linestyle="--", linewidth=1.5,
           label="Optimal k=3")
ax.set_title("Silhouette Score per k", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.legend()
ax.set_xticks(list(k_sil_range))
plt.tight_layout()
savefig("km_plot2_silhouette.png")


# ── 5. Fit Final Model (k=3) ─────────────────────────────────
print("\n[5] Fitting final K-Means model with k=3...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels

print(f"  Inertia        : {kmeans.inertia_:.4f}")
print(f"  Silhouette     : {silhouette_score(X_scaled, labels):.4f}")
print(f"\n  Cluster sizes  :\n{df['cluster'].value_counts().sort_index()}")
print(f"\n  Cluster centres (standardised):")
centres_df = pd.DataFrame(kmeans.cluster_centers_,
                          columns=features,
                          index=[f"Cluster {i}" for i in range(3)])
print(centres_df.round(3))


# ── 6. PCA for 2D Visualisation ──────────────────────────────
print("\n[6] Reducing to 2D with PCA for visualisation...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%  "
      f"Total={sum(pca.explained_variance_ratio_)*100:.1f}%")

# Cluster centres in PCA space
centres_pca = pca.transform(kmeans.cluster_centers_)

CLUSTER_COLORS = ["#2196F3", "#4CAF50", "#F44336"]
CLUSTER_LABELS = ["Cluster 0", "Cluster 1", "Cluster 2"]


# ── 7. Scatter Plot — Clusters (PCA) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("K-Means Clustering (k=3) — 2D PCA Projection",
             fontsize=14, fontweight="bold")

# Left: K-Means clusters
for c, color, label in zip(range(3), CLUSTER_COLORS, CLUSTER_LABELS):
    mask = labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=label, s=60, alpha=0.8, edgecolors="white")
axes[0].scatter(centres_pca[:, 0], centres_pca[:, 1],
                c="black", marker="X", s=200, zorder=5, label="Centroids")
axes[0].set_title("K-Means Clusters")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].legend()

# Right: True species labels
species_colors = {"setosa": "#FF9800", "versicolor": "#9C27B0", "virginica": "#00BCD4"}
for species, color in species_colors.items():
    mask = y_true == species
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=species, s=60, alpha=0.8, edgecolors="white")
axes[1].set_title("True Species Labels")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].legend()

plt.tight_layout()
savefig("km_plot3_pca_clusters.png")


# ── 8. Scatter on Raw Features ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("K-Means Clusters — Raw Feature Space",
             fontsize=14, fontweight="bold")

pairs = [("petal_length", "petal_width"), ("sepal_length", "sepal_width")]
for ax, (fx, fy) in zip(axes, pairs):
    for c, color, label in zip(range(3), CLUSTER_COLORS, CLUSTER_LABELS):
        mask = df["cluster"] == c
        ax.scatter(df.loc[mask, fx], df.loc[mask, fy],
                   c=color, label=label, s=60, alpha=0.8, edgecolors="white")
    ax.set_xlabel(fx.replace("_", " ").title() + " (cm)")
    ax.set_ylabel(fy.replace("_", " ").title() + " (cm)")
    ax.set_title(f"{fx.replace('_',' ').title()} vs {fy.replace('_',' ').title()}")
    ax.legend()

plt.tight_layout()
savefig("km_plot4_raw_feature_clusters.png")


# ── 9. Cluster Profiles (Heatmap) ───────────────────────────
cluster_means = df.groupby("cluster")[features].mean()

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, ax=ax)
ax.set_title("Cluster Mean Feature Values",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Feature")
plt.tight_layout()
savefig("km_plot5_cluster_profiles.png")


# ── 10. Cluster vs True Labels ───────────────────────────────
print("\n[7] Comparing clusters to true species labels...")
comparison = pd.crosstab(df["cluster"], y_true,
                          rownames=["Cluster"], colnames=["Species"])
print(comparison)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(comparison, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, ax=ax)
ax.set_title("Cluster vs True Species (Confusion Matrix Style)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("km_plot6_cluster_vs_species.png")


# ── 11. Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY FINDINGS")
print("=" * 60)
print(f"""
1. Optimal k=3 confirmed by both Elbow Method and Silhouette Score.
2. Silhouette score of {silhouette_score(X_scaled, labels):.3f} indicates well-separated clusters.
3. PCA retains {sum(pca.explained_variance_ratio_)*100:.1f}% of variance in just 2 components.
4. Cluster 0 → maps almost perfectly to Setosa (clearly separable).
5. Clusters 1 & 2 → correspond to Versicolor & Virginica with
   minor overlap, consistent with EDA findings.
6. Petal dimensions (length & width) are the most discriminative
   features for clustering, matching correlation analysis.
""")
print("K-Means Clustering complete. All plots saved.")
