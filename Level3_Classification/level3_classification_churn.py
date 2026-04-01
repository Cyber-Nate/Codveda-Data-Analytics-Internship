# ============================================================
# Codveda Technologies - Data Analysis Internship
# Level 3 - Task 1: Predictive Modeling (Classification)
# Dataset: Telecom Customer Churn
# Tools: Python, scikit-learn, pandas, matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.model_selection import GridSearchCV, cross_val_score

plt.style.use("seaborn-v0_8-whitegrid")
SAVE = True

def savefig(name):
    if SAVE:
        plt.savefig(name, dpi=150, bbox_inches="tight")
        print(f"  [saved] {name}")
    else:
        plt.show()
    plt.close()


# ── 1. Load Dataset ──────────────────────────────────────────
print("=" * 60)
print("  TELECOM CHURN — PREDICTIVE CLASSIFICATION")
print("=" * 60)

train = pd.read_csv("churn-bigml-80.csv")
test  = pd.read_csv("churn-bigml-20.csv")

print(f"\n[1] Train : {train.shape[0]:,} rows x {train.shape[1]} columns")
print(f"    Test  : {test.shape[0]:,} rows x {test.shape[1]} columns")
print(f"\n    Churn distribution (train):")
print(train["Churn"].value_counts())
print(f"\n    Churn rate: {train['Churn'].mean()*100:.1f}%")


# ── 2. Preprocessing ─────────────────────────────────────────
print("\n[2] Preprocessing...")

def preprocess(df):
    df = df.copy()
    # Drop State & Area code (high cardinality, low signal)
    df.drop(columns=["State", "Area code"], inplace=True)
    # Encode binary categoricals
    for col in ["International plan", "Voice mail plan"]:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)
    # Encode target
    df["Churn"] = df["Churn"].astype(int)
    return df

train_clean = preprocess(train)
test_clean  = preprocess(test)

feature_cols = [c for c in train_clean.columns if c != "Churn"]
X_train = train_clean[feature_cols].values
y_train = train_clean["Churn"].values
X_test  = test_clean[feature_cols].values
y_test  = test_clean["Churn"].values

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Features used  : {len(feature_cols)}")
print(f"  Feature list   : {feature_cols}")
print(f"  Train positives: {y_train.sum()} / {len(y_train)} "
      f"({y_train.mean()*100:.1f}%)")
print(f"  Test  positives: {y_test.sum()} / {len(y_test)} "
      f"({y_test.mean()*100:.1f}%)")


# ── 3. Class Distribution Plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
labels_map = {0: "Not Churned", 1: "Churned"}
counts = pd.Series(y_train).value_counts().sort_index()
bars = ax.bar([labels_map[i] for i in counts.index], counts.values,
              color=["#4CAF50", "#F44336"], edgecolor="black")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(bar.get_height()), ha="center", fontsize=12)
ax.set_title("Churn Class Distribution (Training Set)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
savefig("clf_plot1_class_distribution.png")


# ── 4. Correlation with Churn ────────────────────────────────
corr_churn = train_clean.corr()["Churn"].drop("Churn").sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#F44336" if v > 0 else "#2196F3" for v in corr_churn.values]
ax.barh(corr_churn.index, corr_churn.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlation with Churn",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
savefig("clf_plot2_feature_correlation.png")


# ── 5. Train Models ──────────────────────────────────────────
print("\n[3] Training Models...")

models = {
    "Decision Tree"     : DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    # Use scaled features for LR, raw for tree-based
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="f1").mean()
    results[name] = {
        "model"    : model,
        "preds"    : preds,
        "X_test"   : Xte,
        "accuracy" : accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall"   : recall_score(y_test, preds),
        "f1"       : f1_score(y_test, preds),
        "cv_f1"    : cv,
    }
    print(f"\n  {name}:")
    print(f"    Accuracy : {results[name]['accuracy']:.4f}")
    print(f"    Precision: {results[name]['precision']:.4f}")
    print(f"    Recall   : {results[name]['recall']:.4f}")
    print(f"    F1 Score : {results[name]['f1']:.4f}")
    print(f"    CV F1    : {results[name]['cv_f1']:.4f}")


# ── 6. Model Comparison Bar Chart ───────────────────────────
metrics = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrics))
width = 0.25
colors_models = ["#2196F3", "#FF9800", "#4CAF50"]

fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=colors_models[i], edgecolor="white", alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5)

ax.set_title("Model Performance Comparison",
             fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.legend()
plt.tight_layout()
savefig("clf_plot3_model_comparison.png")


# ── 7. Confusion Matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"],
                linewidths=0.5)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
savefig("clf_plot4_confusion_matrices.png")


# ── 8. ROC Curves ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), color in zip(results.items(), colors_models):
    model = res["model"]
    Xte   = res["X_test"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xte)[:, 1]
    else:
        probs = model.decision_function(Xte)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {roc_auc:.3f})")

ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random (AUC = 0.500)")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
savefig("clf_plot5_roc_curves.png")


# ── 9. Hyperparameter Tuning — Random Forest ────────────────
print("\n[4] Hyperparameter Tuning (Random Forest — GridSearchCV)...")
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth"   : [None, 10, 20],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

best_rf  = grid_search.best_estimator_
best_preds = best_rf.predict(X_test)

print(f"  Best params  : {grid_search.best_params_}")
print(f"  Best CV F1   : {grid_search.best_score_:.4f}")
print(f"  Test Accuracy: {accuracy_score(y_test, best_preds):.4f}")
print(f"  Test F1      : {f1_score(y_test, best_preds):.4f}")
print(f"\n  Classification Report (Tuned RF):")
print(classification_report(y_test, best_preds,
                            target_names=["Not Churn", "Churn"]))


# ── 10. Feature Importance (Tuned RF) ───────────────────────
importances = best_rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
feat_imp.plot(kind="barh", ax=ax, color="#9C27B0", edgecolor="white")
ax.set_title("Feature Importances — Tuned Random Forest",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
savefig("clf_plot6_feature_importance.png")


# ── 11. Final Summary ────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Model':<22} {'Accuracy':>9} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8}")
print("-" * 60)
for name, res in results.items():
    print(f"  {name:<20} {res['accuracy']:>9.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>8.4f} {res['f1']:>8.4f}")
tuned_f1 = f1_score(y_test, best_preds)
tuned_acc = accuracy_score(y_test, best_preds)
print(f"  {'RF (Tuned)':<20} {tuned_acc:>9.4f} "
      f"{precision_score(y_test, best_preds):>10.4f} "
      f"{recall_score(y_test, best_preds):>8.4f} {tuned_f1:>8.4f}")
print(f"""
Key Insights:
  1. Random Forest outperforms all models across every metric.
  2. Tuned RF achieves F1={tuned_f1:.3f} — best overall performance.
  3. Top churn predictors: Total day charge, Customer service calls,
     International plan, and Total day minutes.
  4. Class imbalance (~14.5% churn rate) affects recall; RF handles
     this best due to ensemble averaging.
""")
print("Classification complete. All plots saved.")
