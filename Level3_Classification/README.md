# 🤖 Level 3 — Task 1: Predictive Modeling (Classification)

## 📌 Overview
Built and evaluated multiple **classification models** to predict whether a telecom customer will churn. The pipeline covers full preprocessing, training three different classifiers, comparing their performance, and hyperparameter tuning the best model using GridSearchCV.

---

## 📁 Files

| File | Description |
|------|-------------|
| `level3_classification_churn.py` | Main Python script |
| `plots/clf_plot1_class_distribution.png` | Churn class distribution |
| `plots/clf_plot2_feature_correlation.png` | Feature correlations with churn |
| `plots/clf_plot3_model_comparison.png` | Accuracy/Precision/Recall/F1 comparison |
| `plots/clf_plot4_confusion_matrices.png` | Confusion matrices for all 3 models |
| `plots/clf_plot5_roc_curves.png` | ROC curves with AUC scores |
| `plots/clf_plot6_feature_importance.png` | Feature importances (tuned Random Forest) |

---

## 📂 Dataset

**Name:** Telecom Customer Churn  
**Source:** `churn-bigml-80.csv` (training) + `churn-bigml-20.csv` (testing)  
**Shape:** 2,666 train rows + 667 test rows × 20 columns  
**Target:** `Churn` (True = customer churned, False = retained)  
**Churn rate:** ~14.6% (class imbalanced)

| Column | Description |
|--------|-------------|
| `State` | US state of the customer |
| `Account length` | Number of days as a customer |
| `International plan` | Has international plan (Yes/No) |
| `Voice mail plan` | Has voicemail plan (Yes/No) |
| `Total day minutes` | Total daytime call minutes |
| `Total day charge` | Total charge for daytime calls |
| `Customer service calls` | Number of calls to customer service |
| `Churn` | Whether the customer churned *(target)* |
| *(+ more)* | Full feature list in script |

---

## 🎯 Objectives

- Preprocess data (handle categorical variables, feature scaling)
- Train and test multiple classification models
- Evaluate models using accuracy, precision, recall, and F1-score
- Perform hyperparameter tuning using GridSearchCV

---

## 🛠️ Tools Used

```
Python | scikit-learn | pandas | matplotlib | seaborn | NumPy
```

---

## ⚙️ How to Run

```bash
# Place churn-bigml-80.csv and churn-bigml-20.csv in the same folder as the script
python level3_classification_churn.py
```

All 6 plots will be saved automatically. GridSearchCV may take 1–2 minutes.

---

## 📊 Preprocessing Steps

| Step | Action |
|------|--------|
| Dropped columns | `State`, `Area code` (high cardinality, low signal) |
| Encoded binary | `International plan`, `Voice mail plan` → 0/1 |
| Encoded target | `Churn` (bool) → 0/1 integer |
| Feature scaling | `StandardScaler` applied for Logistic Regression |

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 0.9145 | 0.6900 | 0.7263 | 0.7077 |
| Logistic Regression | 0.8546 | 0.4737 | 0.1895 | 0.2707 |
| Random Forest | **0.9535** | **0.9571** | 0.7053 | **0.8121** |
| **RF (Tuned)** | **0.9505** | 0.9189 | **0.7158** | **0.8047** |

### Best Model: Random Forest (Tuned)
```
Best hyperparameters:
  n_estimators     : 100
  max_depth        : None
  min_samples_split: 5
  CV F1 Score      : 0.822
```

### Classification Report (Tuned RF)
```
              Precision  Recall  F1-Score  Support
Not Churn        0.95     0.99     0.97      572
Churn            0.92     0.72     0.80       95
Accuracy                           0.95      667
```

### Top Churn Predictors (Feature Importance)
1. Total day charge
2. Customer service calls
3. International plan
4. Total day minutes
5. Total intl charge

### Key Findings
1. **Random Forest** significantly outperforms both Decision Tree and Logistic Regression
2. Class imbalance (~14.6% churn rate) reduces recall — RF handles this best via ensemble averaging
3. Customers with **high day charges** and **frequent service calls** are most likely to churn
4. Having an **international plan** is a strong churn predictor
5. ROC-AUC for tuned RF: **0.93** — excellent discriminative ability
