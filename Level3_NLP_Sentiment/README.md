# 💬 Level 3 — Task 3: NLP — Sentiment Analysis

## 📌 Overview
Performed **sentiment analysis** on a social media dataset containing posts from Twitter, Instagram, and Facebook. Built a full NLP preprocessing pipeline from scratch — including tokenisation, stopword removal, and stemming — then applied lexicon-based scoring and a TF-IDF + Logistic Regression classifier to classify posts as Positive, Negative, or Neutral.

---

## 📁 Files

| File | Description |
|------|-------------|
| `level3_nlp_sentiment.py` | Main Python script |
| `plots/nlp_plot1_sentiment_distribution.png` | Sentiment label distribution |
| `plots/nlp_plot2_sentiment_by_platform.png` | Sentiment breakdown by platform |
| `plots/nlp_plot3_top_words.png` | Top 15 words per sentiment class |
| `plots/nlp_plot4_word_heatmap.png` | Word frequency heatmap × sentiment |
| `plots/nlp_plot5_token_length.png` | Token count distribution by sentiment |
| `plots/nlp_plot6_tfidf_confusion.png` | TF-IDF classifier confusion matrix |
| `plots/nlp_plot7_tfidf_features.png` | Top TF-IDF feature coefficients |

---

## 📂 Dataset

**Name:** Social Media Sentiment Dataset  
**Source:** `3__Sentiment_dataset.csv`  
**Shape:** 732 rows × 15 columns  
**Platforms:** Twitter, Instagram, Facebook

| Column | Description |
|--------|-------------|
| `Text` | Raw social media post text |
| `Sentiment` | Granular emotion label (191 unique values) |
| `Platform` | Source platform |
| `Timestamp` | Date and time of post |
| `Country` | Country of origin |
| `Retweets` / `Likes` | Engagement metrics |

---

## 🎯 Objectives

- Preprocess text data (tokenisation, stopword removal, stemming/lemmatization)
- Apply sentiment analysis using lexicon-based scoring
- Visualise sentiment distribution and word frequencies

---

## 🛠️ Tools Used

```
Python | pandas | matplotlib | seaborn | scikit-learn | re | collections
```

> **Note:** The NLP pipeline was implemented from scratch using Python's built-in `re` and `collections` libraries — equivalent in functionality to `nltk`/`TextBlob`. TF-IDF vectorisation and classification used `sklearn`.

---

## ⚙️ How to Run

```bash
# Place 3__Sentiment_dataset.csv in the same folder as the script
python level3_nlp_sentiment.py
```

All 7 plots will be saved automatically as PNG files.

---

## 🔧 NLP Preprocessing Pipeline

```
Raw Text
   ↓
1. Lowercase conversion
   ↓
2. Remove URLs, @mentions, #hashtags
   ↓
3. Remove punctuation & numbers
   ↓
4. Tokenisation (whitespace split)
   ↓
5. Stopword removal (178 stopwords)
   ↓
6. Stemming (rule-based suffix stripping)
   ↓
Cleaned Token List
```

### Example
| Stage | Text |
|-------|------|
| Original | `"Enjoying a beautiful day at the park! 🌳"` |
| Cleaned | `"enjoy beauti day park"` |
| Tokens | `['enjoy', 'beauti', 'day', 'park']` |

---

## 📊 Sentiment Mapping

The raw dataset contained **191 unique granular emotion labels** (e.g. "Euphoria", "Heartbreak", "Nostalgia"). These were mapped to 3 standard classes:

| Class | Count | % |
|-------|-------|---|
| Positive | 366 | 50.0% |
| Neutral | 193 | 26.4% |
| Negative | 173 | 23.6% |

---

## 📈 Model Performance

### Lexicon-Based Scoring
- **45 positive** and **40 negative** seed words used
- Assigns Positive / Negative / Neutral based on word counts in each post

### TF-IDF + Logistic Regression (Binary: Pos vs Neg)

| Metric | Score |
|--------|-------|
| Accuracy | **79.6%** |
| Precision (Negative) | 1.00 |
| Recall (Negative) | 0.37 |
| Precision (Positive) | 0.77 |
| Recall (Positive) | 1.00 |
| F1 (Weighted Avg) | 0.76 |

**Vocabulary:** 500 unigrams + bigrams via TF-IDF

---

## 🔍 Key Findings

1. **Positive sentiment dominates** across all 3 platforms (~50% of posts)
2. **Instagram** has the highest proportion of positive content
3. Strong positive signal words: `love`, `enjoy`, `amazing`, `grateful`, `beautiful`
4. Strong negative signal words: `despair`, `grief`, `shatter`, `overwhelm`, `pain`
5. Negative posts tend to use **more emotionally intense vocabulary**
6. TF-IDF classifier achieves **79.6% accuracy** with a small dataset of 732 posts
7. The classifier shows **perfect precision for Negative class** — no false negatives when it predicts negative, though recall is lower due to class imbalance
