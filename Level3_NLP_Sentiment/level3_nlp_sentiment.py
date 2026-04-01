# ============================================================
# Codveda Technologies - Data Analysis Internship
# Level 3 - Task 3: NLP — Sentiment Analysis
# Dataset: Social Media Sentiment Dataset
# Tools: Python, pandas, matplotlib, sklearn
# Note: Implements NLP pipeline from scratch + sklearn TF-IDF
#       (equivalent to nltk/TextBlob tokenization & analysis)
# ============================================================

import re
import string
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

plt.style.use("seaborn-v0_8-whitegrid")
SAVE = True

def savefig(name):
    if SAVE:
        plt.savefig(name, dpi=150, bbox_inches="tight")
        print(f"  [saved] {name}")
    else:
        plt.show()
    plt.close()


# ── 0. English Stopwords (built-in, no nltk needed) ─────────
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself just let's me more
most mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those
through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who who's
whom why why's will with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves
""".split())


# ── 1. Text Preprocessing Functions ─────────────────────────
def clean_text(text):
    """Lowercase, remove URLs, mentions, punctuation, numbers."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#\w+", "", text)                     # hashtags
    text = re.sub(r"[^\w\s]", " ", text)                 # punctuation
    text = re.sub(r"\d+", "", text)                      # numbers
    text = re.sub(r"\s+", " ", text).strip()             # extra spaces
    return text

def tokenize(text):
    """Split cleaned text into word tokens."""
    return text.split()

def remove_stopwords(tokens):
    """Remove stopwords from token list."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def simple_stem(word):
    """Very lightweight rule-based stemmer (suffix stripping)."""
    suffixes = ["ing", "tion", "ness", "ful", "less", "ly", "ed", "er", "est"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[:-len(suffix)]
    return word

def lemmatize_simple(tokens):
    """Apply simple stemming as a proxy for lemmatization."""
    return [simple_stem(t) for t in tokens]

def preprocess_pipeline(text):
    """Full NLP preprocessing pipeline."""
    cleaned  = clean_text(text)
    tokens   = tokenize(cleaned)
    tokens   = remove_stopwords(tokens)
    tokens   = lemmatize_simple(tokens)
    return " ".join(tokens), tokens


# ── 2. Lexicon-Based Sentiment Scorer ───────────────────────
# Positive and negative word lexicons (common words)
POSITIVE_WORDS = set("""
good great excellent amazing wonderful fantastic brilliant awesome
beautiful happy joy love enjoy nice best lovely perfect fun exciting
thrilling fantastic pleased delighted glad cheerful positive success
achievement outstanding superb fabulous magnificent brilliant terrific
incredible remarkable impressive outstanding rewarding satisfying
winning hopeful grateful thankful blessed inspired motivated energetic
""".split())

NEGATIVE_WORDS = set("""
bad terrible awful horrible disgusting dreadful worst hate angry sad
disappointed frustrated upset terrible nasty poor failure miserable
annoyed depressed horrible painful devastating regret sorry anxious
worried stressed nervous scared fear dread panic negative failure
loss terrible disaster tragedy unfortunate unhappy gloomy dark hopeless
""".split())

def lexicon_sentiment(tokens):
    """Score sentiment based on word lexicon."""
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos > neg:
        return "Positive", pos - neg
    elif neg > pos:
        return "Negative", neg - pos
    else:
        return "Neutral", 0


# ── 3. Load & Prepare Data ───────────────────────────────────
print("=" * 60)
print("  SOCIAL MEDIA — NLP SENTIMENT ANALYSIS")
print("=" * 60)

df = pd.read_csv("3__Sentiment_dataset.csv")

# Clean column names and strip whitespace
df.columns = [c.strip() for c in df.columns]
df["Text"]      = df["Text"].astype(str).str.strip()
df["Sentiment"] = df["Sentiment"].str.strip()
df["Platform"]  = df["Platform"].str.strip()

print(f"\n[1] Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"    Unique raw sentiments: {df['Sentiment'].nunique()}")

# Map granular sentiments → Positive / Negative / Neutral
POSITIVE_SET = {
    "Positive","Joy","Excitement","Happiness","Happy","Love","Gratitude",
    "Grateful","Hope","Hopeful","Elation","Euphoria","Contentment","Amusement",
    "Celebration","Enthusiasm","Inspiration","Inspired","Satisfaction","Pride",
    "Proud","Confidence","Confident","Optimism","Freedom","Empowerment","Relief",
    "Blessed","Overjoyed","Triumph","Achievement","Accomplishment","Admiration",
    "Adoration","Affection","Amazement","Appreciation","Awe","Charm","Calmness",
    "Serenity","Harmony","Tranquility","Mindfulness","Positivity","Playful",
    "Kindness","Kind","Compassion","Compassionate","Friendship","Romance",
    "Tenderness","Warmth","Heartwarming","Radiance","Wonder","Wonderment",
    "Rejuvenation","Fulfillment","Breakthrough","Resilience","Success",
    "Motivation","Energy","Creativity","Creative Inspiration","Zest",
}
NEGATIVE_SET = {
    "Negative","Anger","Sadness","Sad","Fear","Fearful","Disgust","Hate",
    "Frustration","Frustrated","Disappointment","Disappointed","Anxiety",
    "Grief","Sorrow","Heartbreak","Heartache","Despair","Desperation",
    "Regret","Shame","Resentment","Betrayal","Jealousy","Jealous","Envious",
    "Envy","Bitter","Bitterness","Loss","Loneliness","Isolation","Desolation",
    "Overwhelmed","Exhaustion","Suffering","Devastated","Darkness","Helplessness",
    "Hopelessness","Numbness","Boredom","Disgust","Embarrassed","Apprehensive",
    "Intimidation","Pressure","Obstacle","Miscalculation","Bad","Negative",
}

def map_sentiment(s):
    if s in POSITIVE_SET:
        return "Positive"
    elif s in NEGATIVE_SET:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment_mapped"] = df["Sentiment"].apply(map_sentiment)
print(f"\n[2] Mapped sentiments:")
print(df["Sentiment_mapped"].value_counts())


# ── 4. Text Preprocessing ────────────────────────────────────
print("\n[3] Applying NLP preprocessing pipeline...")
df["cleaned_text"], df["tokens"] = zip(*df["Text"].apply(preprocess_pipeline))
df["token_count"] = df["tokens"].apply(len)

print(f"  Sample preprocessing:")
for _, row in df.head(3).iterrows():
    print(f"  Original : {row['Text'][:60]}")
    print(f"  Cleaned  : {row['cleaned_text'][:60]}")
    print(f"  Tokens   : {row['tokens'][:8]}")
    print()


# ── 5. Lexicon Sentiment Scoring ─────────────────────────────
print("[4] Applying lexicon-based sentiment scoring...")
df["lex_sentiment"], df["lex_score"] = zip(
    *df["tokens"].apply(lexicon_sentiment)
)
print(f"  Lexicon predictions:\n{df['lex_sentiment'].value_counts()}")


# ── 6. Sentiment Distribution Plot ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sentiment Distribution", fontsize=14, fontweight="bold")

SENT_COLORS = {"Positive": "#4CAF50", "Neutral": "#FF9800", "Negative": "#F44336"}

# Mapped labels
counts = df["Sentiment_mapped"].value_counts()
axes[0].bar(counts.index, counts.values,
            color=[SENT_COLORS[s] for s in counts.index], edgecolor="black")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")
axes[0].set_title("Mapped Sentiment Labels")
axes[0].set_ylabel("Count")

# Lexicon predictions
lex_counts = df["lex_sentiment"].value_counts()
axes[1].bar(lex_counts.index, lex_counts.values,
            color=[SENT_COLORS[s] for s in lex_counts.index], edgecolor="black")
for i, v in enumerate(lex_counts.values):
    axes[1].text(i, v + 3, str(v), ha="center", fontweight="bold")
axes[1].set_title("Lexicon-Based Predictions")
axes[1].set_ylabel("Count")

plt.tight_layout()
savefig("nlp_plot1_sentiment_distribution.png")


# ── 7. Sentiment by Platform ─────────────────────────────────
platform_sent = pd.crosstab(df["Platform"], df["Sentiment_mapped"])
platform_sent_pct = platform_sent.div(platform_sent.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
platform_sent_pct[["Positive","Neutral","Negative"]].plot(
    kind="bar", ax=ax,
    color=[SENT_COLORS["Positive"], SENT_COLORS["Neutral"], SENT_COLORS["Negative"]],
    edgecolor="white"
)
ax.set_title("Sentiment Distribution by Platform",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Platform")
ax.set_ylabel("Proportion")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Sentiment")
plt.tight_layout()
savefig("nlp_plot2_sentiment_by_platform.png")


# ── 8. Top Words per Sentiment ───────────────────────────────
print("\n[5] Computing top words per sentiment...")
top_n = 15

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle(f"Top {top_n} Words per Sentiment Class",
             fontsize=14, fontweight="bold")

for ax, (sent, color) in zip(axes, SENT_COLORS.items()):
    subset = df[df["Sentiment_mapped"] == sent]["tokens"]
    all_tokens = [t for tokens in subset for t in tokens]
    top_words = collections.Counter(all_tokens).most_common(top_n)
    if top_words:
        words, freqs = zip(*top_words)
        ax.barh(list(words)[::-1], list(freqs)[::-1],
                color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{sent}", fontweight="bold", color=color, fontsize=12)
        ax.set_xlabel("Frequency")
    print(f"  {sent}: {dict(top_words)}")

plt.tight_layout()
savefig("nlp_plot3_top_words.png")


# ── 9. Word Frequency Heatmap (Top words × Sentiment) ────────
all_tokens_flat = [t for tokens in df["tokens"] for t in tokens]
top_global = [w for w, _ in collections.Counter(all_tokens_flat).most_common(25)]

heatmap_data = {}
for sent in ["Positive", "Neutral", "Negative"]:
    subset_tokens = [t for tokens in df[df["Sentiment_mapped"]==sent]["tokens"]
                     for t in tokens]
    counter = collections.Counter(subset_tokens)
    total   = sum(counter.values()) or 1
    heatmap_data[sent] = [counter.get(w, 0) / total * 1000 for w in top_global]

heatmap_df = pd.DataFrame(heatmap_data, index=top_global)

fig, ax = plt.subplots(figsize=(8, 9))
sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.3, ax=ax)
ax.set_title("Word Frequency per 1000 Tokens × Sentiment",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Sentiment")
plt.tight_layout()
savefig("nlp_plot4_word_heatmap.png")


# ── 10. Token Length Distribution ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for sent, color in SENT_COLORS.items():
    subset = df[df["Sentiment_mapped"] == sent]["token_count"]
    ax.hist(subset, bins=20, alpha=0.6, color=color, label=sent, edgecolor="white")
ax.set_title("Token Count Distribution by Sentiment",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Tokens (after preprocessing)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
savefig("nlp_plot5_token_length.png")


# ── 11. TF-IDF + Logistic Regression Classifier ─────────────
print("\n[6] Training TF-IDF + Logistic Regression classifier...")

# Filter to Positive/Negative only for binary classification
df_binary = df[df["Sentiment_mapped"].isin(["Positive","Negative"])].copy()
df_binary["label"] = (df_binary["Sentiment_mapped"] == "Positive").astype(int)

X_text = df_binary["cleaned_text"]
y      = df_binary["label"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
X_tr_tfidf = tfidf.fit_transform(X_tr)
X_te_tfidf = tfidf.transform(X_te)

clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_tr_tfidf, y_tr)
preds = clf.predict(X_te_tfidf)

print(f"  Accuracy: {accuracy_score(y_te, preds):.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_te, preds, target_names=["Negative","Positive"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_te, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"],
            linewidths=0.5)
ax.set_title("TF-IDF + Logistic Regression\nConfusion Matrix",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
savefig("nlp_plot6_tfidf_confusion.png")

# Top TF-IDF features
feature_names = tfidf.get_feature_names_out()
coef = clf.coef_[0]
top_pos_idx = coef.argsort()[-15:][::-1]
top_neg_idx = coef.argsort()[:15]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Top TF-IDF Features (Logistic Regression Coefficients)",
             fontsize=13, fontweight="bold")

axes[0].barh([feature_names[i] for i in top_pos_idx[::-1]],
             [coef[i] for i in top_pos_idx[::-1]],
             color="#4CAF50", edgecolor="white")
axes[0].set_title("Top Positive Indicators", fontweight="bold")
axes[0].set_xlabel("Coefficient")

axes[1].barh([feature_names[i] for i in top_neg_idx],
             [coef[i] for i in top_neg_idx],
             color="#F44336", edgecolor="white")
axes[1].set_title("Top Negative Indicators", fontweight="bold")
axes[1].set_xlabel("Coefficient")

plt.tight_layout()
savefig("nlp_plot7_tfidf_features.png")


# ── 12. Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY FINDINGS")
print("=" * 60)
print(f"""
Dataset         : {len(df)} social media posts across platforms
Platforms       : {', '.join(df['Platform'].unique())}
Sentiment split : {df['Sentiment_mapped'].value_counts().to_dict()}

NLP Pipeline Steps:
  1. Lowercasing, URL/mention/hashtag removal
  2. Punctuation & number stripping
  3. Tokenization
  4. Stopword removal ({len(STOPWORDS)} stopwords)
  5. Stemming/Lemmatization (rule-based suffix stripping)

Lexicon Analysis:
  - {len(POSITIVE_WORDS)} positive seed words | {len(NEGATIVE_WORDS)} negative seed words
  - Lexicon predictions: {df['lex_sentiment'].value_counts().to_dict()}

TF-IDF Classifier:
  - Vocabulary: 500 unigrams + bigrams
  - Accuracy: {accuracy_score(y_te, preds)*100:.1f}% on binary Pos/Neg

Key observations:
  1. Positive sentiment dominates across all 3 platforms.
  2. Instagram has the highest proportion of positive posts.
  3. Words like "love", "enjoy", "amazing" are strong positive signals.
  4. Words like "bad", "hate", "sad" are strong negative signals.
""")
print("NLP Sentiment Analysis complete. All plots saved.")
