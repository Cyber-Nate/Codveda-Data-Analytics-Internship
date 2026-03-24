# ============================================================
# Codveda Technologies - Data Analysis Internship
# Level 2 - Task 2: Time Series Analysis
# Dataset: S&P 500 Stock Prices (2014–2017)
# Tools: Python, pandas, matplotlib, statsmodels
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("seaborn-v0_8-whitegrid")
SAVE = True
FOCUS_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NFLX"]
COLORS       = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

def savefig(name):
    if SAVE:
        plt.savefig(name, dpi=150, bbox_inches="tight")
        print(f"  [saved] {name}")
    else:
        plt.show()
    plt.close()


# ── 1. Load & Prepare Data ───────────────────────────────────
print("=" * 60)
print("  STOCK PRICES - TIME SERIES ANALYSIS (2014–2017)")
print("=" * 60)

df_all = pd.read_csv("2__Stock_Prices_Data_Set.csv", parse_dates=["date"])
df_all.sort_values(["symbol", "date"], inplace=True)

print(f"\n[1] Full dataset: {df_all.shape[0]:,} rows, "
      f"{df_all['symbol'].nunique()} unique stocks")
print(f"    Date range  : {df_all['date'].min().date()} → "
      f"{df_all['date'].max().date()}")
print(f"    Missing vals: {df_all.isnull().sum().sum()}")

# Fill minor missing values forward
df_all[["open","high","low"]] = df_all.groupby("symbol")[["open","high","low"]].ffill()

# Filter to focus stocks
df = df_all[df_all["symbol"].isin(FOCUS_STOCKS)].copy()
print(f"\n[2] Filtered to {FOCUS_STOCKS}")
print(df.groupby("symbol")[["date"]].agg(["min","max"]))


# ── 2. Closing Price Trends ──────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for sym, color in zip(FOCUS_STOCKS, COLORS):
    data = df[df["symbol"] == sym]
    ax.plot(data["date"], data["close"], label=sym, color=color, linewidth=1.5)

ax.set_title("Closing Price Trends (2014–2017)", fontsize=15, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (USD)")
ax.legend(loc="upper left", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
plt.tight_layout()
savefig("ts_plot1_closing_prices.png")


# ── 3. Normalised Price (base = 100) ────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
for sym, color in zip(FOCUS_STOCKS, COLORS):
    data = df[df["symbol"] == sym].set_index("date")["close"]
    normalised = (data / data.iloc[0]) * 100
    ax.plot(normalised.index, normalised.values,
            label=sym, color=color, linewidth=1.5)

ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_title("Normalised Closing Price (Base = 100 at Jan 2014)",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Normalised Price")
ax.legend(loc="upper left", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
plt.tight_layout()
savefig("ts_plot2_normalised_prices.png")


# ── 4. Trading Volume ────────────────────────────────────────
fig, axes = plt.subplots(len(FOCUS_STOCKS), 1, figsize=(14, 14), sharex=True)
fig.suptitle("Daily Trading Volume (2014–2017)", fontsize=15, fontweight="bold")

for ax, sym, color in zip(axes, FOCUS_STOCKS, COLORS):
    data = df[df["symbol"] == sym]
    ax.bar(data["date"], data["volume"] / 1e6, color=color, alpha=0.6, width=1)
    ax.set_ylabel("Vol (M)")
    ax.set_title(sym, fontsize=11, fontweight="bold", loc="left")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
plt.tight_layout()
savefig("ts_plot3_volume.png")


# ── 5. Moving Averages (AAPL focus) ─────────────────────────
print("\n[3] Computing Moving Averages for AAPL...")
aapl = df[df["symbol"] == "AAPL"].set_index("date")["close"].copy()

aapl_ma = pd.DataFrame({"Close": aapl})
aapl_ma["MA_20"]  = aapl.rolling(window=20).mean()
aapl_ma["MA_50"]  = aapl.rolling(window=50).mean()
aapl_ma["MA_200"] = aapl.rolling(window=200).mean()

print(aapl_ma.tail(5).round(2))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(aapl_ma.index, aapl_ma["Close"],  color="#B0BEC5", linewidth=1,   label="Close",  alpha=0.8)
ax.plot(aapl_ma.index, aapl_ma["MA_20"],  color="#F44336", linewidth=1.5, label="MA 20")
ax.plot(aapl_ma.index, aapl_ma["MA_50"],  color="#FF9800", linewidth=1.5, label="MA 50")
ax.plot(aapl_ma.index, aapl_ma["MA_200"], color="#2196F3", linewidth=2,   label="MA 200")
ax.set_title("AAPL — Closing Price with Moving Averages",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30)
plt.tight_layout()
savefig("ts_plot4_moving_averages.png")


# ── 6. Seasonal Decomposition (AAPL) — Manual Implementation ─
print("\n[4] Performing Seasonal Decomposition on AAPL...")

# Resample to weekly for cleaner periodicity
aapl_weekly = aapl.resample("W").mean().dropna()

# Trend: 52-week rolling mean (centred)
trend = aapl_weekly.rolling(window=52, center=True).mean()

# Detrended: observed / trend (multiplicative model)
detrended = aapl_weekly / trend

# Seasonality: mean detrended value per week-of-year
week_num = aapl_weekly.index.isocalendar().week.astype(int)
seasonal_avg = detrended.groupby(week_num).transform("mean")
seasonal = pd.Series(seasonal_avg.values, index=aapl_weekly.index)

# Residuals
residual = detrended / seasonal

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("AAPL — Seasonal Decomposition (Weekly, Multiplicative)",
             fontsize=14, fontweight="bold")

components = {
    "Observed"   : aapl_weekly,
    "Trend"      : trend,
    "Seasonality": seasonal,
    "Residuals"  : residual,
}
comp_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

for ax, (label, series), color in zip(axes, components.items(), comp_colors):
    ax.plot(series.index, series.values, color=color, linewidth=1.5)
    ax.set_ylabel(label, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)

plt.tight_layout()
savefig("ts_plot5_decomposition.png")


# ── 7. Daily Returns Distribution ───────────────────────────
fig, axes = plt.subplots(1, len(FOCUS_STOCKS), figsize=(18, 5))
fig.suptitle("Daily Return Distribution per Stock", fontsize=14, fontweight="bold")

for ax, sym, color in zip(axes, FOCUS_STOCKS, COLORS):
    data = df[df["symbol"] == sym].set_index("date")["close"]
    returns = data.pct_change().dropna() * 100
    ax.hist(returns, bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(returns.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"Mean: {returns.mean():.2f}%")
    ax.set_title(sym, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

plt.tight_layout()
savefig("ts_plot6_daily_returns.png")


# ── 8. Correlation of Daily Returns ─────────────────────────
print("\n[5] Computing return correlations...")
returns_df = pd.DataFrame()
for sym in FOCUS_STOCKS:
    data = df[df["symbol"] == sym].set_index("date")["close"]
    returns_df[sym] = data.pct_change()

corr_matrix = returns_df.corr().round(2)
print(corr_matrix)

import seaborn as sns
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Daily Returns Correlation Heatmap",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("ts_plot7_returns_correlation.png")


# ── 9. Summary Statistics ────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY STATISTICS")
print("=" * 60)
for sym in FOCUS_STOCKS:
    data = df[df["symbol"] == sym].set_index("date")["close"]
    ret  = data.pct_change().dropna() * 100
    total_return = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100
    print(f"\n  {sym}:")
    print(f"    Price range     : ${data.min():.2f} – ${data.max():.2f}")
    print(f"    Total return    : {total_return:.1f}%")
    print(f"    Avg daily return: {ret.mean():.4f}%")
    print(f"    Volatility (std): {ret.std():.4f}%")

print("\n\nTime Series Analysis complete. All plots saved.")
