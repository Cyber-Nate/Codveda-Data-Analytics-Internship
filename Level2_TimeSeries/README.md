# 📈 Level 2 — Task 2: Time Series Analysis

## 📌 Overview
Performed a comprehensive time series analysis on **S&P 500 stock price data** spanning 2014–2017. The analysis focuses on 5 major tech stocks — AAPL, GOOGL, MSFT, AMZN, and NFLX — covering price trends, trading volume, moving averages, seasonal decomposition, and daily return distributions.

---

## 📁 Files

| File | Description |
|------|-------------|
| `level2_time_series_stocks.py` | Main Python script |
| `plots/ts_plot1_closing_prices.png` | Closing price trends for all 5 stocks |
| `plots/ts_plot2_normalised_prices.png` | Normalised prices (base = 100 at Jan 2014) |
| `plots/ts_plot3_volume.png` | Daily trading volume per stock |
| `plots/ts_plot4_moving_averages.png` | AAPL with MA-20, MA-50, MA-200 |
| `plots/ts_plot5_decomposition.png` | Seasonal decomposition (AAPL weekly) |
| `plots/ts_plot6_daily_returns.png` | Daily return distributions |
| `plots/ts_plot7_returns_correlation.png` | Return correlation heatmap |

---

## 📂 Dataset

**Name:** S&P 500 Stock Prices  
**Source:** `2__Stock_Prices_Data_Set.csv`  
**Shape:** 497,472 rows × 7 columns (505 unique stocks)  
**Date Range:** 2014-01-02 to 2017-12-29  
**Stocks Analysed:** AAPL, GOOGL, MSFT, AMZN, NFLX

| Column | Description |
|--------|-------------|
| `symbol` | Stock ticker symbol |
| `date` | Trading date |
| `open` | Opening price (USD) |
| `high` | Daily high price (USD) |
| `low` | Daily low price (USD) |
| `close` | Closing price (USD) |
| `volume` | Number of shares traded |

---

## 🎯 Objectives

- Plot time-series data and identify patterns
- Decompose the series into trend, seasonality, and residuals
- Perform moving average smoothing and plot the results

---

## 🛠️ Tools Used

```
Python | pandas | matplotlib | NumPy
```

> **Note:** Seasonal decomposition was implemented manually using rolling averages and week-of-year grouping (multiplicative model), equivalent to `statsmodels.tsa.seasonal_decompose`.

---

## ⚙️ How to Run

```bash
# Place 2__Stock_Prices_Data_Set.csv in the same folder as the script
python level2_time_series_stocks.py
```

All 7 plots will be saved automatically as PNG files.

---

## 📈 Key Findings

### Stock Performance (2014–2017)

| Stock | Price Range | Total Return | Avg Daily Return | Volatility (Std) |
|-------|------------|-------------|-----------------|-----------------|
| AAPL | $71.40 – $176.42 | **+114.2%** | 0.086% | 1.431% |
| GOOGL | $497.06 – $1085.09 | **+89.1%** | 0.073% | 1.382% |
| MSFT | $34.98 – $86.85 | **+130.2%** | 0.092% | 1.370% |
| AMZN | $286.95 – $1195.83 | **+193.9%** | 0.124% | 1.865% |
| NFLX | $44.89 – $202.68 | **+270.4%** | 0.165% | 2.659% |

### Moving Averages (AAPL — end of 2017)
| MA | Value |
|----|-------|
| MA-20 | $171.89 |
| MA-50 | $169.93 |
| MA-200 | $155.96 |

### Return Correlations
- GOOGL ↔ MSFT showed the **highest correlation** (r = 0.55)
- AAPL ↔ NFLX showed the **lowest correlation** (r = 0.25)
- NFLX was the **highest returning** but also the **most volatile** stock

### Key Observations
1. All 5 stocks showed a consistent **upward trend** across 2014–2017
2. NFLX's MA-20 frequently crossed MA-50, signalling active momentum shifts
3. Seasonal decomposition of AAPL reveals a mild **annual cyclical pattern**
4. Higher-returning stocks (NFLX, AMZN) carry significantly higher daily volatility
