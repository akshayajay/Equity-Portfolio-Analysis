# NIFTY50 Equity Portfolio Analyser

[![Tests](https://github.com/akshayajay/Equity-Portfolio-Analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/akshayajay/Equity-Portfolio-Analysis/actions/workflows/tests.yml)

A Streamlit web app that compares a **momentum-based stock selection strategy** against the equal-weight NIFTY50 benchmark and the index itself — letting you tune every parameter interactively and see performance metrics in real time.

---

## What it does

- **Momentum strategy** — ranks all 50 NIFTY50 constituents by historical total return over a configurable look-back window and builds a portfolio from the top N stocks.
- **Equal-weight benchmark** — splits capital evenly across all 50 NIFTY50 stocks.
- **NIFTY Index reference** — tracks the actual ^NSEI index as a passive baseline.

All three are plotted as equity curves, and the app reports CAGR, annualised volatility, Sharpe ratio (6.5% risk-free rate), and max drawdown for each.

---
Example:
<img width="1465" height="645" alt="image" src="https://github.com/user-attachments/assets/d530cff1-3f6d-4d89-9746-2989212b1d8b" />
<img width="1016" height="652" alt="image" src="https://github.com/user-attachments/assets/aeaf4b11-cb24-4eab-9eac-9f9de2e0fee2" />

---
## Tech stack

| Layer | Library |
|---|---|
| Data | `yfinance` (Yahoo Finance) |
| Data wrangling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `altair` |
| App framework | `streamlit` |

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/akshayajay/Equity-Portfolio-Analysis.git
cd Equity-Portfolio-Analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run App.py
```

The app will open automatically at `http://localhost:8501`.

---

## Parameters (sidebar)

| Parameter | Description | Default |
|---|---|---|
| Start / End Date | Date range for the equity curve | 2020-01-01 → today |
| Momentum Lookback | Trading days used to rank stock momentum | 100 |
| Number of Top Stocks | How many stocks the momentum strategy holds | 10 |
| Initial Capital | Starting portfolio value in ₹ | ₹10,00,000 |
| Ranking window | Years of data used to rank top stocks | 1 |

---

## Project structure

```
Equity-Portfolio-Analysis/
├── App.py            # Streamlit application entry point
├── topstock.py       # NIFTY50 momentum ranking logic
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

---

## Notes

- All price data is fetched live from Yahoo Finance on app load (results are cached per session).
- `auto_adjust=True` is used so the `Close` column already reflects corporate actions (splits, dividends).
- Sharpe ratio uses a 6.5% risk-free rate (approximate Indian 10-year government bond yield).
- This project is for educational purposes only and does not constitute financial advice.

---

## Author

**Akshaya J** · [github.com/akshayajay](https://github.com/akshayajay)
