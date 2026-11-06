# ⚡ NEXUS-QUANT: Local Quantitative Backtesting & Fundamental Analysis AI Platform

> A local-first, deterministic quantitative trading research terminal featuring bar-by-bar backtesting, Smart Money Concepts (FVG/IFVG), dynamic S&R pivot breakouts, macroeconomic news blackout shielding, fundamental financial health scoring (Piotroski F-Score), and a zero-cost multi-provider AI advisor.

---

## 🏛️ System Architecture

```
nexus-quant/
├── app.py                  # Streamlit terminal frontend (iOS frosted glass dark UI)
├── data_engine.py          # Market data pipeline (4h cache) & ForexFactory economic calendar
├── indicators.py           # MA 20/50, S&R pivots, Fair Value Gaps (FVG/IFVG), RSI multi-mode
├── backtester.py           # Deterministic event-driven engine (SL buffer, 1:1 BE trailing, TP, slippage)
├── fundamental_engine.py   # Financial statements, key ratios, Piotroski F-Score (0-9)
├── ai_agent.py             # Zero-cost AI advisor switcher (Google Gemini, Groq, local Ollama)
├── requirements.txt        # Lightweight dependencies (local execution, no paid APIs)
├── .env.example            # Environment variables template
├── .gitignore              # Ignores local caches, parquet, pycache, .env
└── README.md               # Architecture documentation & setup guide
```

---

## 🎨 1. UI Palette & Design System (iOS Frosted Glass)
The terminal interface uses custom CSS injected into Streamlit:
- **Canvas Background**: `#0B0C0E` (True Jet)
- **Card Surface**: `#14161A` (Carbon Dark, 82% opacity with `backdrop-filter: blur(12px)`)
- **Border / Frame**: `#272A30` (Titanium Edge, 1px solid)
- **Primary AI Accent**: `#E2F952` (Signal Volt)
- **Secondary Accent**: `#38BDF8` (Ice Blue)
- **Bullish / Profit**: `#22C55E` (Pure Lime)
- **Bearish / Drawdown**: `#EF4444` (Vermilion)
- **Text (Primary)**: `#F3F4F6` (Frost Silver)
- **Text (Muted)**: `#6B7280` (Charcoal Fog)

---

## 📈 2. Multi-Asset Universe & Timeframes
Automatically downloaded via free APIs (`yfinance` with fallback parsing and local caching):
1. **Crypto (15-Minute)**: `BTC-USD`, `ETH-USD` (Max hold period: 3-4 days).
2. **Forex (30-Minute)**: `EURUSD=X`, `GBPUSD=X`, `GBPJPY=X`, `EURAUD=X`.
3. **Indices & Commodities (5-Minute)**:
   - Gold: `GC=F` (XAUUSD proxy)
   - Silver: `SI=F` (XAGUSD proxy)
   - Crude Oil: `CL=F` (USOIL proxy)
   - NASDAQ: `NQ=F`
   - S&P 500: `ES=F`
4. **Stocks (Daily / 1H)**: `NVDA`, `TSLA`, `AAPL`, `MSFT` (equipped with fundamental analysis cards).

---

## ⚙️ 3. Deterministic Trading Engine Rules
No oversimplified vector loops. The engine calculates bar-by-bar execution:
1. **Trend Bias**:
   - `MA 20` (Fast) & `MA 50` (Slow).
   - **LONG Bias**: Price > MA 20 **AND** Price > MA 50.
   - **SHORT Bias**: Price < MA 20 **AND** Price < MA 50.
   - **Neutral / No Trade**: Price sandwiched between MAs.
2. **Signal Triggers**:
   - **Dynamic S&R**: Pivot highs and lows (configurable 5-bar lookback/lookahead) with breakout/retest detection.
   - **SMC Fair Value Gaps (FVG)**:
     - Bullish FVG: `Bar[i-2].High < Bar[i].Low` (gap up).
     - Bearish FVG: `Bar[i-2].Low > Bar[i].High` (gap down).
     - **Inversion FVG (IFVG)**: When a previous FVG is breached in reverse, flipping into opposite support/resistance.
   - **RSI Multi-Mode (14-period)**:
     - Reversal Mode: `RSI < 30` (Oversold Long) or `RSI > 70` (Overbought Short) at S&R levels.
     - Breakout Mode: Momentum trades allowed only when `45 <= RSI <= 60`.
3. **Risk & Trade Management**:
   - **Stop Loss (SL)**: Placed beyond S&R or reversal candle wick + user-defined breathing buffer (e.g., 2–4 pips).
   - **Breakeven Trailing**: As soon as price reaches `1:1 Risk-to-Reward (RR)`, SL trails to exact entry.
   - **Take Profit (TP)**: Configurable fixed target between `1:2.2` and `1:3.0` RR.
   - **Friction**: Real-world slippage and spread simulated on every entry/exit.
4. **Macro / News Shield**:
   - Live economic calendar parser (Forex Factory RSS / free calendar feeds).
   - Flags **High-Impact ("Red Folder")** events (CPI, FOMC, NFP) and imposes a **30-minute blackout** before and after release.

---

## 📊 4. Fundamental Health Analysis
- Fetches balance sheets, income statements, and cash flows for equities via `yfinance`.
- Computes P/E (TTM & Forward), EPS Growth, FCF, FCF Yield, Debt-to-Equity, Current Ratio, ROE, ROIC, and Gross Margins.
- **Piotroski F-Score (0-9)**:
  - Profitability: ROA > 0, Operating Cash Flow > 0, ΔROA > 0, CFO > Net Income.
  - Leverage/Liquidity: ΔLong-term Debt < 0, ΔCurrent Ratio > 0, No dilution.
  - Efficiency: ΔGross Margin > 0, ΔAsset Turnover > 0.
- Overall rating: **STRONG (7-9)**, **MODERATE (4-6)**, or **WEAK (0-3)**.

---

## 🤖 5. Zero-Cost AI Advisor Layer
Switch seamlessly between free local and cloud providers:
1. **Google Gemini**: Free tier via Google AI Studio (`GEMINI_API_KEY`). Default: `gemini-2.0-flash`.
2. **Groq**: Free tier ultra-fast Llama 3 / DeepSeek models (`GROQ_API_KEY`). Default: `llama-3.3-70b-versatile`.
3. **Local Ollama**: 100% offline, zero data leaves your PC (`http://localhost:11434`). Default: `llama3.2`.

The AI receives structured JSON payloads of current setups, backtest metrics, and economic releases to deliver risk-first trade critiques.

---

## 🚀 Quickstart Guide (Local Windows Setup)

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Git for Windows

### 2. Installation
```powershell
# Navigate to the project directory
cd c:\Users\Abdullah\Downloads\nexus-quant

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. API Keys Configuration (Optional for Cloud AI)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and fill in any keys you want to use:
```ini
GEMINI_API_KEY=your_free_gemini_key_here
GROQ_API_KEY=your_free_groq_key_here
```
*Note: If you run Ollama locally (`ollama run llama3.2`), no API keys are required.*

### 4. Launch the Terminal
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📦 Git & GitHub Setup Instructions

Follow these steps to initialize Git and push to your private GitHub repository:

```powershell
cd c:\Users\Abdullah\Downloads\nexus-quant

# 1. Initialize Git repository
git init -b main

# 2. Stage all files (.env and parquet caches are automatically ignored by .gitignore)
git add .

# 3. Create initial commit
git commit -m "feat: initialize NEXUS-QUANT quantitative trading and research platform"

# 4. Link your private GitHub repository
# Replace <YOUR_GITHUB_USERNAME> and <YOUR_REPO_NAME> with your details:
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git

# 5. Push to GitHub
git push -u origin main
```
