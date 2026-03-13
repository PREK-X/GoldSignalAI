You are going to build me a complete, production-quality AI-powered trading signal application called "GoldSignalAI". This is not a demo or portfolio project — it must be accurate, reliable, and good enough to be used as a real trading assistant to pass prop firm challenges and eventually be sold as a product. The bot must perform better than most professional retail traders and achieve a 60-70%+ win rate through disciplined, multi-layered analysis.

═══════════════════════════════════════
PLATFORM & ENVIRONMENT
═══════════════════════════════════════

- OS: Ubuntu Linux
- Language: Python 3.10+
- Broker Platform: MetaTrader 5 (XM broker demo/live account)
- All libraries must be free and open source
- Must run 24/7 without manual intervention
- Use .env file for ALL API keys and credentials

═══════════════════════════════════════
PRIMARY ASSET & TIMEFRAMES
═══════════════════════════════════════

- Asset: XAU/USD (Gold) — primary focus
- Primary timeframe: M15 (15 minutes)
- Confirmation timeframe: H1 (1 hour)
- Only generate signal when BOTH timeframes agree

═══════════════════════════════════════
SIGNAL OUTPUT FORMAT
═══════════════════════════════════════
Every signal must display EXACTLY this:

┌─────────────────────────────────────┐
│ GoldSignalAI 🤖 │
├─────────────────────────────────────┤
│ Asset: XAU/USD (Gold) │
│ Signal: BUY 🟢 / SELL 🔴 / WAIT ⚪ │
│ Entry Price: 2,312.50 │
│ Stop Loss: 2,298.00 (-14.5 pips) │
│ Take Profit 1: 2,341.00 (+28.5 pips)│
│ Take Profit 2: 2,370.00 (+57 pips) │
│ Confidence: 73% │
│ Risk/Reward: 1:2 / 1:3 │
│ Timeframe: M15 + H1 ✅ │
│ ML Confirm: YES ✅ │
│ Indicators: 8/10 Bullish │
│ Timestamp: 2025-03-11 14:30 UTC │
└─────────────────────────────────────┘

WAIT signal = do not trade, market is unclear
Only fire alert if confidence >= 70%
Quality over quantity — fewer, better signals

═══════════════════════════════════════
DATA SOURCES
═══════════════════════════════════════
Primary: MetaTrader5 Python library (live XM connection)
Fallback: yfinance (if MT5 not connected)
News sentiment: Use free NewsAPI or RSS feeds from Reuters/Bloomberg
Economic calendar: Fetch from forexfactory.com or investing.com RSS
Data must be fetched on every new candle close (every 15 minutes)

═══════════════════════════════════════
LAYER 1 — TREND ANALYSIS
═══════════════════════════════════════
Calculate ALL of these using pandas-ta:

- EMA 20, EMA 50, EMA 200
  → BUY if price > EMA20 > EMA50 > EMA200
  → SELL if price < EMA20 < EMA50 < EMA200
- ADX 14
  → Only trade when ADX > 25 (strong trend confirmed)
  → ADX > 40 = very strong trend, increase confidence
- Ichimoku Cloud
  → Price above cloud = bullish
  → Price below cloud = bearish
  → Kumo twist = trend change signal

═══════════════════════════════════════
LAYER 2 — MOMENTUM & ENTRY TIMING
═══════════════════════════════════════

- RSI 14
  → Oversold < 30 = BUY signal
  → Overbought > 70 = SELL signal
  → RSI divergence detection (price makes new high but RSI doesn't = reversal warning)
- MACD (12, 26, 9)
  → Bullish crossover = BUY
  → Bearish crossover = SELL
  → Histogram momentum confirmation
- Stochastic Oscillator (14, 3, 3)
  → Oversold < 20 = BUY
  → Overbought > 80 = SELL
  → %K crosses %D = entry signal
- CCI (Commodity Channel Index) 20
  → Below -100 = oversold BUY
  → Above +100 = overbought SELL

═══════════════════════════════════════
LAYER 3 — VOLATILITY & PRICE LEVELS
═══════════════════════════════════════

- Bollinger Bands (20, 2)
  → Price touches lower band + RSI oversold = strong BUY
  → Price touches upper band + RSI overbought = strong SELL
  → Band squeeze (low volatility) = breakout imminent, prepare signal
- ATR 14 (Average True Range)
  → Used ONLY for stop loss calculation
  → Stop Loss = Entry ± (1.5 × ATR)
  → Never place SL closer than 10 pips on Gold
- Support & Resistance Zones
  → Auto-detect from last 200 candles
  → Mark strong zones where price bounced 3+ times
  → Entry near support = higher BUY confidence
  → Entry near resistance = higher SELL confidence
- Fibonacci Retracement
  → Auto-calculate from last significant swing high/low
  → Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
  → Signal near 61.8% retracement = highest probability entry

═══════════════════════════════════════
LAYER 4 — VOLUME ANALYSIS
═══════════════════════════════════════

- Compare current volume to 20-period average volume
- Signal with above-average volume = stronger confirmation (+1 score)
- Signal with below-average volume = weaker, reduce confidence
- Volume surge on breakout = very strong confirmation (+2 score)

═══════════════════════════════════════
LAYER 5 — CANDLESTICK PATTERN RECOGNITION
═══════════════════════════════════════
Detect and score these patterns using pandas-ta or manual calculation:

- Hammer / Inverted Hammer (bullish reversal)
- Shooting Star / Hanging Man (bearish reversal)
- Bullish/Bearish Engulfing (strong reversal)
- Doji (indecision — reduce confidence, consider WAIT)
- Morning Star / Evening Star (strong reversal)
- Pin Bar (rejection of key level)
  Each confirmed pattern adds +1 to the confidence score

═══════════════════════════════════════
LAYER 6 — SIGNAL SCORING ENGINE
═══════════════════════════════════════
Build a scoring system with these exact rules:

Total indicators monitored: 10
Each bullish signal = +1
Each bearish signal = -1
Neutral = 0

Final score calculation:

- Count bullish signals
- Confidence % = (bullish count / 10) × 100
- BUY: confidence >= 70% (7+ indicators bullish)
- SELL: confidence >= 70% (7+ indicators bearish)
- WAIT: anything between 30–70%

NEVER output a signal below 70% confidence
Display "X/10 indicators bullish/bearish" in every signal

Multi-timeframe rule:

- Calculate score on M15
- Calculate score on H1
- Both must agree (both BUY or both SELL)
- If they disagree → automatic WAIT regardless of score

═══════════════════════════════════════
LAYER 7 — MACHINE LEARNING MODEL
═══════════════════════════════════════
Model: XGBoost Classifier (primary) + Random Forest (verification)

Features (inputs to model):

- All technical indicator values listed above
- Previous 5 candle price changes
- Volume ratio (current vs average)
- Hour of day (market sessions matter)
- Day of week
- Distance from support/resistance
- ATR value (market volatility)

Target (what model predicts):

- Will price go UP or DOWN after 15 candles?
- Binary classification: 1 = up, 0 = down

Training rules:

- Use walk-forward validation (NOT simple train/test split)
- Train on 2 years of historical XAU/USD M15 data
- Minimum 70% accuracy before model is used in production
- If model accuracy drops below 60% → auto-retrain immediately
- Save model to disk as goldSignalAI_model.pkl

Auto-retraining scheduler:

- Retrain every Monday at 00:00 UTC
- Add previous week's data to training set
- Model improves continuously over time
- Log retraining results (accuracy before/after)
- Never use a model with less than 60% accuracy

ML confirmation rule:

- ML must AGREE with technical signal
- If ML says BUY but technicals say SELL → output WAIT
- Display ML confidence % separately in signal output
- ML adds to overall confidence calculation

═══════════════════════════════════════
LAYER 8 — NEWS & ECONOMIC CALENDAR FILTER
═══════════════════════════════════════

- Fetch economic calendar daily (high-impact events for USD and Gold)
- PAUSE all signals 30 minutes before and after:
  → US Non-Farm Payrolls
  → US CPI/Inflation data
  → Federal Reserve interest rate decisions
  → US GDP releases
  → Any high-impact news event
- During pause: output "⚠️ NEWS EVENT — Bot paused" on dashboard
- Resume automatically after event window passes
- This alone will significantly improve win rate

═══════════════════════════════════════
STOP LOSS & TAKE PROFIT RULES
═══════════════════════════════════════
Stop Loss:

- Always use ATR-based SL: Entry ± (1.5 × ATR14)
- Minimum 10 pips on Gold, maximum 30 pips
- Place just beyond nearest support/resistance level

Take Profit:

- TP1: 1:2 risk/reward (close 50% of position here)
- TP2: 1:3 risk/reward (close remaining 50% here)
- Display both TP levels in every signal
- Suggest partial close strategy to user

Risk Management Display:

- Show pip distance for SL and both TPs
- Show risk/reward ratio
- Show suggested lot size based on user's account balance (from settings)

═══════════════════════════════════════
PROP FIRM CHALLENGE COMPLIANCE MODULE
═══════════════════════════════════════
Build a UNIVERSAL prop firm compliance module that works
with ANY prop firm (FundedNext, FTMO, The5ers, E8, MyForexFunds,
Apex, etc). All rules must be configurable via config.py so
the user can change values to match any firm's rules.

CONFIGURABLE RULES (in config.py):

- DAILY_LOSS_LIMIT = 5.0 # % — firm's max daily loss
- DAILY_LOSS_WARNING = 4.0 # % — warn before hitting limit
- MAX_TOTAL_DRAWDOWN = 10.0 # % — firm's max total drawdown
- TOTAL_DRAWDOWN_WARNING = 8.0 # % — warn before hitting limit
- PROFIT_TARGET = 10.0 # % — target to pass challenge
- MIN_TRADING_DAYS = 2 # minimum days required
- MAX_DAILY_LOSS_HARD_STOP = True # stop bot at warning level
- CHALLENGE_ACCOUNT_SIZE = 10000 # challenge account size in USD
- PROP_FIRM_NAME = "FundedNext" # display name (changeable)

MODULE BEHAVIOR:

- Bot automatically stops trading if daily loss reaches warning level
- Bot automatically stops if total drawdown reaches warning level
- Display current drawdown status on dashboard at all times
- Color coding: Green = safe, Yellow = caution, Red = stopped
- Send Telegram alert when approaching ANY limit
- Track and display:
  → Current daily P&L %
  → Total drawdown from peak %
  → Days traded so far
  → Progress toward profit target %
  → Estimated days to complete challenge
- Generate daily compliance report
- Reset daily loss counter at midnight UTC
- Log all limit breaches with timestamps

CHALLENGE PROFILES (pre-configured presets):
Include ready-made presets the user can select:

FTMO:
daily_loss = 5%, max_drawdown = 10%, profit_target = 10%

FundedNext_1Step:
daily_loss = 3%, max_drawdown = 6%, profit_target = 10%

FundedNext_2Step:
daily_loss = 5%, max_drawdown = 10%, profit_target = 8%

The5ers:
daily_loss = 4%, max_drawdown = 6%, profit_target = 6%

E8_Funding:
daily_loss = 5%, max_drawdown = 8%, profit_target = 8%

MyForexFunds:
daily_loss = 5%, max_drawdown = 12%, profit_target = 8%

Apex:
daily_loss = 3%, max_drawdown = 6%, profit_target = 9%

Custom:
all values configurable by user

User selects profile in config.py and bot
automatically applies that firm's rules.

═══════════════════════════════════════
STREAMLIT DASHBOARD
═══════════════════════════════════════
Build a professional web dashboard with these sections:

1. LIVE SIGNAL CARD (top center, most prominent)
   - Current signal with all details
   - Confidence gauge (visual circular meter)
   - Color coded: green/red/grey background

2. LIVE CHART
   - Candlestick chart using Plotly
   - EMA 20, 50, 200 overlaid
   - Bollinger Bands overlaid
   - Support/Resistance zones marked
   - Current SL and TP levels shown as horizontal lines
   - Auto-updates every 60 seconds

3. INDICATOR PANEL
   - Live values for all 10 indicators
   - Each shows bullish/bearish/neutral with color coding
   - Score counter showing X/10

4. PROP FIRM CHALLENGE TRACKER
   - Shows current prop firm name (from config)
   - Current daily P&L %
   - Total drawdown %
   - Days traded
   - Progress toward profit target
   - Visual progress bar
   - Color coded status (green/yellow/red)
   - Quick button to switch between firm presets

5. SIGNAL HISTORY TABLE
   - Last 50 signals
   - Each row: asset, signal type, entry, SL, TP,
     outcome (win/loss/pending), profit/loss in pips

6. PERFORMANCE STATISTICS
   - Overall win rate %
   - Total signals generated
   - Average risk/reward achieved
   - Best/worst trade
   - Profit factor
   - Streak (current win/loss streak)

7. ACCOUNT RISK CALCULATOR
   - User inputs: account balance
   - Bot outputs: suggested lot size for 1% risk per trade
   - Shows potential profit/loss in USD for each signal

8. ML MODEL STATUS
   - Model accuracy %
   - Last retrain date
   - Next scheduled retrain
   - Number of training samples

Auto-refresh: every 60 seconds
Mobile responsive design
Dark theme (professional trading look)

═══════════════════════════════════════
TELEGRAM ALERT BOT
═══════════════════════════════════════
Commands:

- /signal → Get latest signal immediately
- /stats → Win rate, total trades, performance summary
- /status → Bot running status, model accuracy, last signal time
- /drawdown → Current prop firm challenge status
- /setfirm → Change active prop firm preset
- /pause → Manually pause bot
- /resume → Resume bot
- /help → List all commands

Auto alerts:

- New BUY/SELL signal (only >= 70% confidence)
- Include chart image with entry/SL/TP marked
- News pause notification
- Daily summary at market close (17:00 EST)
- Weekly performance report every Sunday
- Alert if model accuracy drops below 65%
- Alert if approaching ANY prop firm limit
- "Market Closed" message on weekends

Message format: Same as signal output format shown above
Never send WAIT signals as alerts — only genuine BUY/SELL

═══════════════════════════════════════
BACKTESTING MODULE
═══════════════════════════════════════

- Backtest on last 2 years of XAU/USD M15 data
- Simulate the full strategy including all layers
- Include transaction costs (spread simulation: 3 pips for Gold)
- Results to display:
  → Total trades
  → Win rate %
  → Average win size (pips)
  → Average loss size (pips)
  → Profit factor
  → Max drawdown %
  → Sharpe ratio
  → Best month / worst month
  → Monthly breakdown table
- Simulate passing each prop firm's challenge
  → Show how many days it would have taken
  → Show if strategy would have passed or failed
  → Show closest breach of rules
- Export full backtest report as PDF
- Export trade history as CSV
- Use backtesting.py library

═══════════════════════════════════════
PROJECT STRUCTURE
═══════════════════════════════════════
GoldSignalAI/
├── main.py
├── config.py
├── .env
├── requirements.txt
├── README.md
│
├── data/
│ ├── fetcher.py
│ ├── processor.py
│ └── news_fetcher.py
│
├── analysis/
│ ├── indicators.py
│ ├── scoring.py
│ ├── sr_levels.py
│ ├── fibonacci.py
│ ├── candlestick.py
│ └── multi_timeframe.py
│
├── ml/
│ ├── features.py
│ ├── model.py
│ ├── trainer.py
│ ├── validator.py
│ └── predictor.py
│
├── signals/
│ ├── generator.py
│ ├── formatter.py
│ └── risk_manager.py
│
├── propfirm/
│ ├── tracker.py
│ ├── profiles.py
│ └── compliance_report.py
│
├── alerts/
│ ├── telegram_bot.py
│ └── chart_generator.py
│
├── dashboard/
│ └── app.py
│
├── backtest/
│ ├── engine.py
│ └── report_generator.py
│
├── scheduler/
│ └── tasks.py
│
├── models/
├── logs/
├── data/historical/
└── reports/

═══════════════════════════════════════
LIBRARIES TO USE
═══════════════════════════════════════
MetaTrader5 — live broker data
yfinance — fallback data source
pandas, numpy — data processing
pandas-ta — all technical indicators
xgboost — primary ML model
scikit-learn — Random Forest + validation
plotly — interactive charts
streamlit — dashboard
python-telegram-bot — telegram alerts
backtesting — backtesting engine
schedule — auto-retraining scheduler
python-dotenv — environment variables
fpdf2 — PDF report generation
matplotlib — chart image generation for Telegram
requests — news/calendar API calls
ta — additional indicator library

═══════════════════════════════════════
CODE QUALITY REQUIREMENTS
═══════════════════════════════════════

- Every function must have docstrings
- Every module must have error handling (try/except)
- Every module must have logging (use Python logging module)
- Handle: network drops, market closed, bad data, MT5 disconnects
- Graceful shutdown (save state before stopping)
- Config file for all tunable parameters (no hardcoded values)
- Each module must work and be testable independently
- Comments explaining WHY not just what

═══════════════════════════════════════
BUILD ORDER (follow exactly)
═══════════════════════════════════════
Build and fully complete each module before moving to next:

1.  Project structure + requirements.txt + .env template + config.py
2.  data/fetcher.py — MT5 + yfinance data fetching
3.  data/processor.py — data cleaning
4.  analysis/indicators.py — all technical indicators
5.  analysis/sr_levels.py — support/resistance
6.  analysis/fibonacci.py — fibonacci levels
7.  analysis/candlestick.py — pattern recognition
8.  analysis/scoring.py — signal scoring engine
9.  analysis/multi_timeframe.py — M15 + H1 confirmation
10. ml/features.py — feature engineering
11. ml/trainer.py — model training
12. ml/predictor.py — live prediction
13. signals/risk_manager.py — SL/TP calculator
14. signals/generator.py — final signal generation
15. signals/formatter.py — signal formatting
16. propfirm/profiles.py — all firm presets
17. propfirm/tracker.py — universal compliance tracker
18. propfirm/compliance_report.py — daily reports
19. data/news_fetcher.py — news filter
20. alerts/chart_generator.py — chart images
21. alerts/telegram_bot.py — telegram bot
22. dashboard/app.py — streamlit dashboard
23. backtest/engine.py — backtesting
24. backtest/report_generator.py — PDF reports
25. scheduler/tasks.py — auto-retraining
26. main.py — tie everything together

After EACH module: pause and ask me to test it before continuing.
If I report an error: fix it completely before moving forward.
Never skip ahead — each module must work before building the next.

Start now with Step 1. Create the complete project structure,
requirements.txt, .env template, and config.py with all
prop firm presets configured and ready to use.
