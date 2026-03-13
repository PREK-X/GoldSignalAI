# GoldSignalAI 🤖

> Production-quality AI-powered trading signal system for XAU/USD (Gold)
> Designed for prop firm challenges and serious retail trading

## Quick Start

```bash
# 1. Clone and enter directory
cd GoldSignalAI

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
nano .env   # Fill in your MT5 credentials and Telegram token

# 5. Select your prop firm in config.py
#    Edit: ACTIVE_PROP_FIRM = "FundedNext_2Step"  (or any preset)

# 6. Run the bot
python main.py

# 7. Open dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
GoldSignalAI/
├── main.py                     # Entry point — ties everything together
├── config.py                   # ALL tunable parameters
├── .env                        # Your credentials (never commit this)
├── requirements.txt
│
├── data/
│   ├── fetcher.py              # MT5 + yfinance data fetching
│   ├── processor.py            # Data cleaning & normalization
│   └── news_fetcher.py         # Economic calendar & news filter
│
├── analysis/
│   ├── indicators.py           # All 10 technical indicators
│   ├── scoring.py              # Signal scoring engine (X/10)
│   ├── sr_levels.py            # Auto support/resistance detection
│   ├── fibonacci.py            # Auto Fibonacci retracement
│   ├── candlestick.py          # Pattern recognition
│   └── multi_timeframe.py      # M15 + H1 agreement check
│
├── ml/
│   ├── features.py             # Feature engineering
│   ├── model.py                # Model definitions
│   ├── trainer.py              # Walk-forward training
│   ├── validator.py            # Accuracy validation
│   └── predictor.py            # Live prediction
│
├── signals/
│   ├── generator.py            # Final signal generation
│   ├── formatter.py            # Signal output formatting
│   └── risk_manager.py         # SL/TP calculator
│
├── propfirm/
│   ├── profiles.py             # All firm presets
│   ├── tracker.py              # Live compliance tracking
│   └── compliance_report.py   # Daily reports
│
├── alerts/
│   ├── telegram_bot.py         # Telegram bot & commands
│   └── chart_generator.py      # Chart image for Telegram
│
├── dashboard/
│   └── app.py                  # Streamlit web dashboard
│
├── backtest/
│   ├── engine.py               # Full strategy backtester
│   └── report_generator.py     # PDF + CSV export
│
├── scheduler/
│   └── tasks.py                # Auto-retraining + scheduled jobs
│
├── models/                     # Saved ML models (.pkl)
├── logs/                       # Rotating log files
├── data/historical/            # Cached historical data
└── reports/                    # Generated PDF/CSV reports
```

## Prop Firm Presets

| Firm | Daily Loss | Max DD | Profit Target |
|------|-----------|--------|---------------|
| FTMO | 5% | 10% | 10% |
| FundedNext 1-Step | 3% | 6% | 10% |
| FundedNext 2-Step | 5% | 10% | 8% |
| The5%ers | 4% | 6% | 6% |
| E8 Funding | 5% | 8% | 8% |
| MyForexFunds | 5% | 12% | 8% |
| Apex | 3% | 6% | 9% |
| Custom | configurable | configurable | configurable |

Switch firms by editing `ACTIVE_PROP_FIRM` in `config.py`.

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/signal` | Get latest signal |
| `/stats` | Win rate & performance |
| `/status` | Bot health check |
| `/drawdown` | Prop firm challenge status |
| `/setfirm` | Switch prop firm preset |
| `/pause` | Pause signal generation |
| `/resume` | Resume bot |
| `/help` | List all commands |
