"""
GoldSignalAI — config.py
========================
Central configuration for all modules.
All tunable parameters live here — never hardcode values in other modules.

Usage:
    from config import Config, PropFirmProfile, PROP_FIRM_PROFILES
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# PROP FIRM PROFILE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PropFirmProfile:
    """
    Represents the risk rules for a specific prop firm challenge.
    All percentage values are expressed as plain floats (5.0 = 5%).
    """
    name: str
    daily_loss_limit: float          # Max daily loss % before hard stop
    daily_loss_warning: float        # Warning threshold % (before hard stop)
    max_total_drawdown: float        # Max total drawdown % (from peak)
    total_drawdown_warning: float    # Warning threshold for total drawdown
    profit_target: float             # % profit needed to pass challenge
    min_trading_days: int            # Minimum number of trading days required
    max_daily_trades: int = 20       # Safety cap on trades per day
    news_filter_enabled: bool = True # Pause bot around high-impact news
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PROP FIRM PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PROP_FIRM_PROFILES: dict[str, PropFirmProfile] = {

    "FTMO": PropFirmProfile(
        name="FTMO",
        daily_loss_limit=5.0,
        daily_loss_warning=4.0,
        max_total_drawdown=10.0,
        total_drawdown_warning=8.0,
        profit_target=10.0,
        min_trading_days=4,
        description="FTMO Standard Challenge — most popular prop firm"
    ),

    "FundedNext_1Step": PropFirmProfile(
        name="FundedNext (1-Step)",
        daily_loss_limit=3.0,
        daily_loss_warning=2.5,
        max_total_drawdown=6.0,
        total_drawdown_warning=5.0,
        profit_target=10.0,
        min_trading_days=5,
        description="FundedNext Stellar 1-Step — strict daily loss rule"
    ),

    "FundedNext_2Step": PropFirmProfile(
        name="FundedNext (2-Step)",
        daily_loss_limit=5.0,
        daily_loss_warning=4.0,
        max_total_drawdown=10.0,
        total_drawdown_warning=8.0,
        profit_target=8.0,
        min_trading_days=5,
        description="FundedNext Stellar 2-Step — standard challenge"
    ),

    "The5ers": PropFirmProfile(
        name="The5%ers",
        daily_loss_limit=4.0,
        daily_loss_warning=3.0,
        max_total_drawdown=6.0,
        total_drawdown_warning=5.0,
        profit_target=6.0,
        min_trading_days=0,   # No minimum days requirement
        description="The5%ers Hyper Growth — low drawdown, low target"
    ),

    "E8_Funding": PropFirmProfile(
        name="E8 Funding",
        daily_loss_limit=5.0,
        daily_loss_warning=4.0,
        max_total_drawdown=8.0,
        total_drawdown_warning=6.5,
        profit_target=8.0,
        min_trading_days=0,
        description="E8 Funding Standard — no minimum trading days"
    ),

    "MyForexFunds": PropFirmProfile(
        name="MyForexFunds",
        daily_loss_limit=5.0,
        daily_loss_warning=4.0,
        max_total_drawdown=12.0,
        total_drawdown_warning=10.0,
        profit_target=8.0,
        min_trading_days=0,
        description="MyForexFunds Rapid — higher drawdown allowance"
    ),

    "Apex": PropFirmProfile(
        name="Apex Trader Funding",
        daily_loss_limit=3.0,
        daily_loss_warning=2.5,
        max_total_drawdown=6.0,
        total_drawdown_warning=5.0,
        profit_target=9.0,
        min_trading_days=0,
        description="Apex — strict daily loss, futures-focused"
    ),

    "Custom": PropFirmProfile(
        name="Custom Firm",
        daily_loss_limit=5.0,
        daily_loss_warning=4.0,
        max_total_drawdown=10.0,
        total_drawdown_warning=8.0,
        profit_target=10.0,
        min_trading_days=0,
        description="User-defined — edit values directly in config.py"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONFIGURATION CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Master configuration object.
    All other modules import from this class.
    Change values here to tune the entire system.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    APP_NAME = "GoldSignalAI"
    VERSION = "1.0.0"

    # ── Asset & Timeframes ────────────────────────────────────────────────────
    SYMBOL = "XAUUSD"                    # MT5 symbol name
    SYMBOL_DISPLAY = "XAU/USD (Gold)"    # Human-readable display name
    PRIMARY_TIMEFRAME = "M15"            # Primary analysis timeframe
    CONFIRMATION_TIMEFRAME = "H1"        # Must agree with primary
    CANDLE_INTERVAL_SECONDS = 900        # 15 minutes = 900 seconds

    # MT5 timeframe constants (set at runtime from mt5 import)
    # These are referenced as strings; fetcher.py resolves them to MT5 constants
    MT5_TIMEFRAME_M15 = "M15"
    MT5_TIMEFRAME_H1  = "H1"

    # ── Data Sources ─────────────────────────────────────────────────────────
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    POLYGON_SYMBOL  = "C:XAUUSD"         # Polygon forex pair format
    DATA_SOURCE_PRIORITY = ["polygon", "mt5", "yfinance"]  # Fallback order
    MIN_BARS_REQUIRED = 100              # Minimum bars for valid dataset
    TIMEZONE = "UTC"                     # All timestamps normalised to UTC

    LOOKBACK_CANDLES = 500               # Candles to fetch per request
    HISTORICAL_YEARS = 2                 # Years of history for ML training
    MIN_CANDLES_FOR_SIGNAL = 200         # Minimum candles before generating signal
    YFINANCE_SYMBOL = "GC=F"            # Gold futures ticker on yfinance

    # ── Indicator Parameters ──────────────────────────────────────────────────
    # Trend
    EMA_FAST   = 20
    EMA_MID    = 50
    EMA_SLOW   = 200
    ADX_PERIOD = 14
    ADX_TREND_THRESHOLD   = 25          # Minimum ADX to confirm trend
    ADX_STRONG_THRESHOLD  = 40          # ADX above this = very strong trend

    # Ichimoku
    ICHIMOKU_TENKAN  = 9
    ICHIMOKU_KIJUN   = 26
    ICHIMOKU_SENKOU  = 52

    # Momentum
    RSI_PERIOD      = 14
    RSI_OVERSOLD    = 30
    RSI_OVERBOUGHT  = 70
    MACD_FAST       = 12
    MACD_SLOW       = 26
    MACD_SIGNAL     = 9
    STOCH_K         = 14
    STOCH_D         = 3
    STOCH_SMOOTH    = 3
    STOCH_OVERSOLD  = 20
    STOCH_OVERBOUGHT = 80
    CCI_PERIOD      = 20
    CCI_OVERSOLD    = -100
    CCI_OVERBOUGHT  = 100

    # Volatility
    BB_PERIOD  = 20
    BB_STDDEV  = 2
    ATR_PERIOD = 14
    ATR_SL_MULTIPLIER     = 1.5         # SL = Entry ± (ATR × this)
    MIN_SL_PIPS           = 50          # Minimum stop loss in pips (Gold)
    MAX_SL_PIPS           = 200         # Maximum stop loss in pips (Gold)
    # Note: Gold M15 median candle range is ~125 pips. SL must be wider
    # than a single candle to avoid noise stop-outs. ATR14 × 1.5 ≈ 130 pips.
    PIP_SIZE              = 0.1         # 1 pip for Gold = $0.10 per 0.01 lot
    GOLD_PIP_VALUE        = 10.0        # $ per pip per standard lot (XAU/USD)

    # Risk/Reward
    TP1_RR_RATIO = 2.0                  # TP1 at 1:2 R/R
    TP2_RR_RATIO = 3.0                  # TP2 at 1:3 R/R

    # Volume
    VOLUME_LOOKBACK     = 20            # Periods for average volume
    VOLUME_SURGE_FACTOR = 2.0           # Volume × this = "surge"

    # Support & Resistance
    SR_LOOKBACK       = 200             # Candles to scan for S/R levels
    SR_MIN_BOUNCES    = 2               # Minimum bounces to confirm a zone
    SR_TOLERANCE_PIPS = 5               # Zone width in pips

    # Fibonacci
    FIBO_LOOKBACK = 100                 # Candles to find swing high/low
    FIBO_LEVELS   = [0.236, 0.382, 0.500, 0.618, 0.786]
    FIBO_KEY_LEVEL = 0.618              # 61.8% — highest probability entry

    # ── Signal Scoring ────────────────────────────────────────────────────────
    # Validated config — 2yr backtest: PF 1.23,
    # DD 10.04%, +17.7% return, 112 trades
    TOTAL_INDICATORS     = 12           # 12 voted indicators (HMA/EMA, ADX, Ichimoku, RSI, MACD, Williams %R, CCI, Supertrend, Connors RSI, Keltner, ATR, Volume; BBands ML-only)
    MIN_CONFIDENCE_PCT   = 65           # Minimum to fire BUY/SELL signal
    MAX_CONFIDENCE_PCT   = 75           # Cap — above 75% indicates over-consensus (lagging)
    WAIT_LOWER_BOUND_PCT = 30           # Below this = strong counter-signal
    WAIT_UPPER_BOUND_PCT = 70           # Above this = tradeable signal

    # ── Machine Learning ─────────────────────────────────────────────────────
    USE_ML_FILTER        = False        # CV 46.7% — no edge; disabled per 2yr backtest validation
    ML_MODEL_PATH        = "models/goldSignalAI_model.pkl"
    ML_RF_MODEL_PATH     = "models/goldSignalAI_rf_model.pkl"
    ML_SCALER_PATH       = "models/goldSignalAI_scaler.pkl"
    ML_MIN_ACCURACY      = 0.45         # Lowered: gold CV accuracy ~46%, model saved for dashboard display
    ML_PRODUCTION_MIN    = 0.70         # Required before going live
    ML_RETRAIN_WEEKDAY   = 0            # 0 = Monday (Python weekday)
    ML_RETRAIN_HOUR_UTC  = 0            # Midnight UTC
    ML_FUTURE_CANDLES    = 15           # Predict price direction after N candles
    ML_WALK_FORWARD_SPLITS = 5          # Walk-forward validation folds
    ML_FEATURES_CANDLE_HISTORY = 5      # Previous N candle changes as features

    # XGBoost hyperparameters
    XGB_N_ESTIMATORS     = 500
    XGB_MAX_DEPTH        = 4    # Reduced from 6 — prevents overfitting
    XGB_MIN_CHILD_WEIGHT = 3    # Minimum samples per leaf — prevents overfitting
    XGB_LEARNING_RATE    = 0.05
    XGB_SUBSAMPLE        = 0.8
    XGB_COLSAMPLE        = 0.8
    XGB_EARLY_STOPPING   = 50
    ML_OVERFIT_WARNING_THRESHOLD = 0.80  # Warn if in-sample accuracy exceeds this

    # Random Forest hyperparameters
    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH    = 10

    # ── News Filter ───────────────────────────────────────────────────────────
    NEWS_PAUSE_MINUTES_BEFORE = 30      # Pause N minutes before high-impact news
    NEWS_PAUSE_MINUTES_AFTER  = 30      # Resume N minutes after news
    NEWS_HIGH_IMPACT_KEYWORDS = [
        "Non-Farm Payroll", "NFP", "CPI", "Inflation",
        "Federal Reserve", "Fed Rate", "Interest Rate",
        "FOMC", "GDP", "Unemployment", "Core PCE",
    ]
    NEWS_CURRENCIES_TO_WATCH = ["USD", "XAU"]
    FOREXFACTORY_RSS = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

    # ── Prop Firm Challenge ───────────────────────────────────────────────────
    # ↓ CHANGE THIS LINE to switch between firm presets ↓
    ACTIVE_PROP_FIRM = "FundedNext_1Step"

    CHALLENGE_ACCOUNT_SIZE = 10_000     # Account size in USD
    RISK_PER_TRADE_PCT     = 1.0        # % of account risked per trade

    # ── Telegram ─────────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_DAILY_SUMMARY_HOUR_EST = 17  # 5 PM EST
    TELEGRAM_WEEKLY_REPORT_WEEKDAY  = 6   # 6 = Sunday

    # ── MT5 Credentials ───────────────────────────────────────────────────────
    MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER   = os.getenv("MT5_SERVER", "XMGlobal-MT5")

    # ── News API ──────────────────────────────────────────────────────────────
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    NEWS_API_URL = "https://newsapi.org/v2/everything"

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR            = os.path.join(BASE_DIR, "data", "historical")
    MODELS_DIR          = os.path.join(BASE_DIR, "models")
    LOGS_DIR            = os.path.join(BASE_DIR, "logs")
    REPORTS_DIR         = os.path.join(BASE_DIR, "reports")
    SIGNAL_HISTORY_FILE = os.path.join(BASE_DIR, "logs", "signal_history.json")
    TRADE_LOG_FILE      = os.path.join(BASE_DIR, "logs", "trades.csv")
    PROP_STATE_FILE     = os.path.join(BASE_DIR, "logs", "prop_firm_state.json")
    ML_TRAINING_LOG     = os.path.join(BASE_DIR, "logs", "ml_training.log")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    DASHBOARD_PORT          = 8501
    DASHBOARD_REFRESH_SEC   = 60        # Auto-refresh interval
    DASHBOARD_MAX_SIGNALS   = 50        # Rows in signal history table

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE         = os.path.join(BASE_DIR, "logs", "goldsignal.log")
    LOG_MAX_BYTES    = 10 * 1024 * 1024  # 10 MB before rotation
    LOG_BACKUP_COUNT = 5

    # ── Market Sessions (UTC) ─────────────────────────────────────────────────
    # Gold is most active during London + New York overlap
    LONDON_OPEN_UTC   = 8
    LONDON_CLOSE_UTC  = 16
    NEW_YORK_OPEN_UTC = 13
    NEW_YORK_CLOSE_UTC = 21
    WEEKEND_DAYS      = [5, 6]          # Saturday=5, Sunday=6


    @classmethod
    def get_active_prop_firm(cls) -> PropFirmProfile:
        """Return the currently active prop firm profile."""
        profile = PROP_FIRM_PROFILES.get(cls.ACTIVE_PROP_FIRM)
        if profile is None:
            raise ValueError(
                f"Unknown prop firm preset '{cls.ACTIVE_PROP_FIRM}'. "
                f"Valid options: {list(PROP_FIRM_PROFILES.keys())}"
            )
        return profile

    @classmethod
    def get_all_prop_firms(cls) -> dict[str, PropFirmProfile]:
        """Return all available prop firm presets."""
        return PROP_FIRM_PROFILES

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate critical config values.
        Returns a list of warning/error strings. Empty list = all good.
        """
        issues = []

        if not cls.POLYGON_API_KEY:
            issues.append("WARNING: POLYGON_API_KEY not set in .env — Polygon data source disabled")
        if cls.MT5_LOGIN == 0:
            issues.append("WARNING: MT5_LOGIN not set in .env — will use yfinance fallback")
        if not cls.MT5_PASSWORD:
            issues.append("WARNING: MT5_PASSWORD not set — will use yfinance fallback")
        if not cls.TELEGRAM_BOT_TOKEN:
            issues.append("WARNING: TELEGRAM_BOT_TOKEN not set — Telegram alerts disabled")
        if not cls.TELEGRAM_CHAT_ID:
            issues.append("WARNING: TELEGRAM_CHAT_ID not set — Telegram alerts disabled")
        if cls.ACTIVE_PROP_FIRM not in PROP_FIRM_PROFILES:
            issues.append(f"ERROR: ACTIVE_PROP_FIRM='{cls.ACTIVE_PROP_FIRM}' is not a valid preset")
        if cls.MIN_CONFIDENCE_PCT < 55:
            issues.append("WARNING: MIN_CONFIDENCE_PCT below 60% — signal quality will degrade")
        if cls.RISK_PER_TRADE_PCT > 2.0:
            issues.append("WARNING: RISK_PER_TRADE_PCT > 2% — risky for prop firm challenges")

        return issues


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE SHORTHAND
# Make it easy to import the active profile directly
# ─────────────────────────────────────────────────────────────────────────────

def get_active_profile() -> PropFirmProfile:
    """Shorthand: returns the active PropFirmProfile."""
    return Config.get_active_prop_firm()
