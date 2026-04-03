# Stage 13 — ML Auto-Retraining Pipeline Design

**Date:** 2026-04-03  
**Status:** Approved

## Goal
Automate weekly LightGBM retraining on latest 2yr data. Add CNN-BiLSTM retrain
trigger after 150+ real trade outcomes. Notify Discord with results. Never deploy
a model that performs worse than the current one.

## Components

| File | Change |
|------|--------|
| `config.py` | Add `RETRAIN_*` constants |
| `ml/retrainer.py` | New: `ModelRetrainer` class |
| `ml/predictor.py` | Add `reload_lgbm_model()` |
| `ml/deep_predictor.py` | Add `reload_deep_model()` |
| `alerts/discord_notifier.py` | Add `send_retrain_report()`, `send_deep_retrain_waiting()` |
| `scheduler/tasks.py` | Add `task_weekly_lgbm_retrain()` — Sunday 02:00 UTC |
| `tests/test_retrainer.py` | 8 tests, mocked training + data |

## Config Constants
```python
RETRAIN_LGBM_ENABLED        = True
RETRAIN_LGBM_INTERVAL_DAYS  = 7
RETRAIN_LGBM_MIN_ACCURACY   = 0.50    # deploy gate
RETRAIN_LGBM_ACCURACY_GATE  = 0.53    # informational only
RETRAIN_DEEP_ENABLED        = True
RETRAIN_DEEP_MIN_TRADES     = 150
RETRAIN_DEEP_MIN_ACCURACY   = 0.52
RETRAIN_STATE_FILE          = "state/retrain_state.json"   # abs path via BASE_DIR
RETRAIN_BACKUP_DIR          = "models/backups/"            # abs path via BASE_DIR
```

## ModelRetrainer — retrain_lgbm()
1. Load old accuracy from retrain_state.json (0.0 if missing)
2. Backup lgbm_direction.pkl, lgbm_scaler.pkl, lgbm_feature_columns.joblib → models/backups/lgbm_YYYYMMDD_HHMMSS.*
3. Fetch 2yr M15 via `data.fetcher.get_candles(n_candles=47000)` + macro
4. Call `ml.trainer.train_lgbm(df, save=False)` — CV only, no disk write
5. Deploy if: `new_cv >= RETRAIN_LGBM_MIN_ACCURACY AND new_cv >= old_accuracy - 0.01`
6. If deployed: write model files, call `reload_lgbm_model()`
7. If not deployed: restore from backup
8. Persist result to retrain_state.json

## ModelRetrainer — retrain_deep_if_ready()
1. Count `SELECT COUNT(*) FROM trades WHERE result IS NOT NULL AND status = 'closed'`
2. Return None if count < RETRAIN_DEEP_MIN_TRADES
3. Backup deep_model.keras + deep_scaler.pkl
4. Retrain via ml.deep_model.build_model() / train_model() / save_model() pipeline
5. Deploy if val_accuracy >= RETRAIN_DEEP_MIN_ACCURACY, else restore

## retrain_state.json schema
```json
{
    "lgbm": {"last_retrain": "ISO", "last_accuracy": 0.521, "deployed": true, "retrain_count": 3},
    "deep": {"last_retrain": null, "last_accuracy": null, "deployed": false, "retrain_count": 0, "trade_outcomes_available": 47}
}
```

## Key Integration Decisions
- `source='live'` proxy: `result IS NOT NULL AND status = 'closed'` (only live trades in SQLite)
- deep_trainer.main() is not reusable — call ml.deep_model APIs directly
- `reload_lgbm_model()` wraps existing `invalidate_lgbm_cache()` for naming consistency
- Sunday 02:00 UTC chosen — outside all trading sessions, safe to run
- Existing XGBoost retrain (Monday 00:00) is unchanged

## Test Plan
8 tests in tests/test_retrainer.py, all mocked:
- test_should_retrain_lgbm_interval
- test_should_retrain_lgbm_not_yet
- test_deep_retrain_not_ready
- test_backup_created
- test_deploy_on_accuracy_improvement
- test_no_deploy_below_minimum
- test_state_persist_load
- test_trade_outcome_count
