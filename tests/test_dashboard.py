"""
Tests for dashboard/app.py — Stage 14

Tests that the module imports correctly and all tab/helper functions exist.
Dashboard itself requires a running Streamlit session, so we test structurally.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.app import (
    STREAMLIT_AVAILABLE,
    run_dashboard,
    render_sidebar,
    tab_trade_history,
    tab_ml_status,
    tab_regime_detection,
    tab_challenge_progress,
    tab_risk_monitor,
    tab_signal_heatmap,
    _db_query,
    _load_trades_df,
    _load_signals_df,
    _load_retrain_state,
    _get_model_info,
    _load_challenge_state,
    _load_state_manager,
    _ensure_tables,
    C_GOLD,
    C_RED,
    C_GREEN,
    PLOTLY_LAYOUT,
)


# ── Test 1: Module imports successfully ──────────────────────────────────────

def test_module_imports():
    assert STREAMLIT_AVAILABLE is not None
    print(f"  ✓ Test 1 passed: Module imports OK (streamlit={'available' if STREAMLIT_AVAILABLE else 'not installed'})")


# ── Test 2: All tab functions exist and are callable ─────────────────────────

def test_tab_functions_exist():
    funcs = [
        run_dashboard,
        render_sidebar,
        tab_trade_history,
        tab_ml_status,
        tab_regime_detection,
        tab_challenge_progress,
        tab_risk_monitor,
        tab_signal_heatmap,
    ]
    for f in funcs:
        assert callable(f), f"{f.__name__} is not callable"
    print(f"  ✓ Test 2 passed: All {len(funcs)} tab/entry functions are callable")


# ── Test 3: Helper functions are callable ────────────────────────────────────

def test_helper_functions():
    helpers = [
        _db_query,
        _load_trades_df,
        _load_signals_df,
        _load_retrain_state,
        _get_model_info,
        _load_challenge_state,
        _load_state_manager,
        _ensure_tables,
    ]
    for h in helpers:
        assert callable(h), f"{h.__name__} is not callable"
    print(f"  ✓ Test 3 passed: All {len(helpers)} helper functions are callable")


# ── Test 4: Theme constants are defined ──────────────────────────────────────

def test_theme_constants():
    assert C_GOLD.startswith("#"), f"C_GOLD should be a hex color, got: {C_GOLD}"
    assert C_RED.startswith("#"),  f"C_RED should be a hex color, got: {C_RED}"
    assert C_GREEN.startswith("#"), f"C_GREEN should be a hex color, got: {C_GREEN}"
    assert isinstance(PLOTLY_LAYOUT, dict)
    assert "template" in PLOTLY_LAYOUT
    assert PLOTLY_LAYOUT["template"] == "plotly_dark"
    print(f"  ✓ Test 4 passed: Theme constants correct (gold={C_GOLD})")


# ── Test 5: _db_query returns list (empty or populated) ─────────────────────

def test_db_query_returns_list():
    result = _db_query("SELECT 1 AS val")
    assert isinstance(result, list)
    print(f"  ✓ Test 5 passed: _db_query returns list (len={len(result)})")


# ── Test 6: _load_retrain_state returns dict ─────────────────────────────────

def test_load_retrain_state():
    state = _load_retrain_state()
    assert isinstance(state, dict)
    print(f"  ✓ Test 6 passed: _load_retrain_state returns dict (keys={list(state.keys())})")


# ── Test 7: _get_model_info returns correct schema ───────────────────────────

def test_get_model_info():
    info = _get_model_info("models/lgbm_direction.pkl")
    assert isinstance(info, dict)
    assert "exists" in info
    assert "path" in info
    assert "mtime" in info
    assert isinstance(info["exists"], bool)
    print(f"  ✓ Test 7 passed: _get_model_info schema correct (exists={info['exists']})")


# ── Test 8: _load_trades_df returns DataFrame ────────────────────────────────

def test_load_trades_df():
    import pandas as pd
    df = _load_trades_df()
    assert isinstance(df, pd.DataFrame)
    print(f"  ✓ Test 8 passed: _load_trades_df returns DataFrame (shape={df.shape})")


# ── Test 9: _load_state_manager returns dict ─────────────────────────────────

def test_load_state_manager():
    state = _load_state_manager()
    assert isinstance(state, dict)
    print(f"  ✓ Test 9 passed: _load_state_manager returns dict")


# ── Test 10: _load_challenge_state returns dict ──────────────────────────────

def test_load_challenge_state():
    state = _load_challenge_state()
    assert isinstance(state, dict)
    print(f"  ✓ Test 10 passed: _load_challenge_state returns dict")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_module_imports,
        test_tab_functions_exist,
        test_helper_functions,
        test_theme_constants,
        test_db_query_returns_list,
        test_load_retrain_state,
        test_get_model_info,
        test_load_trades_df,
        test_load_state_manager,
        test_load_challenge_state,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Dashboard tests: {passed}/{len(tests)} passed")
