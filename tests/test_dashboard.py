"""
Tests for dashboard/app.py

Tests that the module imports correctly and helper functions exist.
Dashboard itself requires Streamlit which we test structurally.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.app import (
    STREAMLIT_AVAILABLE,
    _render_compliance_section,
    _render_latest_signal,
    _render_trade_history,
    _render_ml_status,
    _render_news_section,
)


# ── Test 1: Module imports successfully ──────────────────────────────────────

def test_module_imports():
    assert True  # If we got here, imports worked
    print(f"  ✓ Test 1 passed: Module imports OK (streamlit={'available' if STREAMLIT_AVAILABLE else 'not installed'})")


# ── Test 2: All render functions exist and are callable ──────────────────────

def test_render_functions():
    funcs = [
        _render_compliance_section,
        _render_latest_signal,
        _render_trade_history,
        _render_ml_status,
        _render_news_section,
    ]
    for f in funcs:
        assert callable(f), f"{f.__name__} is not callable"
    print(f"  ✓ Test 2 passed: All {len(funcs)} render functions exist")


# ── Test 3: run_dashboard function exists ────────────────────────────────────

def test_run_dashboard():
    from dashboard.app import run_dashboard
    assert callable(run_dashboard)
    print("  ✓ Test 3 passed: run_dashboard function exists")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_module_imports,
        test_render_functions,
        test_run_dashboard,
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
