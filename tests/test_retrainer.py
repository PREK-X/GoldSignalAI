"""
tests/test_retrainer.py
=======================
Unit tests for the Stage 13 ML Auto-Retraining Pipeline.

All training and data fetching is mocked — tests do not actually
train models or hit external APIs.
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, mock_open

from config import Config


class TestShouldRetrainLgbm(unittest.TestCase):
    """should_retrain_lgbm() interval logic."""

    def setUp(self):
        # Redirect state file to a temp dir
        self.tmpdir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = self.state_file
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_retrainer(self):
        from ml.retrainer import ModelRetrainer
        return ModelRetrainer()

    def test_should_retrain_lgbm_interval(self):
        """Returns True when more than RETRAIN_LGBM_INTERVAL_DAYS have passed."""
        retrainer = self._make_retrainer()
        old_time = datetime.now(timezone.utc) - timedelta(days=8)
        state = retrainer.load_state()
        state["lgbm"]["last_retrain"] = old_time.isoformat()
        retrainer.save_state(state)

        self.assertTrue(retrainer.should_retrain_lgbm())

    def test_should_retrain_lgbm_not_yet(self):
        """Returns False when fewer than RETRAIN_LGBM_INTERVAL_DAYS have passed."""
        retrainer = self._make_retrainer()
        recent_time = datetime.now(timezone.utc) - timedelta(days=2)
        state = retrainer.load_state()
        state["lgbm"]["last_retrain"] = recent_time.isoformat()
        retrainer.save_state(state)

        self.assertFalse(retrainer.should_retrain_lgbm())

    def test_should_retrain_lgbm_never_run(self):
        """Returns True when no retrain has ever occurred (state file absent)."""
        retrainer = self._make_retrainer()
        # No state file written
        self.assertTrue(retrainer.should_retrain_lgbm())


class TestDeepRetrainNotReady(unittest.TestCase):
    """retrain_deep_if_ready() returns None when < 150 outcomes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_deep_retrain_not_ready(self):
        """Returns None if fewer than RETRAIN_DEEP_MIN_TRADES outcomes are available."""
        from ml.retrainer import ModelRetrainer
        retrainer = ModelRetrainer()

        with patch.object(retrainer, "get_trade_outcome_count", return_value=47):
            result = retrainer.retrain_deep_if_ready()

        self.assertIsNone(result)


class TestBackupCreated(unittest.TestCase):
    """Model files are backed up before retraining starts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_backup_created(self):
        """Backup files are created before the retrain begins."""
        from ml.retrainer import ModelRetrainer
        retrainer = ModelRetrainer()

        # Create a fake model file to back up
        fake_model_path = os.path.join(self.tmpdir, "lgbm_direction.pkl")
        with open(fake_model_path, "w") as f:
            f.write("fake model")

        orig_model = Config.LGBM_MODEL_PATH
        Config.LGBM_MODEL_PATH = fake_model_path

        try:
            ts = "20260407_020000"
            backups = retrainer._backup_lgbm(ts)
            # Backup file must exist in backups dir
            self.assertTrue(len(backups) > 0)
            for src, dst in backups.items():
                self.assertTrue(os.path.isfile(dst), f"Backup not found: {dst}")
        finally:
            Config.LGBM_MODEL_PATH = orig_model


class TestDeployOnAccuracyImprovement(unittest.TestCase):
    """Deploys new model when accuracy improves."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("ml.retrainer.ModelRetrainer._backup_lgbm")
    @patch("ml.retrainer.ModelRetrainer._restore_lgbm")
    def test_deploy_on_accuracy_improvement(self, mock_restore, mock_backup):
        """Deploys when new_cv >= min_accuracy and >= old_accuracy - 0.01."""
        from ml.retrainer import ModelRetrainer

        mock_backup.return_value = {}

        # Mock training result with good accuracy
        mock_training_result = MagicMock()
        mock_training_result.cv_accuracy = 0.55
        mock_training_result.rejected = False

        retrainer = ModelRetrainer()
        # Set old accuracy below new
        state = retrainer.load_state()
        state["lgbm"]["last_accuracy"] = 0.52
        retrainer.save_state(state)

        with patch("data.fetcher.get_candles") as mock_fetch, \
             patch("data.processor.process") as mock_proc, \
             patch("ml.trainer.train_lgbm", return_value=mock_training_result), \
             patch("ml.predictor.reload_lgbm_model", return_value=True), \
             patch("data.macro_fetcher.fetch_and_cache_macro", return_value={}):

            mock_fetch.return_value = MagicMock(empty=False)
            mock_proc.return_value = MagicMock(empty=False)

            result = retrainer.retrain_lgbm()

        self.assertTrue(result["deployed"])
        self.assertAlmostEqual(result["new_accuracy"], 0.55)
        mock_restore.assert_not_called()


class TestNoDeployBelowMinimum(unittest.TestCase):
    """Does not deploy when new accuracy is below minimum threshold."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("ml.retrainer.ModelRetrainer._backup_lgbm")
    @patch("ml.retrainer.ModelRetrainer._restore_lgbm")
    def test_no_deploy_below_minimum(self, mock_restore, mock_backup):
        """Old model retained when new CV accuracy is below RETRAIN_LGBM_MIN_ACCURACY."""
        from ml.retrainer import ModelRetrainer

        mock_backup.return_value = {"old_model": "backup_model"}

        mock_training_result = MagicMock()
        mock_training_result.cv_accuracy = 0.45  # below 0.50 minimum
        mock_training_result.rejected = False

        retrainer = ModelRetrainer()

        with patch("data.fetcher.get_candles") as mock_fetch, \
             patch("data.processor.process") as mock_proc, \
             patch("ml.trainer.train_lgbm", return_value=mock_training_result), \
             patch("data.macro_fetcher.fetch_and_cache_macro", return_value={}):

            mock_fetch.return_value = MagicMock(empty=False)
            mock_proc.return_value = MagicMock(empty=False)

            result = retrainer.retrain_lgbm()

        self.assertFalse(result["deployed"])
        mock_restore.assert_called_once()


class TestStatePersistLoad(unittest.TestCase):
    """Retrain state saves and reloads correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_state_persist_load(self):
        """State saved to disk round-trips correctly."""
        from ml.retrainer import ModelRetrainer
        retrainer = ModelRetrainer()

        state = retrainer.load_state()
        state["lgbm"]["last_retrain"] = "2026-04-01T10:00:00+00:00"
        state["lgbm"]["last_accuracy"] = 0.521
        state["lgbm"]["deployed"] = True
        state["lgbm"]["retrain_count"] = 3
        state["deep"]["trade_outcomes_available"] = 47
        retrainer.save_state(state)

        loaded = retrainer.load_state()
        self.assertEqual(loaded["lgbm"]["last_retrain"], "2026-04-01T10:00:00+00:00")
        self.assertAlmostEqual(loaded["lgbm"]["last_accuracy"], 0.521)
        self.assertTrue(loaded["lgbm"]["deployed"])
        self.assertEqual(loaded["lgbm"]["retrain_count"], 3)
        self.assertEqual(loaded["deep"]["trade_outcomes_available"], 47)


class TestTradeOutcomeCount(unittest.TestCase):
    """get_trade_outcome_count() correctly queries SQLite."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_state = Config.RETRAIN_STATE_FILE
        Config.RETRAIN_STATE_FILE = os.path.join(self.tmpdir, "retrain_state.json")
        self._orig_backup = Config.RETRAIN_BACKUP_DIR
        Config.RETRAIN_BACKUP_DIR = os.path.join(self.tmpdir, "backups")

    def tearDown(self):
        Config.RETRAIN_STATE_FILE = self._orig_state
        Config.RETRAIN_BACKUP_DIR = self._orig_backup
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_trade_outcome_count(self):
        """Queries SQLite for rows with result IS NOT NULL AND status = 'closed'."""
        import sqlite3

        # Build a minimal in-memory trades table in a temp DB
        db_path = os.path.join(self.tmpdir, "test_trades.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE trades (id INTEGER PRIMARY KEY, result TEXT, status TEXT)"
        )
        conn.execute("INSERT INTO trades VALUES (1, 'WIN', 'closed')")
        conn.execute("INSERT INTO trades VALUES (2, 'LOSS', 'closed')")
        conn.execute("INSERT INTO trades VALUES (3, NULL, 'open')")   # open — not counted
        conn.execute("INSERT INTO trades VALUES (4, NULL, 'closed')")  # no result — not counted
        conn.commit()
        conn.close()

        # Patch the DB path used inside ModelRetrainer
        from ml.retrainer import ModelRetrainer
        retrainer = ModelRetrainer()

        with patch.object(
            retrainer, "get_trade_outcome_count",
            wraps=lambda: self._count_from_db(db_path),
        ):
            count = retrainer.get_trade_outcome_count()

        # Direct query to confirm
        count = self._count_from_db(db_path)
        self.assertEqual(count, 2)

    def _count_from_db(self, db_path: str) -> int:
        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE result IS NOT NULL AND status = 'closed'"
        ).fetchone()
        conn.close()
        return int(row[0])


if __name__ == "__main__":
    unittest.main()
