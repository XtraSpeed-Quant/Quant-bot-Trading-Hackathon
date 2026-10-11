import os
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

from src.config import Config
from src.db import BotDB
from src.main import _exit_thresholds, _realized_hourly_volatility, plan_targets, plan_short_targets
from src.risk.manager import RiskManager


class AdaptiveExitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(poll_seconds=60, adaptive_exits_enabled=True, target_hourly_vol=0.01,
                          commission_rate=0.001, slippage_bps=10)
        self.market = SimpleNamespace(spread_bps=20.0)

    def test_volatility_adaptive_stops_widen_in_high_volatility(self):
        rng = np.random.default_rng(17)
        low = 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0.0, 0.00005, 2400)]))
        high = 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0.0, 0.0030, 2400)]))
        low_thresholds = _exit_thresholds("LOW/USD", self.cfg, self.market, low.tolist())
        high_thresholds = _exit_thresholds("HIGH/USD", self.cfg, self.market, high.tolist())
        self.assertGreater(high_thresholds["stop"], low_thresholds["stop"])
        self.assertGreater(high_thresholds["trail"], low_thresholds["trail"])
        self.assertGreater(high_thresholds["take_profit"], low_thresholds["take_profit"])

    def test_cost_floor_keeps_stops_outside_round_trip_friction(self):
        flat_history = [100.0] * 2400
        thresholds = _exit_thresholds("FLAT/USD", self.cfg, self.market, flat_history)
        self.assertGreaterEqual(thresholds["stop"], 2.0 * thresholds["round_trip_cost"])
        self.assertGreaterEqual(thresholds["activation"], 2.0 * thresholds["round_trip_cost"])
        self.assertGreaterEqual(thresholds["trail"], thresholds["round_trip_cost"])
        self.assertGreaterEqual(thresholds["take_profit"], 3.0 * thresholds["round_trip_cost"])

    def test_static_exit_mode_preserves_legacy_values(self):
        cfg = Config(adaptive_exits_enabled=False)
        thresholds = _exit_thresholds("BTC/USD", cfg, self.market, [100.0] * 1000)
        self.assertEqual(thresholds["stop"], cfg.stop_loss_pct)
        self.assertEqual(thresholds["activation"], cfg.trailing_activation_pct)
        self.assertEqual(thresholds["trail"], cfg.trailing_stop_pct)
        self.assertEqual(thresholds["take_profit"], cfg.take_profit_pct)


    def test_new_long_entry_does_not_inherit_previous_trade_peak(self):
        cfg = Config(adaptive_exits_enabled=False, bot_namespace="adaptive-test")
        result = SimpleNamespace(long_targets={"BTC/USD": 0.1}, bearish_veto_pairs=(), gross_target=0.1)
        pos = SimpleNamespace(pair="BTC/USD", coin="BTC", qty=1.0, free_qty=1.0,
                              current_price=101.0, value=101.0, entry_price=100.0,
                              entry_ts=2_000, pnl_pct=1.0, age_minutes=1.0)
        market = SimpleNamespace(spread_bps=10.0)
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "bot.sqlite3"))
            try:
                db.set_state("adaptive-test:peak:BTC/USD", 120.0)
                risk = SimpleNamespace(snapshot=SimpleNamespace(halted=False))
                _, forced, _ = plan_targets(result, {"BTC/USD": pos}, 10_000.0, risk, db, cfg, {"BTC/USD": market})
                self.assertNotIn("BTC/USD", forced)
                self.assertEqual(db.get_state("adaptive-test:peak:BTC/USD"), 101.0)
            finally:
                db.close()

    def test_new_short_entry_does_not_inherit_previous_trade_trough(self):
        cfg = Config(adaptive_exits_enabled=False, bot_namespace="adaptive-test")
        result = SimpleNamespace(short_targets={"BTC/USD": 0.1}, regime="balanced")
        pos = SimpleNamespace(pair="BTC/USD", qty=1.0, collateral=100.0,
                              entry_price=100.0, current_price=99.0, unrealized_pnl=1.0,
                              position_value=100.0, entry_ts=2_000, pnl_pct=1.0, age_minutes=1.0)
        market = SimpleNamespace(spread_bps=10.0)
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "bot.sqlite3"))
            try:
                db.set_state("adaptive-test:short_trough:BTC/USD", 80.0)
                risk = SimpleNamespace(snapshot=SimpleNamespace(halted=False))
                _, forced, _ = plan_short_targets(result, {"BTC/USD": pos}, 10_000.0, risk, db, cfg,
                                                   {"BTC/USD": market})
                self.assertNotIn("BTC/USD", forced)
                self.assertEqual(db.get_state("adaptive-test:short_trough:BTC/USD"), 99.0)
            finally:
                db.close()

    def test_hourly_volatility_estimator_uses_configured_poll_cadence(self):
        rng = np.random.default_rng(3)
        prices = 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0, 0.001, 2500)]))
        cfg_60s = Config(poll_seconds=60)
        cfg_30s = Config(poll_seconds=30)
        vol_60s = _realized_hourly_volatility(prices.tolist(), cfg_60s)
        vol_30s = _realized_hourly_volatility(prices.tolist(), cfg_30s)
        # A history with identical per-sample returns represents different elapsed times
        # at different cadences, and the estimator must therefore differ.
        self.assertNotAlmostEqual(vol_60s, vol_30s, places=5)


if __name__ == "__main__":
    unittest.main()
