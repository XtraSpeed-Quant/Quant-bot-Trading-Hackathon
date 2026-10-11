from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.config import Config
from src.db import BotDB
from src.logger.trade_logger import TradeLogger
from src.main import (
    build_markets, execute_order, execute_short_close, execute_short_open,
    fetch_short_positions, portfolio_value, shorts_from_response,
    plan_short_targets, build_short_plan,
)
from src.paper_trader import PaperTrader
from src.predictor import PredictiveEnsemble, PredictorResult
from src.risk.manager import RiskManager
from src.strategy.boss import BossStrategy, Market


def market(pair: str, price: float, volume: float = 2_000_000.0, spread: float = 0.01) -> Market:
    return Market(pair, pair.split("/")[0], price, price - spread, price + spread,
                  -0.01, volume, 6, 1.0, True)


class TimeScalingTests(unittest.TestCase):
    def test_minute_windows_scale_with_poll_cadence(self):
        one_min = Config(poll_seconds=60)
        five_min = Config(poll_seconds=300, rebalance_seconds=300)
        self.assertEqual(one_min.bars(60), 60)
        self.assertEqual(five_min.bars(60), 12)
        self.assertEqual(five_min.bars(240), 48)
        self.assertEqual(five_min.bars(720), 144)
        self.assertEqual(five_min.bars(360), 72)

    def test_predictor_training_windows_scale_with_poll_cadence(self):
        model = PredictiveEnsemble(
            feature_window=360, prediction_horizon=60, min_samples=200,
            max_training_samples=1000, ridge_alpha=1.0, classifier_alpha=1.0,
            validation_fraction=0.25, purge_gap=60,
            min_directional_accuracy=0.51, min_correlation=0.0,
            min_balanced_accuracy=1 / 3, min_top_decile_edge=-1.0,
            class_threshold=0.001, fee_assumption=0.001, poll_seconds=300,
        )
        self.assertEqual(model.feature_window_bars, 72)
        self.assertEqual(model.prediction_horizon_bars, 12)
        self.assertEqual(model.purge_gap_bars, 12)

    def test_zero_turnover_cap_is_valid_unlimited_mode(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(
                paper_mode=True, max_daily_turnover=0.0,
                db_path=os.path.join(td, "db.sqlite3"),
                log_file=os.path.join(td, "bot.log"),
            )
            cfg.validate(require_credentials=False, require_api_url=False)


class ShortPaperExecutionTests(unittest.TestCase):
    def test_short_open_mark_and_close_accounting(self):
        trader = PaperTrader(10_000.0)
        opened = trader.short_open("BTC/USD", 1_000.0, price=100.0)
        self.assertTrue(opened["Success"])
        self.assertAlmostEqual(opened["ShortQty"], 10.0, places=6)
        self.assertAlmostEqual(trader.usd_balance, 8_999.0, places=4)

        marked = trader.short_positions({"BTC/USD": {"MinAsk": 90.0}})
        self.assertTrue(marked["Success"])
        self.assertAlmostEqual(marked["Positions"][0]["UnrealizedPNL"], 100.0, places=4)
        self.assertAlmostEqual(marked["Positions"][0]["PositionValue"], 1_100.0, places=4)

        closed = trader.short_close("BTC/USD", price=90.0)
        self.assertTrue(closed["Success"])
        self.assertTrue(closed["FullyClosed"])
        self.assertEqual(trader.short_positions({})["Positions"], [])
        self.assertAlmostEqual(trader.usd_balance, 10_098.1, places=3)

    def test_short_collateral_and_pnl_count_toward_equity(self):
        value = portfolio_value(
            {"USD": {"Free": 9_000.0, "Lock": 0.0}}, {},
            [{"Collateral": 1_000.0, "UnrealizedPNL": 120.0}],
        )
        self.assertAlmostEqual(value, 10_120.0)

    def test_short_response_schema_parses_expected_fields(self):
        markets = {"BTC/USD": market("BTC/USD", 95.0)}
        positions = shorts_from_response({"Success": True, "Positions": [{
            "Pair": "BTC/USD", "ShortQty": 2.0, "Collateral": 100.0,
            "EntryPrice": 50.0, "CurrentPrice": 45.0, "UnrealizedPNL": 10.0,
            "PositionValue": 110.0, "CreateTimestamp": 1_000,
        }]}, markets, current_ms=61_000)
        self.assertAlmostEqual(positions["BTC/USD"].pnl_pct, 10.0)
        self.assertAlmostEqual(positions["BTC/USD"].position_value, 110.0)
        self.assertAlmostEqual(positions["BTC/USD"].age_minutes, 1.0)

    def test_paper_spot_execution_uses_local_simulator_and_logs_fill(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            trader = PaperTrader(2_000.0)
            cfg = Config(
                paper_mode=True, orders_enabled=False, bot_namespace="test",
                db_path=os.path.join(td, "db.sqlite3"), log_file=os.path.join(td, "bot.log"),
            )
            log = TradeLogger(td)
            ok, response = execute_order(trader, db, log, "BTC/USD", "BUY", 1.0,
                                         cfg, "unit-test", price=100.0)
            self.assertTrue(ok)
            self.assertTrue(response["Success"])
            self.assertAlmostEqual(trader.get_coin_balance("BTC"), 1.0)
            self.assertEqual(db._conn().execute("SELECT COUNT(*) FROM trades").fetchone()[0], 1)
            self.assertEqual(db._conn().execute("SELECT side FROM trades").fetchone()[0], "BUY")
            db.close()

    def test_paper_mode_refuses_nonlocal_execution_client(self):
        class FakeExchange:
            called = False
            def place_order(self, *args, **kwargs):
                self.called = True
                return {"Success": True}
            def short_open(self, *args, **kwargs):
                self.called = True
                return {"Success": True}
            def short_close(self, *args, **kwargs):
                self.called = True
                return {"Success": True}

        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            cfg = Config(paper_mode=True, orders_enabled=False, bot_namespace="paper-isolation",
                         db_path=os.path.join(td, "db.sqlite3"), log_file=os.path.join(td, "bot.log"))
            logger = TradeLogger(td)
            exchange = FakeExchange()
            m = market("BTC/USD", 100.0)
            ok, response = execute_order(exchange, db, logger, "BTC/USD", "BUY", 1.0, cfg, "guard-test", price=100.0)
            self.assertFalse(ok)
            self.assertEqual(response["ErrMsg"], "paper_mode_requires_local_simulator")
            ok, response = execute_short_open(exchange, db, logger, cfg, "BTC/USD", 500.0, m, None, {})
            self.assertFalse(ok)
            self.assertEqual(response["ErrMsg"], "paper_mode_requires_local_simulator")
            self.assertFalse(exchange.called)
            db.close()

    def test_live_short_close_never_sends_unsupported_price_argument(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            cfg = Config(
                paper_mode=False, orders_enabled=True, bot_namespace="live-short-close-contract",
                short_fee_assumption=0.001, db_path=os.path.join(td, "db.sqlite3"),
                log_file=os.path.join(td, "bot.log"),
            )
            logger = TradeLogger(td)
            pos = type("ShortPos", (), {"pair": "BTC/USD", "qty": 1.0, "collateral": 100.0,
                                        "entry_price": 100.0, "current_price": 90.0,
                                        "unrealized_pnl": 10.0, "position_value": 110.0,
                                        "entry_ts": 1, "pnl_pct": 10.0, "age_minutes": 60.0})()
            m = market("BTC/USD", 90.0)
            class API:
                calls = []
                def short_close(self, pair, close_qty=None, close_pct=None):
                    self.calls.append((pair, close_qty, close_pct))
                    return {"Success": True, "ClosePrice": 90.0, "RealizedPNL": 10.0,
                            "CloseFee": 0.09, "ReturnAmount": 109.91, "ClosedQty": 1.0,
                            "FullyClosed": True}
                def short_positions(self):
                    return {"Success": True, "Positions": []}
            api = API()
            ok, result = execute_short_close(api, db, logger, cfg, pos, m, 50.0,
                                             {"BTC/USD": {"MinAsk": 90.0}})
            self.assertTrue(ok, msg=result)
            self.assertEqual(api.calls, [("BTC/USD", None, 50.0)])
            db.close()

    def test_short_open_and_close_execution_paths_normalize_and_log_fills(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            trader = PaperTrader(5_000.0)
            cfg = Config(
                paper_mode=True, orders_enabled=False, bot_namespace="testshort",
                db_path=os.path.join(td, "db.sqlite3"), log_file=os.path.join(td, "bot.log"),
            )
            logger = TradeLogger(td)
            m = market("BTC/USD", 100.0)
            tickers = {"BTC/USD": {"LastPrice": 100.0, "MaxBid": 99.99, "MinAsk": 100.01}}
            ok, opened = execute_short_open(trader, db, logger, cfg, "BTC/USD", 500.0, m, None, tickers)
            self.assertTrue(ok, msg=opened)
            shorts = fetch_short_positions(trader, True, tickers, {"BTC/USD": m})
            self.assertIn("BTC/USD", shorts)
            ok, closed = execute_short_close(trader, db, logger, cfg, shorts["BTC/USD"], m, 100.0, tickers)
            self.assertTrue(ok, msg=closed)
            sides = [row[0] for row in db._conn().execute("SELECT side FROM trades ORDER BY id").fetchall()]
            self.assertEqual(sides, ["SHORT_OPEN", "SHORT_CLOSE"])
            db.close()

    def test_short_fills_are_in_turnover_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            ts = 1_800_000_000_000
            db.record_trade(ts, "BTC/USD", "SHORT_OPEN", 2.0, 100.0, 0.2, "paper", "1", "FILLED")
            db.record_trade(ts + 1000, "BTC/USD", "SHORT_CLOSE", 2.0, 90.0, 0.18, "paper", "2", "FILLED")
            # The current day query uses UTC midnight; verify aggregation includes both short legs.
            from datetime import datetime, timezone
            day = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d")
            self.assertAlmostEqual(db.day_turnover(day), 380.0)
            db.close()


class ShortStrategyTests(unittest.TestCase):
    def test_bearish_cross_section_can_allocate_short_targets(self):
        cfg = Config(
            predictor_enabled=False, poll_seconds=60, min_quote_volume=0,
            max_spread_bps=100, min_short_score=0.40, max_total_gross_target=0.80,
        )
        strategy = BossStrategy(cfg, shorts_available=True)
        rng = np.random.default_rng(44)
        histories = {}
        markets = {}
        # Stronger and weaker downtrends form a meaningful cross-section instead of a tied market.
        for i in range(20):
            pair = f"COIN{i}/USD"
            step = -(0.0002 + i * 0.00011) if i < 16 else (0.00005 + i * 0.00002)
            noise = rng.normal(0.0, 0.0001, size=800)
            prices = 100.0 * np.exp(np.cumsum(np.r_[0.0, step + noise]))
            histories[pair] = prices.tolist()
            markets[pair] = market(pair, float(prices[-1]))
        result = strategy.analyze(histories, markets, 2_000_000)
        self.assertTrue(result.short_targets, msg=result.diagnostics)
        self.assertLessEqual(sum(result.long_targets.values()) + sum(result.short_targets.values()), cfg.max_total_gross_target + 1e-8)
        self.assertFalse(set(result.long_targets) & set(result.short_targets))
        self.assertLessEqual(max(result.short_targets.values()), cfg.max_short_asset_weight + 1e-8)

    def test_short_target_zero_produces_close_plan(self):
        cfg = Config(min_order_usd=10, max_order_usd=10_000)
        short = type("P", (), {
            "pair": "BTC/USD", "qty": 10.0, "collateral": 1_000.0,
            "entry_price": 100.0, "current_price": 105.0, "unrealized_pnl": -50.0,
            "position_value": 950.0, "entry_ts": 1, "pnl_pct": -5.0, "age_minutes": 60.0,
        })()
        result = type("R", (), {"short_targets": {}, "regime": "balanced"})()
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            risk = RiskManager(10_000.0)
            risk.update(10_000.0)
            targets, forced, _ = plan_short_targets(result, {"BTC/USD": short}, 10_000.0, risk, db, cfg)
            closes, opens = build_short_plan(targets, forced, {"BTC/USD": short},
                                              {"BTC/USD": market("BTC/USD", 105.0)}, 10_000.0, cfg)
            self.assertTrue(closes)
            self.assertEqual(closes[0]["pair"], "BTC/USD")
            self.assertGreater(closes[0]["close_pct"], 0)
            self.assertEqual(opens, [])
            db.close()


class MarketUniverseTests(unittest.TestCase):
    def test_all_tradable_usd_pairs_are_retained_without_an_implicit_top_n_cap(self):
        pairs = {f"C{i}/USD": {"CanTrade": True, "AmountPrecision": 4, "MiniOrder": 1}
                 for i in range(45)}
        tickers = {p: {"LastPrice": 10.0, "MaxBid": 9.99, "MinAsk": 10.01,
                       "Change": 0, "UnitTradeValue": 2_000_000} for p in pairs}
        markets = build_markets({"TradePairs": pairs}, tickers, Config())
        self.assertEqual(len(markets), 45)


if __name__ == "__main__":
    unittest.main()

class V8OperatingModeTests(unittest.TestCase):
    def test_zero_order_cap_means_unlimited(self):
        from src.main import _order_cap
        self.assertEqual(_order_cap(25_000.0, Config(max_order_usd=0.0)), 25_000.0)
        self.assertEqual(_order_cap(25_000.0, Config(max_order_usd=4_000.0)), 4_000.0)

    def test_global_halts_are_disabled_when_thresholds_are_zero(self):
        risk = RiskManager(50_000.0, daily_loss_limit=0.0, max_drawdown=0.0)
        risk.update(40_000.0, day_start_equity=50_000.0)
        self.assertFalse(risk.snapshot.halted)
        self.assertAlmostEqual(risk.snapshot.daily_pnl_pct, -0.20)
        self.assertAlmostEqual(risk.snapshot.drawdown_pct, 0.20)

    def test_strategy_features_can_warm_up_before_full_12_hour_window(self):
        cfg = Config(poll_seconds=30, rebalance_seconds=60, predictor_enabled=False,
                     min_quote_volume=0, max_spread_bps=100)
        strat = BossStrategy(cfg)
        prices = (100.0 * np.exp(np.cumsum(np.r_[0.0, np.full(120, 0.0001)]))).tolist()
        feat = strat._features(market("BTC/USD", prices[-1]), prices)
        self.assertIsNotNone(feat)
        self.assertAlmostEqual(feat["fast"], prices[-1] / prices[-121] - 1, places=8)

    def test_predictor_training_sample_cap_spans_time_and_keeps_cross_sections(self):
        model = PredictiveEnsemble(
            feature_window=360, prediction_horizon=60, min_samples=200,
            max_training_samples=500, ridge_alpha=1.0, classifier_alpha=1.0,
            validation_fraction=0.25, purge_gap=60, min_directional_accuracy=0.51,
            min_correlation=0.0, min_balanced_accuracy=1/3, min_top_decile_edge=-1.0,
            class_threshold=0.001, fee_assumption=0.001, poll_seconds=60,
        )
        rng = np.random.default_rng(41)
        histories = {}
        for i in range(5):
            steps = rng.normal(0.00005, 0.0015, 1000)
            histories[f"C{i}/USD"] = (100 * np.exp(np.cumsum(np.r_[0.0, steps]))).tolist()
        x, y, times, classes = model._build_samples(histories)
        self.assertLessEqual(len(y), 500)
        self.assertGreater(np.unique(times).size, 80)
        self.assertLess(float(times.min()), 400)
        self.assertGreater(float(times.max()), 900)
        # Preserve all five assets at the selected timestamps, rather than keeping only
        # a tail of per-asset rows that over-represents the final few hours.
        counts = np.unique(times, return_counts=True)[1]
        self.assertTrue(np.all(counts == 5))

    def test_predictor_training_due_waits_until_signal_warmup(self):
        cfg = Config(predictor_enabled=True, poll_seconds=30, rebalance_seconds=60)
        strat = BossStrategy(cfg)
        self.assertFalse(strat.predictor_training_due(1000, {"BTC/USD": [100.0] * 100}))
        self.assertTrue(strat.predictor_training_due(1000, {"BTC/USD": [100.0] * 121}))


    def test_failed_predictor_warmup_does_not_refit_on_every_poll(self):
        model = PredictiveEnsemble(
            feature_window=360, prediction_horizon=60, min_samples=200,
            max_training_samples=1000, ridge_alpha=1.0, classifier_alpha=1.0,
            validation_fraction=0.25, purge_gap=60, min_directional_accuracy=0.51,
            min_correlation=0.0, min_balanced_accuracy=1 / 3, min_top_decile_edge=-1.0,
            class_threshold=0.001, fee_assumption=0.001, poll_seconds=30,
        )
        model._last_validation = PredictorResult(
            enabled=False, samples=0, validation_directional_accuracy=0.0,
            validation_correlation=0.0, validation_balanced_accuracy=0.0,
            validation_top_decile_edge=0.0, validation_brier=1.0, residual_std=0.0,
            predictions={}, gate_reason="insufficient samples: 0/200",
        )
        with patch.object(model, "_build_samples", side_effect=AssertionError("unexpected fit")):
            result = model.fit_and_predict({"BTC/USD": [100.0] * 1000}, refit=False)
        self.assertEqual(result.gate_reason, "insufficient samples: 0/200")
        self.assertFalse(result.enabled)
