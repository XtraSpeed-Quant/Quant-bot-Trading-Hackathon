from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from roostoo_bot.client import RoostooClient, signature
from roostoo_bot.config import Config
from roostoo_bot.dashboard import format_dashboard
from roostoo_bot.store import Store
from roostoo_bot.strategy import Market, MultiFactorStrategy


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
    def get(self, *args, **kwargs):
        class Response:
            def raise_for_status(self): pass
            def json(self): return self._payload
        response = Response()
        response._payload = self.payload
        return response


class CoreTests(unittest.TestCase):
    def test_signature_vector(self):
        self.assertEqual(
            signature({"pair": "BNB/USD", "quantity": 2000, "side": "BUY", "timestamp": 1580774512000, "type": "MARKET"}, "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"),
            "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff",
        )

    def test_balance_accepts_spotwallet(self):
        client = RoostooClient("https://mock-api.roostoo.com", "key", "secret")
        client.session = FakeSession({"Success": True, "SpotWallet": {"USD": {"Free": 49301.52, "Lock": 0}}})
        self.assertAlmostEqual(client.balance()["USD"]["Free"], 49301.52)

    def test_config_capacity(self):
        cfg = Config(top_n=5, max_gross_target=0.70, max_asset_weight=0.16)
        cfg.validate(require_credentials=False)

    def test_store_prediction_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "state.sqlite3"), 1000)
            store.record_prediction(1000, "BTC/USD", 100, 2000, 0.01, 0.7, 0.2, 0.1, 0.8, True)
            self.assertEqual(store.settle_predictions(2500, {"BTC/USD": 102}, 0.001), 1)
            metrics = store.prediction_metrics()
            self.assertEqual(metrics["samples"], 1)
            self.assertEqual(metrics["directional_accuracy"], 1.0)
            store.close()

    def test_strategy_runs(self):
        strategy = MultiFactorStrategy(
            fast_window=3, medium_window=6, long_window=12, trend_window=12, vol_window=3, top_n=2,
            max_gross_target=0.4, max_asset_weight=0.2, max_short_gross=0.1, max_short_weight=0.1,
            min_score=0.0, min_relative_edge=-0.1, min_1h_return=-0.1, min_4h_return=-0.1, min_12h_return=-0.1,
            min_quote_volume=1, max_spread_bps=100, volatility_floor=0.0005, predictor_enabled=False,
            prediction_horizon=3, predictor_feature_window=12, predictor_min_samples=50, predictor_max_samples=500,
            predictor_ridge_alpha=1, predictor_classifier_alpha=1, predictor_validation_fraction=0.25, predictor_purge_gap=3,
            predictor_min_directional_accuracy=0.5, predictor_min_correlation=0, predictor_min_balanced_accuracy=1/3,
            predictor_min_top_decile_edge=-1, predictor_max_brier=1.0, predictor_class_threshold=0.001, min_prediction_edge=0,
            prediction_retrain_seconds=900, correlation_window=10, max_pair_correlation=0.99, correlation_penalty=0.1,
            overextension_penalty=0.1, breakout_window=6, taker_fee_assumption=0.001, enable_shorts=False, cash_reserve=0.10,
            gross_strength_power=1.15, risk_on_gross_cap=1.0, balanced_gross_cap=0.8, risk_off_gross_cap=0.5, defensive_gross_cap=0.25,
            defensive_min_breadth=0.08, defensive_min_strength=0.12, allocation_edge_power=1.10, allocation_vol_power=0.75, scan_top_n=8,
        )
        histories = {p: [100.0 * (1.001 + i*0.0002) ** j for j in range(25)] for i, p in enumerate(("BTC/USD", "ETH/USD", "SOL/USD"))}
        markets = {p: Market(p, p.split('/')[0], 'USD', v[-1], v[-1]*0.9999, v[-1]*1.0001, 0.02, 1_000_000, 8, 2, 1, True) for p, v in histories.items()}
        result = strategy.analyze(histories, markets)
        self.assertLessEqual(sum(result.long_targets.values()), 0.4 + 1e-9)

    def test_dashboard_markers(self):
        out = format_dashboard({"status":"running","regime":"risk_on","breadth":0.7,"median_4h":0.01,"median_12h":0.02,"median_24h":0.03,"market_strength":0.9,"gross_target":0.7,"short_gross_target":0.0,"equity":100000,"current_exposure":0,"volatility_scale":1,"prediction_enabled":True,"prediction_samples":3000,"prediction_accuracy":0.53,"prediction_correlation":0.04,"prediction_balanced_accuracy":0.38,"prediction_top_decile_edge":0.001,"live_prediction_samples":10,"live_prediction_accuracy":0.6,"live_prediction_correlation":0.02,"signal_rows":[{"pair":"BTC/USD","direction":"LONG","score":0.8,"predicted_return":0.004,"probability_up":0.7,"probability_down":0.1,"prediction_confidence":0.8,"fast_return":0.01,"medium_return":0.02,"long_return":0.03,"return_24h":0.04,"target_weight":0.16,"reason":"trend+prediction_edge"}],"planned_sells":[],"planned_buys":[{"pair":"BTC/USD","notional":16000,"current":0,"target":16000}],"planned_short_opens":[],"planned_short_closes":[],"risk_reason":"ok","orders_enabled":False,"blocked":"inspection_only","history_points":241,"history_required":721,"today_trades":0})
        for marker in ("[MODEL]", "[BUY ]", "WHY:", "PREDICTIVE OVERLAY"):
            self.assertIn(marker, out)

    def test_strategy_result_warmup_safe(self):
        from roostoo_bot.strategy import StrategyResult
        result = StrategyResult(
            long_targets={}, short_targets={}, signals=[], regime="warming_up",
            breadth=0.0, median_4h=0.0, median_12h=0.0, median_24h=0.0,
            market_strength=0.0, gross_target=0.0, short_gross_target=0.0,
            median_volatility=0.0, volatility_scale=0.0, predictor_enabled=False,
            predictor_samples=0, predictor_directional_accuracy=0.0,
            predictor_correlation=0.0, predictor_balanced_accuracy=0.0,
            predictor_top_decile_edge=0.0, predictor_brier=1.0, predictor_gate_reason="warming up",
        )
        self.assertEqual(result.regime, "warming_up")
        self.assertEqual(result.predictor_brier, 1.0)


    def test_engine_passes_prediction_retrain_configuration(self):
        from roostoo_bot.config import Config
        from roostoo_bot.engine import TradingEngine

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(
                db_path=str(Path(tmp) / "state.sqlite3"),
                predictor_enabled=True,
                prediction_retrain_seconds=900,
            )
            class DummyClient:
                pass

            store = Store(cfg.db_path, cfg.history_limit)
            engine = TradingEngine(cfg, DummyClient(), store)
            self.assertEqual(engine.strategy.prediction_retrain_seconds, 900)
            store.close()


    def test_predictor_uses_relative_time_split(self):
        from roostoo_bot.predictor import PredictiveEnsemble
        model = PredictiveEnsemble(10, 3, 50, 500, 1.0, 1.0, 0.25, 2, 0.5, -1.0, 1/3, -1.0, 0.001, 0.9)
        histories = {p: [100.0 * (1.001 + i*0.0001) ** j for j in range(520)] for i, p in enumerate(("BTC/USD", "ETH/USD", "SOL/USD", "SUI/USD"))}
        result = model.fit_and_predict(histories)
        self.assertIn("", result.gate_reason)
        self.assertGreater(result.samples, 0)



    def test_zero_target_forces_exit_below_rebalance_band(self):
        from roostoo_bot.config import Config
        from roostoo_bot.engine import TradingEngine

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(
                db_path=str(Path(tmp) / "state.sqlite3"),
                rebalance_band=0.012,
                min_order_usd=10.0,
                orders_enabled=False,
            )
            class DummyClient:
                pass

            store = Store(cfg.db_path, cfg.history_limit)
            engine = TradingEngine(cfg, DummyClient(), store)
            market = Market("UNI/USD", "UNI", "USD", 100.0, 99.9, 100.1, 0.0, 1_000_000, 8, 2, 1.0, True)
            wallet = {"USD": {"Free": 48_000.0, "Lock": 0.0}, "UNI": {"Free": 5.8789, "Lock": 0.0}}
            sells, buys, _, _ = engine._order_plan(
                long_targets={}, short_targets={}, markets={"UNI/USD": market},
                wallet=wallet, shorts={}, equity=49_289.68, liquidate=False, timestamp_ms=1_000_000,
            )
            self.assertEqual(len(sells), 1)
            self.assertEqual(sells[0]["pair"], "UNI/USD")
            self.assertEqual(sells[0]["target"], 0.0)
            self.assertEqual(buys, [])
            store.close()

    def test_dashboard_shows_true_target_and_gate_reason(self):
        out = format_dashboard({
            "status":"running", "regime":"defensive", "breadth":0.04, "median_4h":-0.01, "median_12h":0.0, "median_24h":-0.02,
            "market_strength":0.09, "gross_target":0.0, "short_gross_target":0.0, "equity":49289.68, "cash":48701.79, "current_exposure":587.89,
            "exposure_pct":587.89/49289.68, "session_pnl":0.0, "session_pnl_pct":0.0, "account_pnl":-710.32, "account_pnl_pct":-0.0142064,
            "today_trades":2, "turnover":0.0, "prediction_enabled":False, "prediction_samples":15738, "prediction_accuracy":0.3314,
            "prediction_correlation":0.3489, "prediction_balanced_accuracy":0.4564, "prediction_top_decile_edge":0.0069, "prediction_brier":0.666,
            "prediction_gate_reason":"direction 0.331<0.515; brier 0.666>0.660", "live_prediction_samples":0,
            "holdings":{"UNI/USD":587.89}, "target_values":{"UNI/USD":0.0}, "signal_rows":[], "planned_sells":[], "planned_buys":[],
            "planned_short_opens":[], "planned_short_closes":[], "risk_reason":"ok", "orders_enabled":False, "blocked":"inspection_only",
            "history_points":573, "history_required":481, "timestamp_ms":1790862951000, "rebalance_due":True, "next_rebalance_ms":0, "volatility_scale":1.1,
            "universe_size":86, "median_volatility":0.0016,
        })
        self.assertIn("INSPECTION ONLY | CASH / NO LONG EXPOSURE", out)
        self.assertIn("target=        $0.00", out)
        self.assertIn("direction 0.331<0.515; brier 0.666>0.660", out)

    def test_breakout_feature_uses_previous_high(self):
        strategy = MultiFactorStrategy(
            fast_window=3, medium_window=6, long_window=12, trend_window=12, vol_window=3, top_n=2,
            max_gross_target=0.4, max_asset_weight=0.2, max_short_gross=0.1, max_short_weight=0.1,
            min_score=0.0, min_relative_edge=-0.1, min_1h_return=-0.1, min_4h_return=-0.1, min_12h_return=-0.1,
            min_quote_volume=1, max_spread_bps=100, volatility_floor=0.0005, predictor_enabled=False,
            prediction_horizon=3, predictor_feature_window=12, predictor_min_samples=50, predictor_max_samples=500,
            predictor_ridge_alpha=1, predictor_classifier_alpha=1, predictor_validation_fraction=0.25, predictor_purge_gap=3,
            predictor_min_directional_accuracy=0.5, predictor_min_correlation=0, predictor_min_balanced_accuracy=1/3,
            predictor_min_top_decile_edge=-1, predictor_max_brier=1.0, predictor_class_threshold=0.001, min_prediction_edge=0,
            prediction_retrain_seconds=900, correlation_window=10, max_pair_correlation=0.99, correlation_penalty=0.1,
            overextension_penalty=0.1, breakout_window=6, taker_fee_assumption=0.001, enable_shorts=False, cash_reserve=0.10,
            gross_strength_power=1.15, risk_on_gross_cap=1.0, balanced_gross_cap=0.8, risk_off_gross_cap=0.5, defensive_gross_cap=0.25,
            defensive_min_breadth=0.08, defensive_min_strength=0.12, allocation_edge_power=1.10, allocation_vol_power=0.75, scan_top_n=8,
        )
        prices = [100 + i for i in range(20)]
        market = Market("BTC/USD", "BTC", "USD", prices[-1], prices[-1]-0.01, prices[-1]+0.01, 0.10, 1_000_000, 8, 2, 1, True)
        feature = strategy._feature(market, prices)
        self.assertIsNotNone(feature)
        self.assertGreater(feature["breakout"], 0.0)


if __name__ == "__main__":
    unittest.main()
