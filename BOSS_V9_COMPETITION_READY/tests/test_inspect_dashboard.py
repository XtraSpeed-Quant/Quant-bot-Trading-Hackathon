import io
import unittest
from contextlib import redirect_stdout

from src.config import Config
from src.main import print_dashboard, now_ms
from src.db import BotDB
from src.risk.manager import RiskManager
from src.strategy.boss import BossStrategy
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import roostoo_bot.__main__ as bot_cli


class EmptyHistoryDashboardTests(unittest.TestCase):
    def test_empty_history_result_has_numeric_predictor_metrics(self):
        cfg = Config(paper_mode=True)
        result = BossStrategy(cfg, shorts_available=True).analyze({}, {}, now_ms())

        self.assertFalse(result.predictor_enabled)
        self.assertEqual(result.predictor_samples, 0)
        self.assertEqual(result.predictor_directional_accuracy, 0.0)
        self.assertEqual(result.predictor_brier, 1.0)
        self.assertEqual(result.predictor_gate_reason, "warming up")
        for name in (
            "predictor_directional_accuracy", "predictor_correlation", "predictor_rank_ic",
            "predictor_balanced_accuracy", "predictor_top_decile_edge",
            "predictor_top_decile_net_return", "predictor_brier",
        ):
            self.assertIsInstance(getattr(result, name), (int, float), name)

    def test_dashboard_prints_without_history(self):
        cfg = Config(paper_mode=True)
        result = BossStrategy(cfg, shorts_available=True).analyze({}, {}, now_ms())
        risk = RiskManager(100_000.0)
        risk.update(100_000.0)
        output = io.StringIO()
        with redirect_stdout(output):
            print_dashboard(
                cfg=cfg,
                equity=100_000.0,
                start_equity=100_000.0,
                wallet={"USD": {"Free": 100_000.0, "Lock": 0.0}},
                markets={},
                result=result,
                positions={},
                turnover_used=0.0,
                turnover_cap=float("inf"),
                next_rebalance_ms=now_ms(),
                executed=[],
                blocked=[],
                risk=risk,
                short_positions={},
            )
        text = output.getvalue()
        self.assertIn("MODEL rank=GATED direction=GATED", text)
        self.assertIn("MODEL GATE: warming up", text)
        self.assertIn("brier=1.000", text)
        self.assertIn("BOSS P&L $+0.00 (+0.00%) from $100,000.00 competition baseline", text)
        self.assertNotIn("BOSS SESSION", text)
        self.assertNotIn("COMPETITION TO DATE", text)

    def test_inspect_mode_is_not_mislabeled_as_live_execution(self):
        cfg = replace(Config(paper_mode=False, orders_enabled=True))
        result = BossStrategy(cfg, shorts_available=True).analyze({}, {}, now_ms())
        risk = RiskManager(83_524.31)
        output = io.StringIO()
        with redirect_stdout(output):
            print_dashboard(
                cfg=cfg, equity=83_524.31, start_equity=100_000.0,
                wallet={"USD": {"Free": 83_523.29, "Lock": 0.0}},
                markets={}, result=result, positions={}, turnover_used=0.0,
                turnover_cap=float("inf"), next_rebalance_ms=now_ms(),
                executed=[], blocked=[], risk=risk, short_positions={},
                execution_override="INSPECT ONLY — NO ORDERS",
            )
        text = output.getvalue()
        self.assertIn("EXECUTION INSPECT ONLY — NO ORDERS", text)
        self.assertIn("NEXT N/A (inspection only)", text)
        self.assertNotIn("EXECUTION LIVE ENABLED", text)

    def test_inspect_reports_empty_history_and_competition_pnl_baseline(self):
        with TemporaryDirectory() as tmp:
            cfg = replace(
                Config(), api_url="https://mock-api.roostoo.com", api_key="test-key",
                api_secret="test-secret", paper_mode=False, orders_enabled=True,
                expected_starting_cash=100_000.0, db_path=f"{tmp}/inspect.sqlite3",
                log_file=f"{tmp}/inspect.log", bot_namespace="inspect-test",
            )
            client = Mock()
            client.exchange_info.return_value = {"TradePairs": {
                "BTC/USD": {"CanTrade": True, "AmountPrecision": 6, "MiniOrder": 1.0}
            }}
            client.get_all_tickers.return_value = {"BTC/USD": {
                "LastPrice": 100.0, "MaxBid": 99.9, "MinAsk": 100.1,
                "Change": 0.0, "UnitTradeValue": 1_000_000.0,
            }}
            client.balance.return_value = {"Success": True, "Wallet": {
                "USD": {"Free": 83_523.29, "Lock": 0.0},
                "BTC": {"Free": 0.01, "Lock": 0.0},
            }}
            client.short_positions.return_value = {"Success": True, "Positions": []}
            out = io.StringIO()
            with patch.object(bot_cli, "Config", return_value=cfg), \
                 patch.object(bot_cli, "RoostooClient", return_value=client), \
                 redirect_stdout(out):
                status = bot_cli.inspect()
            text = out.getvalue()
            self.assertEqual(status, 0)
            self.assertIn("[INSPECT ONLY]", text)
            self.assertIn("[HISTORY] rows=0", text)
            self.assertIn("No stored market snapshots yet", text)
            self.assertIn("[HISTORY] rows=0", text)
            self.assertIn("BOSS P&L $-16,475.71 (-16.48%) from $100,000.00 competition baseline", text)
            self.assertNotIn("BOSS SESSION", text)
            self.assertNotIn("COMPETITION TO DATE", text)
            self.assertIn("GROSS EXPOSURE", text)
            self.assertIn("drawdown=+16.48%", text)
            self.assertNotIn("[ACCOUNT WARNING]", text)
            self.assertIn("EXECUTION INSPECT ONLY — NO ORDERS", text)
            client.place_order.assert_not_called()

    def test_history_summary_reports_empty_and_populated_history(self):
        with TemporaryDirectory() as tmp:
            db = BotDB(f"{tmp}/test.sqlite3", history_limit=20)
            empty = db.history_summary(["BTC/USD"])
            self.assertEqual(empty["rows"], 0)
            self.assertEqual(empty["snapshots"], 0)
            db.add_market_snapshots(1_000, [("BTC/USD", {"LastPrice": 100, "MaxBid": 99, "MinAsk": 101})])
            db.add_market_snapshots(61_000, [("BTC/USD", {"LastPrice": 102, "MaxBid": 101, "MinAsk": 103})])
            summary = db.history_summary(["BTC/USD"])
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["snapshots"], 2)
            self.assertEqual(summary["pairs"], 1)
            self.assertEqual(summary["first_timestamp_ms"], 1_000)
            self.assertEqual(summary["last_timestamp_ms"], 61_000)
            db.close()


if __name__ == "__main__":
    unittest.main()
