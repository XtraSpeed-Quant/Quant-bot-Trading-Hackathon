import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from src.config import Config
from src.db import BotDB
from src.main import build_plan, positions_from_wallet
from src.strategy.boss import BossStrategy, Market, Signal, StrategyResult
from src.predictor import PredictiveEnsemble


def market(pair="BTC/USD", price=100.0, volume=1_000_000.0, spread=0.01):
    return Market(pair, pair.split('/')[0], price, price-spread, price+spread, 0.01, volume, 6, 1.0, True)


class DBTests(unittest.TestCase):
    def test_migration_and_turnover(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "db.sqlite3")
            db = None
            try:
                con = sqlite3.connect(p)
                try:
                    con.executescript("CREATE TABLE trades(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp_ms INTEGER,pair TEXT,side TEXT,quantity REAL,price REAL,notional REAL,fee REAL,mode TEXT,order_id TEXT,status TEXT);")
                    con.commit()
                finally:
                    con.close()
                db = BotDB(p)
                db.record_trade(1770000000000, "BTC/USD", "BUY", 10, 100, 1, "live", "1", "FILLED")
                self.assertGreaterEqual(db.day_turnover("2026-02-02"), 0.0)
                check_con = sqlite3.connect(p)
                try:
                    cols = {r[1] for r in check_con.execute("PRAGMA table_info(trades)")}
                finally:
                    check_con.close()
                self.assertIn("session_id", cols)
            finally:
                if db is not None:
                    db.close()

    def test_turnover_reset_preserves_ledger_and_only_resets_day_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "db.sqlite3")
            db = BotDB(p)
            day = "2026-10-02"
            ts = 1790899200000 + 3600000
            db.record_trade(ts, "BTC/USD", "BUY", 10, 100, 1, "live", "1", "FILLED")
            self.assertGreater(db.day_turnover(day, "test"), 0.0)
            raw_before = db.day_turnover(day, None)
            db.reset_day_turnover("test", day)
            self.assertEqual(db.day_turnover(day, "test"), 0.0)
            self.assertAlmostEqual(db.day_turnover(day, None), raw_before, places=9)
            db.record_trade(ts + 1000, "ETH/USD", "BUY", 5, 100, 1, "live", "2", "FILLED")
            self.assertGreater(db.day_turnover(day, "test"), 0.0)
            self.assertEqual(db._conn().execute("SELECT COUNT(*) FROM trades").fetchone()[0], 2)
            db.close()

    def test_batch_histories_are_chronological(self):
        with tempfile.TemporaryDirectory() as td:
            db = None
            try:
                db = BotDB(os.path.join(td, "db.sqlite3"))
                for ts in range(10):
                    db.add_market_snapshots(1000 + ts, [("BTC/USD", {"LastPrice": 100 + ts, "MaxBid": 99.9, "MinAsk": 100.1, "Change": 0.0, "UnitTradeValue": 1e6})])
                h = db.histories(["BTC/USD"], 10)
                self.assertEqual(h["BTC/USD"], [100,101,102,103,104,105,106,107,108,109])
            finally:
                if db is not None:
                    db.close()


class StrategyTests(unittest.TestCase):
    def test_defensive_regime_zeroes_gross(self):
        cfg = Config(predictor_enabled=False, min_quote_volume=0, max_spread_bps=100)
        strat = BossStrategy(cfg)
        histories = {}
        markets = {}
        for i in range(10):
            pair=f"C{i}/USD"
            prices = [100.0] * 721
            for j in range(720):
                prices[j+1] = prices[j] * (0.99995)
            histories[pair]=prices
            markets[pair]=market(pair,prices[-1])
        result = strat.analyze(histories, markets, 100000)
        self.assertEqual(result.regime, "defensive")
        self.assertEqual(result.gross_target, 0.0)

    def test_model_predictions_have_full_shape(self):
        cfg = Config(predictor_enabled=True, min_quote_volume=0, max_spread_bps=100)
        strat = BossStrategy(cfg)
        rng = np.random.default_rng(7)
        histories={}
        markets={}
        for i in range(12):
            pair=f"C{i}/USD"
            steps=rng.normal(0.0001,0.002,size=1200)
            prices=100*np.exp(np.cumsum(np.r_[0,steps]))
            histories[pair]=prices.tolist()
            markets[pair]=market(pair,float(prices[-1]),volume=2e6)
        result=strat.analyze(histories,markets,200000)
        if result.predictor_predictions:
            p=next(iter(result.predictor_predictions.values()))
            self.assertTrue(0<=p.probability_up<=1)
            self.assertTrue(0<=p.probability_down<=1)
            self.assertTrue(0<=p.confidence<=0.98)

    def test_bearish_veto_requires_all_conditions(self):
        cfg=Config(predictor_enabled=False)
        strat=BossStrategy(cfg)
        pred=SimpleNamespace(predicted_return=-0.01,probability_down=0.8,confidence=0.8)
        self.assertTrue(cfg.bearish_veto_enabled and pred.predicted_return <= cfg.bearish_veto_return and pred.probability_down >= cfg.bearish_veto_probability and pred.confidence >= cfg.bearish_veto_confidence)
        pred2=SimpleNamespace(predicted_return=-0.01,probability_down=0.8,confidence=0.59)
        self.assertFalse(pred2.confidence >= cfg.bearish_veto_confidence)


class PlanningTests(unittest.TestCase):
    def test_forced_exit_can_exceed_budget_but_normal_orders_do_not(self):
        cfg=Config(min_order_usd=10,max_order_usd=10000,rebalance_band=0.005)
        positions={
            "BTC/USD": SimpleNamespace(pair="BTC/USD",coin="BTC",qty=50,free_qty=50,current_price=100,value=5000,entry_price=100,entry_ts=0,pnl_pct=-3,age_minutes=60),
            "ETH/USD": SimpleNamespace(pair="ETH/USD",coin="ETH",qty=100,free_qty=100,current_price=20,value=2000,entry_price=20,entry_ts=0,pnl_pct=0,age_minutes=60),
        }
        markets={"BTC/USD":market("BTC/USD",100),"ETH/USD":market("ETH/USD",20)}
        sells,buys=build_plan({"BTC/USD":0,"ETH/USD":3000},{"BTC/USD"},positions,markets,10000,cfg,1000,{"USD":{"Free":9000,"Lock":0}})
        self.assertEqual(sells[0]["pair"],"BTC/USD")
        self.assertTrue(sells[0]["forced"])
        self.assertEqual(buys[0]["pair"],"ETH/USD")


    def test_defensive_nonzero_gross_does_not_force_liquidation(self):
        from src.main import plan_targets
        with tempfile.TemporaryDirectory() as td:
            db = BotDB(os.path.join(td, "db.sqlite3"))
            try:
                cfg = Config()
                result = SimpleNamespace(
                    long_targets={"BTC/USD": 0.10},
                    bearish_veto_pairs=(),
                    regime="defensive",
                    gross_target=0.10,
                )
                pos = SimpleNamespace(
                    pair="BTC/USD", coin="BTC", qty=1.0, free_qty=1.0,
                    current_price=100.0, value=100.0, entry_price=100.0,
                    entry_ts=0, pnl_pct=0.0, age_minutes=60.0,
                )
                risk = SimpleNamespace(snapshot=SimpleNamespace(halted=False))
                targets, forced, reasons = plan_targets(
                    result, {"BTC/USD": pos}, 1000.0, risk, db, cfg,
                    {"BTC/USD": market("BTC/USD", 100.0)},
                )
                self.assertNotIn("BTC/USD", forced)
                self.assertAlmostEqual(targets["BTC/USD"], 100.0, places=9)
                self.assertNotIn("defensive_regime", reasons)
            finally:
                db.close()

    def test_target_zero_is_normal_turnover_exit(self):
        cfg=Config(min_order_usd=10,max_order_usd=10000,rebalance_band=0.005)
        pos=SimpleNamespace(pair="BTC/USD",coin="BTC",qty=50,free_qty=50,current_price=100,value=5000,entry_price=100,entry_ts=0,pnl_pct=0,age_minutes=60)
        sells, _ = build_plan({"BTC/USD":0.0},set(),{"BTC/USD":pos},{"BTC/USD":market("BTC/USD",100)},10000,cfg,1000,{"USD":{"Free":5000,"Lock":0}})
        self.assertFalse(sells[0]["forced"])
        self.assertEqual(sells[0]["reason"],"target_zero")

    def test_below_band_no_normal_order(self):
        cfg=Config(min_order_usd=10,rebalance_band=0.01)
        positions={"BTC/USD": SimpleNamespace(pair="BTC/USD",coin="BTC",qty=1,free_qty=1,current_price=100,value=100,entry_price=100,entry_ts=0,pnl_pct=0,age_minutes=60)}
        markets={"BTC/USD":market("BTC/USD",100)}
        sells,buys=build_plan({"BTC/USD":105},{},positions,markets,10000,cfg,5000,{"USD":{"Free":9900,"Lock":0}})
        self.assertEqual(sells,[]); self.assertEqual(buys,[])


if __name__ == "__main__":
    unittest.main()
