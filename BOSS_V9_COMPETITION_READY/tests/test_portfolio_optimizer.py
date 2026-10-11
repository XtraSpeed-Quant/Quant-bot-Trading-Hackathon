import unittest
from types import SimpleNamespace

import numpy as np

from src.strategy.portfolio_optimizer import optimize_signed_portfolio
from tools.walkforward_optimize import regularize_longest_segment


def sig(pair, gross_edge=0.02, one_way_cost=0.001, volatility=0.015):
    return SimpleNamespace(pair=pair, gross_expected_edge=gross_edge,
                           expected_edge=gross_edge - 2 * one_way_cost,
                           round_trip_cost=2 * one_way_cost,
                           one_way_cost=one_way_cost, volatility=volatility)


class PortfolioOptimizerTests(unittest.TestCase):
    def _histories(self, seed=7):
        rng = np.random.default_rng(seed)
        n = 800
        factor = rng.normal(0, 0.001, n)
        independent = rng.normal(0, 0.001, n)
        return {
            "A/USD": (100 * np.exp(np.cumsum(factor))).tolist(),
            "B/USD": (100 * np.exp(np.cumsum(0.95 * factor + rng.normal(0, 0.0002, n)))).tolist(),
            "C/USD": (100 * np.exp(np.cumsum(independent))).tolist(),
            "D/USD": (100 * np.exp(np.cumsum(-factor + rng.normal(0, 0.0002, n)))).tolist(),
        }

    def test_optimizer_respects_direction_asset_and_gross_constraints(self):
        hist = self._histories()
        result = optimize_signed_portfolio(
            [sig("A/USD"), sig("B/USD"), sig("C/USD")], [sig("D/USD")], hist, {},
            long_budget=0.55, short_budget=0.30, total_gross_cap=0.70,
            long_asset_cap=0.20, short_asset_cap=0.15,
            covariance_window_bars=300, risk_horizon_bars=100,
            risk_aversion=12, cost_penalty=1, iterations=300,
        )
        self.assertTrue(result.weights)
        self.assertTrue(all(result.weights.get(pair, 0) >= -1e-10 for pair in ("A/USD", "B/USD", "C/USD")))
        self.assertLessEqual(result.weights.get("D/USD", 0), 1e-10)
        self.assertLessEqual(max(result.weights.get(p, 0) for p in ("A/USD", "B/USD", "C/USD")), 0.20 + 1e-8)
        self.assertGreaterEqual(result.weights.get("D/USD", 0), -0.15 - 1e-8)
        self.assertLessEqual(sum(max(0, w) for w in result.weights.values()), 0.55 + 1e-8)
        self.assertLessEqual(sum(max(0, -w) for w in result.weights.values()), 0.30 + 1e-8)
        self.assertLessEqual(sum(abs(w) for w in result.weights.values()), 0.70 + 1e-8)

    def test_high_transaction_cost_reduces_target_changes(self):
        hist = self._histories()
        kwargs = dict(long_budget=0.60, short_budget=0.20, total_gross_cap=0.80,
                      long_asset_cap=0.20, short_asset_cap=0.20,
                      covariance_window_bars=300, risk_horizon_bars=100,
                      risk_aversion=10, iterations=400)
        signals = [sig("A/USD", 0.008, 0.0005), sig("B/USD", 0.008, 0.0005), sig("C/USD", 0.008, 0.0005)]
        previous = {"A/USD": 0.10, "B/USD": 0.10, "C/USD": 0.10}
        low_cost = optimize_signed_portfolio(signals, [], hist, previous, cost_penalty=0.0, **kwargs)
        high_cost = optimize_signed_portfolio(signals, [], hist, previous, cost_penalty=10.0, **kwargs)
        low_distance = sum(abs(low_cost.weights.get(p, 0.0) - previous[p]) for p in previous)
        high_distance = sum(abs(high_cost.weights.get(p, 0.0) - previous[p]) for p in previous)
        self.assertLessEqual(high_distance, low_distance + 1e-7)


    def test_ew_covariance_downweights_recently_volatile_asset(self):
        rng = np.random.default_rng(123)
        n = 700
        a_returns = np.r_[rng.normal(0.0, 0.0008, n - 120), rng.normal(0.0, 0.010, 120)]
        b_returns = rng.normal(0.0, 0.0008, n)
        histories = {
            "A/USD": (100.0 * np.exp(np.cumsum(a_returns))).tolist(),
            "B/USD": (100.0 * np.exp(np.cumsum(b_returns))).tolist(),
        }
        result = optimize_signed_portfolio(
            [sig("A/USD", gross_edge=0.02, one_way_cost=0.0),
             sig("B/USD", gross_edge=0.02, one_way_cost=0.0)], [], histories, {},
            long_budget=0.50, short_budget=0.0, total_gross_cap=0.50,
            long_asset_cap=0.25, short_asset_cap=0.0,
            covariance_window_bars=300, risk_horizon_bars=60,
            covariance_half_life_bars=60, covariance_shrinkage=0.2,
            risk_aversion=60.0, cost_penalty=0.0, iterations=400,
        )
        self.assertLess(result.weights.get("A/USD", 0.0), result.weights.get("B/USD", 0.0))

    def test_historical_replay_does_not_join_across_large_gaps(self):
        # Two synthetic three-bar contiguous segments separated by a one-hour gap.
        times = [0, 60_000, 120_000, 3_600_000, 3_660_000, 3_720_000]
        by_time = {ts: {"BTC/USD": (100.0 + i, 99.9 + i, 100.1 + i, 0.0, 1e6)}
                   for i, ts in enumerate(times)}
        panel, snapshots, regular_times, segment_count = regularize_longest_segment(
            by_time, times, poll_seconds=60, gap_factor=2.5
        )
        self.assertEqual(segment_count, 2)
        self.assertEqual(len(regular_times), 3)
        self.assertLess(regular_times[-1], 1_000_000)
        self.assertEqual(len(panel["BTC/USD"]), 3)
        self.assertEqual(set(regular_times), set(snapshots))

    def test_duplicate_pair_is_not_both_long_and_short(self):
        result = optimize_signed_portfolio(
            [sig("A/USD")], [sig("A/USD"), sig("B/USD")], self._histories(), {},
            long_budget=0.30, short_budget=0.30, total_gross_cap=0.60,
            long_asset_cap=0.30, short_asset_cap=0.30,
            covariance_window_bars=300, risk_horizon_bars=100,
        )
        self.assertGreaterEqual(result.weights.get("A/USD", 0), 0)
        self.assertNotEqual(result.weights.get("A/USD", 0), -0.0)
        self.assertLessEqual(sum(abs(w) for w in result.weights.values()), 0.60 + 1e-8)


if __name__ == "__main__":
    unittest.main()
