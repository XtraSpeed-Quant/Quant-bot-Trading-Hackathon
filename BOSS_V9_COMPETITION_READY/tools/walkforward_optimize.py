"""Chronological optimizer for BOSS's live strategy parameters.

Runs the exact BossStrategy + portfolio optimizer against historical SQLite snapshots.
The first chronological segment selects parameters; a later untouched segment is
reported as out-of-sample. This is research-only: it never calls the exchange API.

Example:
  python tools/walkforward_optimize.py --db data/boss_v9_live.sqlite3 --poll-seconds 60
  python tools/walkforward_optimize.py --db data/bot.sqlite3 --poll-seconds 300 --output reports/boss_optimization.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.strategy.boss import BossStrategy, Market
from src.main import _exit_thresholds


def load_panel(db_path: str):
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    uri = path.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute("""
            SELECT timestamp_ms, pair, price, bid, ask, change_24h, unit_trade_value
            FROM prices
            WHERE price > 0
            ORDER BY timestamp_ms, pair
        """).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError(f"No price snapshots found in {path}")

    panel: dict[str, list[tuple[int, float, float, float, float, float]]] = {}
    by_time: dict[int, dict[str, tuple[float, float, float, float, float]]] = {}
    for ts, pair, price, bid, ask, change, volume in rows:
        ts, pair = int(ts), str(pair)
        record = (float(price), float(bid), float(ask), float(change), float(volume))
        panel.setdefault(pair, []).append((ts, *record))
        by_time.setdefault(ts, {})[pair] = record
    times = sorted(by_time)
    diffs = [b - a for a, b in zip(times[:-1], times[1:]) if b > a]
    cadence_seconds = int(round(statistics.median(diffs) / 1000.0)) if diffs else 60
    return panel, by_time, times, max(1, cadence_seconds)



def regularize_longest_segment(by_time, times, poll_seconds: int, gap_factor: float = 2.5):
    """Resample the longest contiguous segment to a fixed cadence.

    A source gap larger than max(gap_factor * poll, 120 seconds) starts a new
    segment. We never concatenate price history across such gaps because that
    would silently make a bar mean different amounts of elapsed clock time.
    """
    if len(times) < 2:
        return {}, {}, [], 0
    poll_ms = max(1, int(poll_seconds)) * 1000
    gap_threshold = max(int(gap_factor * poll_ms), 120_000)
    cuts = [i for i, (a, b) in enumerate(zip(times[:-1], times[1:])) if b - a > gap_threshold]
    bounds = [0] + [i + 1 for i in cuts] + [len(times)]
    segments = [(times[a:b], a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
    if not segments:
        return {}, {}, [], len(bounds) - 1
    # Pick the segment with the longest elapsed span; ties prefer more snapshots.
    seg_times, start_idx, end_idx = max(segments, key=lambda value: (value[0][-1] - value[0][0], len(value[0])))
    grid = np.arange(int(seg_times[0]), int(seg_times[-1]) + 1, poll_ms, dtype=np.int64)
    seg_arr = np.asarray(seg_times, dtype=np.int64)
    regular_panel: dict[str, list[tuple[int, float, float, float, float, float]]] = {}
    regular_by_time: dict[int, dict[str, tuple[float, float, float, float, float]]] = {}
    max_staleness = max(2 * poll_ms, 120_000)
    for grid_ts in grid:
        source_idx = int(np.searchsorted(seg_arr, grid_ts, side="right") - 1)
        if source_idx < 0:
            continue
        source_ts = int(seg_arr[source_idx])
        if int(grid_ts) - source_ts > max_staleness:
            continue
        source_snapshot = by_time.get(source_ts, {})
        if not source_snapshot:
            continue
        clean_snapshot = dict(source_snapshot)
        gt = int(grid_ts)
        regular_by_time[gt] = clean_snapshot
        for pair, record in clean_snapshot.items():
            regular_panel.setdefault(pair, []).append((gt, *record))
    regular_times = sorted(regular_by_time)
    return regular_panel, regular_by_time, regular_times, len(segments)

def _decision_pairs(times: list[int], warmup_minutes: int, hold_minutes: int, step_minutes: int) -> list[tuple[int, int]]:
    warmup_time = times[0] + int(warmup_minutes * 60_000)
    hold_ms, step_ms = int(hold_minutes * 60_000), int(step_minutes * 60_000)
    candidates = [i for i, ts in enumerate(times) if ts >= warmup_time]
    if not candidates:
        return []
    i = candidates[0]
    decisions: list[tuple[int, int]] = []
    while i < len(times) - 1:
        future = times[i] + hold_ms
        j = int(np.searchsorted(times, future, side="left"))
        if j >= len(times):
            break
        if j <= i:
            j = i + 1
        decisions.append((i, j))
        next_time = times[i] + step_ms
        next_i = int(np.searchsorted(times, next_time, side="left"))
        if next_i <= i:
            next_i = i + 1
        i = next_i
    return decisions


def _snapshot_inputs(panel, by_time, timestamp_ms: int, cfg: Config):
    histories: dict[str, list[float]] = {}
    markets: dict[str, Market] = {}
    costs: dict[str, float] = {}
    snapshot = by_time[timestamp_ms]
    for pair, records in panel.items():
        # Each record is ordered, so the last observation at or before timestamp is safe.
        pair_times = [record[0] for record in records]
        idx = int(np.searchsorted(pair_times, timestamp_ms, side="right"))
        if idx <= 0:
            continue
        selected = records[max(0, idx - cfg.history_limit):idx]
        history = [float(r[1]) for r in selected if r[1] > 0]
        if history:
            histories[pair] = history
        record = snapshot.get(pair)
        if not record:
            continue
        price, bid, ask, change, volume = record
        if price <= 0 or bid <= 0 or ask <= 0 or bid > ask:
            continue
        market = Market(pair, pair.split("/")[0], price, bid, ask, change, volume, 6, 1.0, True)
        # Entry/rebalance cost for 1 unit of portfolio weight (one-way fee + half-spread + slippage).
        cost = cfg.commission_rate + market.spread_bps / 20_000.0 + cfg.slippage_bps / 10_000.0
        markets[pair] = market
        costs[pair] = max(0.0, float(cost))
    return histories, markets, costs


def _metrics(period_returns: list[float], turnover: float, long_pnl: float, short_pnl: float,
             decisions: int, years: float, liquidation_cost: float,
             holding_minutes: int) -> dict[str, float | int]:
    if not period_returns:
        return {"decisions": 0, "total_return_pct": 0.0, "annualized_sharpe": 0.0,
                "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0,
                "turnover_x_equity": turnover, "long_contribution_pct": long_pnl * 100,
                "short_contribution_pct": short_pnl * 100, "liquidation_cost_pct": liquidation_cost * 100}
    arr = np.asarray(period_returns, dtype=float)
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    max_dd = float(np.max((peaks - equity) / np.maximum(peaks, 1e-12))) if equity.size else 0.0
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    periods_per_year = 365.25 * 24.0 * 60.0 / max(int(holding_minutes), 1)
    sharpe = float(np.mean(arr) / sigma * math.sqrt(periods_per_year)) if sigma > 1e-12 else 0.0
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "decisions": int(decisions),
        "total_return_pct": float((np.prod(1.0 + arr) - 1.0) * 100.0),
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "win_rate_pct": float(np.mean(arr > 0) * 100.0),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 1e-12 else (1_000_000.0 if gross_win > 0 else 0.0),
        "turnover_x_equity": float(turnover),
        "long_contribution_pct": float(long_pnl * 100.0),
        "short_contribution_pct": float(short_pnl * 100.0),
        "liquidation_cost_pct": float(liquidation_cost * 100.0),
    }


def simulate(config: Config, panel, by_time, times, decisions, initial_weights=None,
             holding_minutes: int = 60) -> dict[str, Any]:
    """Chronological portfolio replay, including price-path stop/trailing exits."""
    strategy = BossStrategy(config, shorts_available=True)
    previous_weights = dict(initial_weights or {})
    previous_costs: dict[str, float] = {}
    # State persists across rebalance intervals while a signed position remains open.
    position_state: dict[str, dict[str, float]] = {}
    rets: list[float] = []
    turnover = 0.0
    long_contribution = 0.0
    short_contribution = 0.0
    liquidation_cost = 0.0

    for current_i, future_i in decisions:
        timestamp = times[current_i]
        histories, markets, costs = _snapshot_inputs(panel, by_time, timestamp, config)
        result = strategy.analyze(histories, markets, timestamp, current_weights=previous_weights)
        weights = dict(result.long_targets)
        weights.update({pair: -weight for pair, weight in result.short_targets.items()})

        trade_cost = 0.0
        all_pairs = set(previous_weights) | set(weights)
        for pair in all_pairs:
            old_w = float(previous_weights.get(pair, 0.0))
            new_w = float(weights.get(pair, 0.0))
            delta = abs(new_w - old_w)
            if delta <= 1e-12:
                continue
            cost_per_weight = costs.get(pair, previous_costs.get(pair, config.commission_rate + config.slippage_bps / 10_000.0))
            trade_cost += delta * cost_per_weight
            turnover += delta

        # Reset state on closes/reversals. A new signed position starts at this
        # timestamp's observed price; increasing/reducing the same-side weight
        # does not reset the entry/peak/trough reference.
        snapshot_now = by_time[timestamp]
        for pair in set(previous_weights) | set(weights):
            old_w = float(previous_weights.get(pair, 0.0))
            new_w = float(weights.get(pair, 0.0))
            now_rec = snapshot_now.get(pair)
            if abs(new_w) <= 1e-12:
                position_state.pop(pair, None)
                continue
            reversed_side = abs(old_w) > 1e-12 and old_w * new_w < 0
            needs_open = abs(old_w) <= 1e-12 or reversed_side or pair not in position_state
            if reversed_side:
                position_state.pop(pair, None)
            if needs_open and now_rec and now_rec[0] > 0:
                px = float(now_rec[0])
                position_state[pair] = {"entry": px, "peak": px, "trough": px, "sign": 1.0 if new_w > 0 else -1.0}

        gross = 0.0
        long_pnl_segment = 0.0
        short_pnl_segment = 0.0
        post_interval_weights: dict[str, float] = {}
        for pair, weight in weights.items():
            now_rec = snapshot_now.get(pair)
            state = position_state.get(pair)
            if not now_rec or now_rec[0] <= 0 or state is None:
                # If there is no fresh reference price, keep the exposure represented
                # instead of fabricating a zero return.
                if abs(weight) > 1e-12:
                    post_interval_weights[pair] = weight
                continue
            now_price = float(now_rec[0])
            thresholds = _exit_thresholds(pair, config, markets.get(pair), histories.get(pair), is_short=weight < 0)
            last_price = now_price
            exited = False
            for path_i in range(current_i + 1, future_i + 1):
                path_ts = times[path_i]
                rec = by_time[path_ts].get(pair)
                if not rec or rec[0] <= 0:
                    continue
                price = float(rec[0])
                last_price = price
                if weight > 0:
                    state["peak"] = max(float(state["peak"]), price)
                    hard_stop = price <= float(state["entry"]) * (1.0 - thresholds["stop"])
                    take_profit = price >= float(state["entry"]) * (1.0 + thresholds["take_profit"])
                    trailing_hit = (
                        float(state["peak"]) >= float(state["entry"]) * (1.0 + thresholds["activation"])
                        and price <= float(state["peak"]) * (1.0 - thresholds["trail"])
                    )
                else:
                    state["trough"] = min(float(state["trough"]), price)
                    hard_stop = price >= float(state["entry"]) * (1.0 + thresholds["stop"])
                    take_profit = price <= float(state["entry"]) * (1.0 - thresholds["take_profit"])
                    trailing_hit = (
                        float(state["trough"]) <= float(state["entry"]) * (1.0 - thresholds["activation"])
                        and price >= float(state["trough"]) * (1.0 + thresholds["trail"])
                    )
                if hard_stop or trailing_hit or take_profit:
                    # Close at the first observed trigger snapshot and charge a
                    # one-way exit cost using its spread plus fee/slippage assumptions.
                    exit_spread = 0.0
                    bid, ask = float(rec[1]), float(rec[2])
                    mid = (bid + ask) / 2.0
                    if mid > 0 and ask >= bid > 0:
                        exit_spread = (ask - bid) / mid * 10_000.0
                    exit_cost = config.commission_rate + exit_spread / 20_000.0 + config.slippage_bps / 10_000.0
                    trade_cost += abs(weight) * max(0.0, exit_cost)
                    turnover += abs(weight)
                    reason = "stop" if hard_stop else ("take_profit" if take_profit else "trailing")
                    position_state.pop(pair, None)
                    exited = True
                    break

            asset_return = last_price / now_price - 1.0
            contribution = float(weight) * asset_return
            gross += contribution
            if weight > 0:
                long_pnl_segment += contribution
            else:
                short_pnl_segment += contribution
            if not exited:
                post_interval_weights[pair] = weight

        net = gross - trade_cost
        rets.append(net)
        long_contribution += long_pnl_segment
        short_contribution += short_pnl_segment
        previous_weights = post_interval_weights
        previous_costs = costs

    # Mark the final held portfolio for an estimated one-way liquidation, so the
    # last sample does not assume positions can be abandoned without execution cost.
    for pair, weight in previous_weights.items():
        liquidation_cost += abs(weight) * previous_costs.get(pair, config.commission_rate + config.slippage_bps / 10_000.0)
    if rets and liquidation_cost > 0:
        rets[-1] = (1.0 + rets[-1]) * max(0.0, 1.0 - liquidation_cost) - 1.0
    span_years = max((times[decisions[-1][1]] - times[decisions[0][0]]) / (365.25 * 24 * 60 * 60_000), 1e-6) if decisions else 1e-6
    metrics = _metrics(rets, turnover, long_contribution, short_contribution, len(decisions),
                       span_years, liquidation_cost, holding_minutes)
    metrics["compounded_equity_x"] = float(np.prod(1.0 + np.asarray(rets, dtype=float))) if rets else 1.0
    metrics["gross_return_contribution_pct"] = float((long_contribution + short_contribution) * 100.0)
    return metrics


def selection_score(metrics: dict[str, Any]) -> float:
    """Moderate risk-adjusted objective to reduce raw-return-only selection."""
    sharpe = float(metrics.get("annualized_sharpe", 0.0))
    dd = float(metrics.get("max_drawdown_pct", 0.0)) / 100.0
    n = int(metrics.get("decisions", 0))
    ret = float(metrics.get("total_return_pct", 0.0)) / 100.0
    if n < 8:
        return -1e6 + ret
    return sharpe - 2.0 * dd + 2.0 * ret


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/boss_v9_live.sqlite3")
    parser.add_argument("--poll-seconds", type=int, default=0,
                        help="source cadence in seconds; 0 infers median timestamp interval")
    parser.add_argument("--holding-minutes", type=int, default=60)
    parser.add_argument("--step-minutes", type=int, default=60)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--output", default="data/walkforward_optimization.json")
    parser.add_argument("--gap-factor", type=float, default=2.5,
                        help="split history when a snapshot gap exceeds this multiple of polling cadence")
    parser.add_argument("--predictor", action="store_true",
                        help="evaluate the chronological ML ranker; uses a single default parameter set")
    parser.add_argument("--quick", action="store_true", help="run a smaller parameter grid")
    args = parser.parse_args()
    if args.holding_minutes <= 0 or args.step_minutes <= 0:
        parser.error("holding and step minutes must be positive")
    if args.holding_minutes != args.step_minutes:
        parser.error("For this non-overlapping backtest, holding and step minutes must match")
    if not 0.50 <= args.train_fraction <= 0.85:
        parser.error("train fraction must be between 0.50 and 0.85")

    started = time.perf_counter()
    raw_panel, raw_by_time, raw_times, inferred_poll = load_panel(args.db)
    poll_seconds = max(1, args.poll_seconds or inferred_poll)
    panel, by_time, times, segment_count = regularize_longest_segment(
        raw_by_time, raw_times, poll_seconds, gap_factor=args.gap_factor
    )
    if not times:
        raise SystemExit("No contiguous timestamp segment is available for backtesting.")
    base = Config(
        predictor_enabled=bool(args.predictor),
        poll_seconds=poll_seconds,
        rebalance_seconds=max(poll_seconds, args.step_minutes * 60),
        history_limit=max(2000, max((len(v) for v in panel.values()), default=0)),
        predictor_history_limit=max(25_000, max((len(v) for v in panel.values()), default=0)),
        allow_pairs=(),
    )
    base.validate(require_credentials=False, require_api_url=False)
    warmup = max(base.long_window, base.trend_window, base.portfolio_covariance_window_minutes,
                 base.correlation_window, 2 * base.fast_window)
    decisions = _decision_pairs(times, warmup, args.holding_minutes, args.step_minutes)
    if len(decisions) < 48:
        raise SystemExit(
            f"Not enough walk-forward windows ({len(decisions)}). Need >=48 hourly decisions after a {warmup}m warm-up; "
            f"longest continuous segment is {(times[-1]-times[0])/3.6e6:.1f} hours at {poll_seconds}s cadence. "
            f"The source had {len(raw_times)} timestamps across {segment_count} continuity segments, so gaps/cadence changes were not combined. "
            "Collect more uninterrupted history, then rerun."
        )
    # Preserve at least 12 later decisions as a genuinely untouched final holdout.
    split = max(24, min(len(decisions) - 12, int(len(decisions) * args.train_fraction)))
    train_decisions, test_decisions = decisions[:split], decisions[split:]

    exit_presets = [
        {"STOP_VOL_MULTIPLE": 2.0, "TRAILING_ACTIVATION_VOL_MULTIPLE": 1.5,
         "TRAILING_STOP_VOL_MULTIPLE": 0.8, "TAKE_PROFIT_VOL_MULTIPLE": 3.0},
        {"STOP_VOL_MULTIPLE": 2.5, "TRAILING_ACTIVATION_VOL_MULTIPLE": 1.8,
         "TRAILING_STOP_VOL_MULTIPLE": 1.0, "TAKE_PROFIT_VOL_MULTIPLE": 3.5},
        {"STOP_VOL_MULTIPLE": 3.0, "TRAILING_ACTIVATION_VOL_MULTIPLE": 2.2,
         "TRAILING_STOP_VOL_MULTIPLE": 1.25, "TAKE_PROFIT_VOL_MULTIPLE": 4.0},
    ]
    if args.predictor:
        risks, edges, costs, shrinks = [60.0], [0.0025], [1.0], [0.65]
    elif args.quick:
        risks, edges, costs, shrinks = [30.0, 60.0], [0.0015, 0.0030], [0.75, 1.5], [0.5, 0.8]
    else:
        risks, edges, costs, shrinks = [20.0, 60.0, 120.0], [0.0015, 0.0025, 0.0040], [0.5, 1.0, 2.0], [0.4, 0.65, 0.85]

    runs: list[dict[str, Any]] = []
    total = len(risks) * len(edges) * len(costs) * len(shrinks) * len(exit_presets)
    run_n = 0
    for risk in risks:
        for min_edge in edges:
            for cost in costs:
                for shrink in shrinks:
                    for exit_params in exit_presets:
                        run_n += 1
                        short_stop = max(3.0, float(exit_params["STOP_VOL_MULTIPLE"]) * 1.2)
                        short_activation = max(2.0, float(exit_params["TRAILING_ACTIVATION_VOL_MULTIPLE"]) * 1.1)
                        cfg = replace(
                            base,
                            portfolio_risk_aversion=float(risk),
                            min_relative_edge=float(min_edge),
                            min_short_relative_edge=float(min_edge),
                            portfolio_cost_penalty=float(cost),
                            portfolio_covariance_shrinkage=float(shrink),
                            adaptive_exits_enabled=True,
                            stop_vol_multiple=float(exit_params["STOP_VOL_MULTIPLE"]),
                            trailing_activation_vol_multiple=float(exit_params["TRAILING_ACTIVATION_VOL_MULTIPLE"]),
                            trailing_stop_vol_multiple=float(exit_params["TRAILING_STOP_VOL_MULTIPLE"]),
                            take_profit_vol_multiple=float(exit_params["TAKE_PROFIT_VOL_MULTIPLE"]),
                            short_stop_vol_multiple=short_stop,
                            short_trailing_activation_vol_multiple=short_activation,
                            short_trailing_stop_vol_multiple=float(exit_params["TRAILING_STOP_VOL_MULTIPLE"]),
                        )
                        train_metrics = simulate(cfg, panel, by_time, times, train_decisions, holding_minutes=args.holding_minutes)
                        score = selection_score(train_metrics)
                        params = {
                            "PORTFOLIO_RISK_AVERSION": risk,
                            "MIN_RELATIVE_EDGE": min_edge,
                            "MIN_SHORT_RELATIVE_EDGE": min_edge,
                            "PORTFOLIO_COST_PENALTY": cost,
                            "PORTFOLIO_COVARIANCE_SHRINKAGE": shrink,
                            **exit_params,
                            "SHORT_STOP_VOL_MULTIPLE": short_stop,
                            "SHORT_TRAILING_ACTIVATION_VOL_MULTIPLE": short_activation,
                            "SHORT_TRAILING_STOP_VOL_MULTIPLE": float(exit_params["TRAILING_STOP_VOL_MULTIPLE"]),
                        }
                        runs.append({"parameters": params, "selection_score": score, "train": train_metrics})
                        if run_n % max(1, total // 9) == 0:
                            print(f"evaluated {run_n}/{total} parameter combinations")

    runs.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    winner = runs[0]
    p = winner["parameters"]
    best_cfg = replace(
        base,
        portfolio_risk_aversion=float(p["PORTFOLIO_RISK_AVERSION"]),
        min_relative_edge=float(p["MIN_RELATIVE_EDGE"]),
        min_short_relative_edge=float(p["MIN_SHORT_RELATIVE_EDGE"]),
        portfolio_cost_penalty=float(p["PORTFOLIO_COST_PENALTY"]),
        portfolio_covariance_shrinkage=float(p["PORTFOLIO_COVARIANCE_SHRINKAGE"]),
        adaptive_exits_enabled=True,
        stop_vol_multiple=float(p["STOP_VOL_MULTIPLE"]),
        trailing_activation_vol_multiple=float(p["TRAILING_ACTIVATION_VOL_MULTIPLE"]),
        trailing_stop_vol_multiple=float(p["TRAILING_STOP_VOL_MULTIPLE"]),
        take_profit_vol_multiple=float(p["TAKE_PROFIT_VOL_MULTIPLE"]),
        short_stop_vol_multiple=float(p["SHORT_STOP_VOL_MULTIPLE"]),
        short_trailing_activation_vol_multiple=float(p["SHORT_TRAILING_ACTIVATION_VOL_MULTIPLE"]),
        short_trailing_stop_vol_multiple=float(p["SHORT_TRAILING_STOP_VOL_MULTIPLE"]),
    )
    # Report untouched later data for the training-selected parameters only.
    test_metrics = simulate(best_cfg, panel, by_time, times, test_decisions, holding_minutes=args.holding_minutes)
    baseline_cfg = replace(base, portfolio_risk_aversion=60.0,
                           min_relative_edge=0.0025, min_short_relative_edge=0.0025,
                           portfolio_cost_penalty=1.0, portfolio_covariance_shrinkage=0.65,
                           adaptive_exits_enabled=True, stop_vol_multiple=2.5,
                           trailing_activation_vol_multiple=1.8, trailing_stop_vol_multiple=1.0,
                           take_profit_vol_multiple=3.5, short_stop_vol_multiple=3.0,
                           short_trailing_activation_vol_multiple=2.0, short_trailing_stop_vol_multiple=1.0)
    baseline_train = simulate(baseline_cfg, panel, by_time, times, train_decisions, holding_minutes=args.holding_minutes)
    baseline_test = simulate(baseline_cfg, panel, by_time, times, test_decisions, holding_minutes=args.holding_minutes)
    selected_oos_return = float(test_metrics.get("total_return_pct", 0.0))
    baseline_oos_return = float(baseline_test.get("total_return_pct", 0.0))
    selected_oos_sharpe = float(test_metrics.get("annualized_sharpe", 0.0))
    baseline_oos_sharpe = float(baseline_test.get("annualized_sharpe", 0.0))
    selected_oos_drawdown = float(test_metrics.get("max_drawdown_pct", 0.0))
    baseline_oos_drawdown = float(baseline_test.get("max_drawdown_pct", 0.0))
    selected_oos_beats_baseline = (selected_oos_return > baseline_oos_return
                                   and selected_oos_sharpe > baseline_oos_sharpe
                                   and selected_oos_drawdown <= baseline_oos_drawdown)
    if selected_oos_return > 0.0 and selected_oos_beats_baseline:
        promotion_status = "PROMISING_NOT_PROOF: validate against real fills and a later untouched period"
    else:
        promotion_status = "NO_CLEAR_OOS_ADVANTAGE: keep researching; do not call these parameters profitable"

    def format_env_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    result = {
        "title": "BOSS chronological parameter optimization",
        "database": str(Path(args.db).resolve()),
        "markets": len(panel),
        "snapshots": len(times),
        "source_timestamps": len(raw_times),
        "source_continuity_segments": segment_count,
        "longest_segment_only": True,
        "predictor_enabled": bool(args.predictor),
        "start_timestamp_ms": times[0],
        "end_timestamp_ms": times[-1],
        "data_span_hours": (times[-1] - times[0]) / 3_600_000,
        "poll_seconds_used": poll_seconds,
        "holding_minutes": args.holding_minutes,
        "warmup_minutes": warmup,
        "decision_windows": len(decisions),
        "train_windows": len(train_decisions),
        "test_windows": len(test_decisions),
        "selection_objective": "annualized Sharpe - 2*maximum drawdown + 2*net total return; train segment only",
        "minimum_decision_windows": 48,
        "minimum_test_windows": 12,
        "covariance_estimator": "exponentially weighted covariance, half-life configured in minutes, then diagonal shrinkage and PSD projection",
        "selected_parameters": p,
        "selected_env_lines": [f"{key}={format_env_value(value)}" for key, value in p.items()] + [
            "PORTFOLIO_COVARIANCE_HALF_LIFE_MINUTES=180", "PORTFOLIO_RISK_HORIZON_MINUTES=60"],
        "selected_oos_beats_baseline": selected_oos_beats_baseline,
        "promotion_status": promotion_status,
        "selected_train_metrics": winner["train"],
        "selected_out_of_sample_metrics": test_metrics,
        "baseline_parameters": {
            "PORTFOLIO_RISK_AVERSION": 60.0,
            "MIN_RELATIVE_EDGE": 0.0025,
            "MIN_SHORT_RELATIVE_EDGE": 0.0025,
            "PORTFOLIO_COST_PENALTY": 1.0,
            "PORTFOLIO_COVARIANCE_SHRINKAGE": 0.65,
            "STOP_VOL_MULTIPLE": 2.5,
            "TRAILING_ACTIVATION_VOL_MULTIPLE": 1.8,
            "TRAILING_STOP_VOL_MULTIPLE": 1.0,
            "TAKE_PROFIT_VOL_MULTIPLE": 3.5,
            "SHORT_STOP_VOL_MULTIPLE": 3.0,
            "SHORT_TRAILING_ACTIVATION_VOL_MULTIPLE": 2.0,
            "SHORT_TRAILING_STOP_VOL_MULTIPLE": 1.0,
        },
        "baseline_train_metrics": baseline_train,
        "baseline_out_of_sample_metrics": baseline_test,
        "top_10_train_candidates": runs[:10],
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a chronological price-snapshot replay, not a tick-level fill simulation.",
            "Spread, taker fee, configured slippage, and close/trailing stops on sampled prices are modeled; within-snapshot stop paths, order-book impact, partial fills, funding/borrow charges, and API rejection are not represented.",
            "The archive spans the available historical period only; a few days cannot establish stable profitability across market regimes.",
            "At least 48 hourly decisions are required, with at least 12 reserved for one untouched final holdout. The out-of-sample slice is reported once for the training-selected candidate; do not iterate against it and still call it untouched.",
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("\nBOSS WALK-FORWARD OPTIMIZATION")
    print(f"Data: {len(panel)} markets, {len(times)} snapshots, {result['data_span_hours']:.1f}h, poll={poll_seconds}s")
    print(f"Windows: {len(train_decisions)} train / {len(test_decisions)} untouched OOS")
    print("Selected env parameters:")
    for key, val in p.items():
        print(f"  {key}={val}")
    print("\nTrain-selected candidate:")
    print(json.dumps(winner["train"], indent=2))
    print("\nUntouched out-of-sample:")
    print(json.dumps(test_metrics, indent=2))
    print("\nBaseline out-of-sample:")
    print(json.dumps(baseline_test, indent=2))
    print(f"\nPromotion status: {promotion_status}")
    print("Selected environment assignments (review before copying):")
    for line in result["selected_env_lines"]:
        print(f"  {line}")
    print(f"\nFull report: {out_path.resolve()}")
    print("Important: these estimates are conditional on this history and the execution-cost assumptions; they are not a profit guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
