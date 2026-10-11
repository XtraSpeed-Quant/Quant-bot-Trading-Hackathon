"""Read-only evaluation of the strategy/predictor against stored price snapshots.

Usage:
    python tools/research_snapshot.py --db data/boss_v9_live.sqlite3 --poll-seconds 60 --history-limit 25000

This does not connect to an exchange, write to SQLite, or submit orders.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.strategy.boss import BossStrategy, Market


def load_snapshot(db_path: str, history_limit: int, allowed_pairs: tuple[str, ...] = ()):
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            """
            SELECT timestamp_ms, pair, price, bid, ask, change_24h, unit_trade_value
            FROM (
              SELECT timestamp_ms, pair, price, bid, ask, change_24h, unit_trade_value,
                     ROW_NUMBER() OVER (PARTITION BY pair ORDER BY timestamp_ms DESC) AS rn
              FROM prices
            )
            WHERE rn <= ?
            ORDER BY pair, timestamp_ms
            """, (history_limit,)
        ).fetchall()
    finally:
        conn.close()

    histories: dict[str, list[float]] = defaultdict(list)
    latest: dict[str, tuple] = {}
    for ts, pair, price, bid, ask, change, volume in rows:
        histories[pair].append(float(price))
        latest[pair] = (float(price), float(bid), float(ask), float(change), float(volume), int(ts))
    markets: dict[str, Market] = {}
    for pair, (price, bid, ask, change, volume, _) in latest.items():
        if allowed_pairs and pair.upper() not in allowed_pairs:
            continue
        if price <= 0 or bid <= 0 or ask <= 0 or bid > ask:
            continue
        markets[pair] = Market(pair, pair.split("/")[0], price, bid, ask, change, volume, 6, 1.0, True)
    end_ms = max((record[5] for record in latest.values()), default=0)
    return dict(histories), markets, end_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/boss_v7_live.sqlite3")
    parser.add_argument("--poll-seconds", type=int, default=30,
                        help="historical sampling cadence; e.g. 300 for a 5-minute database")
    parser.add_argument("--history-limit", type=int, default=25000)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.history_limit < 50:
        parser.error("poll seconds must be >=1 and history limit >=50")

    cfg = Config(
        poll_seconds=args.poll_seconds,
        rebalance_seconds=max(args.poll_seconds, 2 * args.poll_seconds),
        prediction_log_interval_seconds=max(1, args.poll_seconds),
        history_limit=args.history_limit,
        allow_pairs=(),
    )
    histories, markets, timestamp_ms = load_snapshot(args.db, args.history_limit)
    if not markets:
        print(f"No usable prices found in {args.db}")
        return 1

    strategy = BossStrategy(cfg, shorts_available=True)
    started = time.perf_counter()
    result = strategy.analyze(histories, markets, timestamp_ms)
    elapsed = time.perf_counter() - started
    print("BOSS V9 READ-ONLY HISTORICAL SNAPSHOT")
    print(f"Database: {Path(args.db).resolve()}")
    print(f"Markets: {len(markets)} | histories: {len(histories)} | samples/pair: {min(map(len, histories.values()))}-{max(map(len, histories.values()))}")
    print(f"Poll cadence: {args.poll_seconds}s | analysis runtime: {elapsed:.3f}s")
    print(f"Regime: {result.regime} | breadth={result.breadth:.3f} | strength={result.market_strength:.3f}")
    print(f"Long target gross={result.gross_target:.2%} | short collateral target={result.short_gross_target:.2%} | combined={result.gross_target + result.short_gross_target:.2%}")
    print(f"Predictor samples={result.predictor_samples} | ranker={'PASS' if result.predictor_enabled else 'GATED'} | classifier={'PASS' if result.predictor_directional_enabled else 'GATED'}")
    print(f"Correlation={result.predictor_correlation:.4f} | RankIC={result.predictor_rank_ic:.4f} | balanced accuracy={result.predictor_balanced_accuracy:.4f} | top-decile net={result.predictor_top_decile_net_return:.5%} | normalized Brier={result.predictor_brier:.4f}")
    print(f"Gate: {result.predictor_gate_reason}")
    print("Top long candidates:")
    for signal in result.signals[:10]:
        print(f"  {signal.pair:<14} score={signal.score:.3f} edge_net={signal.expected_edge:+.3%} target={signal.target_weight:.2%}")
    print("Top short candidates:")
    for signal in result.short_signals[:10]:
        print(f"  {signal.pair:<14} score={signal.score:.3f} edge_net={signal.expected_edge:+.3%} collateral={signal.target_weight:.2%}")
    print("Note: this is one stored end-of-window signal snapshot, not a walk-forward P&L backtest and not proof of profitability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
