import argparse
import math
from datetime import datetime, timezone

from src.api.client import RoostooClient
from src.config import Config
from src.db import BotDB
from src.main import (
    _daily_key, build_markets, fetch_short_positions, now_ms, portfolio_value,
    positions_from_wallet, print_dashboard, wallet_from_balance,
)
from src.paper_trader import PaperTrader
from src.risk.manager import RiskManager
from src.strategy.boss import BossStrategy


def check_api() -> int:
    try:
        Config().validate(require_credentials=False, require_api_url=True)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    client = RoostooClient()
    result = client.server_time()
    if not result.get("ServerTime"):
        print("[FAIL] Roostoo API unavailable")
        return 1
    info = client.exchange_info()
    pairs = sum(1 for pair, meta in info.get("TradePairs", {}).items()
                if meta.get("CanTrade", False) and pair.endswith("/USD"))
    print(f"[OK] API reachable | server={result['ServerTime']} | tradable_usd_pairs={pairs}")
    return 0


def inspect() -> int:
    cfg = Config()
    cfg.validate(require_credentials=not cfg.paper_mode)
    client = RoostooClient()
    info = client.exchange_info()
    tickers = client.get_all_tickers()
    markets = build_markets(info, tickers, cfg)
    db = BotDB(cfg.db_path, cfg.history_limit)
    trader = PaperTrader(cfg.expected_starting_cash) if cfg.paper_mode else client
    wallet = wallet_from_balance(trader.balance())
    if cfg.paper_mode:
        shorts_available = True
        shorts = {}
    else:
        probe = client.short_positions()
        shorts_available = probe.get("Success") is True
        if shorts_available:
            shorts = fetch_short_positions(client, False, tickers, markets)
        else:
            shorts = {}
            print(f"[INFO] Short endpoint not available in inspection: {probe.get('ErrMsg', 'unknown error')}")
    equity = portfolio_value(wallet, tickers, list(shorts.values()))
    strategy = BossStrategy(cfg, shorts_available=shorts_available)
    histories = db.histories(list(markets), cfg.history_limit)
    result = strategy.analyze(histories, markets, now_ms())
    summary = db.history_summary(list(markets))
    namespace = cfg.bot_namespace
    saved_session_start = db.get_state(namespace + ":session_start_equity", None)
    saved_day_start = db.get_state(namespace + ":day_start_equity", None)
    saved_peak = db.get_state(namespace + ":peak_equity", None)
    session_start = float(saved_session_start) if saved_session_start is not None else equity
    day_start = float(saved_day_start) if saved_day_start is not None else equity
    risk = RiskManager(session_start, daily_loss_limit=cfg.daily_loss_limit_pct,
                       max_drawdown=cfg.max_drawdown_limit_pct)
    peak_reference = (cfg.expected_starting_cash if not cfg.paper_mode else session_start)
    risk.peak_equity = max(float(saved_peak or peak_reference), peak_reference, session_start)
    risk.update(equity, day_start)
    positions = positions_from_wallet(wallet, markets, db, trader if cfg.paper_mode else None)
    turnover_used = db.day_turnover()
    turnover_cap = math.inf if cfg.max_daily_turnover == 0 else equity * cfg.max_daily_turnover
    print("\n[INSPECT ONLY] This command does not submit orders or append market snapshots.")
    print(f"[HISTORY] rows={summary['rows']} | snapshots={summary['snapshots']} | pairs={summary['pairs']}"
          f" | loaded_per_pair_max={max((len(v) for v in histories.values()), default=0)}")
    if summary["first_timestamp_ms"] is not None and summary["last_timestamp_ms"] is not None:
        span_hours = max(0.0, (summary["last_timestamp_ms"] - summary["first_timestamp_ms"]) / 3_600_000.0)
        age_minutes = max(0.0, (now_ms() - summary["last_timestamp_ms"]) / 60_000.0)
        print(f"[HISTORY] span={span_hours:.2f}h | latest_snapshot_age={age_minutes:.1f}m")
    else:
        print("[HISTORY] No stored market snapshots yet. Run `python -m roostoo_bot` to start collection.")
    print_dashboard(cfg, equity, session_start, wallet, markets, result, positions,
                    turnover_used, turnover_cap, now_ms() + cfg.rebalance_seconds * 1000,
                    [], [], risk, shorts, execution_override="INSPECT ONLY — NO ORDERS")
    db.close()
    return 0


def reset_turnover() -> int:
    cfg = Config()
    db = BotDB(cfg.db_path, cfg.history_limit)
    day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw, after = db.reset_day_turnover(cfg.bot_namespace, day_key)
    db.close()
    print(f"[TURNOVER RESET] namespace={cfg.bot_namespace} day={day_key}")
    print(f"[TURNOVER RESET] raw_turnover_before=${raw:,.2f}")
    print(f"[TURNOVER RESET] adjusted_turnover_used=${after:,.2f}")
    print("[TURNOVER RESET] trade ledger preserved; only today's turnover baseline changed")
    return 0


def clear_execution_halt(confirmed: bool) -> int:
    if not confirmed:
        print("[REFUSED] First reconcile the ambiguous order on the exchange, then pass --confirm-reconciled.")
        return 2
    cfg = Config()
    db = BotDB(cfg.db_path, cfg.history_limit)
    key = cfg.bot_namespace + ":execution_uncertain"
    existing = db.get_state(key, None)
    if not existing:
        print("[OK] No execution halt is set.")
        db.close()
        return 0
    db.delete_state(key)
    db.close()
    print(f"[CLEARED] Execution halt removed for {cfg.bot_namespace}. The unresolved order record was: {existing}")
    print("Only clear this after checking open orders, spot balances, and short positions in the exchange UI.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BOSS V9.4 competition bot")
    parser.add_argument("--check-api", action="store_true", help="check public exchange endpoints")
    parser.add_argument("--inspect", action="store_true", help="show current signals without order placement")
    parser.add_argument("--reset-turnover", action="store_true", help="reset the namespace turnover baseline, not the trade ledger")
    parser.add_argument("--clear-execution-halt", action="store_true", help="clear an unresolved-order halt after manual reconciliation")
    parser.add_argument("--confirm-reconciled", action="store_true", help="confirm ambiguous orders have been reconciled manually")
    args = parser.parse_args()
    if args.check_api:
        return check_api()
    if args.inspect:
        return inspect()
    if args.reset_turnover:
        return reset_turnover()
    if args.clear_execution_halt:
        return clear_execution_halt(args.confirm_reconciled)
    from src.main import run
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
