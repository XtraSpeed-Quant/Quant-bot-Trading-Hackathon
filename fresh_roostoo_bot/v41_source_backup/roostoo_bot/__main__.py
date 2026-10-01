from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import signal
import time

from .client import RoostooClient
from .config import Config
from .dashboard import format_dashboard
from .engine import TradingEngine
from .store import Store


def configure_logging(config: Config) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, config.log_level, logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    handler = RotatingFileHandler(config.log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roostoo real-time quant trading engine")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--account-check", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.validate(require_credentials=not args.check_api)
    configure_logging(config)
    client = RoostooClient(config.api_url, config.api_key, config.api_secret, config.timeout_seconds)

    if args.check_api:
        server = client.sync_time()
        info = client.exchange_info()
        tickers = client.tickers()
        print(f"[API] OK | server_time_ms={server} | exchange_running={info.get('IsRunning')} | tickers={len(tickers.get('Data', {}))}")
        return

    if args.account_check:
        server = client.sync_time()
        info = client.exchange_info()
        wallet = client.balance()
        usd = wallet.get("USD", {})
        current_usd = float(usd.get("Free", 0.0)) + float(usd.get("Lock", 0.0))
        pending = client.pending_count()
        print(f"[API] AUTHENTICATED | server_time_ms={server}")
        print(f"[ACCOUNT] Exchange running: {info.get('IsRunning')}")
        print(f"[ACCOUNT] Exchange InitialWallet: {info.get('InitialWallet', {})}")
        print(f"[ACCOUNT] Current USD: ${current_usd:,.2f}")
        print(f"[ACCOUNT] Pending orders: {pending}")
        if config.enable_shorts:
            print(f"[ACCOUNT] Open shorts: {len(client.short_positions())}")
        return

    store = Store(config.db_path, config.history_limit)
    engine = TradingEngine(config, client, store)

    if args.inspect:
        try:
            print(format_dashboard(engine.cycle(execute=False)))
        finally:
            store.close()
        return

    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False
        logging.getLogger("main").info("[SYSTEM] shutdown requested")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    log = logging.getLogger("main")
    log.info("[SYSTEM] starting namespace=%s poll=%ss rebalance=%ss orders_enabled=%s shorts=%s", config.bot_namespace, config.poll_seconds, config.rebalance_seconds, config.orders_enabled, config.enable_shorts)

    try:
        while running:
            started = time.time()
            try:
                result = engine.cycle(execute=True)
                log.info("[STATUS] state=%s equity=$%.2f exposure=$%.2f regime=%s breadth=%.1f%% signals=%s orders=%s turnover=$%.2f risk=%s blocked=%s history=%s/%s", result["status"], result["equity"], result["current_exposure"], result["regime"], result["breadth"]*100, result["signals"], result["orders"], result["turnover"], result["risk_reason"], result["blocked"] or "none", result["history_points"], result["history_required"])
                if result["rebalance_due"] or result["status"] == "warming_up" or result["orders"] > 0:
                    print(format_dashboard(result))
            except Exception:
                log.exception("[ERROR] trading cycle failed")
            if args.once:
                break
            time.sleep(max(0.0, config.poll_seconds - (time.time() - started)))
    finally:
        store.close()
        log.info("[SYSTEM] stopped")


if __name__ == "__main__":
    main()
