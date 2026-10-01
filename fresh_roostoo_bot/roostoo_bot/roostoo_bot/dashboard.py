from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _countdown(milliseconds: int) -> str:
    seconds = max(0, int(milliseconds / 1000))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def format_dashboard(result: Mapping[str, Any]) -> str:
    width = 100
    timestamp = int(result.get("timestamp_ms", 0) or 0)
    now = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if timestamp else "-"
    plans = list(result.get("planned_sells", [])) + list(result.get("planned_buys", []))
    action = "HOLD / MONITOR"
    if result.get("blocked") == "inspection_only":
        action = "INSPECTION ONLY"
    elif result.get("risk_reason") not in {"ok", "unknown"} and result.get("risk_reason"):
        action = f"RISK: {str(result.get('risk_reason')).upper()}"
    elif plans:
        sell_names = [str(x.get("pair")) for x in result.get("planned_sells", [])]
        buy_names = [str(x.get("pair")) for x in result.get("planned_buys", [])]
        if sell_names and buy_names:
            action = f"ROTATE: {', '.join(sell_names[:2])} -> {', '.join(buy_names[:2])}"
        elif sell_names:
            action = f"REDUCE/EXIT: {', '.join(sell_names[:3])}"
        elif buy_names:
            action = f"ADD: {', '.join(buy_names[:3])}"
    elif float(result.get("gross_target", 0.0)) <= 0.0:
        action = "CASH / NO LONG EXPOSURE"

    lines = [
        "", "=" * width, "ROOSTOO QUANT ENGINE | LIVE PORTFOLIO CONTROL", "=" * width,
        f"  LAST UPDATE       {now}",
        f"  ACTION            {action}",
        f"  STATE             {str(result.get('status', 'unknown')).upper():<12}   REGIME {str(result.get('regime', 'unknown')).upper()}",
        f"  NEXT REBALANCE    {_countdown(int(result.get('next_rebalance_ms', 0)))}{'  (DUE)' if result.get('rebalance_due') else ''}",
        "-" * width,
        "ACCOUNT",
        "-" * width,
        f"  EQUITY            {_money(float(result.get('equity', 0))):>15}    CASH {_money(float(result.get('cash', 0))):>15}    EXPOSURE {_money(float(result.get('current_exposure', 0))):>15}",
        f"  EXPOSURE %        {_pct(float(result.get('exposure_pct', 0))):>15}    SESSION P&L {_money(float(result.get('session_pnl', 0))):>14} ({_pct(float(result.get('session_pnl_pct', 0)))})",
        f"  VS EXCHANGE INIT  {_money(float(result.get('account_pnl', 0))):>15} ({_pct(float(result.get('account_pnl_pct', 0)))})    TRADES {int(result.get('today_trades', 0)):>4}    TURNOVER {_money(float(result.get('turnover', 0))):>12}",
        "-" * width,
        "MARKET",
        "-" * width,
        f"  UNIVERSE          {int(result.get('universe_size', 0)):>4} eligible USD markets     BREADTH {_pct(float(result.get('breadth', 0))):>9}     STRENGTH {float(result.get('market_strength', 0)):.3f}",
        f"  MEDIAN 4H         {_pct(float(result.get('median_4h', 0))):>9}    MEDIAN 12H {_pct(float(result.get('median_12h', 0))):>9}    MEDIAN 24H {_pct(float(result.get('median_24h', 0))):>9}    VOL {float(result.get('median_volatility', 0)):.4f}",
        f"  TARGET GROSS      LONG {_pct(float(result.get('gross_target', 0))):>9}    SHORT {_pct(float(result.get('short_gross_target', 0))):>9}    VOL SCALE {float(result.get('volatility_scale', 0)):.2f}",
        "-" * width,
        "PREDICTIVE MODEL | PREDICTIVE OVERLAY",
        "-" * width,
        f"  [MODEL]            {'ENABLED' if result.get('prediction_enabled') else 'GATED'}    TRAINING SAMPLES {int(result.get('prediction_samples', 0)):>6}    LIVE SETTLED {int(result.get('live_prediction_samples', 0)):>6}",
        f"  VALIDATION         dir={float(result.get('prediction_accuracy', 0))*100:5.2f}%  corr={float(result.get('prediction_correlation', 0)):+.4f}  bal_acc={float(result.get('prediction_balanced_accuracy', 0))*100:5.2f}%  top10={_pct(float(result.get('prediction_top_decile_edge', 0)))}  brier={float(result.get('prediction_brier', 1)):.3f}",
        f"  GATE REASON        {str(result.get('prediction_gate_reason', '') or 'n/a')[:84]}",
        "-" * width,
        "CURRENT HOLDINGS",
        "-" * width,
    ]
    holdings = result.get("holdings", {}) or {}
    if holdings:
        total = float(result.get("equity", 0))
        targets = {}
        for order in result.get("planned_sells", []) + result.get("planned_buys", []):
            targets[order["pair"]] = float(order.get("target", 0.0))
        for pair, value in sorted(holdings.items(), key=lambda kv: kv[1], reverse=True):
            weight = float(value) / total if total > 0 else 0.0
            target = targets.get(pair, value)
            lines.append(f"  {pair:<12} value={_money(float(value)):>13} weight={_pct(weight):>9} target={_money(target):>13}")
    else:
        lines.append("  CASH ONLY / NO SPOT POSITIONS")

    lines += ["-" * width, "SIGNALS / TARGETS", "-" * width]
    rows = list(result.get("signal_rows", []))
    if rows:
        for i, row in enumerate(rows, 1):
            lines.append(
                f"  {i:>2}. {str(row.get('pair')):<11} score={float(row.get('score', 0)):.3f} target={_pct(float(row.get('target_weight', 0))):>8} "
                f"1h={_pct(float(row.get('fast_return', 0))):>8} 4h={_pct(float(row.get('medium_return', 0))):>8} "
                f"12h={_pct(float(row.get('long_return', 0))):>8} 24h={_pct(float(row.get('return_24h', 0))):>8} pred={_pct(float(row.get('predicted_return', 0))):>8}"
            )
            lines.append(f"      WHY: {str(row.get('reason', '')).replace('_', ' ')}")
    else:
        lines.append("  No qualifying signals")

    lines += ["-" * width, "ORDER PLAN", "-" * width]
    for order in result.get("planned_sells", []):
        lines.append(f"  [SELL] {order['pair']:<12} {_money(order['notional']):>13}  current={_money(order['current'])}  target={_money(order['target'])}")
    for order in result.get("planned_buys", []):
        lines.append(f"  [BUY ] {order['pair']:<12} {_money(order['notional']):>13}  current={_money(order['current'])}  target={_money(order['target'])}")
    if not result.get("planned_sells") and not result.get("planned_buys"):
        lines.append("  No trade required")

    lines += [
        "-" * width,
        f"  RISK              {str(result.get('risk_reason', 'unknown')).upper()}    ORDERS {'ENABLED' if result.get('orders_enabled') else 'DISABLED'}    BLOCK {result.get('blocked') or 'none'}",
        f"  HISTORY           {int(result.get('history_points', 0))}/{int(result.get('history_required', 0))}    DAILY TRADES {int(result.get('today_trades', 0))}    EXPOSURE {_pct(float(result.get('exposure_pct', 0)))}",
        "=" * width,
    ]
    return "\n".join(lines)
