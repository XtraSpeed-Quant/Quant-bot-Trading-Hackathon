from __future__ import annotations

from typing import Any, Mapping


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def format_dashboard(result: Mapping[str, Any]) -> str:
    width = 92
    lines = [
        "", "=" * width, "ROOSTOO QUANT ENGINE | LIVE DECISION", "=" * width,
        f"  [STATE]             {str(result.get('status', 'unknown')).upper()}",
        f"  [REGIME]            {str(result.get('regime', 'unknown')).upper()}",
        f"  [BREADTH]           {_pct(float(result.get('breadth', 0)))}",
        f"  [MEDIAN 4H]         {_pct(float(result.get('median_4h', 0)))}",
        f"  [MEDIAN 12H]        {_pct(float(result.get('median_12h', 0)))}",
        f"  [MEDIAN 24H]        {_pct(float(result.get('median_24h', 0)))}",
        f"  [MARKET STRENGTH]   {float(result.get('market_strength', 0)):.3f}",
        f"  [TARGET LONG GROSS] {_pct(float(result.get('gross_target', 0)))}",
        f"  [TARGET SHORT GROSS]{_pct(float(result.get('short_gross_target', 0)))}",
        f"  [EQUITY]            {_money(float(result.get('equity', 0)))}",
        f"  [EXPOSURE]          {_money(float(result.get('current_exposure', 0)))}",
        f"  [VOL SCALE]         {float(result.get('volatility_scale', 0)):.3f}",
        "-" * width,
        "PREDICTIVE OVERLAY",
        "-" * width,
        f"  [MODEL]             {'ENABLED' if result.get('prediction_enabled') else 'GATED'}",
        f"  [TRAINING SAMPLES]  {int(result.get('prediction_samples', 0))}",
        f"  [VALID DIRECTION]   {float(result.get('prediction_accuracy', 0))*100:.2f}%",
        f"  [VALID CORRELATION] {float(result.get('prediction_correlation', 0)):+.4f}",
        f"  [BALANCED ACC]      {float(result.get('prediction_balanced_accuracy', 0))*100:.2f}%",
        f"  [TOP DECILE EDGE]   {_pct(float(result.get('prediction_top_decile_edge', 0)))}",
        f"  [LIVE PREDICTIONS]  {int(result.get('live_prediction_samples', 0))} | acc={float(result.get('live_prediction_accuracy', 0))*100:.2f}% corr={float(result.get('live_prediction_correlation', 0)):+.4f}",
        "-" * width,
        "SIGNALS",
        "-" * width,
    ]
    rows = list(result.get("signal_rows", []))
    for i, row in enumerate(rows, 1):
        lines.append(
            f"  {i:>2}. [{str(row.get('direction', 'HOLD')):<5}] {str(row.get('pair')):<11} "
            f"score={float(row.get('score', 0)):.3f} pred={_pct(float(row.get('predicted_return', 0)))} "
            f"p_up={float(row.get('probability_up', 0)):.2f} p_dn={float(row.get('probability_down', 0)):.2f} "
            f"1h={_pct(float(row.get('fast_return', 0)))} 4h={_pct(float(row.get('medium_return', 0)))} "
            f"12h={_pct(float(row.get('long_return', 0)))} 24h={_pct(float(row.get('return_24h', 0)))} "
            f"target={_pct(float(row.get('target_weight', 0)))}"
        )
        lines.append(f"      WHY: {str(row.get('reason', '')).replace('_', ' ')}")
    if not rows:
        lines.append("  [HOLD ] No qualifying signals")

    lines += ["-" * width, "ORDER PLAN", "-" * width]
    for order in result.get("planned_sells", []):
        lines.append(f"  [SELL] {order['pair']:<12} {_money(order['notional']):>15} current={_money(order['current'])} target={_money(order['target'])}")
    for order in result.get("planned_buys", []):
        lines.append(f"  [BUY ] {order['pair']:<12} {_money(order['notional']):>15} current={_money(order['current'])} target={_money(order['target'])}")
    for order in result.get("planned_short_opens", []):
        lines.append(f"  [SHORT] {order['pair']:<11} {_money(order['collateral']):>15} current={_money(order['current'])} target={_money(order['target'])}")
    for order in result.get("planned_short_closes", []):
        lines.append(f"  [CLOSE] {order['pair']:<11} {_money(order['collateral']):>15}")
    if not result.get("planned_sells") and not result.get("planned_buys") and not result.get("planned_short_opens") and not result.get("planned_short_closes"):
        lines.append("  [HOLD ] No execution required")

    lines += [
        "-" * width,
        f"  [RISK]              {result.get('risk_reason', 'unknown')}",
        f"  [ORDERS]            {'ENABLED' if result.get('orders_enabled') else 'DISABLED'}",
        f"  [BLOCK]             {result.get('blocked') or 'none'}",
        f"  [HISTORY]           {result.get('history_points', 0)}/{result.get('history_required', 0)}",
        f"  [DAILY TRADES]      {result.get('today_trades', 0)}",
        "=" * width,
    ]
    return "\n".join(lines)
