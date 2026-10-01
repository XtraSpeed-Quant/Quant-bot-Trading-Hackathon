from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence


def pct_return(prices: Sequence[float], window: int) -> float | None:
    if len(prices) <= window:
        return None
    start = float(prices[-window - 1])
    end = float(prices[-1])
    if start <= 0 or end <= 0 or not math.isfinite(start) or not math.isfinite(end):
        return None
    return end / start - 1.0


def ema(prices: Sequence[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    alpha = 2.0 / (window + 1.0)
    value = float(prices[0])
    for price in prices[1:]:
        value = alpha * float(price) + (1.0 - alpha) * value
    return value


def rolling_mean(prices: Sequence[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return fmean(float(v) for v in prices[-window:])


def rolling_max(prices: Sequence[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    return max(float(v) for v in prices[-window:])


def realized_volatility(prices: Sequence[float], window: int) -> float | None:
    if len(prices) < window + 1:
        return None
    values = [float(price) for price in prices[-window - 1 :]]
    returns: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous <= 0 or current <= 0 or not math.isfinite(previous) or not math.isfinite(current):
            return None
        returns.append(math.log(current / previous))
    if len(returns) < 2:
        return None
    mean = fmean(returns)
    variance = fmean((value - mean) ** 2 for value in returns)
    return math.sqrt(max(variance, 0.0))


def rank01(values: dict[str, float], reverse: bool = False) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    denominator = len(ordered) - 1
    return {
        key: 1.0 - index / denominator if reverse else index / denominator
        for index, (key, _) in enumerate(ordered)
    }
