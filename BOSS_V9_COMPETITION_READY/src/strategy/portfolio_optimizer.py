"""Covariance-aware, cost-aware signed portfolio allocation.

The optimizer maximizes a one-horizon utility over the strategy's selected long
and short candidates. Expected edges are gross signal estimates; one-way trading
costs enter separately through an L1 turnover penalty. The solver uses proximal
gradient descent with signed per-asset bounds and gross-exposure projections.
It deliberately has no turnover quota.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    weights: dict[str, float]
    objective: float
    expected_edge_contribution: float
    variance: float
    transaction_cost_estimate: float
    covariance_condition: float
    iterations: int
    converged: bool
    method: str = "proximal_mean_variance_l1"


def _sample_covariance(
    pairs: Sequence[str],
    histories: Mapping[str, Sequence[float]],
    window_bars: int,
    horizon_bars: int,
    shrinkage: float,
    fallback_volatility: Mapping[str, float],
    volatility_window_bars: int,
    covariance_half_life_bars: int,
) -> np.ndarray:
    """Estimate horizon covariance from one-bar log returns with diagonal shrinkage."""
    returns_by_pair: list[np.ndarray] = []
    for pair in pairs:
        prices = np.asarray(histories.get(pair, ()), dtype=float)
        prices = prices[np.isfinite(prices) & (prices > 0)]
        prices = prices[-(max(2, int(window_bars)) + 1):]
        if prices.size >= 2:
            returns_by_pair.append(np.diff(np.log(prices)))
        else:
            returns_by_pair.append(np.asarray([], dtype=float))

    n = len(pairs)
    diag_fallback = np.asarray([
        max(float(fallback_volatility.get(pair, 0.01)), 1e-4) ** 2
        * max(int(horizon_bars), 1) / max(int(volatility_window_bars), 1)
        for pair in pairs
    ], dtype=float)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    lengths = [len(values) for values in returns_by_pair]
    common = min(lengths, default=0)
    if common < max(20, min(int(window_bars), 60)):
        covariance = np.diag(diag_fallback)
    else:
        matrix = np.column_stack([values[-common:] for values in returns_by_pair])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        # Exponentially weighted covariance reacts to changing crypto volatility
        # while still averaging enough bars to limit noise. Weight the latest row
        # most heavily; normalize with the effective unbiased denominator.
        half_life = max(1, int(covariance_half_life_bars))
        decay = math.exp(math.log(0.5) / half_life)
        ages = np.arange(common - 1, -1, -1, dtype=float)
        weights = np.power(decay, ages)
        weights /= max(float(weights.sum()), 1e-12)
        weighted_mean = weights @ matrix
        centered = matrix - weighted_mean
        correction = max(1.0 - float(weights @ weights), 1e-8)
        covariance = (centered * weights[:, None]).T @ centered / correction
        covariance = np.atleast_2d(covariance)
        if covariance.shape != (n, n) or not np.all(np.isfinite(covariance)):
            covariance = np.diag(diag_fallback)
        covariance *= max(int(horizon_bars), 1)
        # Shrink toward a diagonal target to reduce noisy cross-asset covariance.
        diagonal = np.diag(np.maximum(np.diag(covariance), diag_fallback * 1e-3))
        shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal

    covariance = (covariance + covariance.T) / 2.0
    # Enforce a positive-semidefinite matrix so the quadratic utility is well-behaved.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.median(np.maximum(np.diag(covariance), 0.0))) * 1e-6, 1e-10)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return (covariance + covariance.T) / 2.0


def optimize_signed_portfolio(
    long_signals: Sequence[object],
    short_signals: Sequence[object],
    histories: Mapping[str, Sequence[float]],
    previous_weights: Mapping[str, float] | None,
    *,
    long_budget: float,
    short_budget: float,
    total_gross_cap: float,
    long_asset_cap: float,
    short_asset_cap: float,
    covariance_window_bars: int,
    risk_horizon_bars: int,
    covariance_shrinkage: float = 0.65,
    volatility_window_bars: int = 60,
    covariance_half_life_bars: int = 120,
    risk_aversion: float = 60.0,
    cost_penalty: float = 1.0,
    iterations: int = 250,
) -> PortfolioOptimizationResult:
    """Return signed target weights (long positive, short negative).

    Utility is
        alpha @ abs(w) - (risk_aversion/2) * w.T @ Sigma @ w
        - cost_penalty * one_way_cost @ abs(w - w_previous)

    Candidate signs are fixed by the strategy's long/short ranking. Exposure
    budgets and per-asset caps are projected after each proximal-gradient step.
    """
    longs = list(long_signals)
    shorts = list(short_signals)
    # Candidates cannot be simultaneously long and short in the same asset.
    long_pairs = {str(getattr(s, "pair")) for s in longs}
    shorts = [s for s in shorts if str(getattr(s, "pair")) not in long_pairs]
    signals = longs + shorts
    pairs = [str(getattr(s, "pair")) for s in signals]
    if not pairs:
        return PortfolioOptimizationResult({}, 0.0, 0.0, 0.0, 0.0, 1.0, 0, True)

    signs = np.asarray([1.0] * len(longs) + [-1.0] * len(shorts), dtype=float)
    gross_alpha = np.asarray([
        max(0.0, float(getattr(s, "gross_expected_edge", 0.0)
                       or (float(getattr(s, "expected_edge", 0.0)) + float(getattr(s, "round_trip_cost", 0.0)))))
        for s in signals
    ], dtype=float)
    one_way_cost = np.asarray([
        max(0.0, float(getattr(s, "one_way_cost", 0.0)
                       or float(getattr(s, "round_trip_cost", 0.0)) / 2.0))
        for s in signals
    ], dtype=float)
    previous = np.asarray([
        float((previous_weights or {}).get(pair, 0.0)) for pair in pairs
    ], dtype=float)
    vol_fallback = {str(getattr(s, "pair")): float(getattr(s, "volatility", 0.01)) for s in signals}
    covariance = _sample_covariance(
        pairs, histories, covariance_window_bars, risk_horizon_bars,
        covariance_shrinkage, vol_fallback, volatility_window_bars, covariance_half_life_bars,
    )
    lam = max(0.0, float(risk_aversion))
    transaction_penalty = max(0.0, float(cost_penalty)) * one_way_cost
    lower = np.where(signs > 0, 0.0, -max(0.0, float(short_asset_cap)))
    upper = np.where(signs > 0, max(0.0, float(long_asset_cap)), 0.0)
    long_budget = max(0.0, float(long_budget))
    short_budget = max(0.0, float(short_budget))
    total_gross_cap = max(0.0, float(total_gross_cap))

    def project(values: np.ndarray) -> np.ndarray:
        out = np.clip(values, lower, upper)
        long_mask, short_mask = signs > 0, signs < 0
        long_sum = float(np.sum(out[long_mask]))
        if long_sum > long_budget > 0:
            out[long_mask] *= long_budget / long_sum
        elif long_sum > 0 and long_budget <= 0:
            out[long_mask] = 0.0
        short_sum = float(-np.sum(out[short_mask]))
        if short_sum > short_budget > 0:
            out[short_mask] *= short_budget / short_sum
        elif short_sum > 0 and short_budget <= 0:
            out[short_mask] = 0.0
        gross = float(np.sum(np.abs(out)))
        if gross > total_gross_cap > 0:
            out *= total_gross_cap / gross
        elif gross > 0 and total_gross_cap <= 0:
            out[:] = 0.0
        return out

    # The alpha term is linear in absolute exposure because candidate direction is fixed.
    signed_alpha = signs * gross_alpha
    prev_feasible = project(previous)
    w = prev_feasible.copy()
    max_eigenvalue = max(float(np.linalg.eigvalsh(covariance)[-1]), 1e-10)
    lipschitz = lam * max_eigenvalue
    step = min(25.0, 0.95 / max(lipschitz, 0.038))
    converged = False
    actual_iterations = max(1, int(iterations))
    for iteration in range(actual_iterations):
        gradient = lam * (covariance @ w) - signed_alpha
        z = w - step * gradient
        # Proximal operator for c * |w - w_previous|.
        delta = z - previous
        w_new = previous + np.sign(delta) * np.maximum(np.abs(delta) - step * transaction_penalty, 0.0)
        w_new = project(w_new)
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            converged = True
            break
        w = w_new

    edge_contribution = float(gross_alpha @ np.abs(w))
    variance = max(0.0, float(w @ covariance @ w))
    cost_estimate = float(one_way_cost @ np.abs(w - previous))
    objective = edge_contribution - 0.5 * lam * variance - max(0.0, float(cost_penalty)) * cost_estimate
    eigen = np.linalg.eigvalsh(covariance)
    positive_eigen = eigen[eigen > 1e-12]
    condition = float(positive_eigen.max() / positive_eigen.min()) if positive_eigen.size else math.inf
    out = {pair: float(value) for pair, value in zip(pairs, w) if abs(value) > 1e-7}
    return PortfolioOptimizationResult(
        weights=out,
        objective=float(objective),
        expected_edge_contribution=edge_contribution,
        variance=variance,
        transaction_cost_estimate=cost_estimate,
        covariance_condition=condition,
        iterations=iteration + 1,
        converged=converged,
    )
