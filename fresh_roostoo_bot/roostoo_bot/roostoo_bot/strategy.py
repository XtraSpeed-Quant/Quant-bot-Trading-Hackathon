from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .indicators import ema, pct_return, rank01, realized_volatility, rolling_max, rolling_mean
from .predictor import Prediction, PredictorResult, PredictiveEnsemble


@dataclass(frozen=True)
class Market:
    pair: str
    coin: str
    unit: str
    price: float
    bid: float
    ask: float
    change_24h: float
    quote_volume_24h: float
    amount_precision: int
    price_precision: int
    min_order: float
    can_trade: bool

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0:
            return math.inf
        return (self.ask - self.bid) / mid * 10_000.0


@dataclass(frozen=True)
class Signal:
    pair: str
    direction: str
    score: float
    fast_return: float
    medium_return: float
    long_return: float
    return_24h: float
    trend: float
    volatility: float
    relative_4h: float
    relative_12h: float
    pullback: float
    breakout: float
    overextension: float
    predicted_return: float
    probability_up: float
    probability_down: float
    prediction_confidence: float
    expected_edge: float
    correlation_penalty: float
    target_weight: float
    reason: str


@dataclass(frozen=True)
class StrategyResult:
    long_targets: dict[str, float]
    short_targets: dict[str, float]
    signals: list[Signal]
    regime: str
    breadth: float
    median_4h: float
    median_12h: float
    median_24h: float
    market_strength: float
    gross_target: float
    short_gross_target: float
    median_volatility: float
    volatility_scale: float
    predictor_enabled: bool
    predictor_samples: int
    predictor_directional_accuracy: float
    predictor_correlation: float
    predictor_balanced_accuracy: float
    predictor_top_decile_edge: float
    predictor_brier: float
    predictor_gate_reason: str


def _empty_predictor() -> PredictorResult:
    return PredictorResult(
        enabled=False,
        samples=0,
        validation_directional_accuracy=0.0,
        validation_correlation=0.0,
        validation_balanced_accuracy=0.0,
        validation_top_decile_edge=0.0,
        validation_brier=1.0,
        residual_std=0.0,
        predictions={},
    )


def _empty_result() -> StrategyResult:
    return StrategyResult(
        long_targets={},
        short_targets={},
        signals=[],
        regime="defensive",
        breadth=0.0,
        median_4h=0.0,
        median_12h=0.0,
        median_24h=0.0,
        market_strength=0.0,
        gross_target=0.0,
        short_gross_target=0.0,
        median_volatility=0.0,
        volatility_scale=0.0,
        predictor_enabled=False,
        predictor_samples=0,
        predictor_directional_accuracy=0.0,
        predictor_correlation=0.0,
        predictor_balanced_accuracy=0.0,
        predictor_top_decile_edge=0.0,
        predictor_brier=1.0,
        predictor_gate_reason="warming up",
    )


class MultiFactorStrategy:
    def __init__(
        self,
        *,
        fast_window: int,
        medium_window: int,
        long_window: int,
        trend_window: int,
        vol_window: int,
        top_n: int,
        max_gross_target: float,
        max_asset_weight: float,
        max_short_gross: float,
        max_short_weight: float,
        min_score: float,
        min_relative_edge: float,
        min_1h_return: float,
        min_4h_return: float,
        min_12h_return: float,
        min_quote_volume: float,
        max_spread_bps: float,
        volatility_floor: float,
        predictor_enabled: bool,
        prediction_horizon: int,
        predictor_feature_window: int,
        predictor_min_samples: int,
        predictor_max_samples: int,
        predictor_ridge_alpha: float,
        predictor_classifier_alpha: float,
        predictor_validation_fraction: float,
        predictor_purge_gap: int,
        predictor_min_directional_accuracy: float,
        predictor_min_correlation: float,
        predictor_min_balanced_accuracy: float,
        predictor_min_top_decile_edge: float,
        predictor_class_threshold: float,
        predictor_max_brier: float,
        min_prediction_edge: float,
        prediction_retrain_seconds: int,
        correlation_window: int,
        max_pair_correlation: float,
        correlation_penalty: float,
        overextension_penalty: float,
        breakout_window: int,
        taker_fee_assumption: float,
        enable_shorts: bool,
        cash_reserve: float,
        gross_strength_power: float,
        risk_on_gross_cap: float,
        balanced_gross_cap: float,
        risk_off_gross_cap: float,
        defensive_gross_cap: float,
        defensive_min_breadth: float,
        defensive_min_strength: float,
        allocation_edge_power: float,
        allocation_vol_power: float,
        scan_top_n: int,
    ) -> None:
        self.fast_window = fast_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.top_n = top_n
        self.max_gross_target = max_gross_target
        self.max_asset_weight = max_asset_weight
        self.max_short_gross = max_short_gross
        self.max_short_weight = max_short_weight
        self.min_score = min_score
        self.min_relative_edge = min_relative_edge
        self.min_1h_return = min_1h_return
        self.min_4h_return = min_4h_return
        self.min_12h_return = min_12h_return
        self.min_quote_volume = min_quote_volume
        self.max_spread_bps = max_spread_bps
        self.volatility_floor = volatility_floor
        self.enable_shorts = enable_shorts
        self.cash_reserve = cash_reserve
        self.min_prediction_edge = min_prediction_edge
        self.prediction_retrain_seconds = prediction_retrain_seconds
        self.correlation_window = correlation_window
        self.max_pair_correlation = max_pair_correlation
        self.correlation_penalty = correlation_penalty
        self.overextension_penalty = overextension_penalty
        self.breakout_window = breakout_window
        self.taker_fee_assumption = taker_fee_assumption
        self.gross_strength_power = gross_strength_power
        self.risk_on_gross_cap = risk_on_gross_cap
        self.balanced_gross_cap = balanced_gross_cap
        self.risk_off_gross_cap = risk_off_gross_cap
        self.defensive_gross_cap = defensive_gross_cap
        self.defensive_min_breadth = defensive_min_breadth
        self.defensive_min_strength = defensive_min_strength
        self.allocation_edge_power = allocation_edge_power
        self.allocation_vol_power = allocation_vol_power
        self.scan_top_n = scan_top_n
        self.predictor = (
            PredictiveEnsemble(
                predictor_feature_window,
                prediction_horizon,
                predictor_min_samples,
                predictor_max_samples,
                predictor_ridge_alpha,
                predictor_classifier_alpha,
                predictor_validation_fraction,
                predictor_purge_gap,
                predictor_min_directional_accuracy,
                predictor_min_correlation,
                predictor_min_balanced_accuracy,
                predictor_min_top_decile_edge,
                predictor_class_threshold,
                taker_fee_assumption,
                predictor_max_brier,
            )
            if predictor_enabled
            else None
        )
        self._last_predictor_result: PredictorResult | None = None
        self._last_predictor_fit_ms = 0

    @property
    def last_predictor_result(self) -> PredictorResult | None:
        return self._last_predictor_result

    @property
    def required_history(self) -> int:
        return max(
            self.long_window,
            self.trend_window,
            self.vol_window,
            self.breakout_window,
            self.predictor.minimum_history if self.predictor else 0,
        ) + 1

    def _feature(self, market: Market, prices: Sequence[float]) -> dict[str, float] | None:
        if len(prices) < max(self.long_window, self.trend_window, self.vol_window, self.breakout_window) + 1:
            return None
        if not market.can_trade or market.unit.upper() != "USD":
            return None
        if market.price <= 0 or market.bid <= 0 or market.ask <= 0 or market.bid > market.ask:
            return None
        if market.quote_volume_24h < self.min_quote_volume or market.spread_bps > self.max_spread_bps:
            return None

        fast = pct_return(prices, self.fast_window)
        medium = pct_return(prices, self.medium_window)
        long_ret = pct_return(prices, self.long_window)
        trend_ema = ema(prices, self.trend_window)
        vol = realized_volatility(prices, self.vol_window)
        previous_prices = prices[:-1]
        previous_high = rolling_max(previous_prices, self.breakout_window)
        mean_fast = rolling_mean(prices, self.fast_window)
        if None in (fast, medium, long_ret, trend_ema, vol, previous_high, mean_fast):
            return None

        vol_v = max(float(vol), self.volatility_floor)
        trend = market.price / float(trend_ema) - 1.0
        extension = max(0.0, float(fast) - 0.35 * max(float(medium), 0.0))
        pullback = max(0.0, -float(fast)) if float(medium) > 0 and trend > 0 else 0.0
        breakout = market.price / max(float(previous_high), 1e-12) - 1.0
        mean_reversion = market.price / max(float(mean_fast), 1e-12) - 1.0
        return {
            "fast": float(fast),
            "medium": float(medium),
            "long": float(long_ret),
            "return_24h": float(market.change_24h),
            "trend": trend,
            "vol": vol_v,
            "liquidity": math.log1p(max(market.quote_volume_24h, 0.0)),
            "spread": market.spread_bps,
            "extension": extension,
            "pullback": pullback,
            "breakout": breakout,
            "mean_reversion": mean_reversion,
        }

    @staticmethod
    def _market_strength(
        breadth: float,
        median_4h: float,
        median_12h: float,
        median_24h: float,
        median_vol: float,
    ) -> float:
        breadth_component = np.clip((breadth - 0.20) / 0.60, 0.0, 1.0)
        m4 = np.clip((median_4h + 0.008) / 0.028, 0.0, 1.0)
        m12 = np.clip((median_12h + 0.015) / 0.050, 0.0, 1.0)
        m24 = np.clip((median_24h + 0.025) / 0.070, 0.0, 1.0)
        vol_drag = np.clip((median_vol - 0.004) / 0.010, 0.0, 1.0)
        return float(
            np.clip(
                0.40 * breadth_component
                + 0.25 * m4
                + 0.18 * m12
                + 0.17 * m24
                - 0.12 * vol_drag,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _regime(strength: float) -> str:
        if strength >= 0.72:
            return "risk_on"
        if strength >= 0.50:
            return "balanced"
        if strength >= 0.28:
            return "risk_off"
        return "defensive"

    def _pair_corr(self, histories: Mapping[str, Sequence[float]], a: str, b: str) -> float:
        va = np.asarray(histories.get(a, ()), dtype=float)
        vb = np.asarray(histories.get(b, ()), dtype=float)
        if len(va) < self.correlation_window + 1 or len(vb) < self.correlation_window + 1:
            return 0.0
        ra = np.diff(np.log(np.maximum(va[-self.correlation_window - 1 :], 1e-12)))
        rb = np.diff(np.log(np.maximum(vb[-self.correlation_window - 1 :], 1e-12)))
        if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
            return 0.0
        value = float(np.corrcoef(ra, rb)[0, 1])
        return value if math.isfinite(value) else 0.0

    def _diversify(self, candidates: list[Signal], histories: Mapping[str, Sequence[float]]) -> list[Signal]:
        chosen: list[Signal] = []
        for candidate in candidates:
            if len(chosen) >= self.top_n:
                break
            if not chosen:
                chosen.append(candidate)
                continue
            corrs = [abs(self._pair_corr(histories, candidate.pair, c.pair)) for c in chosen]
            max_corr = max(corrs, default=0.0)
            avg_corr = float(np.mean(corrs)) if corrs else 0.0
            weakest = min(chosen, key=lambda x: x.score)
            if max_corr > self.max_pair_correlation and candidate.score - weakest.score < 0.06:
                continue
            penalty = self.correlation_penalty * max(0.0, avg_corr - 0.50)
            reason = candidate.reason
            if penalty > 0:
                reason += "+correlation_penalty"
            chosen.append(
                Signal(
                    **{
                        **candidate.__dict__,
                        "correlation_penalty": penalty,
                        "score": max(0.0, candidate.score - penalty),
                        "reason": reason,
                    }
                )
            )
        return chosen

    def _predictor(self, histories: Mapping[str, Sequence[float]], now_ms: int) -> PredictorResult:
        if self.predictor is None:
            result = _empty_predictor()
            self._last_predictor_result = result
            return result
        needs_fit = (
            self._last_predictor_result is None
            or self._last_predictor_fit_ms == 0
            or now_ms - self._last_predictor_fit_ms >= self.prediction_retrain_seconds * 1000
        )
        result = self.predictor.fit_and_predict(histories, refit=needs_fit)
        if needs_fit:
            self._last_predictor_fit_ms = now_ms
        self._last_predictor_result = result
        return result

    def analyze(
        self,
        histories: Mapping[str, Sequence[float]],
        markets: Mapping[str, Market],
        now_ms: int = 0,
    ) -> StrategyResult:
        features = {
            pair: feature
            for pair, market in markets.items()
            if (feature := self._feature(market, histories.get(pair, ()))) is not None
        }
        if not features:
            return _empty_result()

        median_4h = float(np.median([v["medium"] for v in features.values()]))
        median_12h = float(np.median([v["long"] for v in features.values()]))
        median_24h = float(np.median([v["return_24h"] for v in features.values()]))
        median_vol = float(np.median([v["vol"] for v in features.values()]))
        breadth = float(
            np.mean(
                [
                    v["medium"] > 0
                    and v["long"] > 0
                    and v["return_24h"] > 0
                    and v["trend"] > 0
                    for v in features.values()
                ]
            )
        )
        strength = self._market_strength(breadth, median_4h, median_12h, median_24h, median_vol)
        regime = self._regime(strength)
        vol_scale = float(np.clip(0.0030 / max(median_vol, 0.0010), 0.55, 1.10))

        fast_rank = rank01({p: f["fast"] for p, f in features.items()})
        med_rank = rank01({p: f["medium"] for p, f in features.items()})
        long_rank = rank01({p: f["long"] for p, f in features.items()})
        ret24_rank = rank01({p: f["return_24h"] for p, f in features.items()})
        trend_rank = rank01({p: f["trend"] for p, f in features.items()})
        liq_rank = rank01({p: f["liquidity"] for p, f in features.items()})
        spread_rank = rank01({p: f["spread"] for p, f in features.items()})
        vol_rank = rank01({p: f["vol"] for p, f in features.items()})
        pull_rank = rank01({p: f["pullback"] for p, f in features.items()})
        breakout_rank = rank01({p: f["breakout"] for p, f in features.items()})
        relative4_rank = rank01({p: f["medium"] - median_4h for p, f in features.items()})
        relative12_rank = rank01({p: f["long"] - median_12h for p, f in features.items()})

        predictor_result = self._predictor(histories, now_ms or 1)
        prediction_rank = (
            rank01({p: pr.predicted_return for p, pr in predictor_result.predictions.items()})
            if predictor_result.predictions
            else {}
        )

        preliminary: list[Signal] = []
        shorts: list[Signal] = []
        for pair, item in features.items():
            relative_4h = item["medium"] - median_4h
            relative_12h = item["long"] - median_12h
            expected_edge = (
                0.26 * relative_4h
                + 0.22 * relative_12h
                + 0.18 * item["fast"]
                + 0.14 * item["trend"]
                + 0.12 * item["return_24h"]
                + 0.08 * item["breakout"]
            )
            prediction = predictor_result.predictions.get(
                pair,
                Prediction(0.0, 0.5, 1.0, 0.5, 0.0, 0.0),
            )
            round_trip_cost = 2.0 * self.taker_fee_assumption + item["spread"] / 10_000.0
            prediction_edge = prediction.predicted_return - round_trip_cost

            trend_score = (
                0.18 * fast_rank[pair]
                + 0.26 * med_rank[pair]
                + 0.18 * long_rank[pair]
                + 0.20 * trend_rank[pair]
                + 0.18 * ret24_rank[pair]
            )
            relative_score = 0.56 * relative4_rank[pair] + 0.44 * relative12_rank[pair]
            quality = (
                0.45 * liq_rank[pair]
                + 0.25 * (1.0 - spread_rank[pair])
                + 0.20 * (1.0 - vol_rank[pair])
                + 0.10 * (1.0 - pull_rank[pair])
            )
            continuation = 0.60 * pull_rank[pair] + 0.40 * breakout_rank[pair]
            prediction_component = prediction_rank.get(pair, 0.5)
            if predictor_result.enabled:
                prediction_component = 0.60 * prediction_component + 0.40 * prediction.confidence
            else:
                prediction_component = 0.5

            if regime == "risk_on":
                score = (
                    0.28 * trend_score
                    + 0.25 * relative_score
                    + 0.10 * continuation
                    + 0.16 * prediction_component
                    + 0.11 * quality
                    + 0.10 * rank01({p: f["breakout"] for p, f in features.items()})[pair]
                )
            elif regime == "balanced":
                score = (
                    0.25 * trend_score
                    + 0.26 * relative_score
                    + 0.12 * continuation
                    + 0.19 * prediction_component
                    + 0.18 * quality
                )
            elif regime == "risk_off":
                score = (
                    0.18 * trend_score
                    + 0.24 * relative_score
                    + 0.18 * continuation
                    + 0.22 * prediction_component
                    + 0.18 * quality
                )
            else:
                score = (
                    0.14 * trend_score
                    + 0.16 * relative_score
                    + 0.20 * continuation
                    + 0.24 * prediction_component
                    + 0.26 * quality
                )

            extension_penalty = min(
                0.30,
                self.overextension_penalty * max(0.0, item["extension"] / 0.008),
            )
            score = float(max(0.0, score - extension_penalty))

            long_ok = (
                item["fast"] >= self.min_1h_return
                and item["medium"] >= self.min_4h_return
                and item["long"] >= self.min_12h_return
                and item["trend"] > 0
                and expected_edge >= self.min_relative_edge
                and score >= self.min_score
            )
            if predictor_result.enabled and prediction.confidence >= 0.45:
                long_ok = long_ok and prediction_edge >= self.min_prediction_edge

            reasons: list[str] = []
            if item["trend"] > 0:
                reasons.append("trend")
            if relative_4h > 0 or relative_12h > 0:
                reasons.append("relative_strength")
            if item["pullback"] > 0:
                reasons.append("pullback")
            if item["breakout"] > 0:
                reasons.append("breakout")
            if predictor_result.enabled and prediction_edge >= self.min_prediction_edge:
                reasons.append("prediction_edge")
            if extension_penalty > 0:
                reasons.append("extension_penalty")
            reasons.append("liquid")

            if long_ok:
                preliminary.append(
                    Signal(
                        pair,
                        "LONG",
                        score,
                        item["fast"],
                        item["medium"],
                        item["long"],
                        item["return_24h"],
                        item["trend"],
                        item["vol"],
                        relative_4h,
                        relative_12h,
                        item["pullback"],
                        item["breakout"],
                        item["extension"],
                        prediction.predicted_return,
                        prediction.probability_up,
                        prediction.probability_down,
                        prediction.confidence,
                        expected_edge,
                        0.0,
                        0.0,
                        "+".join(reasons),
                    )
                )

            if self.enable_shorts:
                downside_edge = -(
                    0.26 * relative_4h
                    + 0.22 * relative_12h
                    + 0.18 * item["fast"]
                    + 0.14 * item["trend"]
                    + 0.12 * item["return_24h"]
                    + 0.08 * item["breakout"]
                )
                short_score = 1.0 - score
                if predictor_result.enabled:
                    short_score = 0.65 * short_score + 0.35 * prediction.probability_down
                short_ok = (
                    regime in {"balanced", "risk_off", "defensive"}
                    and item["fast"] <= -self.min_1h_return
                    and item["medium"] <= -self.min_4h_return
                    and item["trend"] < 0
                    and downside_edge >= self.min_relative_edge
                    and short_score >= self.min_score
                )
                if predictor_result.enabled and prediction.confidence >= 0.45:
                    short_ok = short_ok and prediction.probability_down > prediction.probability_up and prediction_edge <= -self.min_prediction_edge
                if short_ok:
                    shorts.append(
                        Signal(
                            pair,
                            "SHORT",
                            short_score,
                            item["fast"],
                            item["medium"],
                            item["long"],
                            item["return_24h"],
                            item["trend"],
                            item["vol"],
                            relative_4h,
                            relative_12h,
                            item["pullback"],
                            item["breakout"],
                            item["extension"],
                            prediction.predicted_return,
                            prediction.probability_up,
                            prediction.probability_down,
                            prediction.confidence,
                            downside_edge,
                            0.0,
                            0.0,
                            "trend+relative_weakness+downside_signal+liquid",
                        )
                    )

        preliminary.sort(key=lambda s: (s.score, s.expected_edge), reverse=True)
        shorts.sort(key=lambda s: (s.score, s.expected_edge), reverse=True)
        chosen_longs = self._diversify(preliminary, histories)
        chosen_shorts = self._diversify(shorts, histories) if self.enable_shorts else []

        if breadth < self.defensive_min_breadth or strength < self.defensive_min_strength:
            gross = 0.0
        else:
            regime_cap = {
                "risk_on": self.risk_on_gross_cap,
                "balanced": self.balanced_gross_cap,
                "risk_off": self.risk_off_gross_cap,
                "defensive": self.defensive_gross_cap,
            }.get(regime, self.defensive_gross_cap)
            market_budget = strength ** self.gross_strength_power
            gross = min(
                self.max_gross_target * market_budget * vol_scale,
                self.max_gross_target * regime_cap,
                1.0 - self.cash_reserve if hasattr(self, "cash_reserve") else 0.90,
            )

        def allocate(selected: list[Signal], total: float, cap: float) -> dict[str, float]:
            if not selected or total <= 0:
                return {}
            raw: dict[str, float] = {}
            for signal in selected:
                edge = max(signal.score - self.min_score, 0.015)
                confidence_boost = 1.0 + (0.40 * signal.prediction_confidence if predictor_result.enabled else 0.0)
                raw[signal.pair] = (
                    (edge ** self.allocation_edge_power)
                    * confidence_boost
                    / max(signal.volatility, self.volatility_floor) ** self.allocation_vol_power
                )
            weights = {pair: 0.0 for pair in raw}
            active = set(raw)
            remaining = total
            while active and remaining > 1e-12:
                denominator = sum(raw[pair] for pair in active)
                capped = set()
                for pair in list(active):
                    proposed = remaining * raw[pair] / max(denominator, 1e-12)
                    if proposed >= cap:
                        weights[pair] = cap
                        remaining -= cap
                        capped.add(pair)
                if not capped:
                    for pair in active:
                        weights[pair] = remaining * raw[pair] / max(denominator, 1e-12)
                    break
                active -= capped
            return {pair: weight for pair, weight in weights.items() if weight > 1e-8}

        long_targets = allocate(chosen_longs, gross, self.max_asset_weight)
        strength_short = max(0.0, 0.40 - strength)
        short_gross = min(self.max_short_gross, strength_short * 0.65) if self.enable_shorts else 0.0
        if regime == "defensive" and self.enable_shorts:
            short_gross = self.max_short_gross
        elif regime == "balanced" and self.enable_shorts:
            short_gross = min(short_gross, 0.06)
        max_combined = max(0.0, 1.0 - 0.10)
        if sum(long_targets.values()) + short_gross > max_combined:
            short_gross = max(0.0, max_combined - sum(long_targets.values()))
        short_targets = allocate(chosen_shorts, short_gross, self.max_short_weight)

        weighted: list[Signal] = []
        for signal in chosen_longs + chosen_shorts:
            target_map = long_targets if signal.direction == "LONG" else short_targets
            weighted.append(
                Signal(
                    **{
                        **signal.__dict__,
                        "target_weight": target_map.get(signal.pair, 0.0),
                    }
                )
            )
        weighted.sort(key=lambda s: (s.direction, -s.score))

        return StrategyResult(
            long_targets=long_targets,
            short_targets=short_targets,
            signals=weighted,
            regime=regime,
            breadth=breadth,
            median_4h=median_4h,
            median_12h=median_12h,
            median_24h=median_24h,
            market_strength=strength,
            gross_target=sum(long_targets.values()),
            short_gross_target=sum(short_targets.values()),
            median_volatility=median_vol,
            volatility_scale=vol_scale,
            predictor_enabled=predictor_result.enabled,
            predictor_samples=predictor_result.samples,
            predictor_directional_accuracy=predictor_result.validation_directional_accuracy,
            predictor_correlation=predictor_result.validation_correlation,
            predictor_balanced_accuracy=predictor_result.validation_balanced_accuracy,
            predictor_top_decile_edge=predictor_result.validation_top_decile_edge,
            predictor_brier=predictor_result.validation_brier,
            predictor_gate_reason=predictor_result.gate_reason,
        )
