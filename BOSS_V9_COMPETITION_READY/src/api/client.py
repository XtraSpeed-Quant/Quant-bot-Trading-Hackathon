"""
Roostoo API Client
Handles public, spot-order, and collateralized-short endpoints for the Roostoo competition API.
"""

import hmac
import hashlib
import time
import os
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://mock-api.roostoo.com"


class RoostooClient:
    def __init__(self):
        self.api_key = os.getenv("ROOSTOO_API_KEY", "")
        self.secret_key = os.getenv("ROOSTOO_API_SECRET", "")
        self.base_url = os.getenv("ROOSTOO_API_URL", BASE_URL).rstrip("/")
        self.timeout_seconds = max(1.0, float(os.getenv("TIMEOUT_SECONDS", "12")))
        # API request pacing only; this does not cap trading turnover or number of orders.
        self.min_request_interval = max(0.0, float(os.getenv("API_MIN_REQUEST_INTERVAL_SECONDS", "0.20")))
        self._last_request_monotonic = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Roostoo-BOSS-V9/1.0"})


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pace(self) -> None:
        now = time.monotonic()
        wait = self.min_request_interval - (now - self._last_request_monotonic)
        if wait > 0:
            time.sleep(wait)
        self._last_request_monotonic = time.monotonic()

    def _timestamp(self) -> str:
        """13-digit millisecond timestamp as string."""
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> tuple[dict, str, str]:
        """
        Add timestamp, sort params, build totalParams string, and sign with HMAC-SHA256.
        Returns (params_with_ts, total_params_string, signature).
        """
        params["timestamp"] = self._timestamp()
        sorted_keys = sorted(params.keys())
        total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return params, total_params, signature

    def _signed_headers(self, signature: str) -> dict:
        return {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }

    def _get(self, path: str, params: dict = None, signed: bool = False) -> dict:
        """Send a GET request, optionally signed."""
        url = f"{self.base_url}{path}"
        params = params or {}

        if signed:
            params, _, signature = self._sign(params)
            headers = self._signed_headers(signature)
        else:
            headers = {}

        try:
            self._pace()
            resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("Success", True):  # public endpoints don't have Success key
                logger.warning("API error on GET %s: %s", path, data.get("ErrMsg"))
            return data
        except requests.RequestException as e:
            logger.error("GET %s failed: %s", path, e)
            return {}

    def _post(self, path: str, payload: dict = None) -> dict:
        """Send a signed POST request with form-encoded body."""
        url = f"{self.base_url}{path}"
        payload = payload or {}
        payload, total_params, signature = self._sign(payload)

        headers = self._signed_headers(signature)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            self._pace()
            resp = self.session.post(url, data=total_params, headers=headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("Success", True):
                logger.warning("API error on POST %s: %s", path, data.get("ErrMsg"))
            return data
        except requests.RequestException as e:
            logger.error("POST %s failed: %s", path, e)
            return {}

    # ------------------------------------------------------------------
    # Public endpoints (no auth)
    # ------------------------------------------------------------------

    def server_time(self) -> dict:
        """GET /v3/serverTime -- no auth required."""
        return self._get("/v3/serverTime")

    def exchange_info(self) -> dict:
        """GET /v3/exchangeInfo -- no auth required."""
        return self._get("/v3/exchangeInfo")

    def ticker(self, pair: Optional[str] = None) -> dict:
        """GET /v3/ticker -- timestamp required (RCL_TSCheck)."""
        params = {"timestamp": self._timestamp()}
        if pair:
            params["pair"] = pair
        # RCL_TSCheck: no signature, just timestamp
        url = f"{self.base_url}/v3/ticker"
        try:
            self._pace()
            resp = self.session.get(url, params=params, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error("Ticker request failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Signed GET endpoints (RCL_TopLevelCheck)
    # ------------------------------------------------------------------

    def balance(self) -> dict:
        """GET /v3/balance."""
        return self._get("/v3/balance", signed=True)

    def pending_count(self) -> dict:
        """GET /v3/pending_count."""
        return self._get("/v3/pending_count", signed=True)

    def short_positions(self) -> dict:
        """GET /v6/short_positions."""
        return self._get("/v6/short_positions", signed=True)

    # ------------------------------------------------------------------
    # Signed POST endpoints (RCL_TopLevelCheck)
    # ------------------------------------------------------------------

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> dict:
        """
        POST /v3/place_order
        side: 'BUY' | 'SELL'
        order_type: 'MARKET' | 'LIMIT'
        price: required for LIMIT orders
        """
        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price")
            payload["price"] = str(price)

        logger.info("Placing %s %s order: %s qty=%s", order_type, side, pair, quantity)
        return self._post("/v3/place_order", payload)

    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: bool = False,
        limit: int = 100,
    ) -> dict:
        """POST /v3/query_order."""
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair
            if pending_only:
                payload["pending_only"] = "TRUE"
            payload["limit"] = str(limit)
        return self._post("/v3/query_order", payload)

    def cancel_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
    ) -> dict:
        """POST /v3/cancel_order."""
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        # no args = cancel all pending
        return self._post("/v3/cancel_order", payload)

    def short_open(
        self,
        pair: str,
        collateral: float,
        price: Optional[float] = None,
    ) -> dict:
        """POST /v6/short_open."""
        payload = {"pair": pair, "collateral": str(collateral)}
        if price is not None:
            payload["order_type"] = "LIMIT"
            payload["price"] = str(price)
        return self._post("/v6/short_open", payload)

    def short_close(
        self,
        pair: str,
        close_qty: Optional[float] = None,
        close_pct: Optional[float] = None,
    ) -> dict:
        """POST /v6/short_close. No args = close full position."""
        payload = {"pair": pair}
        if close_qty is not None:
            payload["close_qty"] = str(close_qty)
        elif close_pct is not None:
            payload["close_pct"] = str(close_pct)
        return self._post("/v6/short_close", payload)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_price(self, pair: str) -> Optional[float]:
        """Return the last price for a pair, or None on failure."""
        data = self.ticker(pair)
        return data.get("Data", {}).get(pair, {}).get("LastPrice")

    def get_all_tickers(self) -> dict:
        """Return the Data dict of all pair tickers."""
        data = self.ticker()
        return data.get("Data", {})

    def get_usd_balance(self) -> float:
        """Return free USD balance."""
        data = self.balance()
        # API returns SpotWallet (live) or Wallet (docs example)
        wallet = data.get("SpotWallet") or data.get("Wallet", {})
        return wallet.get("USD", {}).get("Free", 0.0)

    def get_coin_balance(self, coin: str) -> float:
        """Return free balance of a specific coin (e.g. 'BTC')."""
        data = self.balance()
        wallet = data.get("SpotWallet") or data.get("Wallet", {})
        return wallet.get(coin.upper(), {}).get("Free", 0.0)
