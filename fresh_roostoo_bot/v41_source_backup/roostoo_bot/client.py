from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RoostooError(RuntimeError):
    pass


def signature(params: Mapping[str, Any], secret: str) -> str:
    payload = "&".join(f"{key}={params[key]}" for key in sorted(params))
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


class RoostooClient:
    def __init__(self, base_url: str, api_key: str = "", api_secret: str = "", timeout: float = 12.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.clock_offset_ms = 0
        self.session = requests.Session()
        self.log = logging.getLogger(self.__class__.__name__)
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request_timeout(self) -> tuple[float, float]:
        connect = min(4.0, max(1.0, self.timeout / 3.0))
        return connect, max(connect, self.timeout)

    def _get(self, path: str, params: Mapping[str, Any] | None = None, signed: bool = False) -> dict[str, Any]:
        payload = dict(params or {})
        headers: dict[str, str] = {}
        if signed:
            payload["timestamp"] = self._timestamp()
            headers = self._signed_headers(payload)
        try:
            response = self.session.get(
                self.base_url + path,
                params=payload,
                headers=headers,
                timeout=self._request_timeout(),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            self.log.error("GET %s failed: %s", path, exc)
            raise
        self._raise_api_error(data, path)
        return data

    def _post(self, path: str, payload: Mapping[str, Any], retry_safe: bool = False) -> dict[str, Any]:
        data = dict(payload)
        data["timestamp"] = self._timestamp()
        headers = self._signed_headers(data)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        encoded = "&".join(f"{key}={data[key]}" for key in sorted(data))

        attempts = 2 if retry_safe else 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.post(
                    self.base_url + path,
                    data=encoded,
                    headers=headers,
                    timeout=self._request_timeout(),
                )
                response.raise_for_status()
                result = response.json()
                self._raise_api_error(result, path)
                return result
            except (requests.RequestException, ValueError, RoostooError) as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    self.log.error("POST %s failed: %s", path, exc)
                    raise
                time.sleep(0.5)
        raise RuntimeError(str(last_exc))

    @staticmethod
    def _raise_api_error(data: Mapping[str, Any], path: str) -> None:
        if data.get("Success") is False:
            raise RoostooError(f"{path}: {data.get('ErrMsg', 'Roostoo request failed')}")

    def _signed_headers(self, params: Mapping[str, Any]) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise ValueError("Signed Roostoo endpoints require API credentials")
        return {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature(params, self.api_secret),
        }

    def _timestamp(self) -> int:
        return int(time.time() * 1000) + self.clock_offset_ms

    def server_time(self) -> int:
        return int(self._get("/v3/serverTime")["ServerTime"])

    def sync_time(self) -> int:
        started = int(time.time() * 1000)
        server = self.server_time()
        finished = int(time.time() * 1000)
        self.clock_offset_ms = server - ((started + finished) // 2)
        return server

    def exchange_info(self) -> dict[str, Any]:
        return self._get("/v3/exchangeInfo")

    def tickers(self) -> dict[str, Any]:
        return self._get("/v3/ticker", {"timestamp": self._timestamp()})

    def balance(self) -> dict[str, dict[str, float]]:
        result = self._get("/v3/balance", signed=True)
        wallet = result.get("SpotWallet")
        if wallet is None:
            wallet = result.get("Wallet", {})
        if not isinstance(wallet, dict):
            raise RoostooError("/v3/balance returned an invalid wallet")
        return wallet

    def pending_count(self) -> int:
        try:
            result = self._get("/v3/pending_count", signed=True)
        except RoostooError as exc:
            if "no pending order" in str(exc).lower():
                return 0
            raise
        return int(result.get("TotalPending", 0) or 0)

    def place_market_order(self, pair: str, side: str, quantity: str) -> dict[str, Any]:
        return self._post("/v3/place_order", {
            "pair": pair,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
        }, retry_safe=False)

    def query_order(self, order_id: int | str | None = None, pair: str | None = None, pending_only: bool | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)
        elif pending_only is not None:
            payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)
        try:
            result = self._post("/v3/query_order", payload, retry_safe=True)
        except RoostooError as exc:
            if "no order matched" in str(exc).lower():
                return []
            raise
        matched = result.get("OrderMatched", [])
        return matched if isinstance(matched, list) else []

    def cancel_order(self, order_id: int | str | None = None, pair: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        return self._post("/v3/cancel_order", payload, retry_safe=False)

    def short_open(self, pair: str, collateral: str) -> dict[str, Any]:
        return self._post("/v6/short_open", {"pair": pair, "collateral": collateral}, retry_safe=False)

    def short_close(self, pair: str, close_qty: str | None = None, close_pct: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            payload["close_qty"] = close_qty
        elif close_pct is not None:
            payload["close_pct"] = close_pct
        return self._post("/v6/short_close", payload, retry_safe=False)

    def short_positions(self) -> list[dict[str, Any]]:
        result = self._get("/v6/short_positions", signed=True)
        positions = result.get("Positions", [])
        return positions if isinstance(positions, list) else []
