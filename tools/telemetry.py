"""One venue per run. No synthetic fallback prices, funding or sentiment."""
import asyncio
import math
import time
import urllib.request
import json
import pandas as pd
import ta


def indicators(rows, timeframe, now):
    durations = {"15m": 900, "1h": 3600, "1d": 86400}
    # Exclude the still-forming candle to prevent retrospective feature changes.
    rows = [r for r in rows if r[0] / 1000 + durations[timeframe] <= now]
    if len(rows) < 200:
        raise ValueError("At least 200 closed candles required per timeframe")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    close = df.close
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    values = dict(
        close=float(close.iloc[-1]),
        rsi=float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]),
        macd_histogram=float(ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff().iloc[-1]),
        ema_20=float(ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]),
        ema_50=float(ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]),
        sma_200=float(ta.trend.SMAIndicator(close, window=200).sma_indicator().iloc[-1]),
        atr=float(ta.volatility.AverageTrueRange(df.high, df.low, close, window=14).average_true_range().iloc[-1]),
        bb_low=float(bb.bollinger_lband().iloc[-1]), bb_high=float(bb.bollinger_hband().iloc[-1]))
    if not all(math.isfinite(x) for x in values.values()):
        raise ValueError("Non-finite technical features")
    return {**values, "last_closed_candle_ms": int(df.timestamp.iloc[-1])}


def normalize_book(raw, received_at, contract_size=1.):
    sides = {}
    for side in ("bids", "asks"):
        sides[side] = [[float(p), float(v) * contract_size] for p, v, *_ in raw[side][:20]]
        if len(sides[side]) < 20 or any(not math.isfinite(p + v) or p <= 0 or v <= 0 for p, v in sides[side]):
            raise ValueError("Incomplete or invalid top-20 book")
    if sides["bids"][0][0] >= sides["asks"][0][0]:
        raise ValueError("Crossed book")
    bid, ask = (sum(v for _, v in sides[s]) for s in ("bids", "asks"))
    return {**sides, "received_at": received_at, "exchange_timestamp_ms": raw.get("timestamp"),
            "nonce": raw.get("nonce"), "orderbook_imbalance": (bid - ask) / (bid + ask),
            "mid": (sides["bids"][0][0] + sides["asks"][0][0]) / 2}


def sentiment():
    try:
        with urllib.request.urlopen("https://api.alternative.me/fng/?limit=1", timeout=8) as response:
            item = json.load(response)["data"][0]
        return {"value": int(item["value"]), "classification": item["value_classification"],
                "source_timestamp": int(item["timestamp"]), "status": "available"}
    except Exception:
        return {"value": None, "status": "unavailable"}


async def funding_observation(exchange, symbol):
    funding = await exchange.fetch_funding_rate(symbol)
    if funding.get("interval") is None and exchange.id in {"binance", "binanceusdm"}:
        # Binance premiumIndex omits interval; fundingInfo lists adjusted contracts.
        # Only use the documented 8h default after a successful adjustment lookup.
        adjustments = await exchange.fetch_funding_intervals([symbol])
        adjusted = adjustments.get(symbol)
        if adjusted:
            funding["interval"] = adjusted.get("interval")
            funding["interval_source"] = adjusted
        else:
            funding["interval"] = "8h"
            funding["interval_source"] = "Binance documented default; successful fundingInfo lookup found no adjustment"
    return funding


async def snapshot(exchange, symbol, macro):
    frames = ("15m", "1h", "1d")
    results = await asyncio.gather(*(exchange.fetch_ohlcv(symbol, tf, limit=230) for tf in frames),
                                   funding_observation(exchange, symbol), asyncio.to_thread(sentiment))
    now = time.time()
    if macro.get("status") == "provided":
        from datetime import datetime
        as_of = datetime.fromisoformat(macro["as_of"].replace("Z", "+00:00"))
        if as_of.tzinfo is None or not 0 <= now - as_of.timestamp() <= 86400:
            raise ValueError("Provided macro rates must have a timezone and be <=24h old")
    funding = results[3]
    rate = funding.get("fundingRate")
    if rate is None or not math.isfinite(float(rate)):
        raise ValueError("Missing actual perpetual funding rate")
    # Normalize quoted funding periods; do not assume every venue always uses 8h.
    interval = funding.get("interval")
    if interval not in {"8h", "4h", "1h"}:
        raise ValueError("Funding interval unknown; use a venue exposing its actual interval")
    rate_8h = float(rate) * 8 / int(interval[:-1])
    tf = {name: indicators(rows, frame, now) for name, frame, rows in
          zip(("15m_execution", "1h_intermediate", "1d_macro_structure"), frames, results[:3])}
    return {"symbol": symbol, "timeframe_analysis": tf,
            "funding_rate_8h": rate_8h, "funding_source": funding,
            "global_sentiment": results[4], "macro": macro,
            "features_completed_at": now}
