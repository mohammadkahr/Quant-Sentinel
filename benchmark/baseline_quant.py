"""Zero-token control, equation (10), including the RSI collision bound."""
import math
import time

class TraditionalQuantBaseline:
    @staticmethod
    def evaluate(context):
        start = time.perf_counter()
        tf = context["timeframe_analysis"]["15m_execution"]
        price = context["current_price_usd"]
        rsi, low, high, ema = (tf[k] for k in ("rsi", "bb_low", "bb_high", "ema_20"))
        if not all(math.isfinite(x) for x in (price, rsi, low, high, ema)) or min(price, low, high, ema) <= 0:
            raise ValueError("Invalid baseline features")
        direction = "NO-TRADE"
        if (rsi < 33 and price <= 1.002 * low) or (price > ema and 55 < rsi <= 65):
            direction = "LONG"
        elif rsi > 67 and price >= .998 * high:
            direction = "SHORT"
        return dict(direction=direction, confidence=.5, rationale="Deterministic equation (10)",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    tokens_used=0, api_calls=0)

baseline_quant = TraditionalQuantBaseline()
