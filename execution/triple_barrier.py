"""Quote-path labeling and depth-aware prospective paper fills. Never places orders."""
import math
from core.config import settings


def vwap(levels, quantity):
    if not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("Invalid quantity")
    remaining, cost = quantity, 0.
    for price, volume in levels:
        amount = min(volume, remaining)
        cost += price * amount
        remaining -= amount
        if remaining <= quantity * 1e-12:
            return cost / quantity
    raise ValueError("Insufficient visible top-20 depth")


class TripleBarrierPosition:
    def __init__(self, plan, book, funding_rate_8h, hold_seconds=None, fee=None):
        self.plan = plan
        self.sign = 1 if plan["action"] == "LONG" else -1
        self.open_time = book["received_at"]
        self.last_time = self.open_time
        self.horizon = settings.HOLD_SECONDS if hold_seconds is None else hold_seconds
        self.fee = settings.TAKER_FEE if fee is None else fee
        self.funding_rate = funding_rate_8h
        self.quantity = plan["coin_amount"]
        self.entry = vwap(book["asks"] if self.sign == 1 else book["bids"], self.quantity)
        # Re-anchor both barriers at actual fill, preserving the snapshot ATR distance.
        distance = abs(plan["entry_price"] - plan["stop_loss"])
        self.stop = self.entry - self.sign * distance
        self.target = self.entry + self.sign * 2 * distance
        if min(self.stop, self.target) <= 0:
            raise ValueError("Invalid fill-adjusted barriers")
        self.entry_impact = abs(self.entry - book["mid"]) / book["mid"]
        self.result = None

    def update(self, book):
        if self.result is not None:
            return self.result
        now = book["received_at"]
        if now < self.last_time:
            raise ValueError("Quote timestamps must be ordered")
        if now - self.last_time > settings.MAX_QUOTE_GAP_SECONDS:
            raise ValueError("Quote gap: first-touch outcome is unobservable")
        self.last_time = now
        quote = book["bids"][0][0] if self.sign == 1 else book["asks"][0][0]
        deadline = self.open_time + self.horizon
        # A quote strictly after expiry cannot establish a horizontal hit before expiry.
        label = None
        if now <= deadline:
            if self.sign * (quote - self.stop) <= 0:
                label = "LOSS"
            elif self.sign * (quote - self.target) >= 0:
                label = "WIN"
        if label is None and now >= deadline:
            label = "NEUTRAL"
        if label is None:
            return None
        exit_price = vwap(book["bids"] if self.sign == 1 else book["asks"], self.quantity)
        held = min(now - self.open_time, self.horizon)
        funding = self.sign * self.funding_rate * held / (480 * 60)
        gross = self.sign * (exit_price / self.entry - 1)
        fees = self.fee * (1 + exit_price / self.entry)
        net = gross - fees - funding
        self.result = dict(label=label, entry_price=self.entry, exit_price=exit_price,
                           opened_at=self.open_time, exited_at=now, stop_loss=self.stop,
                           take_profit=self.target, entry_impact=self.entry_impact,
                           gross_return=gross, fee_return=fees, funding_return=funding,
                           net_return=net, pnl_usd=net * self.entry * self.quantity,
                           notional_usd=self.entry * self.quantity,
                           expiry_quote_delay_seconds=max(0., now - deadline))
        return self.result
