"""Deterministic sizing from equations (13)-(15) of Quant-Sentinel."""
import math
from core.config import settings

class RiskEngine:
    @staticmethod
    def calculate_trade_parameters(signal_direction, entry_price, atr_value,
                                   account_balance=None, risk_percentage=None, max_leverage=None):
        balance = settings.TOTAL_ACCOUNT_BALANCE if account_balance is None else account_balance
        risk = settings.MAX_RISK_PER_TRADE_PERCENT if risk_percentage is None else risk_percentage
        cap = settings.MAX_LEVERAGE if max_leverage is None else max_leverage
        if signal_direction not in {"LONG", "SHORT", "NO-TRADE"}:
            raise ValueError("Unknown direction")
        if not all(math.isfinite(x) for x in (entry_price, atr_value, balance, risk, cap)):
            raise ValueError("Risk inputs must be finite")
        if entry_price <= 0 or atr_value < 0 or balance <= 0 or not 0 < risk <= 100 or cap < 1 or int(cap) != cap:
            raise ValueError("Invalid risk inputs")
        if signal_direction == "NO-TRADE":
            return dict(action="NO-TRADE", entry_price=entry_price, leverage=1,
                        margin_usd=0., notional_size_usd=0., coin_amount=0.,
                        stop_loss=0., take_profit=0., risk_reward=0., risk_usd=0.)
        distance = max(1.5 * atr_value, .005 * entry_price)
        notional = min(balance * risk / 100 / (distance / entry_price), .15 * balance * cap)
        leverage = min(cap, max(1, math.ceil(notional / (.15 * balance))))
        sign = 1 if signal_direction == "LONG" else -1
        stop, target = entry_price - sign * distance, entry_price + sign * 2 * distance
        if min(stop, target) <= 0:
            raise ValueError("Volatility implies non-positive barriers")
        return dict(action=signal_direction, entry_price=entry_price, leverage=leverage,
                    margin_usd=notional / leverage, notional_size_usd=notional,
                    coin_amount=notional / entry_price, stop_loss=stop, take_profit=target,
                    risk_reward=2., risk_usd=notional * distance / entry_price)
