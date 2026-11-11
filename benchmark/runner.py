"""Concurrent tiers on a shared snapshot; each fills when its own inference finishes."""
import asyncio
import copy
import time
from benchmark.baseline_quant import baseline_quant
from council.single_super_agent import super_agent
from council.engine import CouncilEngine
from core.config import settings
from execution.triple_barrier import TripleBarrierPosition
from risk.risk_engine import RiskEngine
from tools.telemetry import normalize_book, snapshot


class QuoteFeed:
    def __init__(self, exchange, symbol, store, run_id):
        self.exchange, self.symbol = exchange, symbol
        self.store, self.run_id = store, run_id
        self.latest = None
        self.condition = asyncio.Condition()
        self.error = None
        self.positions = []

    async def run(self):
        try:
            market = self.exchange.market(self.symbol)
            if not market.get("linear"):
                raise ValueError("Only linear perpetual contracts are supported")
            while True:
                raw = await asyncio.wait_for(self.exchange.watch_order_book(self.symbol),
                                             timeout=settings.MAX_QUOTE_GAP_SECONDS)
                received_at = time.time()
                if raw.get("timestamp") and abs(received_at - raw["timestamp"] / 1000) > settings.MAX_QUOTE_GAP_SECONDS:
                    raise ValueError("Stale exchange book timestamp")
                book = normalize_book(raw, received_at, market.get("contractSize") or 1.)
                self.store.append(self.run_id, None, "quote", {"symbol": self.symbol, **book})
                # Resolve synchronously on every delivered update; slow inference cannot skip updates.
                for pos, future in list(self.positions):
                    if future.done():
                        self.positions.remove((pos, future))
                        continue
                    try:
                        result = pos.update(book)
                        if result:
                            future.set_result(result)
                    except Exception as error:
                        future.set_exception(error)
                async with self.condition:
                    self.latest = book
                    self.condition.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = error
            for _, future in self.positions:
                if not future.done():
                    future.set_exception(ValueError("Quote stream failed; outcome incomplete"))
            async with self.condition:
                self.condition.notify_all()

    async def next(self, after=0.):
        async with self.condition:
            await asyncio.wait_for(self.condition.wait_for(
                lambda: self.error is not None or self.latest is not None and self.latest["received_at"] > after),
                timeout=settings.MAX_QUOTE_GAP_SECONDS)
            if self.error:
                raise ValueError("Quote stream unavailable") from self.error
            return copy.deepcopy(self.latest)


async def run_epoch(exchange, symbol, feed, store, run_id, epoch, equities, macro, ablation=False):
    state = await snapshot(exchange, symbol, macro)
    book = await feed.next(time.time())
    state.update(snapshot_at=book["received_at"], current_price_usd=book["mid"], orderbook_dynamics=book)
    store.append(run_id, epoch, "snapshot", state)
    council = CouncilEngine()
    engines = {"rule": lambda s: baseline_quant.evaluate(s),
               "single": lambda s: super_agent.evaluate_market(symbol, s),
               "council": lambda s: council.evaluate(symbol, s)}
    if ablation:
        engines["without_cro"] = lambda s: CouncilEngine().evaluate(symbol, s, without_cro=True)

    async def evaluate(tier, evaluate_fn):
        started = time.perf_counter()
        decision = None
        try:
            decision = await asyncio.to_thread(evaluate_fn, copy.deepcopy(state))
            ready_at = time.time()
            decision["latency_ms"] = (time.perf_counter() - started) * 1000
            store.append(run_id, epoch, "decision", {"tier": tier, "symbol": symbol,
                         "ready_at": ready_at, **decision})
            if decision["direction"] == "NO-TRADE":
                result = {"label": "NO-TRADE", "net_return": 0., "pnl_usd": 0.}
            else:
                entry_book = await feed.next(ready_at)
                atr = state["timeframe_analysis"]["15m_execution"]["atr"]
                # Iterate at the depth fill so notional and margin caps hold at the actual entry.
                from execution.triple_barrier import vwap
                price = entry_book["asks" if decision["direction"] == "LONG" else "bids"][0][0]
                side = entry_book["asks" if decision["direction"] == "LONG" else "bids"]
                for _ in range(20):
                    plan = RiskEngine.calculate_trade_parameters(decision["direction"], price, atr,
                                                                account_balance=equities[tier])
                    fill = vwap(side, plan["coin_amount"])
                    if abs(fill - price) <= price * 1e-10:
                        break
                    price = fill
                else:
                    raise ValueError("Depth sizing did not converge")
                pos = TripleBarrierPosition(plan, entry_book, state["funding_rate_8h"])
                store.append(run_id, epoch, "entry", {"tier": tier, "symbol": symbol,
                    "plan": plan, "fill": pos.entry, "timestamp": pos.open_time,
                    "snapshot_to_fill_seconds": pos.open_time - state["snapshot_at"],
                    "mid_drift_return": (entry_book["mid"] / book["mid"] - 1),
                    "entry_impact": pos.entry_impact})
                future = asyncio.get_running_loop().create_future()
                immediate = pos.update(entry_book)
                if immediate:
                    future.set_result(immediate)
                else:
                    feed.positions.append((pos, future))
                result = await asyncio.wait_for(future, settings.HOLD_SECONDS + settings.MAX_QUOTE_GAP_SECONDS * 2)
            equities[tier] += result["pnl_usd"]
            store.append(run_id, epoch, "outcome", {"tier": tier, "symbol": symbol,
                         "equity_usd": equities[tier], **result})
        except Exception as error:
            # Failures stay distinct from a valid defensive NO-TRADE.
            store.append(run_id, epoch, "incomplete", {"tier": tier, "symbol": symbol,
                         "stage": "inference" if decision is None else "execution",
                         "error_type": type(error).__name__})
    await asyncio.gather(*(evaluate(t, f) for t, f in engines.items()))
