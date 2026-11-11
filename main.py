"""Research CLI. Public data only; no exchange credentials or real order execution."""
import argparse
import asyncio
import json
import platform
import uuid
from importlib.metadata import version
from pathlib import Path
from core.config import settings
from database.store import EventStore


async def collect(args):
    import ccxt.pro as ccxtpro
    from benchmark.runner import QuoteFeed, run_epoch
    if not settings.api_keys_list:
        raise ValueError("Set GEMINI_API_KEY in your private .env before live inference")
    exchange = getattr(ccxtpro, args.exchange)({"enableRateLimit": True, "timeout": 15000,
                                               "options": {"defaultType": "swap"}})
    store = EventStore()
    run_id = uuid.uuid4().hex
    equities = {t: settings.TOTAL_ACCOUNT_BALANCE for t in ("rule", "single", "council", "without_cro")}
    macro = {"status": "unavailable", "fiat_premium": None}
    if args.macro:
        macro = json.loads(Path(args.macro).read_text(encoding="utf-8"))
        for key in ("source", "as_of", "usdt_local", "usd_local"):
            if key not in macro:
                raise ValueError(f"Macro input needs {key}")
        if macro["usdt_local"] <= 0 or macro["usd_local"] <= 0:
            raise ValueError("Macro conversion rates must be positive and in the same currency unit")
        macro["fiat_premium"] = macro["usdt_local"] / macro["usd_local"] - 1
        macro["status"] = "provided"
    # Explicit allowlist: settings dumps must never expose keys.
    store.append(run_id, None, "manifest", {
        "exchange": args.exchange, "symbols": args.symbols, "model": settings.GLOBAL_LLM_MODEL,
        "hold_seconds": settings.HOLD_SECONDS, "fee_per_leg": settings.TAKER_FEE,
        "equity": settings.TOTAL_ACCOUNT_BALANCE, "risk_percent": settings.MAX_RISK_PER_TRADE_PERCENT,
        "leverage_cap": settings.MAX_LEVERAGE, "margin_fraction": .15,
        "epochs_per_symbol": args.epochs_per_symbol, "ablation": args.ablation,
        "max_quote_gap_seconds": settings.MAX_QUOTE_GAP_SECONDS,
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in ("ccxt", "google-genai", "ta", "pydantic", "numpy", "scipy")}})
    print(f"Run ID: {run_id}", flush=True)
    try:
        await exchange.load_markets()
        for symbol in args.symbols:
            # Sequential assets keep each tier's portfolio chronology unambiguous.
            feed = QuoteFeed(exchange, symbol, store, run_id)
            task = asyncio.create_task(feed.run())
            try:
                await feed.next()
                for i in range(args.epochs_per_symbol):
                    epoch = f"{symbol}:{i + 1}"
                    print(f"Collecting {epoch}", flush=True)
                    await run_epoch(exchange, symbol, feed, store, run_id, epoch,
                                    equities, macro, args.ablation)
                    if feed.error:
                        raise ValueError("Quote stream failed; restart as a new run")
                    if i + 1 < args.epochs_per_symbol:
                        await asyncio.sleep(args.interval_seconds)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        store.append(run_id, None, "run_finished", {"status": "finished"})
    finally:
        await exchange.close()


def main():
    parser = argparse.ArgumentParser(description="Quant-Sentinel research paper-execution framework")
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("collect", help="Collect new prospective observations (paid LLM calls)")
    live.add_argument("--exchange", choices=["binanceusdm", "bybit", "kucoinfutures"], default="binanceusdm")
    live.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"])
    live.add_argument("--epochs-per-symbol", type=int, default=15)
    live.add_argument("--interval-seconds", type=float, default=0.)
    live.add_argument("--macro", help="JSON with source, as_of, usdt_local and usd_local")
    live.add_argument("--ablation", action="store_true", help="Also run M+L+A without CRO")
    report = sub.add_parser("report")
    report.add_argument("run_id")
    audit = sub.add_parser("audit-export")
    audit.add_argument("run_id")
    audit.add_argument("--output", default="private/audit")
    judge = sub.add_parser("audit-score")
    judge.add_argument("directory")
    args = parser.parse_args()
    if args.command == "collect":
        if args.epochs_per_symbol < 1 or args.interval_seconds < 0 or len(set(args.symbols)) != len(args.symbols):
            parser.error("Use positive epoch counts, nonnegative intervals and unique symbols")
        asyncio.run(collect(args))
    elif args.command == "report":
        from benchmark.statistics import report_run
        print(json.dumps(report_run(EventStore(), args.run_id), indent=2, allow_nan=False))
    elif args.command == "audit-export":
        from benchmark.audit import export_packet
        export_packet(EventStore(), args.run_id, args.output)
    else:
        from benchmark.audit import score_packet
        print(json.dumps(score_packet(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
