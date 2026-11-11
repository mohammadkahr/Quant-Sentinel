# Quant-Sentinel

Reference implementation accompanying **Quant-Sentinel: An Adversarial Multi-Agent LLM Framework for Quantitative Cryptocurrency Trading and Risk Management**.

**Manuscript status: under review.** Conference information and the final citation will be added when available.

This repository implements the decision architecture and prospective paper-execution workflow described in the manuscript.

## Architecture

A shared market snapshot is evaluated concurrently by three decision tiers:

1. **Rule-based baseline:** deterministic technical-indicator rules, without LLM inference.
2. **Monolithic agent:** one model call integrating momentum, liquidity and risk context.
3. **Multi-agent council:** parallel Momentum and Liquidity desks, followed by an adversarial Chief Risk Officer and a consensus Arbiter.

Validated decisions pass through a deterministic risk engine, depth-aware paper execution and Triple-Barrier evaluation. Market observations, decisions and outcomes are stored in SQLite. An optional council ablation removes the CRO.

| Component | Configuration described in the manuscript |
| --- | --- |
| Assets | BTC, ETH and SOL; linear USDT perpetual contracts |
| Market context | 15m / 1h / 1d indicators, top-20 orderbook depth, funding, sentiment and domestic fiat context |
| Generative model | Fixed `gemini-3.8-flash` for both LLM tiers |
| Stop distance | `max(1.5 * ATR15m, 0.005 * entry_price)` |
| Risk and margin | 1.5% equity risk budget, 15% initial-margin cap, leverage up to 10x |
| Exit evaluation | First profit/stop barrier hit, or 45-minute expiry; risk-to-reward ratio 2:1 |
| Execution costs | Observed depth VWAP, 0.04% taker fee per leg and pro-rata funding |
| Qualitative evaluation | Anonymous A/B cases, three evaluator models and four Likert dimensions |

## Setup

Python 3.11 or later is required. Offline checks were run locally on Python 3.14; CI is configured for Python 3.11-3.13.

```bash
python -m venv .venv
# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`, or a comma-separated `GEMINI_API_KEYS` pool. Each user supplies their own credentials. Public exchange telemetry requires no exchange trading key. Model calls may incur charges.

## Run

```bash
# 15 epochs per asset: 45 epochs in total
python main.py collect --epochs-per-symbol 15
# Optional ablation and alternative venue
python main.py collect --ablation --exchange bybit
# Analyze a recorded run using the ID printed at startup
python main.py report YOUR_RUN_ID
```

The venue adapters are `binanceusdm` (default), `bybit` and `kucoinfutures`. Candles use REST and orderbooks use CCXT Pro WebSockets. Assets and epochs are processed sequentially; the three decision tiers run concurrently within each epoch. Positions can remain open for 45 minutes, so a full collection may take more than 33 hours. The default collector runs continuously; the manuscript's non-consecutive session sampling requires an explicit external schedule.

For domestic fiat context, pass `--macro private/macro.json`. The JSON must contain `source`, a timezone-aware ISO-8601 `as_of` timestamp, and positive `usdt_local` and `usd_local` measurements in the same local currency unit. Measurements must be no more than 24 hours old. The premium is `usdt_local / usd_local - 1`. Omitted macro context and unavailable sentiment are marked missing. Invalid market paths or model failures are recorded as incomplete observations.

## Qualitative audit

After at least 20 paired single-agent/council decisions:

```bash
python main.py audit-export YOUR_RUN_ID --output private/audit
# Independent evaluators complete evaluator_1.json through evaluator_3.json.
python main.py audit-score private/audit
```

The exporter creates randomized A/B cases, blank evaluator forms and a private identity mapping. Keep the mapping separate from evaluators until ratings are locked. Use three distinct evaluator models and record their exact IDs and evaluation times. The scoring dimensions are market structure, liquidity awareness, risk invalidation and execution actionability. Scores must be integers from 1 to 5.

## Validation and scope

```bash
python -m unittest discover -s tests -v
```

The 20 offline tests cover decision rules, risk limits, quote-path execution, costs, concurrent tiers and audit analysis. They use synthetic telemetry and mocked inference. Live venue/model integration and reproduction of the manuscript's numerical results require separate validation. The original experimental dataset and completed judge records are not included in this source release; new live collections produce new observations. Reports use realized equity at exits for portfolio drawdown.

## Source layout

```text
benchmark/   baseline, experiment runner, statistics and audit
core/        configuration, model routing and output schemas
council/     monolithic agent and multi-agent council
risk/        deterministic position sizing
execution/   paper fills and Triple-Barrier resolution
tools/       market telemetry and technical indicators
database/    SQLite event store
tests/       offline verification
main.py      research command-line interface
```

Credentials, local environments, generated databases and private audit materials are excluded by `.gitignore`. Paper figures and plotting assets are not required to run the implementation.

**Research use only.** This software is an exploratory proof of concept and does not place real exchange orders. It is not investment advice. Users are responsible for their use, decisions, API costs and any resulting losses; no profitability or fitness for live trading is guaranteed.
