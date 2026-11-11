"""Descriptive metrics and paired epoch analysis; no hard-coded paper results."""
import numpy as np
from scipy import stats


def paired_analysis(a, b, resamples=10000, mean_block=3, seed=2026):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3 or not np.isfinite(a + b).all():
        raise ValueError("Need at least three finite paired epochs")
    difference = a - b
    rng = np.random.default_rng(seed)
    # Stationary bootstrap: geometric block lengths, circular wrapping, expected b=3.
    index = rng.integers(len(a), size=resamples)
    total = difference[index].copy()
    for _ in range(1, len(a)):
        restart = rng.random(resamples) < 1 / mean_block
        index = np.where(restart, rng.integers(len(a), size=resamples), (index + 1) % len(a))
        total += difference[index]
    ci = np.quantile(total / len(a), [.025, .975])
    if np.all(difference == 0):
        t_stat, t_p, w_stat, w_p = 0., 1., 0., 1.
    else:
        t = stats.ttest_rel(a, b)
        w = stats.wilcoxon(a, b)
        t_stat = float(t.statistic) if np.isfinite(t.statistic) else None
        t_p, w_stat, w_p = float(t.pvalue), float(w.statistic), float(w.pvalue)
    return dict(n=len(a), mean_difference=float(difference.mean()), t_statistic=t_stat,
                t_pvalue=t_p, wilcoxon_statistic=w_stat, wilcoxon_pvalue=w_p,
                stationary_bootstrap_ci95=ci.tolist(), resamples=resamples,
                expected_block_length=mean_block, seed=seed)


def report_run(store, run_id):
    manifests = store.read(run_id, "manifest")
    if len(manifests) != 1:
        raise ValueError("Unknown or ambiguous run")
    manifest = manifests[0]
    outcomes = store.read(run_id, "outcome")
    decisions = store.read(run_id, "decision")
    snapshots = store.read(run_id, "snapshot")
    expected = len(manifest["symbols"]) * manifest["epochs_per_symbol"]
    tiers = ["rule", "single", "council"] + (["without_cro"] if manifest["ablation"] else [])
    tables = {}
    series = {}
    for tier in tiers:
        rows = [r for r in outcomes if r["tier"] == tier]
        if len({r["epoch"] for r in rows}) != len(rows):
            raise ValueError("Duplicate epoch outcomes")
        series[tier] = {r["epoch"]: r["net_return"] for r in rows}
        trades = [r for r in rows if r["label"] != "NO-TRADE"]
        pnl = np.array([r["pnl_usd"] for r in trades])
        equity = np.array([manifest["equity"]] + [r["equity_usd"] for r in rows])
        drawdown = equity / np.maximum.accumulate(equity) - 1
        gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        tables[tier] = dict(observed_epochs=len(rows), trades=len(trades),
            win_rate=sum(r["label"] == "WIN" for r in trades) / len(trades) if trades else None,
            mean_trade_return=float(np.mean([r["net_return"] for r in trades])) if trades else None,
            mean_epoch_return=float(np.mean(list(series[tier].values()))) if rows else None,
            profit_factor=float(gains / losses) if losses else None,
            profit_factor_note="undefined without losses" if not losses else None,
            maximum_portfolio_drawdown=float(drawdown.min()), final_equity_usd=float(equity[-1]))
        measured = [r for r in decisions if r["tier"] == tier]
        tokens = [r.get("tokens_used") for r in measured]
        tables[tier]["mean_latency_ms"] = float(np.mean([r["latency_ms"] for r in measured])) if measured else None
        tables[tier]["mean_tokens"] = float(np.mean(tokens)) if tokens and all(t is not None for t in tokens) else None
        tables[tier]["total_request_attempts"] = sum(r["api_calls"] for r in measured)
        if trades:
            returns = np.array([r["net_return"] for r in trades])
            rng = np.random.default_rng(2026)
            bootstrap = np.array([rng.choice(returns, len(returns), replace=True).mean() for _ in range(10000)])
            tables[tier]["trade_mean_bootstrap_ci95"] = np.quantile(bootstrap, [.025, .975]).tolist()
    incomplete = store.read(run_id, "incomplete")
    finished = bool(store.read(run_id, "run_finished"))
    complete = finished and not incomplete and len(snapshots) == expected and all(len(s) == expected for s in series.values())
    result = dict(run_id=run_id, complete=complete, expected_epochs=expected,
                  incomplete_count=len(incomplete), tiers=tables,
                  units="returns and drawdown are fractions; PnL and equity are USD")
    # Never silently impute infrastructure failures as cash returns.
    keys = [s["epoch"] for s in snapshots]
    if complete and len(keys) >= 3:
        result["paired_council_vs_single"] = paired_analysis(
            [series["council"][k] for k in keys], [series["single"][k] for k in keys])
    else:
        result["paired_analysis_status"] = "Not computed: need a completed run with >=3 fully paired epochs"
    return result
