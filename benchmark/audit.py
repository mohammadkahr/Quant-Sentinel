"""Blinded multi-model audit preparation and analysis (no fabricated judge scores)."""
import hashlib
import json
from pathlib import Path
import secrets
import numpy as np
from scipy import stats

DIMENSIONS = ["market_structure", "liquidity_awareness", "risk_invalidation", "execution_actionability"]
PERSONAS = ["Market Structure Specialist", "Chief Risk Officer", "Order Flow Veteran"]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export_packet(store, run_id, output, limit=20):
    directory = Path(output)
    if directory.exists():
        raise ValueError("Choose a new output directory; existing audit packets are immutable")
    snapshots = store.read(run_id, "snapshot")
    decisions = store.read(run_id, "decision")
    lookup = {(d["epoch"], d["tier"]): d for d in decisions}
    eligible = [s for s in snapshots if all((s["epoch"], t) in lookup for t in ("single", "council"))]
    if len(eligible) < limit:
        raise ValueError(f"Need {limit} paired scenarios; found {len(eligible)}")
    selected = secrets.SystemRandom().sample(eligible, limit)
    mapping, packet = {}, {}
    for snapshot in selected:
        case = secrets.token_hex(16)
        order = ["single", "council"]
        secrets.SystemRandom().shuffle(order)
        mapping[case] = {"epoch": snapshot["epoch"], "System_A": order[0], "System_B": order[1]}
        # Common schema and whitespace reduce identity cues; no tier names, timing or outcomes.
        packet[case] = {"market_state": {k: v for k, v in snapshot.items() if k not in {"epoch", "event_timestamp"}}}
        for label, tier in zip(("System_A", "System_B"), order):
            decision = lookup[snapshot["epoch"], tier]
            packet[case][label] = {k: decision[k] for k in ("steps", "direction", "confidence", "rationale")}
    directory.mkdir(parents=True)
    write_json(directory / "packet.json", packet)
    write_json(directory / "SECRET_AUDIT_KEY.json", {"run_id": run_id, "mapping": mapping,
        "packet_sha256": hashlib.sha256((directory / "packet.json").read_bytes()).hexdigest()})
    for i, persona in enumerate(PERSONAS, 1):
        write_json(directory / f"evaluator_{i}.json", {"model": "", "persona": persona, "evaluated_at": "",
            "scores": {case: {"System_A": [None] * 4, "System_B": [None] * 4} for case in packet}})
    (directory / "INSTRUCTIONS.md").write_text(
        "# Blinded research audit\n\nGive only packet.json to independent evaluator models. "
        "Keep SECRET_AUDIT_KEY.json private until all evaluations are locked. "
        "Use three distinct model IDs and the three assigned personas. Record the exact model, "
        "UTC evaluation time, prompt and unedited response for provenance.\n\n"
        "Score each system independently on a 1-5 Likert scale (1 inadequate, 5 exemplary), "
        "in this order: market structure, liquidity/LOB awareness, risk/stop invalidation, execution "
        "actionability. Judge evidence and consistency, not verbosity. All four entries are required. "
        "Do not seek system identity or market outcomes. Analysts should not reveal mappings "
        "to evaluators; a coordinator must maintain separation. Random labels alone do not establish "
        "a double-blind study.\n", encoding="utf-8")
    print(f"Audit packet and unfilled evaluator templates: {directory}")


def fleiss_binary(ratings):
    matrix = np.asarray(ratings, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isin(matrix, [0, 1]).all():
        raise ValueError("Need binary ratings from >=2 evaluators")
    n = matrix.shape[1]
    counts = np.stack([(matrix == k).sum(axis=1) for k in (0, 1)], axis=1)
    observed = np.mean(((counts ** 2).sum(axis=1) - n) / (n * (n - 1)))
    expected = ((counts.sum(axis=0) / counts.sum()) ** 2).sum()
    return float((observed - expected) / (1 - expected)) if expected < 1 else None


def ordinal_alpha(ratings):
    """Krippendorff ordinal alpha for complete unit x rater Likert data (1..5)."""
    matrix = np.asarray(ratings, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isin(matrix, range(1, 6)).all():
        raise ValueError("Need complete 1-5 ratings from >=2 evaluators")
    counts = np.stack([(matrix == k).sum(axis=1) for k in range(1, 6)], axis=1).astype(float)
    coincidence = sum((np.outer(c, c) - np.diag(c)) / (c.sum() - 1) for c in counts)
    marginal = coincidence.sum(axis=0)
    midpoint = np.cumsum(marginal) - marginal / 2
    distances = (midpoint[:, None] - midpoint[None, :]) ** 2
    expected = (np.outer(marginal, marginal) - np.diag(marginal)) / (marginal.sum() - 1)
    denominator = (expected * distances).sum()
    return float(1 - (coincidence * distances).sum() / denominator) if denominator > 0 else None


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def score_packet(directory):
    directory = Path(directory)
    key = json.loads((directory / "SECRET_AUDIT_KEY.json").read_text(encoding="utf-8"))
    if hashlib.sha256((directory / "packet.json").read_bytes()).hexdigest() != key["packet_sha256"]:
        raise ValueError("Packet changed after anonymization")
    judges = [json.loads((directory / f"evaluator_{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    if len({j["model"].strip() for j in judges}) != 3 or any(not j["model"].strip() or not j["evaluated_at"].strip() for j in judges):
        raise ValueError("Record three distinct evaluator models and evaluation times")
    cases = list(key["mapping"])
    scores = {"single": [], "council": []}
    for case in cases:
        values = {"single": [], "council": []}
        for judge in judges:
            if set(judge["scores"]) != set(cases):
                raise ValueError("Evaluator case IDs must match exactly")
            for label in ("System_A", "System_B"):
                ratings = judge["scores"][case][label]
                if len(ratings) != 4 or any(type(x) is not int or not 1 <= x <= 5 for x in ratings):
                    raise ValueError("Scores must be four integer Likert values, 1-5")
                values[key["mapping"][case][label]].append(ratings)
        for tier in scores:
            scores[tier].append(values[tier])
    # Unit is a scenario, not an individual judge: average raters before paired tests.
    c, s = (np.asarray(scores[t], dtype=float) for t in ("council", "single"))
    difference = c.mean(axis=1) - s.mean(axis=1)
    pvalues = [float(stats.wilcoxon(difference[:, i]).pvalue) if np.any(difference[:, i]) else 1. for i in range(4)]
    metrics = {name: {"council_mean": float(c[:, :, i].mean()), "single_mean": float(s[:, :, i].mean()),
                      "paired_wilcoxon_p": pvalues[i], "holm_p": holm(pvalues)[i]} for i, name in enumerate(DIMENSIONS)}
    # Unit x rater; ties count as non-superiority in the explicitly binary Fleiss statistic.
    superiority = (c > s).transpose(0, 2, 1).reshape(-1, 3).astype(int)
    ordinal = np.concatenate([c, s]).transpose(0, 2, 1).reshape(-1, 3).astype(int)
    return {"scenarios": len(cases), "dimensions": metrics,
            "fleiss_binary_superiority_kappa": fleiss_binary(superiority),
            "krippendorff_ordinal_alpha": ordinal_alpha(ordinal),
            "tie_policy": "council score <= single score counts as non-superiority",
            "inference_unit": "scenario mean across three raters"}
