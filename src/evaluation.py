"""M6: evaluation evidence. Contract: docs/contracts.md#m6. Run: python -m src.evaluation

Reporting only: nothing is tuned or selected here. The M5b router is refitted in memory on the train window, so a
models/router.joblib that has seen validation rows (M7) can never be scored by mistake.
Outputs hold aggregates, fixed wording and short common phrases only (ops-policy section 10).
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src import config
from src.audit import product_conflict_mask, vague_mask
from src.baselines import BOT, CI_LEVEL, TARGET, TEXT, paired_difference, round_floats
from src.evaluate import bootstrap_ci, coverage_curve, evaluate
from src.router import CLOSE_CALL, Router, to_records
from src.split import time_split

JSON_NAME, MD_NAME, FIGURES_DIR = "evaluation.json", "evaluation.md", "figures"
SOURCE_NAMES = ("baselines", "experiments", "router")  # reports/<name>.json
FIGURES = {
    "confusion": "confusion_matrices.png",
    "coverage": "coverage_accuracy.png",
    "reliability": "reliability.png",
    "recall": "per_team_recall.png",
}
TOLERANCE = 1e-6
BINS = 10
CURVE_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(20))  # 0.00 to 0.95
WIDTH_FACTOR = math.sqrt(2)  # validation and test are both samples of about the same size
PHRASE_MIN_TEXTS, PHRASES_PER_CATEGORY, PHRASE_MAX_WORDS = 10, 5, 3
QUOTED = re.compile(r"^mentions '([^']*)'$")  # the router's own phrase reasons

CATEGORIES = {  # first match wins in the exclusive view
    "vague": "Vague request (M2 rule)",
    "close_call": "Close call between two teams",
    "product_conflict": "Text names another product than the request field",
    "ivr": "IVR transcript",
    "other": "None of these",
}


# --- Loading -----------------------------------------------------------------
def load_sources(reports_dir: Path) -> dict:
    return {name: json.loads((reports_dir / f"{name}.json").read_text(encoding="utf-8")) for name in SOURCE_NAMES}


def train_router(train_df: pd.DataFrame, sources: dict) -> Router:
    """Same configuration, rows and libraries as M5b, otherwise an error."""
    router = Router.train(train_df, sources["experiments"]["chosen"]["config"])
    expected, version = sources["router"]["router"]["model_version"], router.meta["model_version"]
    if version != expected:
        raise RuntimeError(f"router is {version}, router.json has {expected}: rerun the router target")
    return router


# --- Blocks --------------------------------------------------------------------
def _mean(values) -> float | None:
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) else None


def _accuracy(y_true, y_pred) -> float:
    return float(np.mean(y_true == y_pred))


def _interval(y_true, y_pred) -> dict:
    low, point, high = bootstrap_ci(y_true, y_pred, _accuracy)
    return {"low": low, "point": point, "high": high}


def headline(truth, pred, bot) -> dict:
    def side(guess):
        scores = evaluate(truth, guess)
        return {
            "accuracy": scores["accuracy"],
            "accuracy_ci95": _interval(truth, guess),
            "macro_f1": scores["macro_f1"],
            "misroutes": scores["misroutes"],
        }

    return {
        "rows": len(truth),
        "router": side(pred),
        "bot": side(bot),
        "paired_accuracy_points_router_minus_bot": paired_difference(truth, pred, bot),
    }


def per_team(truth, pred, bot) -> dict:
    router_scores, bot_scores = evaluate(truth, pred)["per_class"], evaluate(truth, bot)["per_class"]
    return {
        team: {
            "support": router_scores[team]["support"],
            "router": {key: router_scores[team][key] for key in ("precision", "recall")},
            "bot": {key: bot_scores[team][key] for key in ("precision", "recall")},
        }
        for team in config.TEAMS
    }


def breakdowns(val: pd.DataFrame, pred, triaged, seen) -> dict:
    truth, bot = val[TARGET].to_numpy(dtype=object), val[BOT].to_numpy(dtype=object)
    dimensions = {
        "channel": val["channel"].to_numpy(dtype=object),
        "month": val["created_month"].to_numpy(dtype=object),
        "warranty_status": val["warranty_status"].to_numpy(dtype=object),
        "text": np.where(seen, "seen", "unseen").astype(object),
    }
    result = {}
    for name, values in dimensions.items():
        groups = ("seen", "unseen") if name == "text" else sorted(set(values))
        rows = []
        for group in groups:
            mask = values == group
            right, bot_right = _mean(pred[mask] == truth[mask]), _mean(bot[mask] == truth[mask])
            rows.append(
                {
                    "group": group,
                    "rows": int(mask.sum()),
                    "router_accuracy": right,
                    "bot_accuracy": bot_right,
                    "lead_points": None if right is None else 100 * (right - bot_right),
                    "triage_rate": _mean(triaged[mask]),
                }
            )
        result[name] = rows
    return result


def triage_block(right, triaged, confidence, vague) -> dict:
    return {
        "count": int(triaged.sum()),
        "rate": float(triaged.mean()),
        "sent_count": int((~triaged).sum()),
        "accuracy_sent": _mean(right[~triaged]),
        "accuracy_triaged": _mean(right[triaged]),
        "misroutes_sent": int((~right & ~triaged).sum()),
        "misroutes_triaged": int((~right & triaged).sum()),
        "mean_confidence_triaged": _mean(confidence[triaged]),
        "threshold": config.TRIAGE_THRESHOLD,
        "vague_count": int(vague.sum()),
        "vague_triaged_count": int((vague & triaged).sum()),
        "vague_triaged_share": _mean(triaged[vague]),
    }


def costs(right, bot_right, triaged, months: int) -> dict:
    def monthly(misroutes, contacts=0):
        return float((misroutes * config.COST_PER_MISROUTE + contacts * config.COST_CLARIFY) / months)

    wrong, n_triaged = ~right, int(triaged.sum())
    sent_wrong, triaged_wrong = int((wrong & ~triaged).sum()), int((wrong & triaged).sum())
    best = monthly(sent_wrong, n_triaged)
    nothing_fixed = monthly(int(wrong.sum()), n_triaged)
    bot = monthly(int((~bot_right).sum()))
    saved = triaged_wrong * config.COST_PER_MISROUTE
    return {
        "per_month_rs": {
            "router_clarification_always_works": best,
            "router_without_triage": monthly(int(wrong.sum())),
            "router_clarification_fixes_nothing": nothing_fixed,
            "bot": bot,
            "bot_licence": config.BOT_LICENCE_PER_YEAR / 12,
        },
        "bot_minus_router_rs": bot - best,
        "bot_minus_router_clarification_fixes_nothing_rs": bot - nothing_fixed,
        "break_even_clarification_rate": float(n_triaged * config.COST_CLARIFY / saved) if saved else None,
        "inputs": {
            "cost_per_transfer_rs": config.COST_PER_TRANSFER,
            "cost_per_misroute_contact_rs": config.COST_PER_MISROUTE_CONTACT,
            "cost_per_misroute_rs": config.COST_PER_MISROUTE,
            "cost_per_misroute_train_rs": config.COST_PER_MISROUTE_TRAIN,
            "cost_clarify_rs": config.COST_CLARIFY,
            "bot_licence_per_year_rs": config.BOT_LICENCE_PER_YEAR,
            "months": months,
        },
    }


def _common_phrases(outputs: list[dict], rows, train_texts: list[str], checked: dict) -> list[dict]:
    """The router's own phrases on these rows, most frequent first. Rare phrases could identify a customer: dropped."""
    counts = Counter()
    for i in rows:
        counts.update(m.group(1) for m in map(QUOTED.match, outputs[i]["reasons"]) if m)
    chosen = []
    for phrase, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(phrase.split()) > PHRASE_MAX_WORDS or any(char.isdigit() for char in phrase):
            continue
        if phrase not in checked:
            whole = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
            checked[phrase] = sum(bool(whole.search(text)) for text in train_texts)
        if checked[phrase] >= PHRASE_MIN_TEXTS:
            chosen.append({"phrase": phrase, "count": count})
        if len(chosen) == PHRASES_PER_CATEGORY:
            break
    return chosen


def failures(val: pd.DataFrame, train_text: pd.Series, outputs: list[dict], right, triaged, vague) -> dict:
    """Overlapping view: every row that matches. Exclusive view: each misroute once, first match in CATEGORIES order."""
    n = len(val)
    masks = {
        "vague": vague,
        "close_call": np.array([o["top3"][0]["prob"] - o["top3"][1]["prob"] <= CLOSE_CALL for o in outputs]),
        "product_conflict": product_conflict_mask(val).to_numpy(dtype=bool),
        "ivr": (val["channel"] == "ivr").to_numpy(),
    }
    masks["other"] = ~np.logical_or.reduce(list(masks.values()))
    wrong, free = ~right, np.ones(n, dtype=bool)
    train_texts, checked = sorted(set(train_text)), {}
    truth = val[TARGET].tolist()
    in_top3 = np.array([t in {x["team"] for x in o["top3"]} for o, t in zip(outputs, truth)])
    rows = []
    for name in CATEGORIES:
        mask = masks[name]
        first = wrong & mask & free
        free &= ~mask
        rows.append(
            {
                "category": name,
                "rows": int(mask.sum()),
                "errors": int((wrong & mask).sum()),
                "error_rate": _mean(wrong[mask]),
                "errors_sent": int((wrong & mask & ~triaged).sum()),
                "exclusive_errors": int(first.sum()),
                "phrases": _common_phrases(outputs, np.flatnonzero(first), train_texts, checked),
            }
        )
    return {
        "categories": rows,
        "errors": int(wrong.sum()),
        "errors_sent": int((wrong & ~triaged).sum()),
        "error_rate": float(wrong.mean()),
        "errors_with_right_team_in_top3": int((wrong & in_top3).sum()),
        "close_call_gap": CLOSE_CALL,
        "phrase_min_train_texts": PHRASE_MIN_TEXTS,
        "phrase_max_words": PHRASE_MAX_WORDS,
    }


def calibration(right, confidence) -> dict:
    """Equal-width confidence bins; empty bins are left out."""
    codes = np.minimum((confidence * BINS).astype(int), BINS - 1)
    bins = []
    for code in range(BINS):
        mask = codes == code
        if mask.any():
            bins.append(
                {
                    "from": code / BINS,
                    "to": (code + 1) / BINS,
                    "rows": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(right[mask].mean()),
                }
            )
    gap = sum(b["rows"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins) / len(right)
    return {
        "bin_count": BINS,
        "bins": bins,
        "expected_calibration_error": float(gap),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(right.mean()),
    }


def expected_score(val: pd.DataFrame, pred, vague, test_text: pd.Series, head: dict, sources: dict) -> dict:
    """Fixed formulas, nothing fitted: CI half-width x root two, and a one-sided drift allowance."""
    truth, bot = val[TARGET].to_numpy(dtype=object), val[BOT].to_numpy(dtype=object)
    interval = head["router"]["accuracy_ci95"]
    inner = sources["experiments"]["supplementary"]["base_inner_val"]["accuracy"]
    drift = max(0.0, inner - interval["point"])
    bot_interval = _interval(bot, pred)
    vague_accuracy = _mean(pred[vague] == truth[vague])
    test_vague = float(vague_mask(test_text).mean())

    def span(ci, allowance=0.0):
        return {
            "point": ci["point"],
            "low": ci["point"] - WIDTH_FACTOR * (ci["point"] - ci["low"]) - allowance,
            "high": ci["point"] + WIDTH_FACTOR * (ci["high"] - ci["point"]),
        }

    month = val["created_month"].to_numpy(dtype=object)
    leads = []
    for name in sorted(set(month)):
        mask = month == name
        leads.append(
            {
                "month": name,
                "router_accuracy": _accuracy(truth[mask], pred[mask]),
                "bot_accuracy": _accuracy(truth[mask], bot[mask]),
                "lead_points": paired_difference(truth[mask], pred[mask], bot[mask]),
            }
        )

    def falling(key):
        values = [row[key] for row in leads]
        return all(a > b for a, b in zip(values, values[1:]))

    points = [row["lead_points"] for row in leads]
    return {
        "final_team": {
            **span(interval, drift),
            "ci95": interval,
            "macro_f1": head["router"]["macro_f1"],
            "width_factor": WIDTH_FACTOR,
            "inner_val_accuracy": inner,
            "drift_allowance_points": 100 * drift,
        },
        "bot_label": {
            **span(bot_interval),
            "ci95": bot_interval,
            "macro_f1": evaluate(bot, pred)["macro_f1"],
        },
        "vague_ceiling": {
            "test_vague_share": test_vague,
            "validation_vague_share": float(vague.mean()),
            "validation_vague_accuracy": vague_accuracy,
            "ceiling_accuracy": None if vague_accuracy is None else 1 - test_vague * (1 - vague_accuracy),
        },
        "monthly_lead": {
            "months": leads,
            "router_declines": falling("router_accuracy"),
            "bot_declines": falling("bot_accuracy"),
            "intervals_overlap": max(p["low"] for p in points) <= min(p["high"] for p in points),
            "lead_rises_every_month": all(a["point"] < b["point"] for a, b in zip(points, points[1:])),
        },
    }


def _lead_sentence(block: dict) -> str:
    """Wording follows the values, so a rerun cannot leave a stale claim behind."""
    leads = ", ".join(f"{row['lead_points']['point']:.2f}" for row in block["months"])
    both = block["router_declines"] and block["bot_declines"]
    start = "The router and the bot both decline over the three months" if both else "Monthly accuracy moves"
    if not block["intervals_overlap"]:
        return f"{start}, and the router's lead over the bot changes: {leads} points, with intervals that do not all overlap."
    middle = "the router's lead over the bot stays roughly constant within sampling noise"
    end = "the paired intervals all overlap"
    if block["lead_rises_every_month"]:
        end += ", although the point estimates rise every month, so the lead is not shrinking"
    return f"{start}, while {middle}: {leads} points; {end}."


# --- Consistency -----------------------------------------------------------------
def _pick(node, path):
    for key in path:
        node = node[key]
    return node


def _pairs(results: dict) -> list[tuple]:
    """(path in evaluation.json, source file, path in that file)"""
    ci, ends = "accuracy_ci95", ("low", "point", "high")
    head, months = ("headline",), ("costs", "per_month_rs")
    cfg = results["meta"]["model_name"]
    final, bot = ("baselines", "tfidf_final"), ("baselines", "bot")
    real = ("real_val", cfg)
    paired = "paired_accuracy_points_router_minus_bot"
    pairs = [
        ((*head, "router", "accuracy"), "baselines", (*final, "evaluate", "accuracy")),
        ((*head, "router", "accuracy"), "experiments", (*real, "accuracy")),
        ((*head, "router", "accuracy"), "router", ("accuracy", "all")),
        ((*head, "router", "macro_f1"), "baselines", (*final, "evaluate", "macro_f1")),
        ((*head, "router", "macro_f1"), "experiments", (*real, "macro_f1")),
        ((*head, "router", "misroutes"), "baselines", (*final, "evaluate", "misroutes")),
        ((*head, "router", "misroutes"), "router", ("misroutes", "all")),
        ((*head, "bot", "accuracy"), "baselines", (*bot, "evaluate", "accuracy")),
        ((*head, "bot", "accuracy"), "experiments", ("bot_real_val_accuracy",)),
        ((*head, "bot", "accuracy"), "router", ("accuracy", "bot")),
        ((*head, "bot", "macro_f1"), "baselines", (*bot, "evaluate", "macro_f1")),
        ((*head, "bot", "misroutes"), "baselines", (*bot, "evaluate", "misroutes")),
        ((*head, "bot", "misroutes"), "router", ("misroutes", "bot")),
        ((*head, "rows"), "router", ("validation", "rows")),
    ]
    for end in ends:
        pairs += [
            ((*head, "router", ci, end), "baselines", (*final, ci, end)),
            ((*head, "router", ci, end), "experiments", (*real, ci, end)),
            ((*head, "bot", ci, end), "baselines", (*bot, ci, end)),
            ((*head, paired, end), "baselines", ("paired_accuracy_points_tfidf_final_minus_bot", end)),
            ((*head, paired, end), "experiments", (*real, "paired_accuracy_points_minus_bot", end)),
        ]
    for ours, theirs in (
        ("count", ("triage", "count")),
        ("rate", ("triage", "rate")),
        ("sent_count", ("triage", "sent_count")),
        ("mean_confidence_triaged", ("triage", "mean_confidence")),
        ("accuracy_sent", ("accuracy", "sent")),
        ("accuracy_triaged", ("accuracy", "triaged")),
        ("misroutes_sent", ("misroutes", "sent")),
        ("misroutes_triaged", ("misroutes", "triaged")),
        ("threshold", ("router", "threshold")),
        ("vague_count", ("vague", "count")),
        ("vague_triaged_count", ("vague", "triaged_count")),
    ):
        pairs.append((("triage", ours), "router", theirs))
    policy = ("coverage_curve", "policy_point")
    pairs.append(((*policy, "accuracy_on_covered"), "router", ("accuracy", "sent")))
    pairs += [
        ((*months, "router_clarification_always_works"), "router", ("cost_per_month_rs", "router")),
        ((*months, "router_without_triage"), "router", ("cost_per_month_rs", "router_without_triage")),
        ((*months, "router_without_triage"), "baselines", (*final, "cost_rs_per_month")),
        (
            (*months, "router_clarification_fixes_nothing"),
            "router",
            ("cost_per_month_rs", "router_if_clarification_fixes_nothing"),
        ),
        ((*months, "bot"), "router", ("cost_per_month_rs", "bot")),
        ((*months, "bot"), "baselines", (*bot, "cost_rs_per_month")),
        ((*months, "bot_licence"), "router", ("cost_per_month_rs", "bot_licence")),
        ((*months, "bot_licence"), "baselines", ("meta", "bot_licence_rs_per_month")),
        (("costs", "bot_minus_router_rs"), "router", ("cost_per_month_rs", "bot_minus_router")),
        (("costs", "break_even_clarification_rate"), "router", ("break_even_clarification_rate",)),
        (("costs", "inputs", "cost_per_misroute_rs"), "router", ("cost_per_misroute_reporting",)),
        (("expected_score", "bot_label", "point"), "baselines", (*final, "agreement_with_team_label")),
        (("expected_score", "final_team", "inner_val_accuracy"), "experiments", ("supplementary", "base_inner_val", "accuracy")),
        (("seen_text", "test_seen_share"), "baselines", ("repeated_text", "test_seen_share")),
    ]
    for index, group in enumerate(("seen", "unseen")):
        for ours, theirs in (("router", "tfidf_final"), ("bot", "bot")):
            pairs.append(
                (("breakdowns", "text", index, f"{ours}_accuracy"), "baselines", ("repeated_text", "accuracy", theirs, group))
            )
    for team in config.TEAMS:
        for side, source in (("router", final), ("bot", bot)):
            for key in ("precision", "recall"):
                pairs.append((("per_team", team, side, key), "baselines", (*source, "evaluate", "per_class", team, key)))
        pairs.append((("per_team", team, "router", "recall"), "experiments", (*real, "per_team_recall", team)))
    pairs.append((("per_team", "Repairs", "router", "precision"), "router", ("repairs", "precision_all")))
    return pairs


def consistency(results: dict, sources: dict) -> dict:
    """Compares rounded results with the M4, M5a and M5b reports. ValueError on any mismatch."""
    checks, bad = [], []
    for path, name, source_path in _pairs(results):
        value, source = _pick(results, path), _pick(sources[name], source_path)
        key = " > ".join(str(p) for p in path)
        checks.append(
            {
                "key": key,
                "value": value,
                "source": f"{name}.json",
                "source_key": " > ".join(source_path),
                "source_value": source,
            }
        )
        if abs(value - source) > TOLERANCE:
            bad.append(f"{key} differs from {name}.json")
    if bad:
        raise ValueError("; ".join(bad))
    return {"tolerance": TOLERANCE, "count": len(checks), "checks": checks}


# --- Everything ---------------------------------------------------------------------
def compute_evaluation(
    train: pd.DataFrame, val: pd.DataFrame, test_text: pd.Series, outputs: list[dict], sources: dict, meta: dict
) -> dict:
    """Rounded results with the consistency block. `outputs` are Router.predict results for val, `meta` the router's."""
    truth, bot = val[TARGET].to_numpy(dtype=object), val[BOT].to_numpy(dtype=object)
    pred = np.array([o["team"] for o in outputs], dtype=object)
    confidence = np.array([o["confidence"] for o in outputs])
    triaged = np.array([o["needs_triage"] for o in outputs], dtype=bool)
    right, vague = pred == truth, vague_mask(val[TEXT]).to_numpy()
    train_texts = set(train[TEXT])
    months = len(pd.period_range(val["created_at_ist"].min(), val["created_at_ist"].max(), freq="M"))
    head = headline(truth, pred, bot)
    curve = coverage_curve(truth, pred, None, thresholds=CURVE_THRESHOLDS, confidence=confidence)
    policy = coverage_curve(truth, pred, None, thresholds=(config.TRIAGE_THRESHOLD,), confidence=confidence)[0]
    results = {
        "meta": {
            "model_name": meta["config"]["name"],
            "model_version": meta["model_version"],
            "training_window": meta["training_window"],
            "validation": {
                "rows": len(val),
                "first_day": str(val["created_at_ist"].min().date()),
                "last_day": str(val["created_at_ist"].max().date()),
                "months": months,
            },
            "teams": list(config.TEAMS),
            "ci_level": CI_LEVEL,
            "labelled_months": len(pd.period_range(config.TRAIN_START, config.VALID_END, freq="M")),
            "seed": config.SEED,
        },
        "headline": head,
        "per_team": per_team(truth, pred, bot),
        "confusion": {
            "router": evaluate(truth, pred)["confusion_matrix"],
            "bot": evaluate(truth, bot)["confusion_matrix"],
        },
        "coverage_curve": {"points": curve, "policy_point": policy},
        "calibration": calibration(right, confidence),
        "breakdowns": breakdowns(val, pred, triaged, val[TEXT].isin(train_texts).to_numpy()),
        "seen_text": {
            "validation_seen_share": float(val[TEXT].isin(train_texts).mean()),
            "test_seen_share": float(test_text.isin(train_texts).mean()),
        },
        "triage": triage_block(right, triaged, confidence, vague),
        "costs": costs(right, bot == truth, triaged, months),
        "failures": failures(val, train[TEXT], outputs, right, triaged, vague),
        "expected_score": expected_score(val, pred, vague, test_text, head, sources),
        "figures": [f"{FIGURES_DIR}/{name}" for name in FIGURES.values()],
    }
    results = round_floats(results)
    results["consistency"] = consistency(results, sources)
    return results


# --- Figures -----------------------------------------------------------------------
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#191919", "#666666", "#e3e3e3"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cce3fd", "#6aabf1", "#2370cd", "#0e3d7e"])
STYLE = {
    "font.family": "DejaVu Sans",  # ships with matplotlib
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
}


def _figure(width, height):
    fig = Figure(figsize=(width, height), dpi=150, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _clean(ax, grid=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _confusion_figure(results):
    teams, fig = results["meta"]["teams"], _figure(15, 6.6)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, side in zip(axes, ("router", "bot")):
        counts = np.array(results["confusion"][side], dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(len(teams)):
            for j in range(len(teams)):
                color = "white" if shares[i, j] > 0.55 else INK
                ax.text(j, i, f"{shares[i, j] * 100:.0f}%", ha="center", va="center", color=color, fontsize=9)
        ax.set_xticks(range(len(teams)), teams, rotation=35, ha="right")
        ax.set_yticks(range(len(teams)), teams)
        ax.tick_params(length=0, labelcolor=INK)
        ax.set_title("Router" if side == "router" else "Bot", loc="left", fontweight="bold")
        ax.set_xlabel("Predicted team")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Final team")
    fig.suptitle("Where each team's requests were sent (share of the final team's requests)")
    return fig


def _coverage_figure(results):
    points = [p for p in results["coverage_curve"]["points"] if p["accuracy_on_covered"] is not None]
    policy, bot = results["coverage_curve"]["policy_point"], results["headline"]["bot"]["accuracy"]
    fig = _figure(8.5, 5.4)
    ax = fig.subplots()
    ax.plot(
        [p["coverage"] * 100 for p in points],
        [p["accuracy_on_covered"] * 100 for p in points],
        color=BLUE, linewidth=2, marker="o", markersize=4, label="Router, by confidence threshold",
    )
    ax.axhline(bot * 100, color=ORANGE, linewidth=2, linestyle="--", label=f"Bot, all requests ({bot * 100:.1f}%)")
    x, y = policy["coverage"] * 100, policy["accuracy_on_covered"] * 100
    ax.plot([x], [y], "o", markersize=11, color=INK, markeredgecolor="white", markeredgewidth=2, zorder=5,
            label="Policy threshold")
    ax.annotate(
        f"Policy threshold {policy['threshold']:.4f}\n{x:.1f}% sent on, {y:.1f}% right",
        (x, y), xytext=(-14, -44), textcoords="offset points", ha="right", color=INK,
    )
    ax.set_xlabel("Requests sent straight on (%)")
    ax.set_ylabel("Accuracy on those requests (%)")
    ax.set_title("Accuracy against the share of requests sent straight on", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 0.1))
    _clean(ax, "y")
    return fig


def _reliability_figure(results):
    bins = results["calibration"]["bins"]
    fig = _figure(7, 6.2)
    ax = fig.subplots()
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.2, linestyle="--", label="Perfectly calibrated")
    x, y = [b["mean_confidence"] * 100 for b in bins], [b["accuracy"] * 100 for b in bins]
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor="white", label="Router")
    for b, px, py in zip(bins, x, y):
        below = py <= px
        ax.annotate(
            f"n={b['rows']:,}", (px, py), xytext=(7, -14) if below else (-7, 8), textcoords="offset points",
            ha="left" if below else "right", color=MUTED, fontsize=8.5,
        )
    ax.set(xlim=(0, 104), ylim=(0, 108))
    ax.set_xlabel("Mean confidence in the bin (%)")
    ax.set_ylabel("Accuracy in the bin (%)")
    ax.set_title("Confidence against accuracy, validation", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    _clean(ax, "both")
    return fig


def _recall_figure(results):
    teams, fig = results["meta"]["teams"], _figure(9.5, 6.2)
    ax = fig.subplots()
    rows = np.arange(len(teams))
    for side, color, shift, label in (("router", BLUE, -0.2, "Router"), ("bot", ORANGE, 0.2, "Bot")):
        values = [results["per_team"][team][side]["recall"] * 100 for team in teams]
        ax.barh(rows + shift, values, height=0.36, color=color, label=label)
        for row, value in zip(rows, values):
            ax.text(value + 0.8, row + shift, f"{value:.1f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(rows, [f"{team} ({results['per_team'][team]['support']})" for team in teams])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Recall (%): share of the final team's requests that were sent to it")
    ax.set_title("Recall per team, router against bot", loc="left", fontweight="bold")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper right", ncols=2)
    _clean(ax, "x")
    return fig


def plot_figures(results: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    makers = {
        "confusion": _confusion_figure,
        "coverage": _coverage_figure,
        "reliability": _reliability_figure,
        "recall": _recall_figure,
    }
    paths = []
    with matplotlib.rc_context({**matplotlib.rcParamsDefault, **STYLE, "backend": matplotlib.rcParams["backend"]}):
        for name, filename in FIGURES.items():
            path = out_dir / filename
            makers[name](results).savefig(path, metadata={"Software": None})  # no version stamp in the file
            paths.append(path)
    return paths


# --- Markdown -------------------------------------------------------------------------
def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _points(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _rs(x: float) -> str:
    return f"{x:,.0f}"


def _value(x) -> str:
    return str(x) if isinstance(x, int) else f"{x:.6f}"


def _range(ci: dict) -> str:
    return f"{_pct(ci['low'])} to {_pct(ci['high'])}"


def _headline_section(r: dict) -> list[str]:
    head, level = r["headline"], f"{r['meta']['ci_level'] * 100:.0f}%"
    paired = head["paired_accuracy_points_router_minus_bot"]
    rows = [
        f"| {name} | {_pct(s['accuracy'])} | {_range(s['accuracy_ci95'])} | {_pct(s['macro_f1'])} | {s['misroutes']} |"
        for name, s in (("Router", head["router"]), ("Bot", head["bot"]))
    ]
    return [
        "## Headline",
        "",
        f"| | Accuracy | {level} interval | Macro F-score | Misroutes |",
        "| --- | ---: | --- | ---: | ---: |",
        *rows,
        "",
        f"The router is {_points(paired['point'])} points above the bot on the same rows ({level} paired interval "
        f"{_points(paired['low'])} to {_points(paired['high'])}). These figures count every request, including those the "
        "router would send to triage.",
        "",
    ]


def _team_section(r: dict) -> list[str]:
    lines = [
        "## Per team",
        "",
        "| Team | Requests | Router precision | Bot precision | Router recall | Bot recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    teams = [(team, r["per_team"][team]) for team in r["meta"]["teams"]]  # stored JSON has sorted keys
    for team, s in teams:
        lines.append(
            f"| {team} | {s['support']} | {_pct(s['router']['precision'])} | {_pct(s['bot']['precision'])} | "
            f"{_pct(s['router']['recall'])} | {_pct(s['bot']['recall'])} |"
        )
    worse = [
        f"{team} {key}"
        for team, s in teams
        for key in ("precision", "recall")
        if s["router"][key] < s["bot"][key]
    ]
    note = f"Where the bot is better: {', '.join(worse)}." if worse else "The router is at least as good in every cell."
    return [*lines, "", note, ""]


def _figure_section(r: dict) -> list[str]:
    cal = r["calibration"]
    return [
        "## Figures",
        "",
        f"![Row-normalised confusion matrices]({FIGURES_DIR}/{FIGURES['confusion']})",
        "",
        f"![Coverage and accuracy]({FIGURES_DIR}/{FIGURES['coverage']})",
        "",
        f"![Reliability plot]({FIGURES_DIR}/{FIGURES['reliability']})",
        "",
        f"![Recall per team]({FIGURES_DIR}/{FIGURES['recall']})",
        "",
        f"Reliability uses {cal['bin_count']} equal-width bins. Mean confidence {_pct(cal['mean_confidence'])} against "
        f"{_pct(cal['accuracy'])} accuracy; expected calibration error {_pct(cal['expected_calibration_error'])}. "
        "Bins with few requests are noisy.",
        "",
        "| Confidence from | to | Requests | Mean confidence | Accuracy |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {b['from']:.1f} | {b['to']:.1f} | {b['rows']:,} | {_pct(b['mean_confidence'])} | {_pct(b['accuracy'])} |"
            for b in cal["bins"]
        ],
        "",
    ]


def _breakdown_section(r: dict) -> list[str]:
    titles = {"channel": "Channel", "month": "Month", "warranty_status": "Warranty status", "text": "Text in train window"}
    lines = ["## Breakdowns", ""]
    for name, title in titles.items():
        rows = r["breakdowns"][name]
        lines += [
            f"| {title} | Requests | Router accuracy | Bot accuracy | Lead, points | Triage rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['group']} | {row['rows']:,} | {_pct(row['router_accuracy'])} | {_pct(row['bot_accuracy'])} | "
                f"{_points(row['lead_points'])} | {_pct(row['triage_rate'])} |"
            )
        lines.append("")
    seen = r["seen_text"]
    lines += [
        f"Seen means the same cleaned text occurs in the train window: {_pct(seen['validation_seen_share'])} of validation, "
        f"{_pct(seen['test_seen_share'])} of test.",
        "",
    ]
    return lines


def _triage_section(r: dict) -> list[str]:
    t, rows = r["triage"], r["headline"]["rows"]
    return [
        "## Triage",
        "",
        "| | Requests | Accuracy | Misroutes |",
        "| --- | ---: | ---: | ---: |",
        f"| Sent straight on | {t['sent_count']:,} | {_pct(t['accuracy_sent'])} | {t['misroutes_sent']} |",
        f"| Triaged | {t['count']:,} | {_pct(t['accuracy_triaged'])} | {t['misroutes_triaged']} |",
        f"| All | {rows:,} | {_pct(r['headline']['router']['accuracy'])} | {r['headline']['router']['misroutes']} |",
        "",
        f"- {_pct(t['rate'])} of requests are triaged (confidence below {t['threshold']:.4f}, from policy costs, not tuned). "
        f"Their mean confidence is {_pct(t['mean_confidence_triaged'])}.",
        f"- Vague requests (M2 rule): {t['vague_triaged_count']} of {t['vague_count']} triaged "
        f"({_pct(t['vague_triaged_share'])}).",
        "",
    ]


def _cost_section(r: dict) -> list[str]:
    c, inputs = r["costs"], r["costs"]["inputs"]
    month = c["per_month_rs"]
    return [
        "## Rupees per month",
        "",
        "| Case | Rs per month |",
        "| --- | ---: |",
        f"| Router with triage, clarification always works (best case) | {_rs(month['router_clarification_always_works'])} |",
        f"| Router without triage | {_rs(month['router_without_triage'])} |",
        f"| Router with triage, clarification fixes nothing (worst case) | "
        f"{_rs(month['router_clarification_fixes_nothing'])} |",
        f"| Bot, same requests | {_rs(month['bot'])} |",
        f"| Bot licence, separate and not included above | {_rs(month['bot_licence'])} |",
        "",
        f"- Best-case saving against the bot: Rs {_rs(c['bot_minus_router_rs'])} per month. If clarification fixes nothing the "
        f"gap is only Rs {_rs(c['bot_minus_router_clarification_fixes_nothing_rs'])} per month. Both leave the licence out.",
        f"- Break-even clarification rate {_pct(c['break_even_clarification_rate'])}: triage beats no triage only if a question "
        "settles more than this share of the triaged misroutes. The data does not say how often it does; ask operations.",
        "",
        "Assumptions:",
        "",
        f"- Rs {inputs['cost_per_misroute_rs']:.2f} per misroute: Rs {inputs['cost_per_transfer_rs']} per transfer times the "
        f"average transfers per misrouted request, plus Rs {inputs['cost_per_misroute_contact_rs']} for the extra contact "
        f"(§4). The average covers all {r['meta']['labelled_months']} labelled months, validation included.",
        f"- The triage threshold uses the train-window figure instead, Rs {inputs['cost_per_misroute_train_rs']:.2f}.",
        "- Our misroutes are assumed to bounce as often as the bot's did.",
        f"- One clarifying contact costs Rs {inputs['cost_clarify_rs']}, the same as an extra customer contact.",
        f"- Validation covers {inputs['months']} months; totals are divided by that.",
        f"- The licence is Rs {inputs['bot_licence_per_year_rs']:,} per year. The cost of running the router and agent time "
        "beyond the contact are not in the data and not included.",
        "",
    ]


def _failure_section(r: dict) -> list[str]:
    f = r["failures"]
    lines = [
        "## When it fails",
        "",
        f"{f['errors']} of {r['headline']['rows']:,} requests are misrouted ({_pct(f['error_rate'])}); {f['errors_sent']} of them "
        f"are sent straight on. The right team is among the top three in {f['errors_with_right_team_in_top3']}.",
        "",
        "Categories overlap, except in the last column, where each misroute counts once in the first category that fits.",
        "",
        "| Category | Requests | Misroutes | Misroute rate | Sent straight on and wrong | Misroutes counted once |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in f["categories"]:
        lines.append(
            f"| {CATEGORIES[row['category']]} | {row['rows']:,} | {row['errors']} | {_pct(row['error_rate'])} | "
            f"{row['errors_sent']} | {row['exclusive_errors']} |"
        )
    lines += [
        "",
        f"A close call is a gap of at most {f['close_call_gap']:.2f} between the first and second probability. A category is "
        "a description, not a proven cause: compare its misroute rate with the overall one.",
        "",
        f"Router phrases seen most on these misroutes, with the number of misroutes (at most {f['phrase_max_words']} words, only "
        f"phrases found in at least {f['phrase_min_train_texts']} different train messages, §10):",
        "",
    ]
    for row in f["categories"]:
        phrases = "; ".join(f"'{p['phrase']}' ({p['count']})" for p in row["phrases"]) or "none common enough"
        lines.append(f"- {CATEGORIES[row['category']]}: {phrases}.")
    return [*lines, ""]


def _expected_section(r: dict) -> list[str]:
    e, level = r["expected_score"], f"{r['meta']['ci_level'] * 100:.0f}%"
    final, bot, vague, lead = e["final_team"], e["bot_label"], e["vague_ceiling"], e["monthly_lead"]
    return [
        "## Expected hidden-test score",
        "",
        f"**If hidden labels are the final team: {_pct(final['point'])}, range {_range(final)}** (macro F-score on validation "
        f"{_pct(final['macro_f1'])}). Accuracy is assumed to be the metric, and every request is scored on the router's first "
        "choice, triaged or not.",
        "",
        f"- Validation interval ({level}): {_range(final['ci95'])}. Its half-widths are multiplied by the square root of two "
        f"({final['width_factor']:.4f}), because the test score is itself a sample of similar size.",
        f"- Time drift: accuracy was {_pct(final['inner_val_accuracy'])} on the earlier inner validation window, "
        f"{_points(final['drift_allowance_points'])} points higher. That much is taken off the low end only. The inner model "
        "was fitted on fewer months, so this is a rough allowance.",
        f"- {_lead_sentence(lead)}",
        f"- Retraining on all {r['meta']['labelled_months']} months: not measured, because M6 fits no new model, so nothing is "
        "added for it.",
        f"- Vague ceiling: {_pct(vague['test_vague_share'])} of test messages are vague (M2 rule), and the router gets "
        f"{_pct(vague['validation_vague_accuracy'])} of vague validation messages right. Even with every other request right, "
        f"accuracy would be about {_pct(vague['ceiling_accuracy'])}.",
        "- The validation score is not untouched: the configuration is the M4 baseline, already scored on validation, and "
        "M5a's selection rule was amended after a validation result had been seen. The range is a judgement, not a "
        "guarantee.",
        "",
        "| Month | Router accuracy | Bot accuracy | Lead, points | Paired interval, points |",
        "| --- | ---: | ---: | ---: | --- |",
        *[
            f"| {row['month']} | {_pct(row['router_accuracy'])} | {_pct(row['bot_accuracy'])} | "
            f"{_points(row['lead_points']['point'])} | {_points(row['lead_points']['low'])} to "
            f"{_points(row['lead_points']['high'])} |"
            for row in lead["months"]
        ],
        "",
        f"**If hidden labels are the bot's `team_label` instead: {_pct(bot['point'])}, range {_range(bot)}** (interval "
        f"{_range(bot['ci95'])}, macro F-score {_pct(bot['macro_f1'])}). This is the router's agreement with the bot on "
        "validation, with the same widening. The bot itself would score perfectly by construction, so this score would reward "
        "copying its mistakes.",
        "",
    ]


def _consistency_section(r: dict) -> list[str]:
    c = r["consistency"]
    return [
        "## Consistency with earlier reports",
        "",
        f"All {c['count']} comparisons match within the tolerance in `{JSON_NAME}`; the build stops otherwise. Keys are in the "
        "JSON form of each report. The per-team rows are in the JSON only.",
        "",
        "| This report | Value | Source | Source key |",
        "| --- | ---: | --- | --- |",
        *[
            f"| {row['key']} | {_value(row['value'])} | {row['source']} | {row['source_key']} |"
            for row in c["checks"]
            if not row["key"].startswith(("per_team", "breakdowns"))
        ],
        "",
    ]


def render_markdown(results: dict) -> str:
    """From the rounded results only."""
    meta = results["meta"]
    window, val = meta["training_window"], meta["validation"]
    lines = [
        "# Evaluation evidence",
        "",
        f"Model `{meta['model_version']}`, fitted on {window['rows']:,} requests ({window['first_day']} to "
        f"{window['last_day']}) and scored against the final team on {val['rows']:,} validation requests "
        f"({val['first_day']} to {val['last_day']}). Reporting only: nothing was tuned on these numbers. Intervals are "
        "bootstrap intervals; repeated messages make them slightly too narrow.",
        "",
        *_headline_section(results),
        *_team_section(results),
        *_figure_section(results),
        *_breakdown_section(results),
        *_triage_section(results),
        *_cost_section(results),
        *_failure_section(results),
        *_expected_section(results),
        *_consistency_section(results),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def build(reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Write evaluation.json, evaluation.md and figures/ to reports_dir. Sources always come from config.REPORTS_DIR."""
    sources = load_sources(config.REPORTS_DIR)
    train_df, val_df = time_split(pd.read_parquet(config.TRAIN_CLEAN_PARQUET))
    test_text = pd.read_parquet(config.TEST_CLEAN_PARQUET, columns=[TEXT])[TEXT]
    router = train_router(train_df, sources)
    outputs = router.predict(to_records(val_df))
    results = compute_evaluation(train_df, val_df, test_text, outputs, sources, router.meta)
    markdown = render_markdown(results)
    payload = json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_figures(results, reports_dir / FIGURES_DIR)
    (reports_dir / JSON_NAME).write_text(payload, encoding="utf-8", newline="\n")
    (reports_dir / MD_NAME).write_text(markdown, encoding="utf-8", newline="\n")
    return results


if __name__ == "__main__":
    build()
