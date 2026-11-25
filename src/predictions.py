"""M7: final model and test predictions. Contract: docs/contracts.md#m7.

Run: python -m src.predictions train, then python -m src.predictions predict.

The final model is the M5a configuration fitted on every labelled row. The triage threshold stays at the M5b value.
Predictions go through the saved model and Router.predict, the same path the API uses.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from src import baselines, config, experiments
from src.audit import vague_mask
from src.baselines import TARGET, TEXT, round_floats
from src.router import Router, to_records

COLUMNS = ["request_id", "team"]
LAST_LABELLED_DAY = str(config.VALID_END)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def train_final(models_dir: Path = config.MODELS_DIR) -> Router:
    """Fit the chosen configuration on all labelled rows and save it. The threshold is not recomputed."""
    labelled = pd.read_parquet(config.TRAIN_CLEAN_PARQUET)
    cfg = _read_json(config.REPORTS_DIR / experiments.JSON_NAME)["chosen"]["config"]
    router = Router.train(labelled, cfg)
    router.save(Path(models_dir) / config.ROUTER_JOBLIB.name)
    return router


def load_final(models_dir: Path = config.MODELS_DIR) -> Router:
    """Load the saved model; refuse one that was not fitted up to the last labelled day."""
    router = Router.load(Path(models_dir) / config.ROUTER_JOBLIB.name)
    last_day = router.meta["training_window"]["last_day"]
    if last_day != LAST_LABELLED_DAY:
        raise RuntimeError(f"not the final model: it was fitted up to {last_day}, expected {LAST_LABELLED_DAY}")
    return router


def predict_test(router: Router, test_df: pd.DataFrame, sample_ids: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Predictions in sample order, and the router's full outputs in test_df order."""
    ids = test_df["request_id"].tolist()
    if len(set(sample_ids)) != len(sample_ids) or len(set(ids)) != len(ids) or set(ids) != set(sample_ids):
        raise ValueError(f"request ids differ: {len(ids)} in test, {len(sample_ids)} in the sample submission")
    outputs = router.predict(to_records(test_df))
    teams = {request_id: output["team"] for request_id, output in zip(ids, outputs, strict=True)}
    return pd.DataFrame({"request_id": sample_ids, "team": [teams[i] for i in sample_ids]}, columns=COLUMNS), outputs


def write_predictions(df: pd.DataFrame, path: Path) -> None:
    """UTF-8 with LF endings. Written next to the target and then moved, so a failure leaves the old file."""
    if list(df.columns) != COLUMNS or df.isna().any().any() or df["request_id"].duplicated().any():
        raise ValueError("predictions need the columns request_id, team, without nulls or repeated ids")
    if not df["team"].isin(config.TEAMS).all():
        raise ValueError("predictions contain a team that is not one of the current names")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validation_bias(reports_dir: Path = config.REPORTS_DIR) -> dict:
    """How far the same configuration over- or under-predicted each team on validation (M4 and M5b reports)."""
    reference = _read_json(reports_dir / baselines.JSON_NAME)["baselines"]["tfidf_final"]["evaluate"]
    accuracy = _read_json(reports_dir / config.ROUTER_JSON.name)["accuracy"]
    if reference["accuracy"] != accuracy["all"]:
        raise ValueError("baselines.json and router.json describe different models")
    matrix, rows = reference["confusion_matrix"], reference["n"]  # rows: final team, columns: predicted
    teams = {}
    for i, team in enumerate(config.TEAMS):
        actual, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
        teams[team] = {"actual": actual, "predicted": predicted, "points": 100 * (predicted - actual) / rows}
    return {
        "rows": rows,
        "accuracy": accuracy["all"],
        "accuracy_auto_routed": accuracy["sent"],
        "accuracy_triaged": accuracy["triaged"],
        "teams": teams,
    }


def summarise(
    outputs: list[dict], test_df: pd.DataFrame, recent_df: pd.DataFrame, meta: dict, val_bias: dict
) -> dict:
    """Aggregates only. outputs are in test_df order; recent_df holds the last labelled months."""
    frame = pd.DataFrame(
        {
            "team": [o["team"] for o in outputs],
            "triaged": [bool(o["needs_triage"]) for o in outputs],
            "month": test_df["created_month"].tolist(),
            "vague": vague_mask(test_df[TEXT]).tolist(),
        }
    )
    rows, months = len(frame), sorted(frame["month"].unique().tolist())
    recent_months = int(recent_df["created_month"].nunique())
    volume, comparison = {}, {}

    def split(part: pd.DataFrame) -> dict:
        triaged = int(part["triaged"].sum())
        return {"total": len(part), "auto_routed": len(part) - triaged, "triaged": triaged}

    for team in config.TEAMS:
        mine = frame[frame["team"] == team]
        volume[team] = {
            **split(mine),
            "share": len(mine) / rows,
            "per_month": len(mine) / len(months),
            "by_month": {month: split(mine[mine["month"] == month]) for month in months},
        }
        recent_count = int((recent_df[TARGET] == team).sum())
        comparison[team] = {
            "recent_count": recent_count,
            "recent_per_month": recent_count / recent_months,
            "recent_share": recent_count / len(recent_df),
            "predicted_share": len(mine) / rows,
            "gap_points": 100 * (len(mine) / rows - recent_count / len(recent_df)),
        }
    sent = int((~frame["triaged"]).sum())
    sent_by_month = {m: sum(volume[t]["by_month"][m]["auto_routed"] for t in config.TEAMS) for m in months}
    triaged_by_month = {m: sum(volume[t]["by_month"][m]["triaged"] for t in config.TEAMS) for m in months}
    auto_routed_only = {
        "total": sent,
        "total_per_month": sent / len(months),
        "total_by_month": sent_by_month,
        "triaged_count": rows - sent,
        "triaged_per_month": (rows - sent) / len(months),
        "triaged_by_month": triaged_by_month,
        "teams": {
            team: {
                "count": volume[team]["auto_routed"],
                "share": volume[team]["auto_routed"] / sent if sent else 0.0,
                "per_month": volume[team]["auto_routed"] / len(months),
                "by_month": {m: volume[team]["by_month"][m]["auto_routed"] for m in months},
            }
            for team in config.TEAMS
        },
    }
    largest = max(config.TEAMS, key=lambda t: abs(comparison[t]["gap_points"]))
    return {
        "model": {key: meta[key] for key in ("model_version", "training_window", "threshold")},
        "test": {
            "rows": rows,
            "first_day": str(test_df["created_at_ist"].min().date()),
            "last_day": str(test_df["created_at_ist"].max().date()),
            "months": months,
            "rows_by_month": {month: int((frame["month"] == month).sum()) for month in months},
        },
        "volume": volume,
        "auto_routed_only": auto_routed_only,
        "triage": {
            "count": int(frame["triaged"].sum()),
            "share": float(frame["triaged"].mean()),
            "by_month": {m: float(frame.loc[frame["month"] == m, "triaged"].mean()) for m in months},
            "allowed_low": config.TRIAGE_SHARE_MIN,
            "allowed_high": config.TRIAGE_SHARE_MAX,
        },
        "vague": {
            "count": int(frame["vague"].sum()),
            "share": float(frame["vague"].mean()),
            "triaged_count": int((frame["vague"] & frame["triaged"]).sum()),
        },
        "comparison": {
            "recent_rows": len(recent_df),
            "recent_months": recent_months,
            "recent_first_day": str(recent_df["created_at_ist"].min().date()),
            "recent_last_day": str(recent_df["created_at_ist"].max().date()),
            "teams": comparison,
            "largest_gap_team": largest,
            "largest_gap_points": abs(comparison[largest]["gap_points"]),
            "limit_points": config.MAX_SHARE_GAP_POINTS,
        },
        "validation": val_bias,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_markdown(results: dict) -> str:
    model, test, volume, triage = results["model"], results["test"], results["volume"], results["triage"]
    vague, comparison, val = results["vague"], results["comparison"], results["validation"]
    window, months = model["training_window"], test["months"]
    by_volume = sorted(config.TEAMS, key=lambda t: -volume[t]["total"])
    busiest = by_volume[0]
    lines = [
        "# Test predictions: summary",
        "",
        f"Model `{model['model_version']}`, fitted on all {window['rows']:,} labelled requests ({window['first_day']} to "
        f"{window['last_day']}). {test['rows']:,} test requests ({test['first_day']} to {test['last_day']}), predicted "
        "through the saved router, as the API does.",
        "",
        f"This model has not been scored: no labels after its training window are available. The validation accuracy of "
        f"{_pct(val['accuracy'])} belongs to the same configuration fitted on the train window only.",
        "",
        "## Predicted volume per team",
        "",
        "Auto-routed requests go straight to the team. For triaged requests the team is only the model's first guess: on "
        f"validation it was right for {_pct(val['accuracy_triaged'])} of them (auto-routed: "
        f"{_pct(val['accuracy_auto_routed'])}). The auto-routed column is the reliable part; the triaged requests will "
        "be spread over the teams in a way that is not yet known.",
        "",
        "| Team | Total | Share | Per month | Auto-routed | Triaged (first guess) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for team in by_volume:
        v = volume[team]
        lines.append(
            f"| {team} | {v['total']:,} | {_pct(v['share'])} | {v['per_month']:.1f} | {v['auto_routed']:,} | "
            f"{v['triaged']:,} |"
        )
    sent = results["auto_routed_only"]
    by_sent = sorted(config.TEAMS, key=lambda t: -sent["teams"][t]["count"])
    month_cells = {team: [sent["teams"][team]["by_month"][m] for m in months] for team in config.TEAMS}

    def cells(counts) -> str:
        return " | ".join(f"{count:,}" for count in counts)

    lines += [
        "",
        "## Auto-routed requests only",
        "",
        "Each team counts only the requests sent straight to it; shares are of all auto-routed requests. Triaged requests "
        "are one separate count because their team is not yet known. They will still reach some team, so each team's count "
        "is not its full workload.",
        "",
        f"| Team | Auto-routed | Share | Per month | {' | '.join(months)} |",
        "| --- | ---: | ---: | ---: |" + " ---: |" * len(months),
        *(
            f"| {team} | {sent['teams'][team]['count']:,} | {_pct(sent['teams'][team]['share'])} | "
            f"{sent['teams'][team]['per_month']:.1f} | {cells(month_cells[team])} |"
            for team in by_sent
        ),
        f"| All auto-routed | {sent['total']:,} | | {sent['total_per_month']:.1f} | "
        f"{cells(sent['total_by_month'][m] for m in months)} |",
        f"| Triaged, team not yet known | {sent['triaged_count']:,} | | {sent['triaged_per_month']:.1f} | "
        f"{cells(sent['triaged_by_month'][m] for m in months)} |",
    ]
    lines += [
        "",
        f"Busiest by total predicted volume: {busiest}, then {by_volume[1]} and {by_volume[2]}. Read with the bias table below.",
        "",
        "## By month (total, of which triaged)",
        "",
        f"| Team | {' | '.join(months)} |",
        "| --- |" + " ---: |" * len(months),
    ]
    for team in by_volume:
        cells = [volume[team]["by_month"][m] for m in months]
        lines.append(f"| {team} | " + " | ".join(f"{c['total']:,} ({c['triaged']:,})" for c in cells) + " |")
    lines += [
        "| All requests | " + " | ".join(f"{test['rows_by_month'][m]:,}" for m in months) + " |",
        "| Triage share | " + " | ".join(_pct(triage["by_month"][m]) for m in months) + " |",
        "",
        "## Triage and vague requests",
        "",
        f"Triaged: {triage['count']:,} requests, {_pct(triage['share'])} (sanity band {_pct(triage['allowed_low'])} to "
        f"{_pct(triage['allowed_high'])}). Confidence threshold {model['threshold']:.4f}, frozen before this model was "
        "fitted.",
        "",
        f"Vague (rule-based estimate): {vague['count']:,} requests, {_pct(vague['share'])}; {vague['triaged_count']:,} of "
        "them are triaged.",
        "",
        "## Compared with the last labelled months",
        "",
        f"Closing team of {comparison['recent_rows']:,} requests, {comparison['recent_first_day']} to "
        f"{comparison['recent_last_day']} ({comparison['recent_months']} months). Largest gap: "
        f"{comparison['largest_gap_team']}, {comparison['largest_gap_points']:.2f} points (limit "
        f"{comparison['limit_points']}).",
        "",
        "| Team | Predicted share | Recent share | Gap, points | Recent per month |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for team in by_volume:
        c = comparison["teams"][team]
        lines.append(
            f"| {team} | {_pct(c['predicted_share'])} | {_pct(c['recent_share'])} | {c['gap_points']:+.2f} | "
            f"{c['recent_per_month']:.1f} |"
        )
    lines += [
        "",
        "## Known bias: do not read the volumes as headcount",
        "",
        f"On the {val['rows']:,} validation requests the same configuration predicted some teams too often, so predicted "
        "volumes are not unbiased. No correction is applied: the bias on the test months is not known.",
        "",
        "| Team | Predicted on validation | Closed by the team | Difference, points |",
        "| --- | ---: | ---: | ---: |",
    ]
    for team in by_volume:
        b = val["teams"][team]
        lines.append(f"| {team} | {b['predicted']:,} | {b['actual']:,} | {b['points']:+.2f} |")
    lines += [
        "",
        "Notes: the predicted team is always the model's first choice, with no rule or lookup on top. Some test messages "
        "also appear, word for word, in the training data. Predictions are reproducible byte for byte on one machine with "
        "the pinned library versions; another processor or maths library could flip a near-tie.",
        "",
    ]
    return "\n".join(lines)


def build(
    root: Path = config.ROOT, reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR
) -> dict:
    """Predict the test set with the saved final model; write predictions.csv and the summary."""
    router = load_final(models_dir)
    test_df = pd.read_parquet(config.TEST_CLEAN_PARQUET)
    labelled = pd.read_parquet(config.TRAIN_CLEAN_PARQUET)
    days = labelled["created_at_ist"].dt.date
    recent = labelled[days.between(*config.LAST_MONTHS)]
    # Only the ids of the sample are read; its team values are placeholders.
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV, usecols=["request_id"], dtype=str)
    predictions, outputs = predict_test(router, test_df, sample["request_id"].tolist())
    results = round_floats(summarise(outputs, test_df, recent, router.meta, validation_bias()))
    write_predictions(predictions, Path(root) / config.PREDICTIONS_CSV.name)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (reports_dir / config.PREDICTIONS_SUMMARY_JSON.name).write_text(payload, encoding="utf-8", newline="\n")
    (reports_dir / config.PREDICTIONS_SUMMARY_MD.name).write_text(
        render_markdown(results), encoding="utf-8", newline="\n"
    )
    return results


if __name__ == "__main__":
    steps = {"train": train_final, "predict": build}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit("usage: python -m src.predictions train|predict")
    steps[sys.argv[1]]()
