"""M2: data audit. Contract: docs/contracts.md#m2. Run: python -m src.audit"""
import json
import re
from pathlib import Path

import pandas as pd

from src import config

MONTHS_PER_YEAR = 12

# Words that name each product family in text_clean (regex alternatives, matched as whole words).
PRODUCT_PATTERNS = {
    "Air Fryer": "fryers?",
    "Ceiling Fan": "fans?",
    "Induction Cooktop": "cooktops?|induction",
    "Mixer Grinder": "mixers?|grinders?|mixies?",
    "Robot Vacuum": "vacuums?|robots?",
    "Room Heater": "heaters?",
    "Water Purifier": "purifiers?|ro",
}
PRODUCT_RULE = (
    "text_clean names at least one other product family and not the row's own product_family "
    "(whole-word match on the patterns listed)"
)

FILLER_WORDS = frozenset(
    """
    hi hello namaste sir madam maam team good morning evening afternoon dear please pls plz kindly thanks thank
    you urgent asap help need call back callback contact me us someone my our the a an i is am to of on for with
    about regarding query issue problem complaint request product order this it and from in ji hai ko se ka ki
    have want support service customer care number no regards
    """.split()
)
PRODUCT_WORDS = frozenset(
    """
    air fryer water purifier robot vacuum mixer grinder room heater ceiling fan induction cooktop ro kestrel home
    """.split()
)
VAGUE_RULE = (
    "no words of text_clean are left after removing the filler words and product words listed "
    "(digits and punctuation are ignored)"
)
_WORD = re.compile(r"[^\W\d_]+")


def _correct(df: pd.DataFrame) -> pd.Series:
    return df["team_label_canon"] == df["final_team_canon"]


def _months() -> list[str]:
    """Every month of the labelled window, including any month without rows."""
    return pd.period_range(config.TRAIN_START, config.VALID_END, freq="M").astype(str).tolist()


def agreement_by(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Share of rows where the bot label equals the final team, per value of col."""
    return {str(k): float(v) for k, v in _correct(df).groupby(df[col]).mean().items()}


def team_volume(teams: pd.Series) -> dict[str, int]:
    """Row count per team, zero-filled. Anything but a current team name is an error."""
    if not teams.isin(config.TEAMS).all():
        raise ValueError("Team name outside config.TEAMS")
    counts = teams.value_counts()
    return {t: int(counts.get(t, 0)) for t in config.TEAMS}


def confusion(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Counts as bot label -> final team."""
    return {b: team_volume(df.loc[df["team_label_canon"] == b, "final_team_canon"]) for b in config.TEAMS}


def product_conflict_mask(df: pd.DataFrame) -> pd.Series:
    """Rows matching PRODUCT_RULE."""
    if not df["product_family"].isin(list(PRODUCT_PATTERNS)).all():
        raise ValueError("product_family without a pattern")
    any_named = pd.Series(False, index=df.index)
    own_named = pd.Series(False, index=df.index)
    for family, pattern in PRODUCT_PATTERNS.items():
        named = df["text_clean"].str.contains(rf"\b(?:{pattern})\b", regex=True)
        any_named |= named
        own_named |= named & (df["product_family"] == family)
    return any_named & ~own_named


def vague_mask(text: pd.Series) -> pd.Series:
    """Rows matching VAGUE_RULE."""
    ignored = FILLER_WORDS | PRODUCT_WORDS
    return text.map(lambda s: set(_WORD.findall(s)) <= ignored).astype(bool)


def _cost_inputs(df: pd.DataFrame) -> dict:
    correct = _correct(df)
    extra = int(df.loc[correct, "transfers"].sum())
    return {
        "rs_per_transfer": config.COST_PER_TRANSFER,
        "rs_per_misroute": config.COST_PER_MISROUTE_CONTACT,
        "transfers": int(df["transfers"].sum()),
        "misroutes": int((~correct).sum()),
        "transfers_on_correct_rows": extra,
        "transfers_on_correct_rows_rs_per_month": extra * config.COST_PER_TRANSFER / len(_months()),
        "licence_rs_per_year": config.BOT_LICENCE_PER_YEAR,
        "licence_rs_per_month": config.BOT_LICENCE_PER_YEAR / MONTHS_PER_YEAR,
    }


def _cost_per_month(inputs: dict, transfers: int) -> float:
    """(transfers * transfer cost + misroutes * contact cost) / months. Licence not included."""
    total = transfers * inputs["rs_per_transfer"] + inputs["misroutes"] * inputs["rs_per_misroute"]
    return total / len(_months())


def compute_metrics(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """All audit metrics. Aggregates only: no request text or ids."""
    correct = _correct(train)
    has_transfers = train["transfers"] > 0
    vague = vague_mask(train["text_clean"])
    months = _months()
    inputs = _cost_inputs(train)
    metrics = {
        "bot_vs_final_agreement": float(correct.mean()),
        "agreement_by_month": agreement_by(train, "created_month"),
        "agreement_by_source": agreement_by(train, "source"),
        "agreement_by_channel": agreement_by(train, "channel"),
        "confusion_bot_to_final": confusion(train),
        "transfers_distribution": {
            str(k): int(v) for k, v in train["transfers"].value_counts().sort_index().items()
        },
        "transfer_consistency": {
            "misrouted_with_0_transfers": int((~correct & ~has_transfers).sum()),
            "correct_with_transfers": int((correct & has_transfers).sum()),
        },
        "final_volume_by_team": team_volume(train["final_team_canon"]),
        "bot_label_volume_by_team": team_volume(train["team_label_canon"]),
        "final_team_volume_by_month": {
            m: team_volume(train.loc[train["created_month"] == m, "final_team_canon"]) for m in months
        },
        "product_text_conflict_count": int(product_conflict_mask(train).sum()),
        "product_text_conflict_rule": PRODUCT_RULE,
        "product_text_conflict_patterns": PRODUCT_PATTERNS,
        "vague_request_estimate": {
            "count": int(vague.sum()),
            "rule": VAGUE_RULE,
            "filler_words": sorted(FILLER_WORDS),
            "product_words": sorted(PRODUCT_WORDS),
            "final_team_distribution": team_volume(train.loc[vague, "final_team_canon"]),
        },
        "bot_cost_rs_per_month": _cost_per_month(inputs, inputs["transfers"]),
        "bot_cost_misrouted_only_rs_per_month": _cost_per_month(
            inputs, inputs["transfers"] - inputs["transfers_on_correct_rows"]
        ),
        "cost_inputs": inputs,
        "months_covered": {"first": months[0], "last": months[-1], "count": len(months)},
    }
    for suffix, col in (("", "text_clean"), ("_raw", "request_text")):
        metrics[f"duplicate_text_count{suffix}"] = int(train[col].duplicated().sum())
        metrics[f"test_text_overlap_count{suffix}"] = int(test[col].isin(train[col]).sum())
    return metrics


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    align = ["---"] + ["---:"] * (len(header) - 1)
    return ["| " + " | ".join(cells) + " |" for cells in [header, align, *rows]] + [""]


def render_markdown(metrics: dict) -> str:
    """Human-readable summary. Every number comes from metrics; the prose must stay digit-free."""
    m = metrics
    teams = list(config.TEAMS)
    months, cost, vague = m["months_covered"], m["cost_inputs"], m["vague_request_estimate"]
    consistency = m["transfer_consistency"]

    lines = [
        "# Data audit",
        "",
        "Generated by `src/audit.py` from `reports/audit_metrics.json`; do not edit by hand.",
        "Train aggregates plus test overlap counts: no request text, no model, no train/validation split.",
        "",
        f"Months covered: {months['first']} to {months['last']} ({months['count']} months).",
        "",
        "## Bot label against final team",
        "",
        f"The bot's label matched the team that closed the request on {_pct(m['bot_vs_final_agreement'])} of requests;",
        f"{_n(cost['misroutes'])} requests were misrouted.",
        "",
    ]
    for label in ("month", "source", "channel"):
        values = m[f"agreement_by_{label}"]
        lines += _table([label.capitalize(), "Agreement"], [[k, _pct(values[k])] for k in sorted(values)])

    lines += ["### Confusion: bot label (rows) to final team (columns)", ""]
    conf = m["confusion_bot_to_final"]
    lines += _table(["Bot label", *teams], [[b, *(_n(conf[b][f]) for f in teams)] for b in teams])

    lines += ["## Volume", ""]
    lines += _table(
        ["Team", "Closed by team", "Sent by bot"],
        [[t, _n(m["final_volume_by_team"][t]), _n(m["bot_label_volume_by_team"][t])] for t in teams],
    )
    lines += ["### Requests closed per team and month", ""]
    by_month = m["final_team_volume_by_month"]
    lines += _table(["Month", *teams], [[k, *(_n(by_month[k][t]) for t in teams)] for k in sorted(by_month)])

    lines += ["## Transfers", ""]
    dist = m["transfers_distribution"]
    lines += _table(["Transfers", "Requests"], [[k, _n(dist[k])] for k in sorted(dist, key=int)])
    lines += [
        "`transfers` is used as recorded and does not fully agree with the team columns:",
        "",
        f"- misrouted but no transfer recorded: {_n(consistency['misrouted_with_0_transfers'])} requests",
        f"- bot label correct but transfers recorded: {_n(consistency['correct_with_transfers'])} requests",
        "",
        "## Text quality",
        "",
        f"**Product conflicts:** {_n(m['product_text_conflict_count'])} requests.",
        f"Rule: {m['product_text_conflict_rule']}. A word such as \"fan\" can also mean a part, so some may be false.",
        "",
    ]
    patterns = m["product_text_conflict_patterns"]
    lines += [f"- {family}: `{patterns[family]}`" for family in sorted(patterns)]
    lines += [
        "",
        "**Repeated text:** repeats count every row after the first use of a text; overlap counts test rows whose text",
        "appears in the labelled data. Cleaned text is the headline because it is what models see.",
        "",
    ]
    lines += _table(
        ["Text", "Repeats", "Test rows overlapping"],
        [
            ["Cleaned", _n(m["duplicate_text_count"]), _n(m["test_text_overlap_count"])],
            ["Raw", _n(m["duplicate_text_count_raw"]), _n(m["test_text_overlap_count_raw"])],
        ],
    )
    lines += [
        f"**Vague requests (estimate):** {_n(vague['count'])} requests.",
        f"Rule: {vague['rule']}. The word lists were chosen by looking at this data, so treat the count as an estimate.",
        "",
        f"- Filler words: {', '.join(vague['filler_words'])}",
        f"- Product words: {', '.join(vague['product_words'])}",
        "",
        "Teams that closed the vague requests:",
        "",
    ]
    lines += _table(["Team", "Vague requests"], [[t, _n(vague["final_team_distribution"][t])] for t in teams])

    lines += [
        "## Cost of the bot",
        "",
        f"Bot transfer and contact cost: **Rs {_n(m['bot_cost_rs_per_month'])} per month**,",
        f"from {_n(cost['transfers'])} transfers at Rs {_n(cost['rs_per_transfer'])} plus {_n(cost['misroutes'])} misroutes at Rs {_n(cost['rs_per_misroute'])},",
        f"divided by {months['count']} months.",
        "",
        f"- From misrouted requests only: Rs {_n(m['bot_cost_misrouted_only_rs_per_month'])} per month.",
        f"- {_n(consistency['misrouted_with_0_transfers'])} misroutes have no recorded transfer, so only their contact",
        "  cost is counted.",
        f"- Includes {_n(cost['transfers_on_correct_rows'])} transfers on requests the bot labelled correctly, worth",
        f"  Rs {_n(cost['transfers_on_correct_rows_rs_per_month'])} per month, which a better router may not avoid.",
        f"- Excludes the bot licence: Rs {_n(cost['licence_rs_per_year'])} per year, or",
        f"  Rs {_n(cost['licence_rs_per_month'])} per month.",
    ]
    return "\n".join(lines) + "\n"


def build(out_dir: Path = config.REPORTS_DIR) -> dict:
    """Write the metrics JSON and its markdown summary to out_dir and return the metrics."""
    metrics = compute_metrics(
        pd.read_parquet(config.TRAIN_CLEAN_PARQUET), pd.read_parquet(config.TEST_CLEAN_PARQUET)
    )
    markdown = render_markdown(metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    (out_dir / config.AUDIT_METRICS_JSON.name).write_text(payload, encoding="utf-8", newline="\n")
    (out_dir / config.DATA_AUDIT_MD.name).write_text(markdown, encoding="utf-8", newline="\n")
    return metrics


if __name__ == "__main__":
    build()
