"""M2: data audit. All example strings here are invented, not customer text."""
import hashlib
import io
import json
import random
import re
import tokenize
from datetime import date

import pandas as pd
import pytest

TEAMS = {
    "Installs & Demo",
    "Repairs",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
}
OLD_TO_NEW = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}
NESTED_KEYS = {
    "transfer_consistency": {"misrouted_with_0_transfers", "correct_with_transfers"},
    "vague_request_estimate": {"count", "rule", "filler_words", "product_words", "final_team_distribution"},
    "months_covered": {"first", "last", "count"},
    "cost_inputs": {
        "rs_per_transfer",
        "rs_per_misroute",
        "transfers",
        "misroutes",
        "transfers_on_correct_rows",
        "transfers_on_correct_rows_rs_per_month",
        "licence_rs_per_year",
        "licence_rs_per_month",
    },
}
TOP_KEYS = {
    "bot_vs_final_agreement",
    "agreement_by_month",
    "agreement_by_source",
    "agreement_by_channel",
    "confusion_bot_to_final",
    "transfers_distribution",
    "final_volume_by_team",
    "bot_label_volume_by_team",
    "final_team_volume_by_month",
    "product_text_conflict_count",
    "product_text_conflict_rule",
    "product_text_conflict_patterns",
    "duplicate_text_count",
    "duplicate_text_count_raw",
    "test_text_overlap_count",
    "test_text_overlap_count_raw",
    "bot_cost_rs_per_month",
    "bot_cost_misrouted_only_rs_per_month",
} | set(NESTED_KEYS)

# Independent copies of the two text rules, as plain words. Kept separate from src/audit.py on purpose.
FAMILY_WORDS = {
    "Air Fryer": {"fryer", "fryers"},
    "Ceiling Fan": {"fan", "fans"},
    "Induction Cooktop": {"cooktop", "cooktops", "induction"},
    "Mixer Grinder": {"mixer", "mixers", "grinder", "grinders", "mixie", "mixies"},
    "Robot Vacuum": {"vacuum", "vacuums", "robot", "robots"},
    "Room Heater": {"heater", "heaters"},
    "Water Purifier": {"purifier", "purifiers", "ro"},
}
IGNORED_WORDS = set(
    """
    hi hello namaste sir madam maam team good morning evening afternoon dear please pls plz kindly thanks thank
    you urgent asap help need call back callback contact me us someone my our the a an i is am to of on for with
    about regarding query issue problem complaint request product order this it and from in ji hai ko se ka ki
    have want support service customer care number no regards
    air fryer water purifier robot vacuum mixer grinder room heater ceiling fan induction cooktop ro kestrel home
    """.split()
)

MONTH = re.compile(r"\b\d{4}-\d{2}\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


@pytest.fixture(scope="module")
def audit():
    from src import audit

    return audit


@pytest.fixture(scope="module")
def train(processed_dir):
    return pd.read_parquet(processed_dir / "train_clean.parquet")


@pytest.fixture(scope="module")
def test_set(processed_dir):
    return pd.read_parquet(processed_dir / "test_clean.parquet")


@pytest.fixture(scope="module")
def out_dir(audit, reports_dir, processed_dir, tmp_path_factory):
    # Never build into reports/: a broken run must not replace the real files.
    path = tmp_path_factory.mktemp("audit")
    before = file_state(reports_dir, processed_dir)
    audit.build(path)
    assert file_state(reports_dir, processed_dir) == before, "build() changed reports/ or data/processed"
    return path


@pytest.fixture(scope="module")
def metrics(out_dir):
    return json.loads((out_dir / "audit_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(out_dir):
    return (out_dir / "data_audit.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def recomputed(train):
    """Canonical bot label from the raw column, so the test does not lean on M1's derived columns."""
    bot = train["team_label"].replace(OLD_TO_NEW)
    return pd.DataFrame({"bot": bot, "final": train["final_team_canon"], "ok": bot == train["final_team_canon"]})


def words(text):
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def letter_words(text):
    return "".join(c if c.isalpha() else " " for c in text).split()


def expected_conflict(family, text):
    named = {f for f, names in FAMILY_WORDS.items() if names & set(words(text))}
    return bool(named) and family not in named


def expected_vague(text):
    return not set(letter_words(text)) - IGNORED_WORDS


def word_runs(text, n=4):
    tokens = words(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def file_state(*folders):
    # The time is included so that rewriting a file with the same bytes is also seen.
    files = [p for d in folders for p in d.rglob("*") if p.is_file()]
    return {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}


def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, path + (key,))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, path + (i,))
    else:
        yield path, node


def unmatched_numbers(md, metrics):
    """Numbers in the markdown that are not in the metrics, allowing for rounding and percentages."""
    values, strings = [], set()
    for path, value in leaves(metrics):
        # List positions are not metrics, so only dict keys count.
        strings.update(p for p in path if isinstance(p, str))
        if isinstance(value, str):
            strings.add(value)
        else:
            values.append(value)
    values += [int(s) for s in strings if s.isdigit()]

    bad = [m for m in MONTH.findall(md) if m not in strings]
    for token in NUMBER.findall(MONTH.sub(" ", md)):
        text = token.rstrip("%").replace(",", "")
        decimals = len(text.partition(".")[2])
        shown = float(text)
        if token.endswith("%"):
            ok = any(round(v * 100, decimals) == shown for v in values if isinstance(v, float) and 0 <= v <= 1)
        else:
            # A share must not count as a match for a bare 0 or 1.
            ok = any(round(v, decimals) == shown for v in values if isinstance(v, int) or v > 1)
        if not ok:
            bad.append(token)
    return bad


# --- Criterion 1 -----------------------------------------------------------
def test_all_keys_present(metrics):
    assert set(metrics) == TOP_KEYS
    for key, nested in NESTED_KEYS.items():
        assert set(metrics[key]) == nested
    assert metrics["product_text_conflict_rule"].strip()
    assert metrics["vague_request_estimate"]["rule"].strip()
    assert metrics["vague_request_estimate"]["filler_words"]


def test_output_paths(config, reports_dir):
    assert config.AUDIT_METRICS_JSON == reports_dir / "audit_metrics.json"
    assert config.DATA_AUDIT_MD == reports_dir / "data_audit.md"


def test_build_writes_exactly_two_files(out_dir):
    assert {p.name for p in out_dir.rglob("*")} == {"audit_metrics.json", "data_audit.md"}


# --- Criterion 2 -----------------------------------------------------------
def test_agreement_and_cost_recomputed(metrics, train, recomputed):
    months = train["created_at_ist"].dt.to_period("M").nunique()
    misroutes = int((~recomputed["ok"]).sum())
    cost = (int(train["transfers"].sum()) * 305 + misroutes * 260) / months
    assert metrics["bot_vs_final_agreement"] == pytest.approx(recomputed["ok"].sum() / len(train), abs=1e-12)
    assert metrics["bot_cost_rs_per_month"] == pytest.approx(cost, abs=1e-6)


def test_agreement_breakdowns_recomputed(metrics, train, recomputed):
    month = train["created_at_ist"].dt.strftime("%Y-%m")
    for key, col in [("month", month), ("source", train["source"]), ("channel", train["channel"])]:
        expected = {k: g.sum() / len(g) for k, g in recomputed["ok"].groupby(col)}
        assert metrics[f"agreement_by_{key}"] == pytest.approx(expected, abs=1e-12)


def test_cost_inputs_recomputed(metrics, train, recomputed):
    inputs, months = metrics["cost_inputs"], 15
    extra = int(train.loc[recomputed["ok"], "transfers"].sum())
    assert inputs["transfers"] == train["transfers"].sum()
    assert inputs["misroutes"] == (~recomputed["ok"]).sum()
    assert inputs["transfers_on_correct_rows"] == extra
    assert inputs["transfers_on_correct_rows_rs_per_month"] == pytest.approx(extra * 305 / months)
    assert inputs["licence_rs_per_month"] == pytest.approx(320_000 / 12)
    assert (inputs["rs_per_transfer"], inputs["rs_per_misroute"], inputs["licence_rs_per_year"]) == (305, 260, 320_000)


def test_misrouted_only_cost_recomputed(metrics, train, recomputed):
    bad = ~recomputed["ok"]
    cost = (int(train.loc[bad, "transfers"].sum()) * 305 + int(bad.sum()) * 260) / 15
    assert metrics["bot_cost_misrouted_only_rs_per_month"] == pytest.approx(cost, abs=1e-6)


def test_months_covered(metrics):
    assert metrics["months_covered"] == {"first": "2025-04", "last": "2026-06", "count": 15}


def test_months_equal_config_range(metrics, config, train):
    expected = [str(p) for p in pd.period_range(config.TRAIN_START, config.VALID_END, freq="M")]
    assert metrics["months_covered"] == {"first": expected[0], "last": expected[-1], "count": len(expected)}
    assert sorted(metrics["final_team_volume_by_month"]) == expected
    # Rows outside the range would be costed but missing from the monthly table.
    assert sorted(set(train["created_at_ist"].dt.strftime("%Y-%m"))) == expected


def test_empty_month_counts_in_cost(audit, config, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_START", date(2025, 4, 1))
    monkeypatch.setattr(config, "VALID_END", date(2025, 5, 31))
    # April: one correct row with a transfer, two misrouted rows (2 transfers and 0). May: no rows.
    df = pd.DataFrame(
        {
            "created_month": "2025-04",
            "source": "crm",
            "channel": "chat",
            "product_family": "Air Fryer",
            "request_text": "the fryer basket is stuck",
            "text_clean": "the fryer basket is stuck",
            "team_label_canon": ["Repairs", "Repairs", "Billing"],
            "final_team_canon": ["Repairs", "Billing", "Repairs"],
            "transfers": [1, 2, 0],
        }
    )
    result = audit.compute_metrics(df, df)
    assert result["months_covered"] == {"first": "2025-04", "last": "2025-05", "count": 2}
    assert result["bot_cost_rs_per_month"] == pytest.approx((3 * 305 + 2 * 260) / 2)
    assert result["bot_cost_misrouted_only_rs_per_month"] == pytest.approx((2 * 305 + 2 * 260) / 2)
    assert result["cost_inputs"]["transfers_on_correct_rows_rs_per_month"] == pytest.approx(305 / 2)
    assert set(result["final_team_volume_by_month"]["2025-05"].values()) == {0}


def test_transfers_recomputed(metrics, train, recomputed):
    expected = {str(k): int((train["transfers"] == k).sum()) for k in set(train["transfers"].tolist())}
    assert metrics["transfers_distribution"] == expected
    zero, ok = train["transfers"] == 0, recomputed["ok"]
    assert metrics["transfer_consistency"] == {
        "misrouted_with_0_transfers": int((~ok & zero).sum()),
        "correct_with_transfers": int((ok & ~zero).sum()),
    }


def test_text_counts_recomputed(metrics, train, test_set):
    for suffix, col in [("", "text_clean"), ("_raw", "request_text")]:
        train_texts = train[col].tolist()
        seen = set(train_texts)
        # Repeats = rows beyond the first use of each text.
        assert metrics[f"duplicate_text_count{suffix}"] == len(train_texts) - len(seen)
        assert metrics[f"test_text_overlap_count{suffix}"] == sum(t in seen for t in test_set[col])


# --- Criterion 3 -----------------------------------------------------------
def test_agreement_in_range(metrics):
    assert 0.70 <= metrics["bot_vs_final_agreement"] <= 0.85


# --- Criterion 4 -----------------------------------------------------------
def test_volumes_sum_and_canonical(metrics, train, recomputed):
    for key, col in [("final_volume_by_team", "final"), ("bot_label_volume_by_team", "bot")]:
        assert set(metrics[key]) == TEAMS
        assert sum(metrics[key].values()) == len(train)
        assert metrics[key] == {t: int((recomputed[col] == t).sum()) for t in TEAMS}

    confusion = metrics["confusion_bot_to_final"]
    assert set(confusion) == TEAMS
    for bot, row in confusion.items():
        assert set(row) == TEAMS
        for final, count in row.items():
            assert count == ((recomputed["bot"] == bot) & (recomputed["final"] == final)).sum()
    assert sum(sum(row.values()) for row in confusion.values()) == len(train)

    by_month = metrics["final_team_volume_by_month"]
    month = train["created_at_ist"].dt.strftime("%Y-%m")
    assert set(by_month) == set(month)
    for m, row in by_month.items():
        assert set(row) == TEAMS
        for team, count in row.items():
            assert count == ((month == m) & (recomputed["final"] == team)).sum()

    vague = metrics["vague_request_estimate"]
    assert set(vague["final_team_distribution"]) == TEAMS
    assert sum(vague["final_team_distribution"].values()) == vague["count"]


@pytest.mark.parametrize("old_name", ["Installations", "Consumables"])
def test_team_volume_rejects_old_names(audit, old_name):
    with pytest.raises(ValueError):
        audit.team_volume(pd.Series(["Repairs", old_name]))


# --- Criterion 5 -----------------------------------------------------------
def test_md_numbers_in_json(md, metrics):
    assert len(NUMBER.findall(md)) > 100
    assert not unmatched_numbers(md, metrics)


def test_md_rerenders_from_json(audit, config, md, metrics):
    assert audit.render_markdown(metrics) == md

    # Replace every number: a hand-typed one would no longer be found in the metrics.
    rng = random.Random(config.SEED)
    changed = json.loads(json.dumps(metrics))
    for path, value in leaves(metrics):
        if isinstance(value, str):
            continue
        node = changed
        for p in path[:-1]:
            node = node[p]
        if isinstance(value, int):
            node[path[-1]] = rng.randrange(10**7, 10**8)
        else:
            node[path[-1]] = rng.random() if value <= 1 else rng.uniform(10**7, 10**8)
    assert not unmatched_numbers(audit.render_markdown(changed), changed)


def test_md_headline_lines_match_json(md, metrics):
    lines = md.splitlines()
    cost, months = metrics["cost_inputs"], metrics["months_covered"]
    for expected in (
        f"Months covered: {months['first']} to {months['last']} ({months['count']} months).",
        "The bot's label matched the team that closed the request on "
        f"{metrics['bot_vs_final_agreement'] * 100:.2f}% of requests;",
        f"{cost['misroutes']:,} requests were misrouted.",
        f"Bot transfer and contact cost: **Rs {metrics['bot_cost_rs_per_month']:,.0f} per month**,",
        f"from {cost['transfers']:,} transfers at Rs {cost['rs_per_transfer']:,} plus "
        f"{cost['misroutes']:,} misroutes at Rs {cost['rs_per_misroute']:,},",
        f"divided by {months['count']} months.",
        f"- From misrouted requests only: Rs {metrics['bot_cost_misrouted_only_rs_per_month']:,.0f} per month.",
        f"- {metrics['transfer_consistency']['misrouted_with_0_transfers']:,} misroutes have no recorded transfer, "
        "so only their contact",
    ):
        assert lines.count(expected) == 1, expected


# --- Criterion 6 -----------------------------------------------------------
def test_costs_from_config(audit, config, train, metrics, monkeypatch):
    transfers, misroutes = metrics["cost_inputs"]["transfers"], metrics["cost_inputs"]["misroutes"]
    monkeypatch.setattr(config, "COST_PER_TRANSFER", 7)
    monkeypatch.setattr(config, "COST_PER_MISROUTE_CONTACT", 11)
    monkeypatch.setattr(config, "BOT_LICENCE_PER_YEAR", 1_200)
    patched = audit.compute_metrics(train, train)
    assert patched["bot_cost_rs_per_month"] == pytest.approx((transfers * 7 + misroutes * 11) / 15)
    inputs = patched["cost_inputs"]
    assert (inputs["rs_per_transfer"], inputs["rs_per_misroute"]) == (7, 11)
    assert inputs["licence_rs_per_month"] == pytest.approx(100)
    assert inputs["transfers_on_correct_rows_rs_per_month"] == pytest.approx(
        inputs["transfers_on_correct_rows"] * 7 / 15
    )

    with open(audit.__file__, encoding="utf-8") as f:
        tokens = list(tokenize.generate_tokens(io.StringIO(f.read()).readline))
    literals = {t.string.replace("_", "") for t in tokens if t.type == tokenize.NUMBER}
    assert not literals & {"305", "260", "320000"}


# --- Other -----------------------------------------------------------------
def test_committed_reports_current(out_dir, reports_dir):
    for name in ("audit_metrics.json", "data_audit.md"):
        # Universal newlines: git may check the files out with CRLF.
        fresh = (out_dir / name).read_text(encoding="utf-8")
        stored = (reports_dir / name).read_text(encoding="utf-8")
        assert stored == fresh, f"{name} is stale: run clean, then audit"


def test_no_request_text_in_reports(out_dir, train, test_set):
    report = " ".join(p.read_text(encoding="utf-8") for p in out_dir.iterdir())
    report = " ".join(report.lower().split())
    leaked = 0
    for df in (train, test_set):
        for col in ("request_id", "request_text", "text_clean"):
            leaked += sum(" ".join(v.lower().split()) in report for v in set(df[col]))
    # Count only: a failure must not print customer text.
    assert leaked == 0


def test_no_shared_word_runs_in_reports(out_dir, train, test_set):
    report = set()
    for path in out_dir.iterdir():
        report |= word_runs(path.read_text(encoding="utf-8"))
    shared = 0
    for df in (train, test_set):
        for col in ("text_clean", "request_text"):
            shared += sum(len(word_runs(text) & report) for text in set(df[col]))
    # Count only: a failure must not print customer text.
    if shared:
        pytest.fail(f"{shared} four-word runs shared with request text", pytrace=False)


@pytest.mark.parametrize(
    "family, text, expected",
    [
        ("Room Heater", "the fryer basket is stuck", True),
        ("Air Fryer", "two fans are noisy", True),
        ("Air Fryer", "the fryer and the heater both failed", False),
        ("Mixer Grinder", "grinder jar leaks", False),
        ("Robot Vacuum", "robot stopped under the sofa", False),
        ("Induction Cooktop", "the ro needs a new membrane", True),
        ("Air Fryer", "nothing named here", False),
        ("Air Fryer", "fantastic support", False),
        ("Air Fryer", "the induction plate is cracked", True),
        ("Induction Cooktop", "the induction plate is cracked", False),
        ("Room Heater", "my air fryer trips the mcb", True),
        ("Room Heater", "no air comes out", False),
    ],
)
def test_product_rule_on_invented_text(audit, family, text, expected):
    df = pd.DataFrame({"product_family": [family], "text_clean": [text]}, index=[99])
    mask = audit.product_conflict_mask(df)
    assert mask.index.tolist() == [99]
    assert mask.tolist() == [expected]
    assert expected_conflict(family, text) == expected


def test_product_rule_rejects_unknown_family(audit):
    df = pd.DataFrame({"product_family": ["Toaster"], "text_clean": ["the fryer"]})
    with pytest.raises(ValueError):
        audit.product_conflict_mask(df)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("please call me about my purifier", True),
        ("hi, asap!!", True),
        ("need help with order 12345", True),
        ("pls call back", True),
        ("heater is not warming", False),
        ("invoice missing, pls call back", False),
        ("dear team, query regarding the air fryer, regards", True),
        ("query about induction cooktop", True),
        ("air fryer basket is stuck", False),
        ("induction plate is cracked", False),
        ("पानी", False),
    ],
)
def test_vague_rule_on_invented_text(audit, text, expected):
    mask = audit.vague_mask(pd.Series([text], index=[99]))
    assert mask.index.tolist() == [99]
    assert mask.tolist() == [expected]
    assert expected_vague(text) == expected


def test_rule_counts_recomputed(metrics, train):
    conflict = [expected_conflict(f, t) for f, t in zip(train["product_family"], train["text_clean"])]
    vague = train["text_clean"].map(expected_vague)
    assert metrics["product_text_conflict_count"] == sum(conflict)
    assert metrics["vague_request_estimate"]["count"] == vague.sum()
    assert metrics["vague_request_estimate"]["final_team_distribution"] == {
        t: int((train.loc[vague, "final_team_canon"] == t).sum()) for t in TEAMS
    }
    assert 0 < sum(conflict) < len(train) and 0 < vague.sum() < len(train)
