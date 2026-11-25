"""M1: data cleaning. All example strings here are invented, not customer text."""
import ftfy
import pandas as pd
import pytest
from pandas.api import types as pdt

TEAMS = {
    "Installs & Demo",
    "Repairs",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
}
CANON_COLUMNS = ["team_label_canon", "final_team_canon", "first_team_canon"]
BASE_COLUMNS = [
    "request_id",
    "created_at_ist",
    "channel",
    "product_family",
    "warranty_status",
    "request_text",
    "source",
]
TRAIN_COLUMNS = BASE_COLUMNS + [
    "team_label",
    "text_clean",
    "team_label_canon",
    "final_team_canon",
    "first_team_canon",
    "transfers",
    "resolved_at_ist",
    "resolution_hours",
    "misrouted",
    "created_month",
]
TEST_COLUMNS = BASE_COLUMNS + ["text_clean", "created_month"]

# clean_text lowercases, so a leftover marker could survive in either case.
MOJIBAKE_MARKERS = [m for marker in ("Ã", "â€", "Â") for m in {marker, marker.lower()}]

IDEMPOTENCE_CASES = [
    "thÃ© mcb",
    "  Mixed   CASE\ttext\n",
    "waiting Ã¢â‚¬Â¦",
    "help Ã¢â‚¬â€œ heater",
    "café – ‘quoted’ …",
    "a &amp;amp; b",
    "पानी का filter",
    "plain text",
]


def has_marker(text):
    return any(m in text for m in MOJIBAKE_MARKERS)


@pytest.fixture(scope="module")
def clean_module(config):
    from src import clean

    return clean


@pytest.fixture(scope="module")
def summary(clean_module):
    return clean_module.build()


@pytest.fixture(scope="module")
def train(summary, processed_dir):
    return pd.read_parquet(processed_dir / "train_clean.parquet")


@pytest.fixture(scope="module")
def test_set(summary, processed_dir):
    return pd.read_parquet(processed_dir / "test_clean.parquet")


@pytest.fixture(scope="module")
def raw_train(config):
    return pd.read_csv(config.TRAIN_CSV)


@pytest.fixture(scope="module")
def raw_log(config):
    return pd.read_csv(config.RESOLUTION_LOG_CSV)


# --- Criterion 1 -----------------------------------------------------------
def test_row_counts_and_unique_ids(train, test_set, raw_train):
    assert len(train) == 10822
    assert len(test_set) == 2178
    assert train["request_id"].is_unique
    assert test_set["request_id"].is_unique
    assert train["request_id"].tolist() == raw_train["request_id"].tolist()


def test_one_resolution_row_per_train_id(train, raw_log):
    counts = raw_log["request_id"].value_counts()
    assert (train["request_id"].map(counts) == 1).all()
    assert train[["first_team_canon", "final_team_canon", "transfers", "resolved_at_ist"]].notna().all().all()
    by_id = raw_log.set_index("request_id")["transfers"]
    assert (train["request_id"].map(by_id) == train["transfers"]).all()


# --- Criterion 2 -----------------------------------------------------------
def test_canon_columns_only_current_teams(train, raw_log, config):
    for column in CANON_COLUMNS:
        assert set(train[column]) == TEAMS, column
    expected = raw_log.set_index("request_id")["final_team"].replace(config.TEAM_NAME_MAP)
    assert (train["request_id"].map(expected) == train["final_team_canon"]).all()


def test_canon_team_maps_old_names_and_rejects_unknown(clean_module):
    assert clean_module.canon_team("Installations") == "Installs & Demo"
    assert clean_module.canon_team("Consumables") == "Filters & Consumables"
    assert clean_module.canon_team("Billing") == "Billing"
    with pytest.raises(ValueError):
        clean_module.canon_team("Unknown Team")


# --- Criterion 3 -----------------------------------------------------------
def test_text_clean_no_mojibake_or_empty(train, test_set):
    for df in (train, test_set):
        assert df["text_clean"].notna().all()
        assert (df["text_clean"].str.strip() != "").all()
        assert not df["text_clean"].map(has_marker).any()
        # Folding accents could hide an unrepaired marker, so also check before folding,
        # and require that no non-ASCII residue is left in this dataset.
        assert not df["request_text"].map(lambda s: has_marker(ftfy.fix_text(s))).any()
        assert df["text_clean"].map(str.isascii).all()


# --- Criterion 4 -----------------------------------------------------------
def test_clean_text_example(clean_module):
    assert clean_module.clean_text("thÃ© mcb") == "the mcb"
    assert clean_module.clean_text("  Mixed   CASE\ttext\n") == "mixed case text"
    assert clean_module.clean_text("help Ã¢â‚¬â€œ heater") == "help - heater"


def test_clean_text_keeps_hindi(clean_module):
    text = "पानी का"
    assert clean_module.clean_text(text) == text


def test_clean_text_idempotent(clean_module, train, test_set):
    texts = IDEMPOTENCE_CASES + train["request_text"].tolist() + test_set["request_text"].tolist()
    for text in texts:
        once = clean_module.clean_text(text)
        assert clean_module.clean_text(once) == once


# --- Criterion 5 -----------------------------------------------------------
def test_no_negative_resolution_hours(train, raw_train, raw_log, summary):
    merged = raw_train.merge(raw_log, on="request_id")
    raw_negative = pd.to_datetime(merged["resolved_at"]) < pd.to_datetime(merged["created_at_ist"])
    before = int(raw_negative.sum())
    after = int((train["resolution_hours"] < 0).sum())
    print(f"negative resolution_hours: before={before} after={after}")
    assert before == 1140
    assert set(merged.loc[raw_negative, "source"]) == {"legacy_zoho"}
    assert after == 0
    assert summary["negative_before"] == before
    assert summary["negative_after"] == after


def test_timezone_shift_only_on_legacy_rows(train, raw_log):
    stored = pd.to_datetime(train["request_id"].map(raw_log.set_index("request_id")["resolved_at"]))
    shift = train["resolved_at_ist"] - stored
    legacy = train["source"] == "legacy_zoho"
    assert (shift[legacy] == pd.Timedelta(hours=5, minutes=30)).all()
    assert (shift[~legacy] == pd.Timedelta(0)).all()
    hours = (train["resolved_at_ist"] - train["created_at_ist"]).dt.total_seconds() / 3600
    assert (hours - train["resolution_hours"]).abs().max() < 1e-9


# --- Criterion 6 -----------------------------------------------------------
def test_test_set_is_crm_jul_to_sep_2026(test_set, config):
    assert set(test_set["source"]) == {"crm"}
    days = test_set["created_at_ist"].dt.date
    assert days.min() >= config.TEST_START
    assert days.max() <= config.TEST_END
    assert set(test_set["created_month"]) == {"2026-07", "2026-08", "2026-09"}


# --- Schema ----------------------------------------------------------------
def test_output_schema(train, test_set):
    assert list(train.columns) == TRAIN_COLUMNS
    assert list(test_set.columns) == TEST_COLUMNS
    for df in (train, test_set):
        assert pdt.is_datetime64_any_dtype(df["created_at_ist"])
        assert df["created_at_ist"].dt.tz is None
        assert pdt.is_string_dtype(df["text_clean"])
        assert (df["created_month"] == df["created_at_ist"].dt.strftime("%Y-%m")).all()
    assert pdt.is_datetime64_any_dtype(train["resolved_at_ist"])
    assert train["resolved_at_ist"].dt.tz is None
    assert pdt.is_integer_dtype(train["transfers"])
    assert pdt.is_float_dtype(train["resolution_hours"])
    assert pdt.is_bool_dtype(train["misrouted"])
    assert (train["misrouted"] == (train["team_label_canon"] != train["final_team_canon"])).all()
