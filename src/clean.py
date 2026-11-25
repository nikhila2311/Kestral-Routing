"""M1: data cleaning. Contract: docs/contracts.md#m1. Run: python -m src.clean"""
import unicodedata

import ftfy
import pandas as pd

from src import config

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
LEGACY_SOURCE = "legacy_zoho"

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})
_MAX_PASSES = 10


def _clean_once(s: str) -> str:
    # ftfy must run before lowercasing: lowercased mojibake can no longer be decoded.
    s = ftfy.fix_text(s)
    # Drop combining diacritics (U+0300-U+036F) only, so Hindi vowel signs survive.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not "̀" <= c <= "ͯ")
    s = unicodedata.normalize("NFC", s).translate(_DASHES)
    return " ".join(s.lower().split())


def clean_text(s: str) -> str:
    """Repair mojibake, fold accents and dashes, lowercase and collapse whitespace."""
    # One pass can expose more to fix (e.g. nested HTML entities), so repeat until stable.
    for _ in range(_MAX_PASSES):
        cleaned = _clean_once(s)
        if cleaned == s:
            break
        s = cleaned
    return s


def canon_team(name: str) -> str:
    """Map an old or current team name to its current name."""
    team = config.TEAM_NAME_MAP.get(name, name)
    if team not in config.TEAMS:
        raise ValueError(f"Unknown team: {name!r}")
    return team


def to_ist(resolved_at: pd.Series, source: pd.Series) -> pd.Series:
    """Parse resolved_at and convert legacy_zoho rows from UTC to IST (ops-policy section 9)."""
    parsed = pd.to_datetime(resolved_at)
    return parsed.where(source != LEGACY_SOURCE, parsed + IST_OFFSET)


def _add_shared_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at_ist"] = pd.to_datetime(df["created_at_ist"])
    df["text_clean"] = df["request_text"].map(clean_text)
    if (df["text_clean"] == "").any():
        raise ValueError("Blank request_text after cleaning")
    df["created_month"] = df["created_at_ist"].dt.strftime("%Y-%m")
    return df


def clean_train(train: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    if set(train["request_id"]) != set(log["request_id"]):
        raise ValueError("train and resolution_log have different request_ids")
    df = _add_shared_columns(train).merge(log, on="request_id", how="left", validate="one_to_one")
    for column in ("team_label", "first_team", "final_team"):
        df[f"{column}_canon"] = df[column].map(canon_team)
    df["transfers"] = df["transfers"].astype("int64")
    df["resolved_at_ist"] = to_ist(df["resolved_at"], df["source"])
    df["resolution_hours"] = (df["resolved_at_ist"] - df["created_at_ist"]).dt.total_seconds() / 3600
    df["misrouted"] = df["team_label_canon"] != df["final_team_canon"]
    return df[list(config.TRAIN_CLEAN_COLUMNS)]


def clean_test(test: pd.DataFrame) -> pd.DataFrame:
    if not test["request_id"].is_unique:
        raise ValueError("Duplicate request_id in test")
    return _add_shared_columns(test)[list(config.TEST_CLEAN_COLUMNS)]


def _read_raw(path) -> pd.DataFrame:
    # Strings as-is: a message such as "NA" must not become a missing value.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build() -> dict:
    """Write both cleaned files and return (and print) a summary."""
    raw_train = _read_raw(config.TRAIN_CSV)
    log = _read_raw(config.RESOLUTION_LOG_CSV)
    train = clean_train(raw_train, log)
    test = clean_test(_read_raw(config.TEST_CSV))

    stored = pd.to_datetime(train["request_id"].map(log.set_index("request_id")["resolved_at"]))
    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "negative_before": int((stored < train["created_at_ist"]).sum()),
        "negative_after": int((train["resolution_hours"] < 0).sum()),
    }

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(config.TRAIN_CLEAN_PARQUET, index=False)
    test.to_parquet(config.TEST_CLEAN_PARQUET, index=False)
    print(summary)
    return summary


if __name__ == "__main__":
    build()
