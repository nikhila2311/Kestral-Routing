"""M7: final predictions. All example strings here are invented, not customer text.

Everything is built into temp folders: tests never write to models/, reports/ or the repo root.
Assertions on real rows compare counts, so a failing test does not print customer text.
"""
import csv
import inspect
import json
import os
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

OLD_NAMES = {"Installations", "Consumables"}
FROZEN_THRESHOLD = 0.6283009502968999  # the M5b value, written again; must never be recomputed
MAX_GAP_POINTS, TRIAGE_LOW, TRIAGE_HIGH = 10, 0.10, 0.30  # written again from the spec
MONTHS = ["2026-07", "2026-08", "2026-09"]
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"(?<![\w.§])[+-]?\d[\d,]*(?:\.\d+)?%?")
BUILD = (
    "import sys; from pathlib import Path; from src import predictions as p; "
    "d = Path(sys.argv[1]); p.train_final(d); p.build(d, d, d)"
)


@pytest.fixture(scope="module")
def mod():
    from src import predictions

    return predictions


@pytest.fixture(scope="module")
def train(processed_dir):
    return pd.read_parquet(processed_dir / "train_clean.parquet")


@pytest.fixture(scope="module")
def test_df(processed_dir):
    return pd.read_parquet(processed_dir / "test_clean.parquet")


@pytest.fixture(scope="module")
def recent(train):
    return train[train["created_month"].isin(["2026-04", "2026-05", "2026-06"])]


@pytest.fixture(scope="module")
def models(mod, tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    mod.train_final(folder)
    return folder


@pytest.fixture(scope="module")
def built(mod, models, tmp_path_factory):
    """(folder of predictions.csv, folder of the reports)"""
    root, reports = tmp_path_factory.mktemp("root"), tmp_path_factory.mktemp("reports")
    mod.build(root, reports, models)
    return root, reports


@pytest.fixture(scope="module")
def meta(models):
    return json.loads((models / "router_meta.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def outputs(models, test_df):
    """Straight through the public interface, in test order; not through src.predictions."""
    from src.router import Router

    records = test_df[["request_text", "channel", "product_family", "warranty_status"]].to_dict("records")
    return Router.load(models / "router.joblib").predict(records)


@pytest.fixture(scope="module")
def committed(config):
    with open(config.PREDICTIONS_CSV, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


@pytest.fixture(scope="module")
def sample_ids(config):
    with open(config.SAMPLE_SUBMISSION_CSV, encoding="utf-8", newline="") as f:
        return [row["request_id"] for row in csv.DictReader(f)]


@pytest.fixture(scope="module")
def results(built):
    return json.loads((built[1] / "predictions_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(built):
    return (built[1] / "predictions_summary.md").read_text(encoding="utf-8")


# --- Criterion 1: rows, ids, order ---------------------------------------------------------------
def test_rows_ids_order_nulls_dupes(committed, sample_ids):
    assert committed[0] == ["request_id", "team"]
    rows = committed[1:]
    assert len(rows) == 2178
    assert all(len(row) == 2 and all(cell.strip() for cell in row) for row in rows)
    ids = [row[0] for row in rows]
    assert len(set(ids)) == 2178
    assert ids == sample_ids


def test_order_follows_the_sample_not_the_test_file(mod, models, test_df):
    from src.router import Router

    small = test_df.head(30)
    flipped = small["request_id"].tolist()[::-1]
    frame, outs = mod.predict_test(Router.load(models / "router.joblib"), small, flipped)
    assert frame["request_id"].tolist() == flipped
    assert list(frame.columns) == ["request_id", "team"]
    # Outputs stay in test order; each id keeps its own team.
    assert frame["team"].tolist()[::-1] == [o["team"] for o in outs]


@pytest.mark.parametrize("change", ["missing", "extra", "repeated"])
def test_ids_must_match_the_sample(mod, models, test_df, change):
    from src.router import Router

    small = test_df.head(5)
    ids = small["request_id"].tolist()
    ids = {"missing": ids[:-1], "extra": [*ids, "SR000000"], "repeated": [*ids[:-1], ids[0]]}[change]
    with pytest.raises(ValueError):
        mod.predict_test(Router.load(models / "router.joblib"), small, ids)


# --- Criterion 2: team names ---------------------------------------------------------------------
def test_only_current_names(committed, config):
    teams = {row[1] for row in committed[1:]}
    assert teams <= set(config.TEAMS)
    assert not teams & OLD_NAMES
    raw = config.PREDICTIONS_CSV.read_bytes().replace(b"Filters & Consumables", b"")
    for name in OLD_NAMES:
        assert name.encode() not in raw


def test_write_predictions_rejects_bad_frames(mod, tmp_path):
    good = pd.DataFrame({"request_id": ["SR1", "SR2"], "team": ["Repairs", "Billing"]})
    path = tmp_path / "predictions.csv"
    mod.write_predictions(good, path)
    expected = b"request_id,team\nSR1,Repairs\nSR2,Billing\n"
    assert path.read_bytes() == expected
    for bad in (
        good.assign(team=["Repairs", "Installations"]),
        good.assign(team=["Repairs", None]),
        good.assign(request_id=["SR1", "SR1"]),
        good.assign(request_id=["SR1", None]),
        good[["team", "request_id"]],
    ):
        with pytest.raises(ValueError):
            mod.write_predictions(bad, path)
    assert path.read_bytes() == expected  # a refused frame leaves the old file
    assert [p.name for p in tmp_path.iterdir()] == ["predictions.csv"]  # no temp file left


# --- Criterion 3: git ----------------------------------------------------------------------------
def test_tracked_not_ignored(root_dir):
    if shutil.which("git") is None or not (root_dir / ".git").exists():
        pytest.skip("not a git checkout")

    def git(*args):
        return subprocess.run(["git", *args], cwd=root_dir, capture_output=True, text=True)

    assert git("ls-files", "--error-unmatch", "predictions.csv").returncode == 0, "predictions.csv is not tracked"
    ignored = git("check-ignore", "predictions.csv")  # without -v, which would print the ! rule
    assert (ignored.returncode, ignored.stdout) == (1, "")
    # core.autocrlf must not turn the file into CRLF on checkout: criterion 7 compares bytes.
    assert git("check-attr", "eol", "predictions.csv").stdout.strip() == "predictions.csv: eol: lf"
    for name in ("models/router.joblib", "models/router_meta.json", "models/train_window/router.joblib"):
        assert git("check-ignore", "-q", name).returncode == 0, f"{name} is not git-ignored"


# --- Criterion 4: the final model ----------------------------------------------------------------
def test_final_meta(meta, reports_dir, config):
    chosen = json.loads((reports_dir / "experiments.json").read_text(encoding="utf-8"))["chosen"]["config"]
    m5b = json.loads((reports_dir / "router.json").read_text(encoding="utf-8"))["router"]
    assert meta["training_window"] == {"first_day": "2025-04-01", "last_day": "2026-06-30", "rows": 10822}
    assert meta["config"] == chosen
    assert meta["threshold"] == FROZEN_THRESHOLD == config.TRIAGE_THRESHOLD
    assert round(meta["threshold"], 6) == m5b["threshold"]
    assert {k: round(v, 6) for k, v in meta["cost_inputs"].items()} == m5b["cost_inputs"]
    assert meta["model_version"] != m5b["model_version"]
    assert m5b["training_window"]["last_day"] == "2026-03-31"


def test_model_on_disk_is_the_final_one(meta, config):
    if not config.ROUTER_META_JSON.exists():
        pytest.skip("models/ not built on this machine: run the train target")
    assert json.loads(config.ROUTER_META_JSON.read_text(encoding="utf-8")) == meta


def test_router_target_cannot_overwrite_the_final_model(config):
    from src import router

    default = inspect.signature(router.build).parameters["models_dir"].default
    assert default == config.MODELS_DIR / "train_window"


def test_train_window_model_is_refused(mod, models, tmp_path):
    for name in ("router.joblib", "router_meta.json"):
        shutil.copy(models / name, tmp_path / name)
    path = tmp_path / "router_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["training_window"]["last_day"] = "2026-03-31"
    path.write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="2026-03-31"):
        mod.build(out, out, tmp_path)
    assert not out.exists()


# --- Criterion 5: same answers as the service ----------------------------------------------------
def test_service_consistency(committed, outputs, test_df, models, sample_ids):
    from src.router import Router

    by_id = dict(zip(test_df["request_id"], outputs))
    assert sum(row[1] != by_id[row[0]]["team"] for row in committed[1:]) == 0
    # The API gets one record at a time: a sample of rows, alone.
    router = Router.load(models / "router.joblib")
    fields = ["request_text", "channel", "product_family", "warranty_status"]
    records = dict(zip(test_df["request_id"], test_df[fields].to_dict("records")))
    written = dict(committed[1:])
    differ = sum(router.predict([records[i]])[0]["team"] != written[i] for i in sample_ids[::25])
    assert differ == 0


# --- Criterion 6: sanity of the distribution -----------------------------------------------------
def test_share_gap_and_triage_band(committed, outputs, recent, config):
    assert len(recent) == 2135
    predicted = pd.Series([row[1] for row in committed[1:]]).value_counts(normalize=True)
    actual = recent["final_team_canon"].value_counts(normalize=True)
    gaps = {t: 100 * (predicted.get(t, 0) - actual.get(t, 0)) for t in config.TEAMS}
    assert max(abs(g) for g in gaps.values()) <= MAX_GAP_POINTS, gaps
    triaged = sum(o["needs_triage"] for o in outputs) / len(outputs)
    assert TRIAGE_LOW <= triaged <= TRIAGE_HIGH


def test_limits_in_config_match_the_spec(config):
    assert config.MAX_SHARE_GAP_POINTS == MAX_GAP_POINTS
    assert (config.TRIAGE_SHARE_MIN, config.TRIAGE_SHARE_MAX) == (TRIAGE_LOW, TRIAGE_HIGH)
    assert tuple(str(d) for d in config.LAST_MONTHS) == ("2026-04-01", "2026-06-30")


# --- Criterion 7: deterministic ------------------------------------------------------------------
@pytest.mark.slow
def test_rebuild_byte_identical(root_dir, config, tmp_path):
    folders = []
    for seed in ("1", "2"):  # two separate processes, different hash seeds
        folder = tmp_path / seed
        env = {**os.environ, "PYTHONHASHSEED": seed}
        subprocess.run([sys.executable, "-c", BUILD, str(folder)], cwd=root_dir, env=env, check=True)
        folders.append(folder)
    for name in ("predictions.csv", "predictions_summary.json", "predictions_summary.md", "router_meta.json"):
        assert (folders[0] / name).read_bytes() == (folders[1] / name).read_bytes(), name
    assert (folders[0] / "predictions.csv").read_bytes() == config.PREDICTIONS_CSV.read_bytes()
    assert b"\r" not in config.PREDICTIONS_CSV.read_bytes()


# --- Criterion 8 and the summary -----------------------------------------------------------------
def leaves(node):
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        return [leaf for child in node for leaf in leaves(child)]
    return [node]


def unmatched_numbers(md, results):
    strings = {v for v in leaves(results) if isinstance(v, str)}
    values = [v for v in leaves(results) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    bad = [d for d in DATE.findall(md) if d not in strings]
    md = DATE.sub(" ", md)
    for text in sorted(strings, key=len, reverse=True):  # months and versions
        md = md.replace(text, " ") if any(char.isdigit() for char in text) else md
    for token in NUMBER.findall(md):
        text = token.rstrip("%").replace(",", "")
        decimals = len(text.partition(".")[2])
        scale = 100 if token.endswith("%") else 1
        sign = "+" if text[0] in "+-" else ""
        # Shares are only ever shown as percentages, so a bare integer cannot match a rounded share.
        usable = [v for v in values if scale == 100 or decimals or isinstance(v, int) or v > 100]
        if not any(f"{v * scale:{sign}.{decimals}f}" == text for v in usable):
            bad.append(token)
    return bad


def test_md_numbers_in_json(mod, md, results):
    assert md == mod.render_markdown(results)
    assert unmatched_numbers(md, results) == []
    added = "\n12,345 and 99.87% and 777.77 and +88.88 and 2031-01-01\n"
    assert unmatched_numbers(md + added, results) == ["2031-01-01", "12,345", "99.87%", "777.77", "+88.88"]


def test_committed_reports_match_a_fresh_build(mod, results, config):
    stored = json.loads(config.PREDICTIONS_SUMMARY_JSON.read_text(encoding="utf-8"))
    assert config.PREDICTIONS_SUMMARY_MD.read_text(encoding="utf-8") == mod.render_markdown(stored)

    def same(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(map(same, a, b))
        return a == pytest.approx(b, abs=1e-4) if isinstance(a, float) else a == b

    assert same(stored, results)


def test_summary_numbers_recomputed(results, outputs, test_df, recent, config):
    from src.audit import vague_mask

    frame = test_df.assign(team=[o["team"] for o in outputs], triaged=[o["needs_triage"] for o in outputs])
    assert results["test"]["rows"] == 2178
    assert results["test"]["months"] == MONTHS
    for team in config.TEAMS:
        mine, block = frame[frame["team"] == team], results["volume"][team]
        assert block["total"] == len(mine)
        assert block["share"] == pytest.approx(len(mine) / 2178, abs=1e-6)
        assert block["auto_routed"] == int((~mine["triaged"]).sum())
        assert block["triaged"] == int(mine["triaged"].sum())
        assert block["per_month"] == pytest.approx(len(mine) / 3, abs=1e-6)
        for month in MONTHS:
            rows = mine[mine["created_month"] == month]
            triaged = int(rows["triaged"].sum())
            expected = {"total": len(rows), "auto_routed": len(rows) - triaged, "triaged": triaged}
            assert block["by_month"][month] == expected
        gap = results["comparison"]["teams"][team]
        actual = int((recent["final_team_canon"] == team).sum())
        assert gap["recent_count"] == actual
        assert gap["recent_share"] == pytest.approx(actual / 2135, abs=1e-6)
        assert gap["predicted_share"] == block["share"]
        assert gap["gap_points"] == pytest.approx(100 * (len(mine) / 2178 - actual / 2135), abs=1e-4)
    assert sum(b["total"] for b in results["volume"].values()) == 2178
    gaps = [abs(g["gap_points"]) for g in results["comparison"]["teams"].values()]
    assert results["comparison"]["largest_gap_points"] == pytest.approx(max(gaps), abs=1e-4)
    for month in MONTHS:
        rows = frame[frame["created_month"] == month]
        assert results["test"]["rows_by_month"][month] == len(rows)
        assert results["triage"]["by_month"][month] == pytest.approx(rows["triaged"].mean(), abs=1e-6)
    assert results["triage"]["count"] == int(frame["triaged"].sum())
    assert results["triage"]["share"] == pytest.approx(frame["triaged"].mean(), abs=1e-6)
    vague = vague_mask(test_df["text_clean"])
    assert results["vague"]["count"] == int(vague.sum())
    assert results["vague"]["share"] == pytest.approx(vague.mean(), abs=1e-6)
    assert results["vague"]["triaged_count"] == int((vague & frame["triaged"]).sum())


def test_auto_routed_only_volume(results, outputs, test_df, config, md):
    frame = test_df.assign(team=[o["team"] for o in outputs], triaged=[o["needs_triage"] for o in outputs])
    sent, block = frame[~frame["triaged"]], results["auto_routed_only"]
    assert set(block["teams"]) == set(config.TEAMS)
    assert block["total"] == len(sent) == sum(t["count"] for t in block["teams"].values())
    assert block["triaged_count"] == int(frame["triaged"].sum()) == results["triage"]["count"]
    assert block["total"] + block["triaged_count"] == 2178
    for month in MONTHS:
        rows = frame[frame["created_month"] == month]
        assert block["total_by_month"][month] == int((~rows["triaged"]).sum())
        assert block["triaged_by_month"][month] == int(rows["triaged"].sum())
    for team in config.TEAMS:
        mine, entry = sent[sent["team"] == team], block["teams"][team]
        assert entry["count"] == len(mine) == results["volume"][team]["auto_routed"]
        assert entry["share"] == pytest.approx(len(mine) / len(sent), abs=1e-6)
        assert entry["per_month"] == pytest.approx(len(mine) / 3, abs=1e-6)
        assert entry["by_month"] == {m: int((mine["created_month"] == m).sum()) for m in MONTHS}
        cells = " | ".join(f"{entry['by_month'][m]:,}" for m in MONTHS)
        row = f"| {team} | {entry['count']:,} | {entry['share'] * 100:.2f}% | {entry['per_month']:.1f} | {cells} |"
        assert md.count(row) == 1
    cells = " | ".join(f"{block['triaged_by_month'][m]:,}" for m in MONTHS)
    triaged = f"| Triaged, team not yet known | {block['triaged_count']:,} | | {block['triaged_per_month']:.1f} | {cells} |"
    assert md.count(triaged) == 1


def test_nothing_auto_routed_does_not_crash(mod, results, meta, recent, test_df):
    outs = [{"team": "Repairs", "needs_triage": True}] * len(test_df)
    out = mod.summarise(outs, test_df, recent, meta, results["validation"])
    assert out["auto_routed_only"]["total"] == 0
    assert out["auto_routed_only"]["teams"]["Repairs"]["share"] == 0.0
    assert mod.render_markdown(out)


def test_validation_bias_from_verified_reports(results, reports_dir, recent, config):
    baselines = json.loads((reports_dir / "baselines.json").read_text(encoding="utf-8"))
    matrix = baselines["baselines"]["tfidf_final"]["evaluate"]["confusion_matrix"]
    m5b = json.loads((reports_dir / "router.json").read_text(encoding="utf-8"))
    bias = results["validation"]
    for i, team in enumerate(config.TEAMS):
        actual, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
        assert actual == int((recent["final_team_canon"] == team).sum())  # matrix rows are the final team
        assert bias["teams"][team]["actual"] == actual
        assert bias["teams"][team]["predicted"] == predicted
        assert bias["teams"][team]["points"] == pytest.approx(100 * (predicted - actual) / 2135, abs=1e-4)
    assert (bias["teams"]["Repairs"]["predicted"], bias["teams"]["Repairs"]["actual"]) == (582, 498)
    assert bias["rows"] == 2135
    assert bias["accuracy"] == m5b["accuracy"]["all"]
    assert bias["accuracy_triaged"] == m5b["accuracy"]["triaged"]


def test_team_never_predicted_still_counts(mod, results, meta, recent, config):
    """Invented rows: every message goes to Repairs, so the other six teams are 0, not missing."""
    frame = pd.DataFrame(
        {
            "created_at_ist": pd.to_datetime(["2026-07-01", "2026-08-01", "2026-09-01"]),
            "created_month": MONTHS,
            "text_clean": ["mixer is dead", "call me", "cooler makes a noise"],
        }
    )
    outs = [{"team": "Repairs", "needs_triage": i == 1} for i in range(3)]
    out = mod.summarise(outs, frame, recent, meta, results["validation"])
    assert set(out["volume"]) == set(out["comparison"]["teams"]) == set(config.TEAMS)
    assert out["volume"]["Billing"]["total"] == 0
    assert out["volume"]["Billing"]["by_month"]["2026-08"] == {"total": 0, "auto_routed": 0, "triaged": 0}
    assert out["comparison"]["teams"]["Billing"]["gap_points"] == pytest.approx(-100 * 255 / 2135)
    assert out["comparison"]["largest_gap_points"] > 70


def test_md_states_the_caveats(md):
    assert "83.89%" in md and "has not been scored" in md
    assert "first guess" in md  # triaged volume is not a routing decision
    assert "No correction is applied" in md


def test_reports_have_no_rows_or_text(md, built, test_df, train):
    blob = md + (built[1] / "predictions_summary.json").read_text(encoding="utf-8")
    assert not re.search(r"\bSR\d+", blob)
    for frame in (test_df, train):
        for column in ("request_text", "text_clean"):
            texts = frame.loc[frame[column].str.len() > 25, column]
            assert sum(text in blob for text in texts) == 0
    stripped = blob.replace("Filters & Consumables", "")
    assert not [name for name in OLD_NAMES if name in stripped]
