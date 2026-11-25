"""M5a: model selection. All example strings here are invented, not customer text."""
import json
import random
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ALLOWED = ["text_clean", "channel", "product_family", "warranty_status"]
REFERENCE = "logreg_c1_none"
GROUPS = {"payment", "install", "fault", "order_id", "conflict", "ivr_conflict"}
BASE_NAMES = {
    *(f"logreg_c{c}_{w}" for c in ("0.3", "1", "3", "10") for w in ("none", "balanced")),
    *(f"linear_svc_c{c}_{m}" for c in ("0.1", "1") for m in ("sigmoid", "isotonic")),
    *(f"complement_nb_a{a}" for a in ("0.1", "0.3", "1")),
}
REAL_VAL_KEYS = {
    "accuracy",
    "accuracy_ci95",
    "macro_f1",
    "log_loss",
    "paired_accuracy_points_minus_bot",
    "per_team_recall",
    "repairs_precision",
    "top_confusions",
}
TOP_KEYS = {
    "candidates",
    "chosen",
    "discarded",
    "selection",
    "cue_candidates",
    "supplementary",
    "original_rule",
    "inner_split",
    "reference",
    "real_val",
    "bot_real_val_accuracy",
    "meta",
}
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


@pytest.fixture(scope="module")
def mod():
    from src import experiments

    return experiments


@pytest.fixture(scope="module")
def df(processed_dir):
    return pd.read_parquet(processed_dir / "train_clean.parquet")


@pytest.fixture(scope="module")
def windows(df):
    # Filtered here, not with the module's splits, so a split error cannot hide in both places.
    day = df["created_at_ist"]
    return {
        "inner_train": df[day < "2026-01-01"],
        "inner_val": df[(day >= "2026-01-01") & (day < "2026-04-01")],
        "train": df[day < "2026-04-01"],
        "val": df[(day >= "2026-04-01") & (day < "2026-07-01")],
    }


@pytest.fixture(scope="module")
def recorded(mod, tmp_path_factory):
    """One full build into a temp folder, with every fit_predict call and its output recorded."""
    path = tmp_path_factory.mktemp("experiments")
    calls, original = [], mod.fit_predict

    def recording(cfg, fit_df, score_df):
        pred, proba = original(cfg, fit_df, score_df)
        calls.append(
            {
                "name": cfg["name"],
                "fit_ids": list(fit_df["request_id"]),
                "score_ids": list(score_df["request_id"]),
                "pred": pred,
                "proba": proba,
            }
        )
        return pred, proba

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "fit_predict", recording)
        mod.build(path)
    return path, calls


@pytest.fixture(scope="module")
def out_dir(recorded):
    return recorded[0]


@pytest.fixture(scope="module")
def calls(recorded):
    return recorded[1]


@pytest.fixture(scope="module")
def results(out_dir):
    return json.loads((out_dir / "experiments.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(out_dir):
    return (out_dir / "experiments.md").read_text(encoding="utf-8")


def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, path + (key,))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, path + (i,))
    else:
        yield path, node


def final_scores(results):
    """Every real-val score: chosen and reference, plus the original choice kept as evidence."""
    original = results["original_rule"]
    return {**results["real_val"], original["chosen"]["name"]: original["real_val"]}


def by_name(results):
    return {row["name"]: row for row in results["candidates"]}


def word_runs(text, n=4):
    tokens = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# --- Criterion 1: inner split ----------------------------------------------
def test_inner_split_is_chronological_and_excludes_validation(mod, windows):
    inner_train, inner_val = mod.inner_split(windows["train"])
    assert inner_train["created_at_ist"].max() < inner_val["created_at_ist"].min()
    assert len(inner_train) + len(inner_val) == len(windows["train"])
    for part, months in ((inner_train, 9), (inner_val, 3)):
        assert (part["created_at_ist"] < "2026-04-01").all()
        assert part["created_month"].nunique() == months
    assert list(inner_train["request_id"]) == list(windows["inner_train"]["request_id"])
    assert list(inner_val["request_id"]) == list(windows["inner_val"]["request_id"])


def test_inner_split_rejects_validation_rows(mod, df):
    with pytest.raises(ValueError):
        mod.inner_split(df)


def test_inner_split_matches_config(config):
    assert str(config.INNER_TRAIN_END) == "2025-12-31"
    assert str(config.INNER_VAL_START) == "2026-01-01"


def test_corrupting_validation_rows_does_not_change_selection(mod, df, windows):
    # A few fast candidates are enough: this checks which rows reach the selection, not the models.
    corrupted = df.copy()
    rows = corrupted["request_id"].isin(windows["val"]["request_id"])
    corrupted.loc[rows, "text_clean"] = "zzqx corrupted"
    corrupted.loc[rows, "final_team_canon"] = "Billing"
    few = [c for c in mod.base_candidates() if c["model_type"] == "complement_nb"]
    few.append(next(c for c in mod.base_candidates() if c["name"] == REFERENCE))
    outputs = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "base_candidates", lambda: few)
        patch.setattr(mod, "score_final", lambda cfg, train, val: {"accuracy": 0.0})
        for frame in (df, corrupted):
            result = mod.compute_experiments(frame)
            outputs.append({k: result[k] for k in ("candidates", "chosen", "discarded", "selection")})
            outputs[-1]["original"] = result["original_rule"]["chosen"]
            outputs[-1]["supplementary"] = result["supplementary"]
    assert outputs[0] == outputs[1]


# --- Criterion 2: which rows score which candidate --------------------------
def test_every_candidate_has_inner_metrics(results):
    rows = by_name(results)
    assert len(rows) == len(results["candidates"]) == results["selection"]["candidate_count"]
    base = results["selection"]["feature_base"]
    assert set(rows) == BASE_NAMES | {f"{base}+{group}" for group in GROUPS | {"all"}}
    for row in rows.values():
        assert set(row["inner_val"]) == {"accuracy", "macro_f1", "log_loss"}
        assert all(isinstance(v, float) and np.isfinite(v) for v in row["inner_val"].values())
        assert 0 < row["inner_val"]["accuracy"] < 1
        assert row["complexity"] == len(row["config"]["feature_groups"]) + (row["config"]["calibration"] is not None)


def test_real_val_only_for_chosen_and_reference(results):
    assert set(results) == TOP_KEYS
    assert results["reference"] == REFERENCE
    assert set(results["real_val"]) == {results["chosen"]["name"], REFERENCE}
    for score in results["real_val"].values():
        assert set(score) == REAL_VAL_KEYS
    for row in results["candidates"]:
        assert set(row) == {
            "name", "config", "complexity", "inner_val", "log_loss_above_best", "points_minus_chosen"
        }
    # The only other real-val score is the original choice, inside its own evidence block.
    assert set(results["original_rule"]["real_val"]) == REAL_VAL_KEYS


def test_each_fit_uses_its_own_window_and_validation_is_scored_once(calls, results, windows):
    ids = {name: list(frame["request_id"]) for name, frame in windows.items()}
    final = len(final_scores(results))
    inner_calls, final_calls = calls[:-final], calls[-final:]
    inner_rows = results["candidates"] + results["supplementary"]["candidates"]
    assert [c["name"] for c in inner_calls] == [row["name"] for row in inner_rows]
    for call in inner_calls:
        assert call["fit_ids"] == ids["inner_train"] and call["score_ids"] == ids["inner_val"]
    names = [results["chosen"]["name"], REFERENCE, results["original_rule"]["chosen"]["name"]]
    assert [c["name"] for c in final_calls] == list(dict.fromkeys(names))
    assert {c["name"] for c in final_calls} == {REFERENCE, "complement_nb_a0.1"}
    for call in final_calls:
        assert call["fit_ids"] == ids["train"] and call["score_ids"] == ids["val"]


def test_stored_metrics_recomputed_from_predictions(calls, results, windows):
    n_final = len(final_scores(results))
    rows = by_name(results) | {row["name"]: row for row in results["supplementary"]["candidates"]}
    truth = windows["inner_val"]["final_team_canon"].to_numpy()
    for call in calls[:-n_final]:
        assert rows[call["name"]]["inner_val"]["accuracy"] == pytest.approx(np.mean(call["pred"] == truth), abs=1e-6)
    truth = windows["val"]["final_team_canon"].to_numpy()
    for call in calls[-n_final:]:
        score, pred = final_scores(results)[call["name"]], call["pred"]
        assert score["accuracy"] == pytest.approx(np.mean(pred == truth), abs=1e-6)
        for team, recall in score["per_team_recall"].items():
            assert recall == pytest.approx(np.mean(pred[truth == team] == team), abs=1e-6)
        precision = np.mean(truth[pred == "Repairs"] == "Repairs")
        assert score["repairs_precision"] == pytest.approx(precision, abs=1e-6)
        pairs = pd.Series(list(zip(truth[pred != truth], pred[pred != truth]))).value_counts()
        assert [c["count"] for c in score["top_confusions"]] == pairs.head(5).tolist()
        for cell in score["top_confusions"]:
            assert pairs[(cell["true"], cell["predicted"])] == cell["count"]


def test_intervals_recomputed(calls, results, windows, config):
    from src.evaluate import bootstrap_ci

    truth = windows["val"]["final_team_canon"].to_numpy(dtype=object)
    bot = windows["val"]["team_label_canon"].to_numpy(dtype=object)
    for call in calls[-len(final_scores(results)) :]:
        score, pred = final_scores(results)[call["name"]], call["pred"]
        ci = bootstrap_ci(truth, pred, lambda t, p: float(np.mean(t == p)), seed=config.SEED)
        assert list(score["accuracy_ci95"].values()) == pytest.approx([ci[2], ci[0], ci[1]], abs=1e-6)  # sorted keys
        paired = bootstrap_ci(pred == truth, bot == truth, lambda a, b: 100 * (np.mean(a) - np.mean(b)))
        stored = score["paired_accuracy_points_minus_bot"]
        assert [stored["low"], stored["point"], stored["high"]] == pytest.approx(list(paired), abs=1e-6)


# --- Criterion 3: the selection rule ---------------------------------------
def complexity_of(row):
    return len(row["config"]["feature_groups"]) + bool(row["config"]["calibration"])


def near_top(rows):
    best = max(r["inner_val"]["accuracy"] for r in rows)
    return [r for r in rows if (best - r["inner_val"]["accuracy"]) * 100 <= 1.0 + 1e-9]


def original_rule(rows):
    """The original rule written again from the spec, on the stored numbers."""
    near = near_top(rows)
    near = [r for r in near if complexity_of(r) == min(map(complexity_of, near))]
    return sorted(near, key=lambda r: (-r["inner_val"]["accuracy"], r["inner_val"]["log_loss"], rows.index(r)))[0]


def amended_rule(rows):
    """The amended rule written again from the spec, on the stored numbers."""
    best = min(r["inner_val"]["log_loss"] for r in rows)
    kept = [r for r in near_top(rows) if r["inner_val"]["log_loss"] - best <= 0.1 + 1e-9]
    kept = [r for r in kept if complexity_of(r) == min(map(complexity_of, kept))]
    return sorted(kept, key=lambda r: (r["inner_val"]["log_loss"], -r["inner_val"]["accuracy"], rows.index(r)))[0]


def test_amended_rule_reproduces_choice(results):
    rows = results["candidates"]
    chosen = amended_rule(rows)
    assert chosen["name"] == "logreg_c1_none"
    assert results["chosen"]["name"] == chosen["name"]
    assert results["chosen"]["config"] == chosen["config"]
    assert results["selection"]["margin_points"] == 1.0
    assert results["selection"]["log_loss_limit"] == 0.1
    top = max(r["inner_val"]["accuracy"] for r in rows)
    best = min(r["inner_val"]["log_loss"] for r in rows)
    assert by_name(results)[results["chosen"]["top_candidate"]]["inner_val"]["accuracy"] == top
    assert results["chosen"]["points_below_top"] == pytest.approx(
        100 * (top - chosen["inner_val"]["accuracy"]), abs=1e-6
    )
    for row in rows:
        expected = 100 * (row["inner_val"]["accuracy"] - chosen["inner_val"]["accuracy"])
        assert row["points_minus_chosen"] == pytest.approx(expected, abs=1e-6)
        assert row["log_loss_above_best"] == pytest.approx(row["inner_val"]["log_loss"] - best, abs=1e-6)


def test_walk_through_counts(results):
    rows, s = results["candidates"], results["selection"]
    best = min(r["inner_val"]["log_loss"] for r in rows)
    near = near_top(rows)
    kept = [r for r in near if r["inner_val"]["log_loss"] - best <= 0.1]
    simplest = [r for r in kept if complexity_of(r) == min(map(complexity_of, kept))]
    assert s["within_margin_count"] == len(near)
    assert s["within_log_loss_limit_count"] == len(kept)
    assert s["simplest_count"] == len(simplest)
    assert s["best_log_loss"] == best
    assert s["log_loss_ceiling"] == pytest.approx(best + 0.1, abs=1e-6)
    assert s["accuracy_floor"] == pytest.approx(max(r["inner_val"]["accuracy"] for r in rows) - 0.01, abs=1e-6)
    assert s["accuracy_first_choice"] == max(simplest, key=lambda r: r["inner_val"]["accuracy"])["name"]


def test_original_rule_block_records_failure(results, md, reports_dir):
    original, rows = results["original_rule"], results["candidates"]
    assert set(original) == {"rule", "chosen", "discarded", "real_val", "criterion_4"}
    assert original["chosen"]["name"] == original_rule(rows)["name"] == "complement_nb_a0.1"
    assert original["chosen"]["config"] == original_rule(rows)["config"]
    assert results["selection"]["feature_base"] == original_rule(rows[: len(BASE_NAMES)])["name"]
    # Against M4's stored result, not this module's own reference score.
    d = json.loads((reports_dir / "baselines.json").read_text(encoding="utf-8"))["baselines"]["tfidf_final"]["evaluate"]
    gate = original["criterion_4"]
    assert gate["result"] == "failed"
    assert gate["allowed_drop_points"] == 1.0
    assert gate["threshold_accuracy"] == pytest.approx(d["accuracy"] - 0.01, abs=1e-4)
    assert original["real_val"]["accuracy"] < d["accuracy"] - 0.01
    assert gate["points_minus_reference"] == pytest.approx(
        100 * (original["real_val"]["accuracy"] - d["accuracy"]), abs=1e-2
    )
    assert original["real_val"]["accuracy"] == pytest.approx(0.8258, abs=1e-3)
    section = md.split("## Original rule")[1].split("\n## ")[0]
    for text in ("`complement_nb_a0.1`", "82.58%", "82.89%", "**It failed criterion 4**"):
        assert text in section


def handmade(name, accuracy, complexity, log_loss=0.5):
    return {"name": name, "complexity": complexity, "inner_val": {"accuracy": accuracy, "log_loss": log_loss}}


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([("a", 0.80, 0), ("b", 0.85, 3)], "b"),  # simpler, but outside the margin
        ([("a", 0.841, 0), ("b", 0.85, 3)], "a"),  # simpler and within the margin
        ([("a", 0.84, 0), ("b", 0.85, 3)], "a"),  # exactly 1.0 point counts as within
        ([("a", 0.835, 0), ("b", 0.843, 1), ("c", 0.85, 3)], "b"),  # no chaining through b
        ([("a", 0.84, 1), ("b", 0.845, 1), ("c", 0.85, 3)], "b"),  # equally simple: more accurate
        ([("a", 0.85, 0, 0.6), ("b", 0.85, 0, 0.5)], "b"),  # then log loss
        ([("a", 0.85, 0), ("b", 0.85, 0)], "a"),  # then list order
        ([("a", 0.84, 2), ("b", 0.85, 1)], "b"),  # more complex never wins on a lower accuracy
    ],
)
def test_original_rule_on_handmade_rows(mod, rows, expected):
    assert mod.select_original([handmade(*row) for row in rows]) == expected


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([("a", 0.80, 0), ("b", 0.85, 3)], "b"),  # simpler, but outside the margin
        ([("a", 0.84, 0), ("b", 0.85, 3)], "a"),  # exactly 1.0 point counts as within
        ([("a", 0.85, 0, 0.62), ("b", 0.845, 1, 0.5)], "b"),  # simpler, but log loss over the limit
        ([("a", 0.85, 0, 0.6), ("b", 0.845, 1, 0.5)], "a"),  # exactly 0.1 above the best is kept
        ([("a", 0.85, 0, 0.56), ("b", 0.845, 0, 0.5)], "b"),  # equally simple: log loss before accuracy
        ([("a", 0.84, 0, 0.5), ("b", 0.85, 0, 0.5)], "b"),  # then accuracy
        ([("a", 0.85, 0), ("b", 0.85, 0)], "a"),  # then list order
        ([("a", 0.85, 1, 0.45), ("b", 0.845, 0, 0.5)], "b"),  # complexity before log loss
        # The best log loss is over all candidates, also those outside the accuracy margin.
        ([("a", 0.70, 0, 0.3), ("b", 0.85, 0, 0.45), ("c", 0.845, 1, 0.39)], "c"),
    ],
)
def test_amended_rule_on_handmade_rows(mod, rows, expected):
    assert mod.select([handmade(*row) for row in rows]) == expected


def test_discard_reasons_match_numbers(results):
    rows, chosen = results["candidates"], by_name(results)[results["chosen"]["name"]]
    best = min(r["inner_val"]["log_loss"] for r in rows)
    near = [r["name"] for r in near_top(rows)]
    assert set(results["discarded"]) == set(by_name(results)) - {chosen["name"]}
    for name, reason in results["discarded"].items():
        row = by_name(results)[name]
        over = row["inner_val"]["log_loss"] - best > 0.1
        assert reason.startswith("outside the margin") == (name not in near)
        assert reason.startswith("within the margin, but log loss") == (name in near and over)
        if reason.startswith("passes"):
            assert reason.endswith("more complex") == (row["complexity"] > chosen["complexity"])


def test_original_discard_reasons_match_numbers(results):
    original, rows = results["original_rule"], results["candidates"]
    chosen = by_name(results)[original["chosen"]["name"]]
    near = [r["name"] for r in near_top(rows)]
    assert set(original["discarded"]) == set(by_name(results)) - {chosen["name"]}
    for name, reason in original["discarded"].items():
        row = by_name(results)[name]
        gap = 100 * (row["inner_val"]["accuracy"] - chosen["inner_val"]["accuracy"])
        assert reason.startswith("outside the margin") == (name not in near)
        if reason.startswith("more accurate"):
            assert gap > 0 and row["complexity"] > chosen["complexity"]
        elif reason.startswith("more complex"):
            assert gap <= 0 and row["complexity"] > chosen["complexity"]
        elif "less accurate" in reason:
            assert gap < 0 and row["complexity"] == chosen["complexity"]


def test_cue_candidates_reported(results, md):
    cues = [r for r in results["candidates"] if "+" in r["name"]]
    block = results["cue_candidates"]
    near = [r["name"] for r in near_top(results["candidates"]) if r in cues]
    assert block["count"] == len(cues) == 7
    assert block["within_margin"] == near and block["within_margin_count"] == len(near)
    assert block["within_log_loss_limit_count"] == sum(r["log_loss_above_best"] <= 0.1 for r in cues)
    assert f"Cue features: {len(near)} of {len(cues)} cue candidates came within 1.0 point" in md


# --- Supplementary, report-only ---------------------------------------------
def test_chosen_config_unchanged_by_supplementary(mod, results):
    expected = next(c for c in mod.base_candidates() if c["name"] == "logreg_c1_none")
    assert results["chosen"]["name"] == "logreg_c1_none"
    assert results["chosen"]["config"] == expected
    assert results["chosen"]["config"]["feature_groups"] == []
    # Never part of the selection or of real validation.
    names = {row["name"] for row in results["supplementary"]["candidates"]}
    assert not names & (set(by_name(results)) | set(results["discarded"]) | set(final_scores(results)))
    assert results["selection"]["candidate_count"] == len(results["candidates"]) == 22


def test_supplementary_cannot_change_selection(mod, df):
    few = [c for c in mod.base_candidates() if c["name"] in (REFERENCE, "complement_nb_a0.1")]
    perfect = {"accuracy": 1.0, "macro_f1": 1.0, "log_loss": 0.0}
    original = mod.score_inner
    outputs = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "base_candidates", lambda: few)
        patch.setattr(mod, "score_final", lambda cfg, train, val: {"accuracy": 0.0})
        for fake in (False, True):
            patch.setattr(
                mod,
                "score_inner",
                lambda cfg, a, b: perfect if fake and cfg["name"].startswith(f"{REFERENCE}+") else original(cfg, a, b),
            )
            outputs.append(mod.compute_experiments(df))
    assert all(row["inner_val"] == perfect for row in outputs[1]["supplementary"]["candidates"])  # control
    for result in outputs:
        del result["supplementary"]
    assert outputs[0] == outputs[1]


def test_supplementary_block(results, md):
    s, base = results["supplementary"], by_name(results)[REFERENCE]
    assert set(s) == {"base", "base_inner_val", "count", "candidates", "note"}
    assert s["base"] == REFERENCE and s["base_inner_val"] == base["inner_val"]
    assert {r["name"] for r in s["candidates"]} == {f"{REFERENCE}+{g}" for g in GROUPS | {"all"}}
    assert s["count"] == len(s["candidates"]) == 7
    section = md.split("## Supplementary")[1].split("\n## ")[0]
    assert "The selection is unchanged" in s["note"] and s["note"] in section
    for row in s["candidates"]:
        assert set(row) == {
            "name", "feature_groups", "inner_val", "accuracy_points_minus_base", "log_loss_minus_base"
        }
        m, b = row["inner_val"], base["inner_val"]
        points, loss = 100 * (m["accuracy"] - b["accuracy"]), m["log_loss"] - b["log_loss"]
        assert row["accuracy_points_minus_base"] == pytest.approx(points, abs=1e-6)
        assert row["log_loss_minus_base"] == pytest.approx(loss, abs=1e-6)
        cells = [
            f"`{row['name']}`",
            f"{m['accuracy'] * 100:.2f}%",
            f"{row['accuracy_points_minus_base']:+.2f}",
            f"{m['log_loss']:.4f}",
            f"{row['log_loss_minus_base']:+.4f}",
        ]
        assert section.splitlines().count("| " + " | ".join(cells) + " |") == 1


# --- Criterion 4: result on real validation ---------------------------------
def test_chosen_not_worse_than_d_and_beats_bot(results, reports_dir, windows):
    d = json.loads((reports_dir / "baselines.json").read_text(encoding="utf-8"))["baselines"]["tfidf_final"]["evaluate"]
    val = windows["val"]
    bot = float(np.mean(val["team_label_canon"] == val["final_team_canon"]))
    chosen = results["real_val"][results["chosen"]["name"]]["accuracy"]
    assert results["real_val"][REFERENCE]["accuracy"] == pytest.approx(d["accuracy"], abs=1e-4)
    assert results["real_val"][REFERENCE]["log_loss"] == pytest.approx(d["log_loss"], abs=1e-4)
    assert results["bot_real_val_accuracy"] == pytest.approx(bot, abs=1e-6)
    assert chosen >= d["accuracy"] - 0.01
    assert chosen > bot


# --- Criterion 5: allowed columns only --------------------------------------
def scrambled(frame, seed, keep):
    """Every column outside keep permuted on its own."""
    rng, out = np.random.default_rng(seed), frame.copy()
    for col in frame.columns:
        if col not in keep:
            out[col] = frame[col].to_numpy()[rng.permutation(len(frame))]
    return out


def test_scrambled_forbidden_columns_leave_predictions_unchanged(mod, config, results, windows):
    # Fit rows keep their target. Score rows do not need one, so it is scrambled too.
    fit = scrambled(windows["train"], config.SEED, [*ALLOWED, "final_team_canon"])
    score = scrambled(windows["val"], config.SEED + 1, ALLOWED)
    assert (score["team_label_canon"].to_numpy() != windows["val"]["team_label_canon"].to_numpy()).any()
    rows = by_name(results)
    names = {results["chosen"]["name"], REFERENCE, f"{results['selection']['feature_base']}+all"}
    for name in names:
        cfg = rows[name]["config"]
        assert cfg["features"] == ALLOWED
        before = mod.fit_predict(cfg, windows["train"], windows["val"])
        after = mod.fit_predict(cfg, fit, score)
        assert (before[0] == after[0]).all()
        np.testing.assert_array_equal(before[1], after[1])


def test_cue_features_need_only_allowed_columns(mod, windows):
    val = windows["val"]
    full = mod.cue_features(val)
    pd.testing.assert_frame_equal(full, mod.cue_features(val[ALLOWED]))
    assert set(full.to_numpy().ravel()) <= {0.0, 1.0}
    columns = [c for group in mod.FEATURE_GROUPS.values() for c in group]
    assert sorted(columns) == sorted(full.columns) and set(mod.FEATURE_GROUPS) == GROUPS


def test_allowed_column_change_is_detected(mod, results, windows):
    # Control for the scramble test: it would miss a change if predictions ignored the columns.
    val = windows["val"]
    cfg = by_name(results)[f"{results['selection']['feature_base']}+all"]["config"]
    before = mod.fit_predict(cfg, windows["train"], val)[1]
    for col in ALLOWED:
        changed = val.copy()
        changed[col] = np.roll(val[col].to_numpy(), 1)
        assert not np.array_equal(before, mod.fit_predict(cfg, windows["train"], changed)[1]), col


CUE_CASES = [
    ("please send the gst invoice", {"payment_is_problem"}),
    ("i was charged twice", {"payment_is_problem"}),
    ("i paid online for installation", {"payment_only_mentioned", "install_cue"}),
    ("paid by upi, it is not working", {"payment_only_mentioned", "fault_cue"}),
    ("when will the technician come for wall mounting", {"install_cue"}),
    ("installer did not come for demo", {"install_cue"}),
    ("it shows an error and is leaking", {"fault_cue"}),
    ("where is ko1234567", {"order_id_present"}),
    ("ref sr12345", set()),
    ("where is my warranty card", set()),
]


@pytest.mark.parametrize("text, expected", CUE_CASES)
def test_cue_columns(mod, text, expected):
    row = pd.DataFrame(
        {"text_clean": [text], "channel": ["chat"], "product_family": ["Air Fryer"], "warranty_status": ["shield"]}
    )
    cues = mod.cue_features(row).iloc[0]
    assert {c for c in cues.index if cues[c]} == expected


def test_conflict_columns(mod):
    frame = pd.DataFrame(
        {
            "text_clean": ["the fan is here", "the fan is here", "the fryer is here"],
            "channel": ["ivr", "chat", "ivr"],
            "product_family": ["Air Fryer"] * 3,
            "warranty_status": ["shield"] * 3,
        }
    )
    cues = mod.cue_features(frame)
    assert cues["product_conflict"].tolist() == [1, 1, 0]
    assert cues["product_conflict_on_ivr"].tolist() == [1, 0, 0]


# --- Models ----------------------------------------------------------------
def test_reference_is_m4_pipeline(mod, results):
    from src.baselines import make_pipeline

    model, expected = mod.build_model(by_name(results)[REFERENCE]["config"]), make_pipeline()
    assert isinstance(model.named_steps["classifier"], LogisticRegression)
    assert repr(model) == repr(expected)
    assert model.named_steps["classifier"].get_params() == expected.named_steps["classifier"].get_params()


def test_calibration_never_shuffles_and_refits_everything(mod):
    for cfg in mod.base_candidates():
        model = mod.build_model(cfg)
        if cfg["calibration"] is None:
            assert not isinstance(model, CalibratedClassifierCV)
            continue
        assert isinstance(model.cv, KFold) and model.cv.shuffle is False and model.cv.n_splits == 5
        assert model.method == cfg["calibration"]["method"] and model.ensemble is False
        assert "features" in model.estimator.named_steps  # vectorisers are inside the folds


def test_feature_candidates_add_exactly_their_columns(mod, windows):
    sample = windows["inner_train"].tail(400)
    base = mod.base_candidates()[0]
    for cfg in mod.feature_candidates(base):
        added = sum(len(mod.FEATURE_GROUPS[g]) for g in cfg["feature_groups"])
        sizes = []
        for c in (base, cfg):
            features = mod.build_model(c).named_steps["features"]
            sizes.append(features.fit_transform(sample[ALLOWED]).shape[1])
        assert sizes[1] - sizes[0] == added


def test_chosen_config_round_trips_through_json(mod, calls, results, windows):
    cfg = json.loads(json.dumps(results["chosen"]["config"]))
    n_final = len(final_scores(results))
    recorded = next(c for c in calls[-n_final:] if c["name"] == cfg["name"])
    pred, proba = mod.fit_predict(cfg, windows["train"], windows["val"])
    assert (pred == recorded["pred"]).all()
    np.testing.assert_array_equal(proba, recorded["proba"])


# --- Criterion 6: determinism ------------------------------------------------
def test_deterministic_and_rounded(mod, out_dir, results, tmp_path, config):
    mod.build(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"experiments.json", "experiments.md"}
    for name in ("experiments.json", "experiments.md"):
        assert (tmp_path / name).read_bytes() == (out_dir / name).read_bytes()
    for path, value in leaves(results):
        if isinstance(value, float):
            assert round(value, 6) == value, path
    assert results["meta"]["seed"] == config.SEED
    for row in results["candidates"]:
        if "random_state" in row["config"]["params"]:
            assert row["config"]["params"]["random_state"] == config.SEED


# --- Criterion 7: the report ---------------------------------------------------
def unmatched_numbers(md, results):
    """Numbers in the markdown that are not numeric values in the results, allowing for rounding and percentages."""
    values = [v for _, v in leaves(results) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    strings = {v for _, v in leaves(results) if isinstance(v, str)}
    for name in sorted(by_name(results), key=len, reverse=True):  # names contain hyperparameters
        md = md.replace(f"`{name}`", " ")
    bad = [d for d in DATE.findall(md) if d not in strings]
    for token in NUMBER.findall(DATE.sub(" ", md)):
        text = token.rstrip("%").replace(",", "")
        decimals = len(text.partition(".")[2])
        scale = 100 if token.endswith("%") else 1
        if not any(abs(float(f"{v * scale:.{decimals}f}")) == abs(float(text)) for v in values):
            bad.append(token)
    return bad


def test_every_md_number_in_json(mod, md, results, config):
    assert len(NUMBER.findall(md)) > 100
    assert not unmatched_numbers(md, results)
    assert mod.render_markdown(results) == md
    assert unmatched_numbers(md + "\n- 98.7654% of 31,415\n", results)  # control

    # Replace every number: a hand-typed one would no longer be found in the results.
    rng = random.Random(config.SEED)
    changed = json.loads(json.dumps(results))
    for path, value in leaves(results):
        if isinstance(value, str) or value is None or isinstance(value, bool):
            continue
        node = changed
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = rng.randrange(10**7, 10**8) if isinstance(value, int) else rng.uniform(10**5, 10**6)
    assert not unmatched_numbers(mod.render_markdown(changed), changed)


def test_md_table_cells_match_keys(md, results, config):
    lines = md.splitlines()
    for row in results["candidates"]:
        m = row["inner_val"]
        cells = [
            f"`{row['name']}`",
            str(row["complexity"]),
            f"{m['accuracy'] * 100:.2f}%",
            f"{m['macro_f1'] * 100:.2f}%",
            f"{m['log_loss']:.4f}",
            f"{row['points_minus_chosen']:.2f}",
        ]
        assert lines.count("| " + " | ".join(cells) + " |") == 1
    for r in results["candidates"]:
        if 100 * (max(x["inner_val"]["accuracy"] for x in results["candidates"]) - r["inner_val"]["accuracy"]) > 1.0:
            continue
        m, outcome = r["inner_val"], results["discarded"].get(r["name"], "chosen")
        cells = [f"`{r['name']}`", f"{m['accuracy'] * 100:.2f}%", f"{m['log_loss']:.4f}"]
        cells += [f"{r['log_loss_above_best']:.4f}", outcome]
        assert lines.count("| " + " | ".join(cells) + " |") == 1
    for name, s in results["real_val"].items():
        ci = s["accuracy_ci95"]
        cells = [
            f"`{name}`",
            f"{s['accuracy'] * 100:.2f}%",
            f"{ci['low'] * 100:.2f}% to {ci['high'] * 100:.2f}%",
            f"{s['macro_f1'] * 100:.2f}%",
            f"{s['log_loss']:.4f}",
        ]
        assert lines.count("| " + " | ".join(cells) + " |") == 1
    final = results["real_val"][results["chosen"]["name"]]
    for team in config.TEAMS:
        assert f"| {team} | {final['per_team_recall'][team] * 100:.2f}% |" in lines
    for cell in final["top_confusions"]:
        assert f"| {cell['true']} | {cell['predicted']} | {cell['count']} |" in lines


def test_md_states_choice_reasons_and_caveats(mod, md, results):
    assert md.count("**Chosen: ") == 1
    assert f"**Chosen: `{results['chosen']['name']}`**, because it {results['chosen']['reason']}" in md
    for name, reason in results["discarded"].items():
        assert f"- `{name}`: {reason}." in md
    assert results["selection"]["amendment_reason"] in md
    assert results["selection"]["rule"] in md and results["original_rule"]["rule"] in md
    for name, reason in results["original_rule"]["discarded"].items():
        assert f"- `{name}`: {reason}." in md
    disclosure = md.split("## Disclosure")[1].split("\n## ")[0]
    assert "amended after one real validation result" in disclosure
    assert "already seen in M4, so it is not an untouched estimate" in disclosure
    assert "within noise" in disclosure and results["selection"]["accuracy_first_choice"] in disclosure
    caveat = next(line for line in md.splitlines() if "technician" in line)
    assert "not blind" in caveat and "inner-val" in caveat and "after validation" in caveat


def test_no_request_text_or_old_names(out_dir, df):
    report = set()
    for path in out_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        report |= word_runs(text)
        text = text.replace("Filters & Consumables", "")
        assert "Installations" not in text and "Consumables" not in text
    shared = sum(len(word_runs(t) & report) for c in ("text_clean", "request_text") for t in set(df[c]))
    # Count only: a failure must not print customer text.
    if shared:
        pytest.fail(f"{shared} four-word runs shared with request text", pytrace=False)


def test_committed_reports_current(mod, results, reports_dir):
    # Tolerance: another BLAS or CPU may move model floats slightly. One row of validation moves accuracy by 4.7e-4.
    stored = json.loads((reports_dir / "experiments.json").read_text(encoding="utf-8"))
    stored_leaves, fresh_leaves = dict(leaves(stored)), dict(leaves(results))
    assert set(stored_leaves) == set(fresh_leaves), "experiments.json is stale: run clean, then experiments"
    for path, fresh in fresh_leaves.items():
        # Point differences are 100 times a share, so 1e-4 there would be stricter than for shares.
        tolerance = 1e-2 if any("points" in str(p) for p in path) else 1e-4
        expected = pytest.approx(fresh, rel=1e-9, abs=tolerance) if isinstance(fresh, float) else fresh
        assert stored_leaves[path] == expected, f"experiments.json is stale at {path}: run clean, then experiments"
    stored_md = (reports_dir / "experiments.md").read_text(encoding="utf-8")
    assert stored_md == mod.render_markdown(stored), "experiments.md is stale: run clean, then experiments"
