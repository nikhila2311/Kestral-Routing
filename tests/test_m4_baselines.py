"""M4: baselines. All example strings here are invented, not customer text."""
import json
import random
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

NAMES = {"majority", "bot", "keywords", "tfidf_final", "tfidf_bot"}
TITLES = {
    "majority": "(a) Majority team",
    "bot": "(b) Bot label",
    "keywords": "(c) Keyword rules",
    "tfidf_final": "(d) TF-IDF, trained on the final team",
    "tfidf_bot": "(e) TF-IDF, trained on the bot label",
}
SCORE_KEYS = {"evaluate", "accuracy_ci95", "agreement_with_team_label", "cost_rs", "cost_rs_per_month"}
EVALUATE_KEYS = {"accuracy", "macro_f1", "per_class", "confusion_matrix", "n", "misroutes", "cost_rs"}
PROBA_KEYS = {"log_loss", "coverage_curve"}
COMPARISON_KEYS = {
    "accuracy_points_tfidf_final_minus_bot",
    "accuracy_points_tfidf_final_minus_tfidf_bot",
    "accuracy_points_keywords_minus_bot",
    "agreement_points_tfidf_bot_minus_tfidf_final",
    "cost_rs_per_month_bot_minus_tfidf_final",
}
ALLOWED = {"text_clean", "channel", "product_family", "warranty_status"}
PLANTED = "zzqxplanted"
FORBIDDEN = (
    "team_label",
    "team_label_canon",
    "first_team_canon",
    "transfers",
    "misrouted",
    "resolved_at_ist",
    "resolution_hours",
)
TOP_KEYS = {
    "baselines",
    "comparisons",
    "paired_accuracy_points_tfidf_final_minus_bot",
    "cost_sensitivity",
    "repeated_text",
    "meta",
}

DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


@pytest.fixture(scope="module")
def mod():
    from src import baselines

    return baselines


@pytest.fixture(scope="module")
def df(processed_dir):
    return pd.read_parquet(processed_dir / "train_clean.parquet")


@pytest.fixture(scope="module")
def test_text(processed_dir):
    return pd.read_parquet(processed_dir / "test_clean.parquet")["text_clean"]


@pytest.fixture(scope="module")
def val(df):
    # Filtered here, not with time_split, so a split error cannot hide in both places.
    created = df["created_at_ist"]
    return df[(created >= "2026-04-01") & (created < "2026-07-01")]


@pytest.fixture(scope="module")
def out_dir(mod, tmp_path_factory):
    # Never build into reports/: a broken run must not replace the real files.
    path = tmp_path_factory.mktemp("baselines")
    mod.build(path)
    return path


@pytest.fixture(scope="module")
def results(out_dir):
    return json.loads((out_dir / "baselines.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(out_dir):
    return (out_dir / "baselines.md").read_text(encoding="utf-8")


def run_and_record(mod, df, test_text):
    """compute_baselines, plus the (predictions, probabilities) each baseline was scored on."""
    calls, original = [], mod.score

    def recording_score(val_df, pred, proba=None):
        calls.append((np.asarray(pred, dtype=object).copy(), proba))
        return original(val_df, pred, proba)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "score", recording_score)
        result = mod.compute_baselines(df, test_text)
    assert len(calls) == len(result["baselines"])
    return result, dict(zip(result["baselines"], calls))


@pytest.fixture(scope="module")
def full_run(mod, df, test_text):
    return run_and_record(mod, df, test_text)


@pytest.fixture(scope="module")
def full(full_run):
    """Unrounded results, so values are checked exactly; test_json_is_rounded_results ties the JSON to them."""
    return full_run[0]


@pytest.fixture(scope="module")
def pipelines():
    """Every pipeline make_pipeline returned during planted_run."""
    return []


@pytest.fixture(scope="module")
def planted_run(mod, df, test_text, pipelines):
    """compute_baselines with an invented token in every validation text; returns the fits it made."""
    planted = df.copy()
    in_val = planted["created_at_ist"] >= "2026-04-01"
    planted.loc[in_val, "text_clean"] = planted.loc[in_val, "text_clean"] + " " + PLANTED

    fits, original = [], mod.fit_model

    def recording_fit(train_df, target_col):
        model = original(train_df, target_col)
        fits.append((train_df, target_col, model))
        return model

    original_make = mod.make_pipeline

    def recording_make():
        pipelines.append(original_make())
        return pipelines[-1]

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-converged fit fails here
        patch.setattr(mod, "fit_model", recording_fit)
        patch.setattr(mod, "make_pipeline", recording_make)
        mod.compute_baselines(planted, test_text)
    return fits


def accuracy(results, name):
    return results["baselines"][name]["evaluate"]["accuracy"]


def agreement(results, name):
    return results["baselines"][name]["agreement_with_team_label"]


def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, path + (key,))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, path + (i,))
    else:
        yield path, node


def unmatched_numbers(md, results):
    """Numbers in the markdown that are not in the results, allowing for rounding and percentages."""
    values, strings = [], set()
    for path, value in leaves(results):
        # These hundreds of numbers are not shown in the markdown and would match almost anything by accident.
        if {"confusion_matrix", "per_class", "coverage_curve"} & set(path):
            continue
        if isinstance(value, str):
            strings.add(value)
        else:
            values.append(value)

    bad = [d for d in DATE.findall(md) if d not in strings]
    for token in NUMBER.findall(DATE.sub(" ", md)):
        text = token.rstrip("%").replace(",", "")
        decimals = len(text.partition(".")[2])
        shown = float(text)
        if token.endswith("%"):
            ok = any(round(v * 100, decimals) == shown for v in values if isinstance(v, float) and 0 <= v <= 1)
        else:
            # A share must not count as a match for a bare 0 or 1.
            ok = any(round(abs(v), decimals) == shown for v in values if isinstance(v, int) or abs(v) > 1)
        if not ok:
            bad.append(token)
    return bad


def word_runs(text, n=4):
    tokens = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# --- Criterion 1 -----------------------------------------------------------
def test_all_baselines_and_keys(results, config, val):
    assert set(results) == TOP_KEYS
    assert set(results["baselines"]) == NAMES
    assert set(results["comparisons"]) == COMPARISON_KEYS
    for name, scores in results["baselines"].items():
        assert set(scores) == SCORE_KEYS | ({"rule_fired_share"} if name == "keywords" else set())
        expected = EVALUATE_KEYS | (PROBA_KEYS if name.startswith("tfidf") else set())
        assert set(scores["evaluate"]) == expected
        assert set(scores["evaluate"]["per_class"]) == set(config.TEAMS)
        assert scores["evaluate"]["n"] == len(val) == 2135
        ci = scores["accuracy_ci95"]
        assert ci["low"] < ci["point"] == scores["evaluate"]["accuracy"] < ci["high"]


def test_costs_and_comparisons_recomputed(full, config):
    for scores in full["baselines"].values():
        cost = scores["evaluate"]["misroutes"] * config.COST_PER_MISROUTE
        assert scores["cost_rs"] == scores["evaluate"]["cost_rs"] == pytest.approx(cost)
        assert scores["cost_rs_per_month"] == pytest.approx(cost / 3)
    c = full["comparisons"]
    points = 100 * (accuracy(full, "tfidf_final") - accuracy(full, "bot"))
    assert c["accuracy_points_tfidf_final_minus_bot"] == pytest.approx(points)
    points = 100 * (accuracy(full, "tfidf_final") - accuracy(full, "tfidf_bot"))
    assert c["accuracy_points_tfidf_final_minus_tfidf_bot"] == pytest.approx(points)
    points = 100 * (accuracy(full, "keywords") - accuracy(full, "bot"))
    assert c["accuracy_points_keywords_minus_bot"] == pytest.approx(points)
    points = 100 * (agreement(full, "tfidf_bot") - agreement(full, "tfidf_final"))
    assert c["agreement_points_tfidf_bot_minus_tfidf_final"] == pytest.approx(points)
    monthly = {name: s["cost_rs_per_month"] for name, s in full["baselines"].items()}
    assert c["cost_rs_per_month_bot_minus_tfidf_final"] == pytest.approx(monthly["bot"] - monthly["tfidf_final"])
    assert full["meta"]["bot_licence_rs_per_month"] == pytest.approx(320_000 / 12)


def test_interval_matches_m3_bootstrap(full, val, config):
    from src.evaluate import bootstrap_ci

    low, _, high = bootstrap_ci(
        val["final_team_canon"], val["team_label_canon"], lambda t, p: float((t == p).mean()), seed=config.SEED
    )
    ci = full["baselines"]["bot"]["accuracy_ci95"]
    assert (ci["low"], ci["high"]) == (low, high)


def test_json_is_rounded_results(results, full):
    json_leaves, full_leaves = dict(leaves(results)), dict(leaves(full))
    assert set(json_leaves) == set(full_leaves)
    for path, value in full_leaves.items():
        assert json_leaves[path] == (round(value, 6) if isinstance(value, float) else value), path


def test_paired_interval_recomputed(full, full_run, val, config):
    truth = val["final_team_canon"].to_numpy(dtype=object)
    final_ok = (full_run[1]["tfidf_final"][0] == truth).astype(int)
    bot_ok = (val["team_label_canon"].to_numpy(dtype=object) == truth).astype(int)
    difference = final_ok - bot_ok
    rng = np.random.default_rng(config.SEED)
    samples = [difference[rng.integers(0, len(val), len(val))].mean() for _ in range(1000)]
    low, high = 100 * np.percentile(samples, [2.5, 97.5])
    paired = full["paired_accuracy_points_tfidf_final_minus_bot"]
    assert set(paired) == {"low", "point", "high"}
    assert (paired["low"], paired["high"]) == (pytest.approx(low), pytest.approx(high))
    assert paired["point"] == pytest.approx(full["comparisons"]["accuracy_points_tfidf_final_minus_bot"])
    assert paired["low"] < paired["point"] < paired["high"]


def test_cost_sensitivity_recomputed(full, config):
    assert set(full["cost_sensitivity"]) == {"average", "simple"}
    for key, cost in (("average", config.COST_PER_MISROUTE), ("simple", 565)):
        s = full["cost_sensitivity"][key]
        assert s["cost_per_misroute_rs"] == cost
        assert set(s["cost_rs_per_month"]) == NAMES
        for name, monthly in s["cost_rs_per_month"].items():
            assert monthly == pytest.approx(full["baselines"][name]["evaluate"]["misroutes"] * cost / 3)
        saving = s["cost_rs_per_month"]["bot"] - s["cost_rs_per_month"]["tfidf_final"]
        assert s["saving_rs_per_month_bot_minus_tfidf_final"] == pytest.approx(saving)


# --- Criterion 2 -----------------------------------------------------------
def test_bot_accuracy_matches_parquet(full, val):
    assert accuracy(full, "bot") == (val["team_label_canon"] == val["final_team_canon"]).mean()
    assert agreement(full, "bot") == 1.0


# --- Criterion 3 -----------------------------------------------------------
def test_tfidf_beats_bot(results):
    assert accuracy(results, "tfidf_final") > accuracy(results, "bot")


# --- Criterion 4 -----------------------------------------------------------
def test_final_target_vs_bot_target(results):
    assert accuracy(results, "tfidf_final") > accuracy(results, "tfidf_bot")
    assert agreement(results, "tfidf_bot") > agreement(results, "tfidf_final")


# --- Criterion 5 -----------------------------------------------------------
def test_majority_is_lowest(full, df, val):
    for name in NAMES - {"majority"}:
        assert accuracy(full, "majority") < accuracy(full, name)
    # The majority team comes from the train window, and its score is that team's share of validation.
    team = df.loc[df["created_at_ist"] < "2026-04-01", "final_team_canon"].value_counts().idxmax()
    assert full["meta"]["majority_team"] == team
    assert accuracy(full, "majority") == (val["final_team_canon"] == team).mean()


def test_majority_tie(mod):
    assert mod.majority_class(["Billing", "Repairs"]) == "Repairs"  # first in config.TEAMS
    assert mod.majority_class(["Billing", "Billing", "Repairs"]) == "Billing"


def test_majority_comes_from_train_window(mod, df, test_text, results):
    # The real majority team is the same in both windows, so give validation a different one.
    changed = df.copy()
    changed.loc[changed["created_at_ist"] >= "2026-04-01", "final_team_canon"] = "Product Advice"
    assert results["meta"]["majority_team"] != "Product Advice"
    assert mod.compute_baselines(changed, test_text)["meta"]["majority_team"] == results["meta"]["majority_team"]


# --- Criterion 6 -----------------------------------------------------------
def test_only_allowed_features(mod, results, planted_run):
    assert set(mod.ALLOWED_FEATURES) == set(results["meta"]["features"]) == ALLOWED
    for _, _, model in planted_run:
        assert set(model.feature_names_in_) == ALLOWED
        columns = model.named_steps["features"]
        used = set()
        for _, _, cols in columns.transformers_:
            used |= {cols} if isinstance(cols, str) else set(cols)
        assert used == ALLOWED


def test_predictions_ignore_forbidden_columns(mod, df, val, planted_run):
    model = planted_run[0][2]
    _, expected = mod.predict_with_proba(model, val)
    _, actual = mod.predict_with_proba(model, val[sorted(ALLOWED)])
    assert np.array_equal(actual, expected)


def test_forbidden_columns_do_not_change_predictions(mod, config, df, test_text, full_run):
    rng = np.random.default_rng(config.SEED)
    scrambled = df.copy()
    for column in FORBIDDEN:
        # One permutation per column, so the columns do not stay consistent with each other.
        scrambled[column] = df[column].to_numpy()[rng.permutation(len(df))]
    _, before = full_run
    _, after = run_and_record(mod, scrambled, test_text)
    for name in ("majority", "keywords", "tfidf_final"):
        assert bool((before[name][0] == after[name][0]).all()), name
    assert np.array_equal(before["tfidf_final"][1], after["tfidf_final"][1])
    # Control: the scramble did reach the baselines that are meant to depend on the bot label.
    for name in ("bot", "tfidf_bot"):
        assert bool((before[name][0] != after[name][0]).any()), name


def test_vocab_fitted_on_train_only(mod, planted_run):
    assert len(planted_run) == 2
    for train_df, _, model in planted_run:
        features = model.named_steps["features"]
        assert PLANTED not in features.named_transformers_["word"].vocabulary_
        # zzqxp holds the character runs zzq, zqx, qxp, which appear only in the planted token.
        train_text = " ".join(train_df["text_clean"])
        char_vocab = features.named_transformers_["char"].vocabulary_
        for run in ("zzqx", "zqxp"):
            assert run not in train_text
            assert run not in char_vocab


def test_planted_token_would_be_seen(mod, df):
    # Control: if validation rows were fitted on, the check above would fail.
    leaky = df.copy()
    leaky["text_clean"] = leaky["text_clean"] + " " + PLANTED
    features = mod.fit_model(leaky.tail(600), "final_team_canon").named_steps["features"]
    assert PLANTED in features.named_transformers_["word"].vocabulary_
    assert "zzqx" in features.named_transformers_["char"].vocabulary_


def test_build_fits_on_train_window(planted_run, df):
    assert [target for _, target, _ in planted_run] == ["final_team_canon", "team_label_canon"]
    expected = df.index[df["created_at_ist"] < "2026-04-01"]
    for train_df, _, _ in planted_run:
        assert train_df["created_at_ist"].max() < pd.Timestamp("2026-04-01")
        assert train_df.index.equals(expected)


def test_model_settings(mod, config, planted_run):
    for _, _, model in planted_run:
        classifier = model.named_steps["classifier"]
        assert classifier.class_weight is None
        assert classifier.random_state == config.SEED
        assert classifier.n_iter_.max() < classifier.max_iter
        features = model.named_steps["features"].named_transformers_
        assert features["word"].ngram_range == (1, 2)
        assert features["char"].ngram_range == (3, 5)
        assert features["char"].analyzer == "char_wb"


def comparable(params):
    """get_params as plain values: estimators become their type, step lists become names, types and columns."""

    def plain(value):
        if hasattr(value, "get_params"):
            return type(value)
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    return {key: plain(value) for key, value in params.items()}


def test_final_and_bot_models_differ_only_in_target(planted_run, pipelines):
    (_, _, final), (_, _, as_bot) = planted_run
    # Both come straight from the shared make_pipeline, which was called once for each.
    assert len(pipelines) == 2 and final is pipelines[0] and as_bot is pipelines[1]
    assert comparable(final.get_params()) == comparable(as_bot.get_params())
    assert list(final.named_steps) == list(as_bot.named_steps) == ["features", "classifier"]
    # ColumnTransformer fits clones, so the fitted sub-steps are compared as well.
    fitted = [m.named_steps["features"].named_transformers_ for m in (final, as_bot)]
    assert set(fitted[0]) == set(fitted[1]) >= {"word", "char", "category"}
    for step in ("word", "char", "category"):
        assert type(fitted[0][step]) is type(fitted[1][step])
        assert comparable(fitted[0][step].get_params()) == comparable(fitted[1][step].get_params())


def test_settings_pinned(planted_run):
    for _, _, model in planted_run:
        features, classifier = model.named_steps["features"], model.named_steps["classifier"]
        assert features.transformer_weights is None and features.remainder == "drop"
        fitted = features.named_transformers_
        assert type(fitted["word"]) is type(fitted["char"]) is TfidfVectorizer
        assert type(fitted["category"]) is OneHotEncoder
        assert (fitted["word"].analyzer, fitted["char"].analyzer) == ("word", "char_wb")
        for step in ("word", "char"):
            assert fitted[step].min_df == 2
            assert fitted[step].sublinear_tf is True
        assert type(classifier) is LogisticRegression
        assert classifier.C == 1.0
        assert classifier.max_iter == 2000


def test_proba_columns_follow_teams(mod, config, df, planted_run):
    model = planted_run[0][2]
    sample = df.tail(50)
    pred, proba = mod.predict_with_proba(model, sample)
    assert list(pred) == list(model.predict(sample[list(mod.ALLOWED_FEATURES)]))
    raw = model.predict_proba(sample[list(mod.ALLOWED_FEATURES)])
    for i, team in enumerate(config.TEAMS):
        assert np.array_equal(proba[:, i], raw[:, list(model.classes_).index(team)])


# --- Criterion 7 -----------------------------------------------------------
def test_build_deterministic(mod, out_dir, tmp_path):
    mod.build(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"baselines.json", "baselines.md"}
    for name in ("baselines.json", "baselines.md"):
        assert (tmp_path / name).read_bytes() == (out_dir / name).read_bytes()


# --- Criterion 8 -----------------------------------------------------------
def test_md_numbers_in_json(mod, md, results, config):
    assert len(NUMBER.findall(md)) > 40
    assert not unmatched_numbers(md, results)
    assert mod.render_markdown(results) == md

    # Replace every number: a hand-typed one would no longer be found in the results.
    rng = random.Random(config.SEED)
    changed = json.loads(json.dumps(results))
    for path, value in leaves(results):
        if isinstance(value, str):
            continue
        node = changed
        for p in path[:-1]:
            node = node[p]
        if isinstance(value, int):
            node[path[-1]] = rng.randrange(10**7, 10**8)
        elif path[0] == "comparisons":
            node[path[-1]] = rng.choice((-1, 1)) * rng.uniform(10**7, 10**8)
        else:
            node[path[-1]] = rng.random() if value <= 1 else rng.uniform(10**7, 10**8)
    assert not unmatched_numbers(mod.render_markdown(changed), changed)


def test_md_table_cells_match_keys(md, results):
    lines = md.splitlines()
    for name, title in TITLES.items():
        s = results["baselines"][name]
        e, ci = s["evaluate"], s["accuracy_ci95"]
        cells = [
            title,
            f"{e['accuracy'] * 100:.2f}%",
            f"{ci['low'] * 100:.2f}% to {ci['high'] * 100:.2f}%",
            f"{e['macro_f1'] * 100:.2f}%",
            f"{s['agreement_with_team_label'] * 100:.2f}%",
            f"{e['misroutes']:,}",
            f"{s['cost_rs']:,.0f}",
            f"{s['cost_rs_per_month']:,.0f}",
        ]
        assert lines.count("| " + " | ".join(cells) + " |") == 1


def test_findings_count(md):
    findings = md.split("## Findings")[1].strip().splitlines()
    assert 3 <= len(findings) <= 5


def test_findings_wording_follows_values(mod, results):
    flipped = json.loads(json.dumps(results))
    for key, value in results["comparisons"].items():
        flipped["comparisons"][key] = -value
    before, after = mod.render_markdown(results), mod.render_markdown(flipped)
    assert "points above the bot's" in before and "points below the bot's" in after
    assert "a monthly saving of" in before and "a monthly increase of" in after
    assert "do not overlap" in before
    flipped["baselines"]["tfidf_final"]["accuracy_ci95"] = results["baselines"]["bot"]["accuracy_ci95"]
    assert "do not overlap" not in mod.render_markdown(flipped)


# --- Repeated text --------------------------------------------------------
def test_repeated_text_shares_recomputed(full, df, val, test_text):
    train_texts = set(df.loc[df["created_at_ist"] < "2026-04-01", "text_clean"])
    val_seen = sum(t in train_texts for t in val["text_clean"])
    test_seen = sum(t in train_texts for t in test_text)
    r = full["repeated_text"]
    # Numbers only: a failure must not print customer text.
    assert r["validation_seen_rows"] == val_seen
    assert r["validation_unseen_rows"] == len(val) - val_seen
    assert r["validation_seen_share"] == val_seen / len(val)
    assert r["test_seen_share"] == test_seen / len(test_text)


def test_repeated_text_accuracy(full, df, val):
    r = full["repeated_text"]
    assert set(r["accuracy"]) == {"bot", "tfidf_final"}
    train_texts = set(df.loc[df["created_at_ist"] < "2026-04-01", "text_clean"])
    seen = np.array([t in train_texts for t in val["text_clean"]])
    correct = (val["team_label_canon"] == val["final_team_canon"]).to_numpy()
    assert r["accuracy"]["bot"] == {"seen": correct[seen].mean(), "unseen": correct[~seen].mean()}

    # Both groups together must give the overall accuracy.
    for name, acc in r["accuracy"].items():
        total = acc["seen"] * r["validation_seen_rows"] + acc["unseen"] * r["validation_unseen_rows"]
        assert total / len(val) == pytest.approx(accuracy(full, name))
    final, share = r["accuracy"]["tfidf_final"], r["test_seen_share"]
    expected = share * final["seen"] + (1 - share) * final["unseen"]
    assert r["tfidf_final_accuracy_at_test_share"] == pytest.approx(expected)


def test_repeated_text_lines_match_json(md, results):
    r, lines = results["repeated_text"], md.splitlines()
    for name in ("bot", "tfidf_final"):
        acc = r["accuracy"][name]
        assert lines.count(f"| {TITLES[name]} | {acc['seen'] * 100:.2f}% | {acc['unseen'] * 100:.2f}% |") == 1
    assert lines.count(f"- Test texts also in the train window: {r['test_seen_share'] * 100:.2f}%.") == 1
    assert (
        lines.count(
            f"- Validation texts also in the train window: {r['validation_seen_share'] * 100:.2f}% "
            f"({r['validation_seen_rows']:,} seen, {r['validation_unseen_rows']:,} unseen)."
        )
        == 1
    )


def test_no_repeat_verdict(md):
    assert "optimistic" not in md and "pessimistic" not in md


# --- Cost caveats and paired interval --------------------------------------
def test_md_cost_caveats_and_sensitivity(md, results):
    lines = md.splitlines()
    assert lines.count("- Our misroutes are assumed to bounce between teams as often as the bot's did.") == 1
    assert sum("average covers all labelled months, including the validation months" in line for line in lines) == 1
    order = ["majority", "bot", "keywords", "tfidf_final", "tfidf_bot"]
    for key, shown in (("average", "698.06"), ("simple", "565.00")):
        s = results["cost_sensitivity"][key]
        costs = [f"{s['cost_rs_per_month'][name]:,.0f}" for name in order]
        cells = [shown, *costs, f"{s['saving_rs_per_month_bot_minus_tfidf_final']:,.0f}"]
        assert lines.count("| " + " | ".join(cells) + " |") == 1


def test_md_paired_interval(md, results):
    p = results["paired_accuracy_points_tfidf_final_minus_bot"]
    assert f"interval for the difference is {p['low']:.2f} to {p['high']:.2f} points." in md


# --- Keyword rules ---------------------------------------------------------
@pytest.mark.parametrize(
    "text, team",
    [
        ("i paid for installation last week, nobody has come", "Installs & Demo"),
        ("already paid for repair and the mixer is still not working", "Repairs"),
        ("settled through emi, when is the demo", "Installs & Demo"),
        ("my card was charged twice for one order", "Billing"),
        ("kindly send the gst invoice", "Billing"),
        ("emi conversion is pending since monday", "Billing"),
        ("refund of my payment has not come", "Billing"),
        ("the box came damaged, please exchange it", "Returns & Replacement"),
        ("is the motor covered under warranty", "Warranty Claims"),
        ("in warranty and leaking from the tap", "Repairs"),
        ("filter is leaking since morning", "Repairs"),
        ("need a new candle and membrane", "Filters & Consumables"),
        ("wall mounting visit needed", "Installs & Demo"),
        ("how to set the timer", "Product Advice"),
        ("technician did not turn up for the demo", "Installs & Demo"),
    ],
)
def test_keyword_rules(mod, text, team):
    predicted, fired = mod.keyword_predict([text], "fallback")
    assert (predicted[0], bool(fired[0])) == (team, True)


@pytest.mark.parametrize(
    "text", ["i have paid", "settled through emi", "it is in warranty", "please call me", "awaiting your technician"]
)
def test_mention_alone_fires_nothing(mod, text):
    predicted, fired = mod.keyword_predict([text], "fallback")
    assert (predicted[0], bool(fired[0])) == ("fallback", False)


def test_keyword_rule_teams_are_canonical(mod, config):
    assert {team for team, _ in mod.KEYWORD_RULES} <= set(config.TEAMS)


# --- Other -----------------------------------------------------------------
def test_committed_reports_current(mod, results, reports_dir):
    # Tolerance: another BLAS or CPU may move model floats slightly. One row of validation moves accuracy by 4.7e-4.
    stored = json.loads((reports_dir / "baselines.json").read_text(encoding="utf-8"))
    stored_leaves, fresh_leaves = dict(leaves(stored)), dict(leaves(results))
    assert set(stored_leaves) == set(fresh_leaves), "baselines.json is stale: run clean, then baselines"
    for path, fresh in fresh_leaves.items():
        expected = pytest.approx(fresh, rel=1e-9, abs=1e-4) if isinstance(fresh, float) else fresh
        assert stored_leaves[path] == expected, f"baselines.json is stale at {path}: run clean, then baselines"
    # The markdown is checked exactly against its own JSON. Universal newlines: git may check it out with CRLF.
    stored_md = (reports_dir / "baselines.md").read_text(encoding="utf-8")
    assert stored_md == mod.render_markdown(stored), "baselines.md is stale: run clean, then baselines"


def test_no_request_text_in_reports(out_dir, df):
    report = set()
    for path in out_dir.iterdir():
        report |= word_runs(path.read_text(encoding="utf-8"))
    shared = sum(len(word_runs(t) & report) for c in ("text_clean", "request_text") for t in set(df[c]))
    # Count only: a failure must not print customer text.
    if shared:
        pytest.fail(f"{shared} four-word runs shared with request text", pytrace=False)


def test_no_old_team_names_in_reports(out_dir):
    for path in out_dir.iterdir():
        text = path.read_text(encoding="utf-8").replace("Filters & Consumables", "")
        assert "Installations" not in text and "Consumables" not in text
