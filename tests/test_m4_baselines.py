"""M4: baselines. All example strings here are invented, not customer text."""
import json
import random
import re
import warnings

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture(scope="module")
def planted_run(mod, df):
    """compute_baselines with an invented token in every validation text; returns the fits it made."""
    planted = df.copy()
    in_val = planted["created_at_ist"] >= "2026-04-01"
    planted.loc[in_val, "text_clean"] = planted.loc[in_val, "text_clean"] + " " + PLANTED

    fits, original = [], mod.fit_model

    def recording_fit(train_df, target_col):
        model = original(train_df, target_col)
        fits.append((train_df, target_col, model))
        return model

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-converged fit fails here
        patch.setattr(mod, "fit_model", recording_fit)
        mod.compute_baselines(planted)
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
    assert set(results) == {"baselines", "comparisons", "meta"}
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


def test_costs_and_comparisons_recomputed(results, config):
    for scores in results["baselines"].values():
        cost = scores["evaluate"]["misroutes"] * config.COST_PER_MISROUTE
        assert scores["cost_rs"] == scores["evaluate"]["cost_rs"] == pytest.approx(cost)
        assert scores["cost_rs_per_month"] == pytest.approx(cost / 3)
    c = results["comparisons"]
    points = 100 * (accuracy(results, "tfidf_final") - accuracy(results, "bot"))
    assert c["accuracy_points_tfidf_final_minus_bot"] == pytest.approx(points)
    points = 100 * (accuracy(results, "tfidf_final") - accuracy(results, "tfidf_bot"))
    assert c["accuracy_points_tfidf_final_minus_tfidf_bot"] == pytest.approx(points)
    points = 100 * (accuracy(results, "keywords") - accuracy(results, "bot"))
    assert c["accuracy_points_keywords_minus_bot"] == pytest.approx(points)
    points = 100 * (agreement(results, "tfidf_bot") - agreement(results, "tfidf_final"))
    assert c["agreement_points_tfidf_bot_minus_tfidf_final"] == pytest.approx(points)
    monthly = {name: s["cost_rs_per_month"] for name, s in results["baselines"].items()}
    assert c["cost_rs_per_month_bot_minus_tfidf_final"] == pytest.approx(monthly["bot"] - monthly["tfidf_final"])
    assert results["meta"]["bot_licence_rs_per_month"] == pytest.approx(320_000 / 12)


def test_interval_matches_m3_bootstrap(results, val, config):
    from src.evaluate import bootstrap_ci

    low, _, high = bootstrap_ci(
        val["final_team_canon"], val["team_label_canon"], lambda t, p: float((t == p).mean()), seed=config.SEED
    )
    ci = results["baselines"]["bot"]["accuracy_ci95"]
    assert (ci["low"], ci["high"]) == (low, high)


# --- Criterion 2 -----------------------------------------------------------
def test_bot_accuracy_matches_parquet(results, val):
    assert accuracy(results, "bot") == (val["team_label_canon"] == val["final_team_canon"]).mean()
    assert agreement(results, "bot") == 1.0


# --- Criterion 3 -----------------------------------------------------------
def test_tfidf_beats_bot(results):
    assert accuracy(results, "tfidf_final") > accuracy(results, "bot")


# --- Criterion 4 -----------------------------------------------------------
def test_final_target_vs_bot_target(results):
    assert accuracy(results, "tfidf_final") > accuracy(results, "tfidf_bot")
    assert agreement(results, "tfidf_bot") > agreement(results, "tfidf_final")


# --- Criterion 5 -----------------------------------------------------------
def test_majority_is_lowest(results, df, val):
    for name in NAMES - {"majority"}:
        assert accuracy(results, "majority") < accuracy(results, name)
    # The majority team comes from the train window, and its score is that team's share of validation.
    team = df.loc[df["created_at_ist"] < "2026-04-01", "final_team_canon"].value_counts().idxmax()
    assert results["meta"]["majority_team"] == team
    assert accuracy(results, "majority") == (val["final_team_canon"] == team).mean()


def test_majority_tie(mod):
    assert mod.majority_class(["Billing", "Repairs"]) == "Repairs"  # first in config.TEAMS
    assert mod.majority_class(["Billing", "Billing", "Repairs"]) == "Billing"


def test_majority_comes_from_train_window(mod, df, results):
    # The real majority team is the same in both windows, so give validation a different one.
    changed = df.copy()
    changed.loc[changed["created_at_ist"] >= "2026-04-01", "final_team_canon"] = "Product Advice"
    assert results["meta"]["majority_team"] != "Product Advice"
    assert mod.compute_baselines(changed)["meta"]["majority_team"] == results["meta"]["majority_team"]


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
    ],
)
def test_keyword_rules(mod, text, team):
    predicted, fired = mod.keyword_predict([text], "fallback")
    assert (predicted[0], bool(fired[0])) == (team, True)


@pytest.mark.parametrize("text", ["i have paid", "settled through emi", "it is in warranty", "please call me"])
def test_mention_alone_fires_nothing(mod, text):
    predicted, fired = mod.keyword_predict([text], "fallback")
    assert (predicted[0], bool(fired[0])) == ("fallback", False)


def test_keyword_rule_teams_are_canonical(mod, config):
    assert {team for team, _ in mod.KEYWORD_RULES} <= set(config.TEAMS)


# --- Other -----------------------------------------------------------------
def test_committed_reports_current(out_dir, reports_dir):
    for name in ("baselines.json", "baselines.md"):
        # Universal newlines: git may check the files out with CRLF.
        fresh = (out_dir / name).read_text(encoding="utf-8")
        stored = (reports_dir / name).read_text(encoding="utf-8")
        assert stored == fresh, f"{name} is stale: run clean, then baselines"


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
