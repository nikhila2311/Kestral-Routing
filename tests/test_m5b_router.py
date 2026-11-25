"""M5b: router. All example strings here are invented, not customer text.

Assertions on real rows compare counts or row numbers, so a failing test does not print customer text.
"""
import copy
import hashlib
import json
import re
import time
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

KEYS = {"team", "confidence", "top3", "needs_triage", "reasons", "clarifying_question", "model_version"}
META_KEYS = {
    "model_version",
    "training_window",
    "threshold",
    "cost_inputs",
    "sklearn_version",
    "joblib_version",
    "numpy_version",
    "config",
}
PHRASES = {  # written again from the spec
    "Repairs": "something has stopped working or is faulty",
    "Installs & Demo": "you're waiting for an installation or demo",
    "Returns & Replacement": "you want to return or replace it",
    "Billing": "there's a problem with a payment, invoice or refund",
    "Warranty Claims": "it's about warranty or Shield cover",
    "Filters & Consumables": "you need filters or spare parts",
    "Product Advice": "you have a question about using it",
}
AGENT_PHRASES = {  # written again: what an agent reads
    "Repairs": "something has stopped working or is faulty",
    "Installs & Demo": "the customer is waiting for an installation or demo",
    "Returns & Replacement": "the customer wants to return or replace the product",
    "Warranty Claims": "it is about warranty or Shield cover",
    "Filters & Consumables": "the customer needs filters or spare parts",
    "Product Advice": "the customer has a question about using the product",
}
OTHER_TEAM = "mentions a payment, but the problem {} {}, not the payment itself (policy §3)"
# Team and triage of 500 validation rows, hashed before the wording of reasons and questions was changed.
FROZEN_DIGEST = "fa94399908f03d4f75ed108494cdd577c24d67b004a47a15a231d647c11eb4f0"
FROZEN_TRIAGED, FROZEN_MEAN_CONFIDENCE = 91, 0.8318091382907701
NEVER_ALONE = {"not", "no", "never", "cannot", "without", "the", "is", "a", "an", "it", "and", "to", "of", "my", "for"}
TEMPLATES = {  # every sentence a reason may contain outside the quotes, written again from the contract
    *(OTHER_TEAM.format(hedge, phrase) for phrase in AGENT_PHRASES.values() for hedge in ("looks like", "may be that")),
    "the message has no readable text, so this team is only a guess",
    "no single word stands out; the team is the best overall match",
    "mentions a payment, but the problem may not be the payment itself (policy §3)",
    "talks about a payment problem",
    "mentions an installation, demo or technician visit",
    "mentions a fault or breakdown",
    "includes an order number",
    "names a different product from the one on the request",
    "mentions ''",
}
WARNING = "mentions a payment, but the problem may not be the payment itself (policy §3)"
PAID_TEXT = "I paid for installation but technician never came"
QUOTED = re.compile(r"^mentions '([^']*)'$")  # team phrases contain apostrophes
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"(?<![\w.§])\d[\d,]*(?:\.\d+)?%?")


@pytest.fixture(scope="module")
def mod():
    from src import router

    return router


@pytest.fixture(scope="module")
def windows(processed_dir):
    from src.split import time_split

    return time_split(pd.read_parquet(processed_dir / "train_clean.parquet"))


@pytest.fixture(scope="module")
def val(windows):
    return windows[1]


@pytest.fixture(scope="module")
def built(mod, tmp_path_factory):
    """Everything is built into temp folders: tests never touch models/ or reports/."""
    reports, models = tmp_path_factory.mktemp("reports"), tmp_path_factory.mktemp("models")
    mod.build(reports, models)
    return reports, models


@pytest.fixture(scope="module")
def router(mod, built):
    return mod.Router.load(built[1] / "router.joblib")


@pytest.fixture(scope="module")
def results(built):
    return json.loads((built[0] / "router.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(built):
    return (built[0] / "router.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def records(mod, val):
    return mod.to_records(val)


@pytest.fixture(scope="module")
def outputs(router, records):
    return router.predict(records)


@pytest.fixture
def saved(router, tmp_path):
    router.save(tmp_path / "router.joblib")
    return tmp_path / "router.joblib"


def one(router, text, **fields):
    return router.predict([{"request_text": text, **fields}])[0]


def biased_to(router, team):
    """A copy whose model always predicts the team."""
    twin = copy.deepcopy(router)
    classifier = twin.model.named_steps["classifier"]
    classifier.intercept_[list(classifier.classes_).index(team)] += 100
    return twin


# --- Criterion 1: interface ---------------------------------------------------
def test_output_schema(outputs, records, config):
    assert len(outputs) == len(records)
    for out in outputs:
        assert set(out) == KEYS
        assert type(out["team"]) is str and out["team"] in config.TEAMS
        assert type(out["confidence"]) is float and type(out["needs_triage"]) is bool
        assert type(out["reasons"]) is list and all(type(r) is str for r in out["reasons"])
        assert out["clarifying_question"] is None or type(out["clarifying_question"]) is str
        assert type(out["model_version"]) is str and out["model_version"]
        top = out["top3"]
        assert len(top) == 3 and all(set(t) == {"team", "prob"} for t in top)
        assert len({t["team"] for t in top}) == 3 and all(t["team"] in config.TEAMS for t in top)
        probs = [t["prob"] for t in top]
        assert all(type(p) is float and 0 <= p <= 1 for p in probs) and probs == sorted(probs, reverse=True)
        assert out["confidence"] == probs[0] and out["team"] == top[0]["team"]
    json.dumps(outputs)


def test_empty_batch_and_extra_keys(router):
    assert router.predict([]) == []
    plain = one(router, "the fan is noisy", channel="chat", product_family="Ceiling Fan", warranty_status="shield")
    extra = one(
        router,
        "the fan is noisy",
        channel="chat",
        product_family="Ceiling Fan",
        warranty_status="shield",
        team_label="Billing",
        request_id=7,
        anything=None,
    )
    assert plain == extra


# --- Criterion 2: no drift from the model -------------------------------------
def test_team_matches_pipeline_argmax_500_val_rows(router, val, records):
    sample = val.sample(500, random_state=42)
    out = router.predict([records[i] for i in val.index.get_indexer(sample.index)])
    proba = router.model.predict_proba(sample[["text_clean", "channel", "product_family", "warranty_status"]])
    expected = router.model.classes_[proba.argmax(axis=1)]
    assert sum(o["team"] != e for o, e in zip(out, expected)) == 0
    assert np.abs(np.array([o["confidence"] for o in out]) - proba.max(axis=1)).max() < 1e-12


def test_model_is_the_chosen_config_on_the_train_window(router, windows, val, records, outputs, reports_dir):
    from src.experiments import fit_predict

    cfg = json.loads((reports_dir / "experiments.json").read_text(encoding="utf-8"))["chosen"]["config"]
    assert router.meta["config"] == cfg
    teams, _ = fit_predict(cfg, windows[0], val)
    assert sum(o["team"] != t for o, t in zip(outputs, teams)) == 0


def test_other_model_types_are_rejected(mod, windows):
    cfg = {"name": "x", "calibration": None, "feature_groups": [], "features": [], "params": {}, "text": {}}
    for change in ({"model_type": "complement_nb"}, {"model_type": "logreg", "calibration": {"method": "sigmoid"}}):
        with pytest.raises(ValueError, match="plain logreg"):
            mod.Router.train(windows[0], {**cfg, **change})


# --- Criterion 3: the triage threshold -----------------------------------------
def test_threshold_from_costs(config, router, windows):
    missed = windows[0][windows[0]["misrouted"]]
    assert (len(missed), int(missed["transfers"].sum())) == (1973, 2843)
    cost = 305 * missed["transfers"].sum() / len(missed) + 260
    assert config.COST_PER_MISROUTE_TRAIN == pytest.approx(cost, abs=1e-9)
    assert config.COST_CLARIFY == 260
    assert config.TRIAGE_THRESHOLD == pytest.approx(1 - 260 / cost, abs=1e-12)
    assert router.meta["threshold"] == config.TRIAGE_THRESHOLD
    assert router.meta["cost_inputs"] == {
        "cost_per_transfer": 305,
        "cost_per_misroute_contact": 260,
        "cost_per_misroute_train": config.COST_PER_MISROUTE_TRAIN,
        "cost_clarify": 260,
    }


def test_triage_formula_row_by_row(outputs, config, val):
    assert val["request_text"].str.contains(r"\w").all()  # no empty text here, so the formula alone decides
    flags = [o["needs_triage"] for o in outputs]
    by_cost = [(1 - o["confidence"]) * config.COST_PER_MISROUTE_TRAIN > 260 for o in outputs]
    by_threshold = [o["confidence"] < config.TRIAGE_THRESHOLD for o in outputs]
    assert flags == by_cost == by_threshold
    assert 0 < sum(flags) < len(flags)


def test_triage_never_changes_the_team(router, records, outputs):
    always = copy.deepcopy(router)
    always.meta["cost_inputs"]["cost_clarify"] = -1.0
    out = always.predict(records)
    assert all(o["needs_triage"] for o in out)
    assert [o["team"] for o in out] == [o["team"] for o in outputs]


# --- Criterion 4: awkward input --------------------------------------------------
@pytest.mark.parametrize("text", ["", "   ", "\n\t ", " ", "?!...", "\U0001f600"])
def test_empty_text_triaged(router, config, text):
    for r in (router, biased_to(router, "Repairs")):  # also when the model is certain
        out = one(r, text)
        assert out["needs_triage"] is True
        assert out["team"] in config.TEAMS and out["reasons"] and out["clarifying_question"]


def test_empty_text_is_triaged_beside_confident_rows(router):
    out = biased_to(router, "Repairs").predict([{"request_text": "the fan is noisy"}, {"request_text": " "}])
    assert [o["needs_triage"] for o in out] == [False, True]


@pytest.mark.parametrize("value", ["", "fax", "Water Purifier ", "\ud800", None, 5, float("nan"), ["chat"]])
@pytest.mark.parametrize("field", ["channel", "product_family", "warranty_status"])
def test_missing_or_unknown_fields(router, config, field, value):
    assert one(router, "the filter needs changing")["team"] in config.TEAMS
    assert one(router, "the filter needs changing", **{field: value})["team"] in config.TEAMS


def test_unknown_field_equals_missing_field(router):
    missing = one(router, "the filter needs changing")
    for field in ("channel", "product_family", "warranty_status"):
        assert one(router, "the filter needs changing", **{field: "never seen"}) == missing


def test_unknown_product_gives_no_conflict_reason(router):
    conflict = "names a different product from the one on the request"
    assert conflict in one(router, "the fan is noisy", product_family="Air Fryer")["reasons"]
    for product in ("Ceiling Fan", "Toaster", None):
        assert conflict not in one(router, "the fan is noisy", product_family=product)["reasons"]


@pytest.mark.parametrize("text", ["\ud800 broken", "a\x00b leak", "गीज़र काम नहीं कर रहा", "noise " * 5000, "[(*+?{\\"])
def test_odd_text_does_not_crash(router, config, text):
    assert one(router, text)["team"] in config.TEAMS


@pytest.mark.parametrize("record", [{}, {"request_text": None}, {"request_text": 5}])
def test_request_text_is_required(router, record):
    with pytest.raises(ValueError, match="record 1: request_text is required"):
        router.predict([{"request_text": "fine"}, record])


def test_errors_do_not_echo_text(router):
    with pytest.raises(ValueError) as error:
        router.predict([{"request_text": ["secret words"]}])
    assert "secret" not in str(error.value)


# --- Criterion 5: save and load ----------------------------------------------------
def test_save_load_round_trip(mod, saved, records, outputs, router):
    loaded = mod.Router.load(saved)
    assert loaded.predict(records[:500]) == router.predict(records[:500])
    assert loaded.meta == router.meta


def test_trained_router_equals_loaded_one(mod, router, windows, records, outputs):
    trained = mod.Router.train(windows[0], router.meta["config"])
    assert trained.meta == router.meta
    assert sum(a != b for a, b in zip(trained.predict(records), outputs)) == 0


def test_meta_fields(built, router, config):
    meta = json.loads((built[1] / "router_meta.json").read_text(encoding="utf-8"))
    assert set(meta) == META_KEYS
    assert meta["training_window"] == {"first_day": "2025-04-01", "last_day": "2026-03-31", "rows": 8687}
    assert meta["sklearn_version"] == version("scikit-learn")
    assert meta["joblib_version"] == version("joblib")
    assert meta["numpy_version"] == version("numpy")
    assert meta["model_version"].startswith("logreg_c1_none-")
    assert meta["model_version"] == router.predict([{"request_text": "hello"}])[0]["model_version"]


def test_model_version_follows_the_training_rows(mod, windows, router):
    fewer = mod.Router.train(windows[0].head(2000), router.meta["config"])
    assert fewer.meta["model_version"] != router.meta["model_version"]


@pytest.mark.parametrize(
    "key, library", [("sklearn_version", "scikit-learn"), ("joblib_version", "joblib"), ("numpy_version", "numpy")]
)
def test_version_mismatch_message(mod, saved, monkeypatch, key, library):
    path = saved.with_name("router_meta.json")
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), key: "0.0.1"}), encoding="utf-8")

    def unpickled(*args, **kwargs):
        raise AssertionError("the model was unpickled before the version check")

    monkeypatch.setattr(mod.joblib, "load", unpickled)
    with pytest.raises(RuntimeError) as error:
        mod.Router.load(saved)
    assert str(error.value) == (
        f"Model was built with {library} 0.0.1; installed is {version(library)}. "
        "Run the train step (see README) to rebuild."
    )


def test_model_and_meta_must_match(mod, saved):
    path = saved.with_name("router_meta.json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, "model_version": "other"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different builds"):
        mod.Router.load(saved)


def test_requirements_pin_installed_versions(root_dir):
    lines = (root_dir / "requirements.txt").read_text(encoding="utf-8").split()
    for library in ("scikit-learn", "joblib", "numpy"):
        assert f"{library}=={version(library)}" in lines


# --- Criterion 6: reasons ------------------------------------------------------------
def test_reasons_quote_only_own_words(outputs, val, config):
    close = {f"close call between {a} and {b}" for a in config.TEAMS for b in config.TEAMS if a != b}
    bad = []
    for i, (out, text) in enumerate(zip(outputs, val["text_clean"])):
        words = set(re.findall(r"\w+", text))
        quotes = [q for reason in out["reasons"] for q in QUOTED.findall(reason)]
        templates = {QUOTED.sub("mentions ''", reason) for reason in out["reasons"]}
        ok = (
            bool(out["reasons"])
            and templates <= TEMPLATES | close  # nothing but fixed wording outside the quotes
            and all(q and q in text and set(q.split()) <= words for q in quotes)
            and len(quotes) <= 3
            and not NEVER_ALONE & set(quotes)
            and len({w for q in quotes for w in q.split()}) == sum(len(set(q.split())) for q in quotes)  # no word in two phrases
        )
        if not ok:
            bad.append(i)
    assert bad == []


def test_phrases_are_the_strongest_pushes(mod, router, outputs, val):
    """The first phrase has positive weight for the team, and no usable word of the text has a larger push."""
    vectoriser = router.model.named_steps["features"].named_transformers_["word"]
    classifier = router.model.named_steps["classifier"]
    names, where = vectoriser.get_feature_names_out(), router.model.named_steps["features"].output_indices_["word"]
    matrix, checked = vectoriser.transform(val["text_clean"].head(200)).tocsr(), 0
    for i, out in enumerate(outputs[:200]):
        coef = classifier.coef_[list(classifier.classes_).index(out["team"]), where]
        pushes = {names[c]: v * coef[c] for c, v in zip(matrix[i].indices, matrix[i].data)}
        # The phrases as chosen; the reasons show them with neighbours merged into one span.
        text = val["text_clean"].iloc[i]
        shown = router._phrases(matrix[i], text, out["team"])
        assert [q for r in out["reasons"] for q in QUOTED.findall(r)] == mod._merge_phrases(shown, text)
        assert sum(pushes[q] <= 0 for q in shown) == 0
        if len(shown) == 3:
            hidden = [w for w, p in pushes.items() if router._word_usable[list(names).index(w)] and " " not in w
                      and not any(w in q.split() for q in shown) and p > max(pushes[q] for q in shown)]
            assert len(hidden) == 0
            checked += 1
    assert checked > 0


def test_longer_phrase_replaces_its_word(router):
    assert "mentions 'not working'" in one(router, "the mixer is not working")["reasons"]


@pytest.mark.parametrize(
    "scores, expected",
    [
        ({"noise": 1.0, "loud noise": 0.66}, ["loud noise"]),  # at least 65%: the longer phrase
        ({"noise": 1.0, "loud noise": 0.64}, ["noise"]),  # less: the word, and the longer phrase is dropped
        ({"loud noise": 1.0, "noise": 0.9, "loud": 0.8}, ["loud noise"]),  # parts of a chosen phrase are dropped
        ({"noise": 1.0, "not noise": 0.1}, ["not noise"]),  # a negation always wins
        ({"not": 1.0, "no": 0.9, "the": 0.8, "noise": 0.1}, ["noise"]),  # never alone
        ({"noise": 1.0, "loud noise": -0.1}, ["noise"]),  # pushes away from the team
        ({"fan": 1.0, "noise": 0.9, "loud": 0.8, "heater": 0.7}, ["fan", "noise", "loud"]),  # at most three
    ],
)
def test_phrase_choice(mod, scores, expected):
    words = list(scores)
    stub = SimpleNamespace(
        model=SimpleNamespace(named_steps={"classifier": SimpleNamespace(coef_=np.array([list(scores.values())]))}),
        _class_row={"Repairs": 0},
        _word_slice=slice(None),
        _words=words,
        _word_usable=[mod._usable(word) for word in words],
    )
    row = SimpleNamespace(data=np.ones(len(words)), indices=np.arange(len(words)))
    assert mod.Router._phrases(stub, row, ", ".join(words), "Repairs") == expected


@pytest.mark.parametrize(
    "phrases, text, expected",
    [
        (["working", "smell", "grinder stopped"], "my grinder stopped working, odd smell", ["grinder stopped working", "smell"]),
        (["smell", "working", "grinder stopped"], "my grinder stopped working, odd smell", ["smell", "grinder stopped working"]),
        (["fan", "loud", "noise"], "loud noise fan", ["loud noise fan"]),  # in the order of the text
        (["fan", "noise"], "noise, fan", ["fan", "noise"]),  # punctuation between: not adjacent
        (["air", "fryer"], "air-fryer", ["air", "fryer"]),  # quoted words stay whole words
        (["fan", "noise"], "fan makes noise", ["fan", "noise"]),  # an unchosen word between
        (["fan", "noise"], "noise from the kitchen. the fan noise", ["fan noise"]),  # any adjacent place counts
        (["fan", "loud", "noise"], "fan noise. loud noise", ["fan noise", "loud"]),  # no word in two spans
        (["fan loud", "loud noise"], "fan loud noise", ["fan loud noise"]),  # overlapping
        (["fan"], "fan fan", ["fan"]),  # a repeated word is not two phrases
        (["fan", "loud"], "fans loud fan", ["loud fan"]),  # whole words only
        ([], "fan", []),
    ],
)
def test_adjacent_phrases_are_merged(mod, phrases, text, expected):
    merged = mod._merge_phrases(phrases, text)
    assert merged == expected
    assert all(span in text for span in merged) and len(merged) <= len(phrases)


def test_grinder_stopped_working(router, monkeypatch):
    """The phrases the final model chooses; the train-window model picks 'stopped working'. M8 tests the final model."""
    monkeypatch.setattr(router, "_phrases", lambda *args: ["working", "smell", "grinder stopped"])
    reasons = one(router, "My mixer grinder stopped working and makes a burning smell")["reasons"]
    quotes = [q for r in reasons for q in QUOTED.findall(r)]
    assert quotes == ["grinder stopped working", "smell"]


def test_no_adjacent_quotes_are_left(outputs, val):
    """Two quotes of one request are never next to each other in its text; some spans were merged."""
    left = 0
    for out, text in zip(outputs, val["text_clean"]):
        quotes = [q for r in out["reasons"] for q in QUOTED.findall(r)]
        for a in quotes:
            for b in quotes:
                found = a != b and re.search(rf"(?<![\w'’]){re.escape(a)} {re.escape(b)}(?![\w'’])", text)
                left += bool(found)
    assert left == 0
    assert any(len(q.split()) > 2 for o in outputs for r in o["reasons"] for q in QUOTED.findall(r))


def test_no_phrase_reason(router):
    assert "no single word stands out; the team is the best overall match" in one(router, "zzqx vvkw")["reasons"]


@pytest.mark.parametrize(
    "text, sentence",
    [
        ("i was charged twice", "talks about a payment problem"),
        ("please send a technician", "mentions an installation, demo or technician visit"),
        ("the heater is leaking", "mentions a fault or breakdown"),
        ("my order is ko1234567", "includes an order number"),
    ],
)
def test_cue_sentences(router, text, sentence):
    assert sentence in one(router, text)["reasons"]
    assert sentence not in one(router, "which model do you suggest")["reasons"]


def test_close_call(router, outputs):
    for out in outputs:
        first, second = out["top3"][:2]
        sentence = f"close call between {first['team']} and {second['team']}"
        assert (sentence in out["reasons"]) == (first["prob"] - second["prob"] <= 0.10)
    assert any("close call" in r for o in outputs for r in o["reasons"])


# --- Criterion 7: policy section 3 -----------------------------------------------------
def test_policy3_warning(router):
    natural = one(router, PAID_TEXT)
    assert (WARNING in natural["reasons"]) == (natural["team"] == "Billing")
    billing = one(biased_to(router, "Billing"), PAID_TEXT)
    assert billing["team"] == "Billing" and WARNING in billing["reasons"]
    assert "mentions a payment" not in billing["reasons"]  # the warning says it
    problem = one(biased_to(router, "Billing"), "I paid and was charged twice")
    assert problem["team"] == "Billing" and WARNING not in problem["reasons"]
    other = one(biased_to(router, "Repairs"), PAID_TEXT)
    assert other["team"] == "Repairs" and WARNING not in other["reasons"]
    assert OTHER_TEAM.format("looks like", AGENT_PHRASES["Repairs"]) in other["reasons"]
    for team in PHRASES:
        reasons = one(biased_to(router, team), PAID_TEXT)["reasons"]
        expected = None if team == "Billing" else OTHER_TEAM.format("looks like", AGENT_PHRASES[team])
        assert [r for r in reasons if "looks like" in r or "may be that" in r] == ([expected] if expected else [])
        for text in ("I was charged twice", "the fan is noisy"):  # a payment problem, no payment
            assert not any("policy" in r for r in one(biased_to(router, team), text)["reasons"])


def test_policy3_warning_on_validation(outputs, val):
    from src.experiments import cue_features

    mention = cue_features(val)["payment_only_mentioned"].to_numpy() == 1
    expected = [o["team"] == "Billing" and bool(m) for o, m in zip(outputs, mention)]
    assert [WARNING in o["reasons"] for o in outputs] == expected
    assert any(expected)
    # Firm when sent straight on, hedged when triaged; compared as counts of rows that differ.
    hedges, bad = {"looks like": 0, "may be that": 0}, 0
    for o, m in zip(outputs, mention):
        shown = [r for r in o["reasons"] if "looks like" in r or "may be that" in r]
        wanted = []
        if o["team"] != "Billing" and m:
            hedge = "may be that" if o["needs_triage"] else "looks like"
            hedges[hedge] += 1
            wanted = [OTHER_TEAM.format(hedge, AGENT_PHRASES[o["team"]])]
        bad += shown != wanted
    assert bad == 0
    assert all(hedges.values())  # both wordings occurred


def test_reasons_never_address_the_customer(outputs, router):
    sentences = [QUOTED.sub("", r) for o in outputs for r in o["reasons"]]
    sentences += [r for team in PHRASES for r in one(biased_to(router, team), PAID_TEXT)["reasons"]]
    assert sum(bool(re.search(r"\byou(?:r|'re)?\b", s)) for s in sentences) == 0


def test_policy3_cases_reported(results, router, mod):
    cases = results["policy3_cases"]
    assert len(cases) == 5 and cases[0]["text"] == PAID_TEXT
    for case in cases:
        out = one(router, case["text"])
        assert (case["team"], case["needs_triage"]) == (out["team"], out["needs_triage"])
        assert case["confidence"] == pytest.approx(out["confidence"], abs=1e-6)
        assert case["warning"] == (WARNING in out["reasons"])


# --- Criterion 8: clarifying question -----------------------------------------------------
def test_question_iff_triage(outputs):
    assert [bool(o["clarifying_question"]) for o in outputs] == [o["needs_triage"] for o in outputs]


def join(parts, last):
    return ", ".join(parts[:-1]) + last + parts[-1] if len(parts) > 1 else parts[0]


def test_question_builder_on_validation(router, outputs, val, mod):
    from src.audit import vague_mask

    proba = router.model.predict_proba(val[["text_clean", "channel", "product_family", "warranty_status"]])
    vague, general, bad = vague_mask(val["text_clean"]).tolist(), 0, []
    for i, out in enumerate(outputs):
        if not out["needs_triage"]:
            continue
        ranked = sorted(zip(proba[i], router.model.classes_), key=lambda x: -x[0])
        if vague[i] or ranked[0][0] < 0.40:
            general += 1
            product = mod._product_name(val["text_clean"].iloc[i], val["product_family"].iloc[i])
            about = f" about your {product}" if product else ""
            options = join([PHRASES[team] for _, team in ranked[:4]], ", or ")
            expected = f"So we can send you to the right team{about}: is this because {options}?"
        else:
            options = join([PHRASES[team] for p, team in ranked if p >= 0.10][:3], " or ")
            expected = f"Can you confirm whether {options}?"
        if out["clarifying_question"] != expected:
            bad.append(i)
    assert bad == []
    assert 0 < general < sum(o["needs_triage"] for o in outputs)  # both kinds of question were checked


def test_question_wording(mod, config):
    assert mod.TEAM_PHRASES == PHRASES and set(PHRASES) == set(config.TEAMS)

    def ask(vague=False, product=None, **shares):
        return mod._clarifying_question(
            np.array([shares.get(team.split()[0], 0.0) for team in config.TEAMS]), vague, product
        )

    assert ask(Repairs=0.5, Installs=0.4, Billing=0.1) == (
        "Can you confirm whether something has stopped working or is faulty, you're waiting for an installation or demo or "
        "there's a problem with a payment, invoice or refund?"
    )
    assert ask(Repairs=0.6, Installs=0.31, Billing=0.09).count("you're waiting") == 1
    assert "payment" not in ask(Repairs=0.6, Installs=0.31, Billing=0.09)
    assert ask(Repairs=0.45, Installs=0.25, Billing=0.16, Warranty=0.14).count("warranty") == 0  # at most three
    assert ask(Repairs=0.6) == "Can you confirm whether something has stopped working or is faulty?"
    general = ask(Repairs=0.39, Installs=0.3, Billing=0.2, Warranty=0.11, product="purifier")
    assert general == (
        "So we can send you to the right team about your purifier: is this because something has stopped working or is "
        "faulty, you're waiting for an installation or demo, there's a problem with a payment, invoice or refund, or it's "
        "about warranty or Shield cover?"
    )
    assert ask(Repairs=0.4, Installs=0.3).startswith("Can you confirm")  # the limit itself is not below it
    assert ask(vague=True, Repairs=0.6, Installs=0.4).startswith("So we can send you to the right team: is this because")


@pytest.mark.parametrize(
    "text, family, expected",
    [
        ("please call me about my purifier", "", "purifier"),
        ("please call me", "Robot Vacuum", "vacuum"),
        ("my fan is here", "Air Fryer", "fan"),  # the customer's own words come first
        ("the fan and the heater", "Ceiling Fan", None),
        ("please call me", "Toaster", None),
        ("", "", None),
    ],
)
def test_product_name(mod, text, family, expected):
    assert mod._product_name(text, family) == expected


def test_product_names_cover_every_family(mod):
    from src.audit import PRODUCT_PATTERNS

    assert set(mod.PRODUCT_NAMES) == set(PRODUCT_PATTERNS)


def test_vague_text_gets_the_general_question(router):
    question = one(router, "please call me about my purifier")["clarifying_question"]
    assert question.startswith("So we can send you to the right team about your purifier: is this because ")


# --- The wording change touched nothing else ------------------------------------------------
def test_only_reasons_and_questions_changed(router, val, records, config):
    """Against the outputs frozen before the change, and row by row against the model."""
    sample = val.sample(500, random_state=42)
    out = router.predict([records[i] for i in val.index.get_indexer(sample.index)])
    lines = [f"{rid}|{o['team']}|{o['needs_triage']}" for rid, o in zip(sample["request_id"], out)]
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == FROZEN_DIGEST
    assert sum(o["needs_triage"] for o in out) == FROZEN_TRIAGED
    assert np.mean([o["confidence"] for o in out]) == pytest.approx(FROZEN_MEAN_CONFIDENCE, abs=1e-6)
    proba = router.model.predict_proba(sample[["text_clean", "channel", "product_family", "warranty_status"]])
    ranked = np.argsort(-proba, axis=1, kind="stable")[:, :3]
    top = [[t["team"] for t in o["top3"]] for o in out]
    assert sum(t != router.model.classes_[r].tolist() for t, r in zip(top, ranked)) == 0
    confidence = np.array([o["confidence"] for o in out])
    assert np.abs(confidence - proba.max(axis=1)).max() < 1e-12
    assert [o["needs_triage"] for o in out] == ((1 - confidence) * config.COST_PER_MISROUTE_TRAIN > 260).tolist()


# --- Criterion 9: speed and determinism ------------------------------------------------------
def test_1000_under_5s(router, records):
    start = time.perf_counter()
    assert len(router.predict(records[:1000])) == 1000
    assert time.perf_counter() - start < 5


def test_deterministic(router, records, outputs):
    assert sum(a != b for a, b in zip(router.predict(records), outputs)) == 0


def test_rows_do_not_depend_on_the_batch(router, records, outputs):
    for i in range(0, 200, 10):
        alone = router.predict([records[i]])[0]
        for key in KEYS - {"confidence", "top3"}:
            assert alone[key] == outputs[i][key], f"row {i}, {key}"
        assert alone["confidence"] == pytest.approx(outputs[i]["confidence"], abs=1e-12)


def test_build_is_deterministic(mod, built, tmp_path):
    mod.build(tmp_path, tmp_path)
    for name in ("router.json", "router.md"):
        assert (tmp_path / name).read_bytes() == (built[0] / name).read_bytes()
    assert (tmp_path / "router_meta.json").read_bytes() == (built[1] / "router_meta.json").read_bytes()


# --- Criterion 10: the report -------------------------------------------------------------------
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
    md = DATE.sub(" ", md).replace("(1 - confidence)", " ")
    for text in sorted(strings, key=len, reverse=True):  # versions and messages
        md = md.replace(text, " ") if any(char.isdigit() for char in text) else md
    for token in NUMBER.findall(md):
        text = token.rstrip("%").replace(",", "")
        decimals = len(text.partition(".")[2])
        scale = 100 if token.endswith("%") else 1
        # Shares are only ever shown as percentages, so a bare integer cannot match a rounded share.
        usable = [v for v in values if scale == 100 or decimals or isinstance(v, int) or v > 100]
        if not any(f"{v * scale:.{decimals}f}" == text for v in usable):
            bad.append(token)
    return bad


def test_md_numbers_in_json(md, results, mod):
    assert md == mod.render_markdown(results)
    assert unmatched_numbers(md, results) == []
    assert unmatched_numbers(md + "\n12,345 and 99.87% and 7.5\n", results) == ["12,345", "99.87%", "7.5"]


def test_report_numbers_recomputed(results, outputs, val, config):
    truth = val["final_team_canon"].tolist()
    right = np.array([o["team"] == t for o, t in zip(outputs, truth)])
    triaged = np.array([o["needs_triage"] for o in outputs])
    repairs = np.array([o["team"] == "Repairs" for o in outputs]) & ~triaged
    bot = int((val["team_label_canon"] != val["final_team_canon"]).sum())
    sent_wrong = int((~right & ~triaged).sum())

    def close(value):
        return pytest.approx(value, abs=1e-5)

    assert results["triage"]["rate"] == close(triaged.mean())
    confidence = np.array([o["confidence"] for o in outputs])
    assert results["triage"]["mean_confidence"] == close(confidence[triaged].mean())
    assert results["accuracy"]["sent"] == close(right[~triaged].mean())
    assert results["accuracy"]["triaged"] == close(right[triaged].mean())
    assert results["repairs"]["precision_sent"] == close(right[repairs].mean())
    cost, rate = results["cost_per_month_rs"], config.COST_PER_MISROUTE
    assert cost["router"] == close((sent_wrong * rate + triaged.sum() * 260) / 3)
    assert cost["router_if_clarification_fixes_nothing"] == close(((~right).sum() * rate + triaged.sum() * 260) / 3)
    assert cost["router_without_triage"] == close((~right).sum() * rate / 3)
    assert cost["bot"] == close(498 * rate / 3) and bot == 498
    assert cost["bot_minus_router"] == close(cost["bot"] - cost["router"])
    assert cost["bot_licence"] == close(320_000 / 12)


def test_vague_counts(results, outputs, val):
    from src.audit import vague_mask

    vague = vague_mask(val["text_clean"]).tolist()
    triaged = sum(v and o["needs_triage"] for v, o in zip(vague, outputs))
    assert results["vague"] == {"count": sum(vague), "triaged_count": triaged}
    assert 0 < triaged <= sum(vague)


def test_break_even_clarification_rate(results, md, config):
    triaged, wrong = results["triage"]["count"], results["misroutes"]["triaged"]
    rate = results["break_even_clarification_rate"]
    assert rate == pytest.approx(triaged * 260 / (wrong * config.COST_PER_MISROUTE), abs=1e-6)
    assert results["break_even_clarification_formula"] in md
    # At the break-even rate, triage costs the same as sending everything straight on.
    cost = results["cost_per_month_rs"]
    left = wrong * (1 - rate) * config.COST_PER_MISROUTE / results["validation"]["months"]
    assert cost["router"] + left == pytest.approx(cost["router_without_triage"], abs=0.1)


def test_assumption_is_stated(md):
    assert "Assumption: a clarified request is never misrouted" in md


def test_report_contains_no_customer_text(md, results, val, built):
    blob = md + (built[0] / "router.json").read_text(encoding="utf-8")
    long_texts = val.loc[val["request_text"].str.len() > 25, "request_text"]
    assert sum(text in blob for text in long_texts) == 0
