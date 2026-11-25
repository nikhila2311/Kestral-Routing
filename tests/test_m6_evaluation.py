"""M6: evaluation evidence. Assertions on real rows compare counts, so a failing test does not print customer text."""
import copy
import json
import re
import struct
import zlib

import numpy as np
import pandas as pd
import pytest

FIGURES = {"confusion_matrices.png", "coverage_accuracy.png", "reliability.png", "per_team_recall.png"}
NAMES = ("baselines", "experiments", "router")
DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
NUMBER = re.compile(r"(?<![\w.§])\d[\d,]*(?:\.\d+)?%?")
TOP_KEYS = {
    "meta", "headline", "per_team", "confusion", "coverage_curve", "calibration", "breakdowns", "seen_text", "triage",
    "costs", "failures", "expected_score", "figures", "consistency",
}  # fmt: skip
COST_KEYS = {
    "router_clarification_always_works", "router_without_triage", "router_clarification_fixes_nothing", "bot", "bot_licence",
}  # fmt: skip


@pytest.fixture(scope="module")
def mod():
    from src import evaluation

    return evaluation


@pytest.fixture(scope="module")
def windows(processed_dir):
    from src.split import time_split

    return time_split(pd.read_parquet(processed_dir / "train_clean.parquet"))


@pytest.fixture(scope="module")
def val(windows):
    return windows[1]


@pytest.fixture(scope="module")
def test_set(processed_dir):
    return pd.read_parquet(processed_dir / "test_clean.parquet")


@pytest.fixture(scope="module")
def sources(reports_dir):
    return {n: json.loads((reports_dir / f"{n}.json").read_text(encoding="utf-8")) for n in NAMES}


@pytest.fixture(scope="module")
def built(mod, tmp_path_factory):
    """Built into a temp folder: tests never touch reports/."""
    out = tmp_path_factory.mktemp("reports")
    mod.build(out)
    return out


@pytest.fixture(scope="module")
def results(built):
    return json.loads((built / "evaluation.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md(built):
    return (built / "evaluation.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def router(windows, sources):
    """Trained here, not taken from src.evaluation or models/."""
    from src.router import Router

    return Router.train(windows[0], sources["experiments"]["chosen"]["config"])


@pytest.fixture(scope="module")
def outputs(router, val):
    fields = ["request_text", "channel", "product_family", "warranty_status"]
    return router.predict(val[fields].to_dict("records"))


@pytest.fixture(scope="module")
def arrays(outputs, val):
    """Everything the recomputation tests need, from the Router output and the parquet only."""
    return {
        "truth": val["final_team_canon"].to_numpy(dtype=object),
        "bot": val["team_label_canon"].to_numpy(dtype=object),
        "router": np.array([o["team"] for o in outputs], dtype=object),
        "confidence": np.array([o["confidence"] for o in outputs]),
        "triaged": np.array([o["needs_triage"] for o in outputs]),
    }


def close(value):
    return pytest.approx(value, abs=1e-6)


def precision_recall(truth, guess, team):
    hit = int(((truth == team) & (guess == team)).sum())
    return hit / int((guess == team).sum()), hit / int((truth == team).sum())


def leaves(node):
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        return [leaf for child in node for leaf in leaves(child)]
    return [node]


# --- Criterion 1: figures ------------------------------------------------------
def png_chunks(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, chunks = 8, []
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position : position + 8])
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length : position + 12 + length])
        assert crc == zlib.crc32(kind + body)
        chunks.append((kind, body))
        position += 12 + length
    return chunks


def check_figures(folder):
    assert {p.name for p in folder.iterdir()} == FIGURES
    for name in FIGURES:
        data = (folder / name).read_bytes()
        assert len(data) > 10_000, name
        chunks = png_chunks(data)
        assert chunks[0][0] == b"IHDR" and chunks[-1][0] == b"IEND"
        width, height, depth, colour = struct.unpack(">IIBB", chunks[0][1][:10])
        pixels = zlib.decompress(b"".join(body for kind, body in chunks if kind == b"IDAT"))
        assert depth == 8 and colour == 6 and len(pixels) == height * (1 + 4 * width)
        assert len(set(pixels)) > 50, f"{name} looks blank"


def test_figures_valid_png_over_10kb(built, results):
    check_figures(built / "figures")
    assert set(results["figures"]) == {f"figures/{name}" for name in FIGURES}


def test_committed_figures_valid(reports_dir):
    check_figures(reports_dir / "figures")


def test_figures_linked_in_md(md):
    assert set(re.findall(r"\]\(figures/([^)]+)\)", md)) == FIGURES


def test_figures_follow_the_results(mod, results, built, tmp_path):
    """A changed number must change the picture that shows it."""
    changed = copy.deepcopy(results)
    changed["confusion"]["bot"][0] = [50] * 7
    changed["coverage_curve"]["policy_point"]["accuracy_on_covered"] = 0.9
    changed["calibration"]["bins"][0]["accuracy"] = 0.5
    changed["per_team"]["Billing"]["bot"]["recall"] = 0.1
    mod.plot_figures(changed, tmp_path)
    for name in FIGURES:
        assert (tmp_path / name).read_bytes() != (built / "figures" / name).read_bytes(), name


# --- Criterion 2: recomputed independently -------------------------------------
def test_structure(results):
    assert set(results) == TOP_KEYS


def test_headline_recomputed(results, arrays, config):
    truth = arrays["truth"]
    assert results["headline"]["rows"] == len(truth) == 2135
    for side in ("router", "bot"):
        guess, stored = arrays[side], results["headline"][side]
        f1 = []
        for team in config.TEAMS:
            p, r = precision_recall(truth, guess, team)
            f1.append(2 * p * r / (p + r))
        assert stored["accuracy"] == close((truth == guess).mean())
        assert stored["macro_f1"] == close(np.mean(f1))
        assert stored["misroutes"] == int((truth != guess).sum())


def test_intervals_recomputed(results, arrays):
    """Percentile bootstrap written again: 1,000 resamples of the rows, seed 42, the same rows for both sides."""
    right = {side: (arrays[side] == arrays["truth"]).astype(float) for side in ("router", "bot")}
    agree = (arrays["router"] == arrays["bot"]).astype(float)
    n, rng = len(agree), np.random.default_rng(42)
    samples = {"router": [], "bot": [], "gap": [], "agree": []}
    for _ in range(1000):
        rows = rng.integers(0, n, n)
        a, b = right["router"][rows].mean(), right["bot"][rows].mean()
        for key, value in (("router", a), ("bot", b), ("gap", 100 * (a - b)), ("agree", agree[rows].mean())):
            samples[key].append(value)
    stored = {
        "router": results["headline"]["router"]["accuracy_ci95"],
        "bot": results["headline"]["bot"]["accuracy_ci95"],
        "gap": results["headline"]["paired_accuracy_points_router_minus_bot"],
        "agree": results["expected_score"]["bot_label"]["ci95"],
    }
    for key, values in samples.items():
        low, high = np.percentile(values, [2.5, 97.5])
        assert stored[key]["low"] == close(low) and stored[key]["high"] == close(high), key


def test_per_team_recomputed(results, arrays, config):
    assert results["meta"]["teams"] == list(config.TEAMS) and set(results["per_team"]) == set(config.TEAMS)
    for team, stored in results["per_team"].items():
        assert stored["support"] == int((arrays["truth"] == team).sum())
        for side in ("router", "bot"):
            p, r = precision_recall(arrays["truth"], arrays[side], team)
            assert stored[side] == {"precision": close(p), "recall": close(r)}, (team, side)


def test_confusion_recomputed(results, arrays, config):
    for side in ("router", "bot"):
        for i, actual in enumerate(config.TEAMS):
            for j, sent in enumerate(config.TEAMS):
                expected = int(((arrays["truth"] == actual) & (arrays[side] == sent)).sum())
                assert results["confusion"][side][i][j] == expected


def test_breakdowns_recomputed(results, arrays, windows, val):
    seen = val["text_clean"].isin(set(windows[0]["text_clean"])).map({True: "seen", False: "unseen"})
    columns = {"channel": val["channel"], "month": val["created_month"], "warranty_status": val["warranty_status"]}
    columns["text"] = seen
    assert set(results["breakdowns"]) == set(columns)
    right, bot_right = arrays["router"] == arrays["truth"], arrays["bot"] == arrays["truth"]
    for name, column in columns.items():
        rows = results["breakdowns"][name]
        assert {r["group"] for r in rows} == set(column) and len(rows) == column.nunique()
        assert sum(r["rows"] for r in rows) == 2135
        for row in rows:
            mask = (column == row["group"]).to_numpy()
            assert row["rows"] == int(mask.sum())
            assert row["router_accuracy"] == close(right[mask].mean())
            assert row["bot_accuracy"] == close(bot_right[mask].mean())
            assert row["lead_points"] == close(100 * (right[mask].mean() - bot_right[mask].mean()))
            assert row["triage_rate"] == close(arrays["triaged"][mask].mean())


def test_triage_recomputed(results, arrays, val, config):
    from src.audit import vague_mask

    t, right, triaged = results["triage"], arrays["router"] == arrays["truth"], arrays["triaged"]
    vague = vague_mask(val["text_clean"]).to_numpy()
    assert (triaged == (arrays["confidence"] < config.TRIAGE_THRESHOLD)).all()
    assert (t["count"], t["sent_count"]) == (int(triaged.sum()), int((~triaged).sum()))
    assert t["rate"] == close(triaged.mean())
    assert t["accuracy_sent"] == close(right[~triaged].mean()) and t["accuracy_triaged"] == close(right[triaged].mean())
    assert t["mean_confidence_triaged"] == close(arrays["confidence"][triaged].mean())
    assert (t["misroutes_sent"], t["misroutes_triaged"]) == (int((~right[~triaged]).sum()), int((~right[triaged]).sum()))
    assert (t["vague_count"], t["vague_triaged_count"]) == (int(vague.sum()), int((vague & triaged).sum()))
    assert t["vague_triaged_share"] == close(triaged[vague].mean())


def test_costs_recomputed(results, arrays):
    c, right = results["costs"], arrays["router"] == arrays["truth"]
    triaged, each = arrays["triaged"], 305 * 3549 / 2471 + 260
    month = c["per_month_rs"]
    asked = triaged.sum() * 260
    assert month["router_clarification_always_works"] == close(((~right & ~triaged).sum() * each + asked) / 3)
    assert month["router_without_triage"] == close((~right).sum() * each / 3)
    assert month["router_clarification_fixes_nothing"] == close(((~right).sum() * each + asked) / 3)
    assert month["bot"] == close((arrays["bot"] != arrays["truth"]).sum() * each / 3)
    assert month["bot_licence"] == close(320_000 / 12)
    assert c["break_even_clarification_rate"] == close(asked / ((~right & triaged).sum() * each))
    worst_gap = month["bot"] - month["router_clarification_fixes_nothing"]
    assert c["bot_minus_router_clarification_fixes_nothing_rs"] == pytest.approx(worst_gap, abs=2e-6)
    best_gap = month["bot"] - month["router_clarification_always_works"]
    assert c["bot_minus_router_rs"] == pytest.approx(best_gap, abs=2e-6)
    assert set(month) == COST_KEYS  # the licence is a line of its own: no combined total


def test_curve_and_bins_recomputed(results, arrays, config):
    right, confidence = arrays["router"] == arrays["truth"], arrays["confidence"]
    policy = results["coverage_curve"]["policy_point"]
    assert policy["threshold"] == close(config.TRIAGE_THRESHOLD)
    thresholds = [(p, p["threshold"]) for p in results["coverage_curve"]["points"]] + [(policy, config.TRIAGE_THRESHOLD)]
    for point, threshold in thresholds:
        covered = confidence >= threshold
        assert point["coverage"] == close(covered.mean())
        assert point["accuracy_on_covered"] == (close(right[covered].mean()) if covered.any() else None)
    assert policy["coverage"] == close(1 - results["triage"]["rate"])
    assert policy["accuracy_on_covered"] == close(results["triage"]["accuracy_sent"])
    cal, gap = results["calibration"], 0.0
    assert sum(b["rows"] for b in cal["bins"]) == 2135
    for b in cal["bins"]:
        mask = (confidence >= b["from"]) & ((confidence < b["to"]) | (b["to"] == 1.0))
        assert b["rows"] == int(mask.sum())
        assert b["mean_confidence"] == close(confidence[mask].mean()) and b["accuracy"] == close(right[mask].mean())
        gap += mask.mean() * abs(right[mask].mean() - confidence[mask].mean())
    assert cal["expected_calibration_error"] == close(gap)


def test_calibration_bins_edges(mod):
    out = mod.calibration(np.array([True, False, True]), np.array([0.0, 0.1, 1.0]))
    assert [(b["from"], b["rows"]) for b in out["bins"]] == [(0.0, 1), (0.1, 1), (0.9, 1)]


def test_failures_recomputed(results, arrays, outputs, val):
    from src.audit import product_conflict_mask, vague_mask

    wrong, triaged = arrays["router"] != arrays["truth"], arrays["triaged"]
    masks = {
        "vague": vague_mask(val["text_clean"]).to_numpy(),
        "close_call": np.array([o["top3"][0]["prob"] - o["top3"][1]["prob"] <= 0.10 for o in outputs]),
        "product_conflict": product_conflict_mask(val).to_numpy(),
        "ivr": (val["channel"] == "ivr").to_numpy(),
    }
    # The router's own wording must agree with the close-call rule.
    said = np.array([any(r.startswith("close call between") for r in o["reasons"]) for o in outputs])
    assert int((said != masks["close_call"]).sum()) == 0
    masks["other"] = ~(masks["vague"] | masks["close_call"] | masks["product_conflict"] | masks["ivr"])
    f = results["failures"]
    assert [row["category"] for row in f["categories"]] == list(masks)
    taken = np.zeros(len(wrong), dtype=bool)
    for row in f["categories"]:
        mask = masks[row["category"]]
        assert (row["rows"], row["errors"]) == (int(mask.sum()), int((mask & wrong).sum()))
        assert row["error_rate"] == close(wrong[mask].mean())
        assert row["errors_sent"] == int((mask & wrong & ~triaged).sum())
        assert row["exclusive_errors"] == int((mask & wrong & ~taken).sum())
        taken |= mask
    assert sum(row["exclusive_errors"] for row in f["categories"]) == f["errors"] == int(wrong.sum())
    listed = np.array([t in [x["team"] for x in o["top3"]] for o, t in zip(outputs, arrays["truth"])])
    assert f["errors_with_right_team_in_top3"] == int((wrong & listed).sum())


# --- Criterion 3: consistency --------------------------------------------------
def test_consistency_with_m4_m5a_m5b(results, sources):
    """Written out by hand, independent of the module's own list."""
    b, e, r = sources["baselines"]["baselines"], sources["experiments"], sources["router"]
    head, triage, month = results["headline"], results["triage"], results["costs"]["per_month_rs"]
    real, rs = e["real_val"][e["chosen"]["name"]], r["cost_per_month_rs"]
    final, bot = b["tfidf_final"], b["bot"]
    equal = [
        (head["router"]["accuracy"], final["evaluate"]["accuracy"], real["accuracy"], r["accuracy"]["all"]),
        (head["bot"]["accuracy"], bot["evaluate"]["accuracy"], e["bot_real_val_accuracy"], r["accuracy"]["bot"]),
        (head["router"]["macro_f1"], final["evaluate"]["macro_f1"], real["macro_f1"]),
        (head["bot"]["macro_f1"], bot["evaluate"]["macro_f1"]),
        (triage["rate"], r["triage"]["rate"]),
        (triage["count"], r["triage"]["count"]),
        (triage["accuracy_sent"], r["accuracy"]["sent"]),
        (triage["accuracy_triaged"], r["accuracy"]["triaged"]),
        (month["router_clarification_always_works"], rs["router"]),
        (month["router_without_triage"], rs["router_without_triage"], final["cost_rs_per_month"]),
        (month["router_clarification_fixes_nothing"], rs["router_if_clarification_fixes_nothing"]),
        (month["bot"], rs["bot"], bot["cost_rs_per_month"]),
        (month["bot_licence"], rs["bot_licence"], sources["baselines"]["meta"]["bot_licence_rs_per_month"]),
        (results["costs"]["bot_minus_router_rs"], rs["bot_minus_router"]),
        (results["costs"]["break_even_clarification_rate"], r["break_even_clarification_rate"]),
        (results["expected_score"]["bot_label"]["point"], final["agreement_with_team_label"]),
    ]
    for end in ("low", "point", "high"):
        ours = head["paired_accuracy_points_router_minus_bot"][end]
        theirs = sources["baselines"]["paired_accuracy_points_tfidf_final_minus_bot"][end]
        equal += [
            (head["router"]["accuracy_ci95"][end], final["accuracy_ci95"][end], real["accuracy_ci95"][end]),
            (head["bot"]["accuracy_ci95"][end], bot["accuracy_ci95"][end]),
            (ours, theirs, real["paired_accuracy_points_minus_bot"][end]),
        ]
    for ours, *theirs in equal:
        for value in theirs:
            assert ours == close(value)
    assert results["meta"]["model_version"] == r["router"]["model_version"]


def test_consistency_block_is_complete(results):
    block = results["consistency"]
    assert block["count"] == len(block["checks"]) > 50
    for check in block["checks"]:
        node = results
        for key in check["key"].split(" > "):
            node = node[int(key)] if isinstance(node, list) else node[key]
        assert node == check["value"] and abs(node - check["source_value"]) <= 1e-6
    assert {c["source"] for c in block["checks"]} == {f"{n}.json" for n in NAMES}
    covered = {c["key"] for c in block["checks"]}
    for side in ("router", "bot"):
        for key in ("accuracy", "macro_f1", "misroutes", *(f"accuracy_ci95 > {e}" for e in ("low", "point", "high"))):
            assert f"headline > {side} > {key}" in covered
    assert all(f"costs > per_month_rs > {key}" in covered for key in COST_KEYS)
    for team in results["per_team"]:
        for side in ("router", "bot"):
            assert {f"per_team > {team} > {side} > {k}" for k in ("precision", "recall")} <= covered


@pytest.mark.parametrize("path", [("accuracy", "all"), ("triage", "rate"), ("cost_per_month_rs", "bot_licence")])
def test_consistency_detects_changed_value(mod, results, sources, path):
    changed = copy.deepcopy(sources)
    changed["router"][path[0]][path[1]] += 1e-5
    with pytest.raises(ValueError, match="differs from router.json"):
        mod.consistency(results, changed)
    assert mod.consistency(results, sources) == results["consistency"]


@pytest.mark.slow
def test_mismatch_writes_nothing(mod, sources, monkeypatch, tmp_path):
    changed = copy.deepcopy(sources)
    changed["baselines"]["baselines"]["bot"]["cost_rs_per_month"] += 1
    monkeypatch.setattr(mod, "load_sources", lambda _: changed)
    with pytest.raises(ValueError, match="differs from baselines.json"):
        mod.build(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_other_model_is_refused(mod, sources, windows):
    with pytest.raises(RuntimeError, match="rerun the router target"):
        mod.train_router(windows[0].head(2000), sources)


def test_scored_router_is_fitted_on_the_train_window(mod, results, outputs):
    assert results["meta"]["training_window"] == {"first_day": "2025-04-01", "last_day": "2026-03-31", "rows": 8687}
    assert results["meta"]["model_version"] == outputs[0]["model_version"]
    assert "Router.load" not in open(mod.__file__, encoding="utf-8").read()  # models/ may hold the M7 model


# --- Criterion 4: intervals and the expected score -----------------------------
def test_ci_ordering(results):
    head, expected = results["headline"], results["expected_score"]
    intervals = [
        head["router"]["accuracy_ci95"],
        head["bot"]["accuracy_ci95"],
        head["paired_accuracy_points_router_minus_bot"],
        expected["final_team"],
        expected["bot_label"],
        expected["bot_label"]["ci95"],
        *(row["lead_points"] for row in expected["monthly_lead"]["months"]),
    ]
    for ci in intervals:
        assert ci["low"] <= ci["point"] <= ci["high"]
    for side in ("router", "bot"):
        assert head[side]["accuracy_ci95"]["point"] == head[side]["accuracy"]


def test_expected_range_contains_point(results, sources, arrays):
    final, label = results["expected_score"]["final_team"], results["expected_score"]["bot_label"]
    point = results["headline"]["router"]["accuracy"]
    assert final["low"] < point < final["high"] and final["point"] == point
    ci, inner = final["ci95"], sources["experiments"]["supplementary"]["base_inner_val"]["accuracy"]
    drift = inner - point
    assert ci == results["headline"]["router"]["accuracy_ci95"]
    assert drift > 0 and final["drift_allowance_points"] == pytest.approx(100 * drift, abs=1e-3)
    assert final["low"] == pytest.approx(point - 2**0.5 * (point - ci["low"]) - drift, abs=5e-6)
    assert final["high"] == pytest.approx(point + 2**0.5 * (ci["high"] - point), abs=5e-6)
    assert label["point"] == close((arrays["router"] == arrays["bot"]).mean())
    assert label["low"] == pytest.approx(label["point"] - 2**0.5 * (label["point"] - label["ci95"]["low"]), abs=5e-6)
    assert label["high"] == pytest.approx(label["point"] + 2**0.5 * (label["ci95"]["high"] - label["point"]), abs=5e-6)


def test_vague_ceiling_recomputed(results, arrays, val, test_set):
    from src.audit import vague_mask

    v, vague = results["expected_score"]["vague_ceiling"], vague_mask(val["text_clean"]).to_numpy()
    share = vague_mask(test_set["text_clean"]).mean()
    right = (arrays["router"] == arrays["truth"])[vague].mean()
    assert v["test_vague_share"] == close(share) and v["validation_vague_accuracy"] == close(right)
    assert v["ceiling_accuracy"] == close(1 - share * (1 - right))


def test_lead_sentence_matches_the_numbers(mod, md, results):
    block = results["expected_score"]["monthly_lead"]
    months = block["months"]
    by_month = {row["group"]: row for row in results["breakdowns"]["month"]}
    for row in months:
        assert row["lead_points"]["point"] == close(by_month[row["month"]]["lead_points"])
        assert row["router_accuracy"] == by_month[row["month"]]["router_accuracy"]
    for key, flag in (("router_accuracy", "router_declines"), ("bot_accuracy", "bot_declines")):
        values = [row[key] for row in months]
        assert block[flag] == (values == sorted(set(values), reverse=True))
    pairs = [(a["lead_points"], b["lead_points"]) for a in months for b in months]
    assert block["intervals_overlap"] == all(a["low"] <= b["high"] for a, b in pairs)
    section = md.split("## Expected hidden-test score")[1].split("\n## ")[0]
    assert "both decline over the three months" in section
    assert "lead over the bot stays roughly constant within sampling noise" in section
    assert "point estimates rise every month" in section  # not flat: the point leads widen
    # Wording follows the values.
    changed = copy.deepcopy(block)
    changed.update(intervals_overlap=False, router_declines=False)
    other = mod._lead_sentence(changed)
    assert "roughly constant" not in other and "both decline" not in other


# --- Criterion 5: no customer text ---------------------------------------------
def word_runs(text, n=4):  # the M2 leak check
    tokens = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def shared_runs(report, *frames):
    return sum(
        len(word_runs(text) & report) for df in frames for col in ("text_clean", "request_text") for text in set(df[col])
    )


def test_no_shared_word_runs(built, windows, test_set):
    train, val = windows
    for name in ("evaluation.json", "evaluation.md"):
        shared = shared_runs(word_runs((built / name).read_text(encoding="utf-8")), train, val, test_set)
        # Count only: a failure must not print customer text.
        if shared:
            pytest.fail(f"{shared} four-word runs shared with request text in {name}", pytrace=False)
    # Control: the check sees a leaked message.
    long_text = next(t for t in val["text_clean"] if len(t.split()) >= 8)
    assert bool(shared_runs(word_runs(long_text), val))


def test_no_request_ids_or_old_names(built, windows, test_set):
    ids = set(windows[0]["request_id"]) | set(windows[1]["request_id"]) | set(test_set["request_id"])
    for name in ("evaluation.json", "evaluation.md"):
        text = (built / name).read_text(encoding="utf-8")
        assert sum(i in text for i in ids) == 0
        text = text.replace("Filters & Consumables", "")
        assert "Installations" not in text and "Consumables" not in text


def spaced(text):
    return f" {' '.join(re.findall(r'[^\W_]+', text))} "


def test_quoted_phrases_short_and_common(results, md, windows, outputs):
    texts = [spaced(t) for t in set(windows[0]["text_clean"])]
    quoted = {r[len("mentions '") : -1] for o in outputs for r in o["reasons"] if r.startswith("mentions '")}
    phrases = [p for row in results["failures"]["categories"] for p in row["phrases"]]
    assert phrases
    bad = 0
    for item in phrases:
        phrase = item["phrase"]
        short = len(phrase.split()) <= 3 and not any(c.isdigit() for c in phrase)
        common = sum(spaced(phrase) in text for text in texts) >= 10
        bad += not (short and common and phrase in quoted and item["count"] > 0)
    if bad:
        pytest.fail(f"{bad} quoted phrases are too long, rare, have digits or are not the router's own", pytrace=False)
    section = md.split("## When it fails")[1].split("\n## ")[0]
    assert re.findall(r"'([^']*)' \(\d+\)", section) == [p["phrase"] for p in phrases]


def test_rare_phrase_is_dropped(mod):
    """Invented texts: one phrase in 10 different messages, one in 9, one with a digit, one of four words."""
    train = [f"red lamp number{i}" for i in range(10)] + [f"odd hinge number{i}" for i in range(9)]
    train += [f"model 77x item{i}" for i in range(20)] + [f"one two three four item{i}" for i in range(20)]
    reasons = ["mentions 'red lamp'", "mentions 'odd hinge'", "mentions '77x'", "mentions 'one two three four'", "close call"]
    assert mod._common_phrases([{"reasons": reasons}], [0], train, {}) == [{"phrase": "red lamp", "count": 1}]


# --- Criterion 6: numbers in the markdown --------------------------------------
def unmatched_numbers(md, results):
    strings = {v for v in leaves(results) if isinstance(v, str)}
    values = [v for v in leaves(results) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    bad = [d for d in DATE.findall(md) if d not in strings]
    md = DATE.sub(" ", md)
    for text in sorted(strings, key=len, reverse=True):  # model version and similar
        md = md.replace(text, " ") if any(char.isdigit() for char in text) else md
    for token in NUMBER.findall(md):
        text = token.rstrip("%").replace(",", "")
        decimals = len(text.partition(".")[2])
        scale = 100 if token.endswith("%") else 1
        usable = [v for v in values if scale == 100 or decimals or isinstance(v, int) or v > 100]
        if not any(f"{v * scale:.{decimals}f}" == text for v in usable):
            bad.append(token)
    return bad


def test_md_numbers_in_json(md, results, mod):
    assert md == mod.render_markdown(results)
    assert unmatched_numbers(md, results) == []
    planted = md + "\n12,346 and 99.871% and 7.512 on 2026-07\n"
    assert unmatched_numbers(planted, results) == ["2026-07", "12,346", "99.871%", "7.512"]


def test_md_follows_the_results(mod, results):
    """No number is typed into the template: change every value and the rendering must still match."""

    def shift(node):
        if isinstance(node, dict):
            return {k: shift(v) for k, v in node.items()}
        if isinstance(node, list):
            return [shift(v) for v in node]
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            return node
        return node + 1000 if isinstance(node, int) else round(node * 0.9, 6)

    changed = shift(results)
    assert unmatched_numbers(mod.render_markdown(changed), changed) == []


def test_md_tables(md, results):
    def pct(x):
        return f"{x * 100:.2f}%"

    for team, s in results["per_team"].items():
        cells = [s["router"]["precision"], s["bot"]["precision"], s["router"]["recall"], s["bot"]["recall"]]
        assert md.count(f"| {team} | {s['support']} | " + " | ".join(map(pct, cells)) + " |") == 1
    month = results["costs"]["per_month_rs"]
    section = md.split("## Rupees per month")[1].split("\n## ")[0]
    for key, start in (
        ("router_clarification_always_works", "| Router with triage, clarification always works"),
        ("router_without_triage", "| Router without triage"),
        ("router_clarification_fixes_nothing", "| Router with triage, clarification fixes nothing"),
        ("bot", "| Bot, same requests"),
        ("bot_licence", "| Bot licence, separate"),
    ):
        lines = [line for line in section.splitlines() if line.startswith(start)]
        assert len(lines) == 1 and lines[0].endswith(f"| {month[key]:,.0f} |")


def test_md_discloses(md):
    assert "not untouched" in md and "not measured, because M6 fits no new model" in md
    assert "ask operations" in md and "bounce as often as the bot's did" in md
    assert "would score perfectly by construction" in md


# --- Criterion 7: deterministic, rounded, current ------------------------------
@pytest.mark.slow
def test_deterministic(mod, built, tmp_path):
    mod.build(tmp_path)
    for path in (p for p in built.rglob("*") if p.is_file()):
        assert (tmp_path / path.relative_to(built)).read_bytes() == path.read_bytes(), path.name


def test_floats_6dp(results):
    floats = [v for v in leaves(results) if isinstance(v, float)]
    assert len(floats) > 100 and all(np.isfinite(v) and v == round(v, 6) for v in floats)


def test_committed_reports_current(mod, results, reports_dir):
    # Tolerance: another BLAS or CPU may move model floats slightly. One validation row moves accuracy by 4.7e-4.
    stored = json.loads((reports_dir / "evaluation.json").read_text(encoding="utf-8"))
    fresh_leaves, stored_leaves = leaves(results), leaves(stored)
    assert len(fresh_leaves) == len(stored_leaves), "evaluation.json is stale: run evaluate"
    for a, b in zip(stored_leaves, fresh_leaves):
        same = a == pytest.approx(b, abs=1e-4) if isinstance(b, float) else a == b
        assert same, "evaluation.json is stale: run evaluate"
    assert (reports_dir / "evaluation.md").read_text(encoding="utf-8") == mod.render_markdown(stored)
