"""M9 acceptance tests: memo. Contract: docs/contracts.md#m9.

Every number in the memo is bound to its wording: a CLAIMS template is filled from reports/*.json and the resulting
sentence fragment must be in the document. Digits left over must be in ALLOWED (not measurements) or in a sentence marked
as an estimate or proposal.
"""

import json
import re

import pandas as pd
import pytest

from src import config

MEMO = "memo.md"
DOCS = (MEMO,)
MEMO_HEADINGS = ["The decision", "The number", "The rupees", "Next week"]
JARGON = ["F1", "logistic", "TF-IDF", "calibration", "log loss", "bootstrap", "confusion matrix", "parquet", "sklearn"]


def read(name):
    return (config.ROOT / name).read_text(encoding="utf-8")


def flat(text):
    """One line, without bold marks, so that wrapping and emphasis do not matter."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------- numbers from the reports


def load_facts():
    names = ("audit_metrics", "baselines", "experiments", "router", "evaluation", "predictions_summary")
    r = {n: json.loads((config.REPORTS_DIR / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    audit, base, exp, router = r["audit_metrics"], r["baselines"], r["experiments"], r["router"]
    ev, pred = r["evaluation"], r["predictions_summary"]

    def pct(x, digits=1):
        return f"{100 * x:.{digits}f}%"

    def rs(x):
        return f"{x:,.0f}"

    costs = ev["costs"]
    score, bot_score = ev["expected_score"]["final_team"], ev["expected_score"]["bot_label"]
    months = ev["expected_score"]["monthly_lead"]["months"]
    lead = ev["headline"]["paired_accuracy_points_router_minus_bot"]
    ci = ev["headline"]["router"]["accuracy_ci95"]
    copy = base["baselines"]["tfidf_bot"]
    recent = pred["comparison"]["teams"]
    top3 = sorted(recent, key=lambda t: -recent[t]["recent_count"])[:3]
    assert top3 == ["Repairs", "Returns & Replacement", "Installs & Demo"]
    label_gap = {
        t: audit["bot_label_volume_by_team"][t] / n - 1 for t, n in audit["final_volume_by_team"].items()
    }
    assert {t for t, g in label_gap.items() if g > 0} == {"Repairs", "Billing", "Filters & Consumables"}
    under = [-100 * label_gap[t] for t in ("Installs & Demo", "Returns & Replacement", "Warranty Claims")]
    failures = {c["category"]: c for c in ev["failures"]["categories"]}
    inner_best = max(c["inner_val"]["accuracy"] for c in exp["candidates"] if c["name"].startswith("linear_svc"))
    chosen_inner = next(c for c in exp["candidates"] if c["name"] == exp["chosen"]["name"])["inner_val"]["accuracy"]
    original = exp["original_rule"]
    assert original["criterion_4"]["result"] == "failed" and exp["cue_candidates"]["within_log_loss_limit_count"] == 0
    bot_better = {t for t, v in ev["per_team"].items() if v["bot"]["precision"] > v["router"]["precision"]}
    assert bot_better == {"Installs & Demo", "Returns & Replacement", "Warranty Claims", "Product Advice"}

    return {
        "bot_all": pct(audit["bot_vs_final_agreement"]),
        "bot_all0": pct(audit["bot_vs_final_agreement"], 0),
        "label_months": audit["months_covered"]["count"],
        "bot": pct(ev["headline"]["bot"]["accuracy"]),
        "router": pct(router["accuracy"]["all"]),
        "copy_agree": pct(copy["agreement_with_team_label"]),
        "copy": pct(copy["evaluate"]["accuracy"]),
        "keywords": pct(base["baselines"]["keywords"]["evaluate"]["accuracy"]),
        "low": pct(score["low"]),
        "high": pct(score["high"]),
        "ci_low": pct(ci["low"]),
        "ci_high": pct(ci["high"]),
        "drift": f"{score['drift_allowance_points']:.1f}",
        "f1": f"{score['macro_f1']:.3f}",
        "bl": pct(bot_score["point"]),
        "bl_low": pct(bot_score["low"]),
        "bl_high": pct(bot_score["high"]),
        "bl_f1": f"{bot_score['macro_f1']:.3f}",
        "lead": f"{lead['point']:.1f}",
        "lead_low": f"{lead['low']:.1f}",
        "lead_high": f"{lead['high']:.1f}",
        "ci_level": pct(ev["meta"]["ci_level"], 0),
        "m_router": [pct(m["router_accuracy"]) for m in months],
        "m_bot": [pct(m["bot_accuracy"]) for m in months],
        "val_rows": f"{ev['headline']['rows']:,}",
        "ceiling": pct(ev["expected_score"]["vague_ceiling"]["ceiling_accuracy"]),
        "vague_share": pct(ev["expected_score"]["vague_ceiling"]["validation_vague_share"]),
        "sent": pct(router["triage"]["sent_count"] / router["validation"]["rows"]),
        "sent_acc": pct(router["accuracy"]["sent"]),
        "flagged": pct(router["triage"]["rate"]),
        "flagged_acc": pct(router["accuracy"]["triaged"]),
        "threshold": f"{router['router']['threshold']:.3f}",
        "vague": all_flagged(router["vague"]),
        "vague_test": all_flagged(pred["vague"]),
        "flagged_month": f"{pred['triage']['count'] / len(pred['test']['months']):.0f}",
        "train_rows": f"{pred['model']['training_window']['rows']:,}",
        "test_rows": f"{pred['test']['rows']:,}",
        "c_bot": rs(costs["per_month_rs"]["bot"]),
        "c_plain": rs(costs["per_month_rs"]["router_without_triage"]),
        "c_best": rs(costs["per_month_rs"]["router_clarification_always_works"]),
        "c_worst": rs(costs["per_month_rs"]["router_clarification_fixes_nothing"]),
        "save": rs(base["comparisons"]["cost_rs_per_month_bot_minus_tfidf_final"]),
        "save_best": rs(costs["bot_minus_router_rs"]),
        "save_worst": rs(costs["bot_minus_router_clarification_fixes_nothing_rs"]),
        "save_simple": rs(base["cost_sensitivity"]["simple"]["saving_rs_per_month_bot_minus_tfidf_final"]),
        "cost_simple": rs(base["cost_sensitivity"]["simple"]["cost_per_misroute_rs"]),
        "cost_avg": rs(base["cost_sensitivity"]["average"]["cost_per_misroute_rs"]),
        "clarify": rs(costs["inputs"]["cost_clarify_rs"]),
        "break_even": pct(costs["break_even_clarification_rate"]),
        "licence": f"{audit['cost_inputs']['licence_rs_per_year'] / 100_000:.1f} lakh",
        "n_rep": f"{recent['Repairs']['recent_per_month']:.0f}",
        "n_ret": f"{recent['Returns & Replacement']['recent_per_month']:.0f}",
        "n_ins": f"{recent['Installs & Demo']['recent_per_month']:.0f}",
        "g_rep": pct(label_gap["Repairs"], 0),
        "g_bil": pct(label_gap["Billing"], 0),
        "g_fil": pct(label_gap["Filters & Consumables"], 0),
        "g_under": f"{min(under):.0f}–{max(under):.0f}%",
        "g_adv": pct(-label_gap["Product Advice"], 0),
        "first_rule": pct(original["real_val"]["accuracy"]),
        "first_drop": f"{-original['criterion_4']['points_minus_reference']:.1f}",
        "allowed_drop": f"{original['criterion_4']['allowed_drop_points']:.1f}",
        "n_cand": len(exp["candidates"]),
        "svm_gain": f"{100 * (inner_best - chosen_inner):.1f}",
        "bias": f"{pred['validation']['teams']['Repairs']['points']:.1f}",
        "errors": ev["failures"]["errors"],
        "vague_close": failures["vague"]["exclusive_errors"] + failures["close_call"]["exclusive_errors"],
        "seen_test": pct(ev["seen_text"]["test_seen_share"]),
        "seen_val": pct(ev["seen_text"]["validation_seen_share"]),
    }


def all_flagged(vague):
    """The count, only if every vague request was flagged (the documents say "all")."""
    assert vague["count"] == vague["triaged_count"]
    return vague["count"]


CLAIMS = {
    MEMO: [
        "the bot was right {bot_all} of the time over {label_months} months",
        "it matches the labels {copy_agree} of the time",
        "yet is right only {copy} of the time",
        "the router was right on {router} of requests, the bot on {bot}.",
        "Expect {low} to {high} on new requests",
        "The router sends {sent} straight on, {sent_acc} of them correctly",
        "The other {flagged} get one clarifying question",
        'all {vague} vague "please call me" requests',
        "| Bot | {c_bot} |",
        "| Router, no question asked | {c_plain} |",
        "| Router, question always works | {c_best} |",
        "| Router, question never helps | {c_worst} |",
        "Plan on saving Rs {save} a month, which assumes no question is asked.",
        "The question could lift that to Rs {save_best}",
        "each costs Rs {clarify}: if it never helped, the saving would be Rs {save_worst}",
        "It pays only if it fixes over {break_even} of flagged requests whose first guess was wrong",
        "licence saves Rs {licence} a year",
        "flagged requests (about {flagged_month} a month)",
        "stop if it misses {break_even}",
        "Repairs {n_rep} a month, Returns {n_ret}, Installs {n_ins} are busiest",
        "overstate Repairs by {g_rep}, Billing by {g_bil} and Filters by {g_fil}",
        "understate Installs, Returns and Warranty by {g_under} and Product Advice by {g_adv}",
    ],
}

# Not measurements: the client's bar, policy sections, calendar years, confidence bins named in the M6 findings.
ALLOWED = {
    MEMO: [r"Head of D2C Operations", r'"90% match to the old labels"', r"April–June 2026", r"policy §10"],
}
MARKER = re.compile(r"\bestimate\b|\bproposed\b", re.IGNORECASE)
# Marked sentences: the exact unsourced figures the owner accepted, so that the marker cannot hide a measurement.
MARKED = {MEMO: ["3", "93%", "2,000", "4,000"]}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def unsourced(name, text, facts):
    """(missing claims, numbers without a source) for one document."""
    rest = flat(text)
    missing = []
    for template in CLAIMS[name]:
        claim = template.format(**facts)
        if claim not in rest:
            missing.append(claim)
        rest = rest.replace(claim, " ~ ")
    for pattern in ALLOWED[name]:
        rest = re.sub(pattern, " ~ ", rest)
    unmarked, marked = [], []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z*#|-])", rest):
        found = [n.rstrip(",") for n in NUMBER.findall(sentence)]
        (marked if MARKER.search(sentence) else unmarked).extend(found)
    extra = [n for n in marked if n not in MARKED[name]]
    return missing, unmarked + extra


@pytest.fixture(scope="module")
def facts():
    return load_facts()


# ---------------------------------------------------------------- criteria


def sections(text):
    parts = re.split(r"^## +(.+)$", text, flags=re.MULTILINE)
    return dict(zip((h.strip() for h in parts[1::2]), (b.strip() for b in parts[2::2])))


def test_memo_length_and_headings():
    """Criterion 1."""
    text = read(MEMO)
    assert len(text.split()) <= 450
    found = sections(text)
    assert list(found) == MEMO_HEADINGS
    assert all(found.values())


@pytest.mark.parametrize("name", DOCS)
def test_every_number_is_sourced(name, facts):
    """Criterion 2."""
    missing, left = unsourced(name, read(name), facts)
    assert missing == []
    assert left == []


@pytest.mark.parametrize("name", DOCS)
def test_number_check_catches_a_changed_or_added_number(name, facts):
    text = flat(read(name))
    for template in CLAIMS[name]:
        claim = template.format(**facts)
        start = max(i for i, ch in enumerate(claim) if ch.isdigit())
        changed = claim[:start] + str((int(claim[start]) + 1) % 10) + claim[start + 1 :]
        assert unsourced(name, text.replace(claim, changed), facts)[0] != [], template
    assert unsourced(name, text + " It is right 99.9% of the time.", facts)[1] == ["99.9%"]
    assert unsourced(name, text + " Estimate: it is right 99.9% of the time.", facts)[1] == ["99.9%"]


def test_memo_estimates_are_marked_for_confirmation():
    assert "Estimate, to confirm with Tanmay: Rs 2,000–4,000 a month" in flat(read(MEMO))
    assert "Proposed conditions" in read(MEMO)


def test_memo_no_jargon():
    """Criterion 3."""
    text = flat(read(MEMO)).lower()
    assert [j for j in JARGON if j.lower() in text] == []


def four_word_runs(text):
    w = words(text)
    return {tuple(w[i : i + 4]) for i in range(len(w) - 3)}


def test_no_customer_text():
    """Criterion 4. Failures show counts only, never the text."""
    customer = set()
    for path in (config.TRAIN_CLEAN_PARQUET, config.TEST_CLEAN_PARQUET):
        for text in pd.read_parquet(path, columns=["text_clean"])["text_clean"]:
            customer |= four_word_runs(text)
    assert len(customer) > 1000
    overlaps = {name: len(four_word_runs(read(name)) & customer) for name in DOCS}
    assert overlaps == dict.fromkeys(DOCS, 0)
