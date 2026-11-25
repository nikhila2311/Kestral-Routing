"""M4: baselines on the validation window. Contract: docs/contracts.md#m4. Run: python -m src.baselines"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.evaluate import align_proba, bootstrap_ci, evaluate
from src.split import time_split

JSON_NAME, MD_NAME = "baselines.json", "baselines.md"
MONTHS_PER_YEAR = 12
CI_LEVEL = 0.95  # what bootstrap_ci returns

TARGET, BOT = "final_team_canon", "team_label_canon"
TEXT, CATEGORICAL = "text_clean", ("channel", "product_family", "warranty_status")
ALLOWED_FEATURES = (TEXT, *CATEGORICAL)  # known when the request is created

# Fixed up front, never tuned.
WORD_NGRAMS, CHAR_NGRAMS = (1, 2), (3, 5)
MIN_DF, C, MAX_ITER = 2, 1.0, 2000

BASELINES = {
    "majority": "(a) Majority team",
    "bot": "(b) Bot label",
    "keywords": "(c) Keyword rules",
    "tfidf_final": "(d) TF-IDF, trained on the final team",
    "tfidf_bot": "(e) TF-IDF, trained on the bot label",
}

# Written from the teams.csv descriptions and ops-policy section 3, not tuned on any scores. First match wins.
# A mention of paying, EMI or warranty matches nothing on its own. Faults come before spares ("Not faults").
KEYWORD_RULES = (
    (
        "Billing",
        r"\binvoices?\b|\bgst\b|\bcoupons?\b|\bdouble charg\w*|\b(?:double|duplicate) payment"
        r"|\b(?:charged|deducted|debited|billed|paid) (?:twice|double|two times)\b"
        r"|\bemi (?:\w+ ){0,2}conver\w+|\bconver\w+ (?:\w+ ){0,4}emi\b"
        r"|\brefund\w* (?:\w+ ){0,3}(?:payment|money|amount)\b"
        r"|\b(?:payment|money|amount) (?:\w+ ){0,3}refund\w*",
    ),
    (
        "Returns & Replacement",
        r"\bdamaged\b|\bexchang\w+|\breturn(?:s|ed|ing)?\b|\bincomplete\b"
        r"|\bwrong (?:item|product|model|colou?r|piece|order|delivery)\b"
        r"|\bwrong (?:\w+ ){1,3}(?:delivered|received|sent)\b"
        r"|\b(?:delivered|received|sent) (?:\w+ ){0,2}wrong\b"
        r"|\bmissing (?:\w+ ){0,3}(?:box|package|parcel|delivery)\b"
        r"|\breplace(?:ment)? (?:\w+ ){0,2}(?:unit|product|item)\b",
    ),
    (
        "Warranty Claims",
        r"\bclaims?\b|\bregist(?:er|ration)\w* (?:\w+ ){0,5}(?:warranty|shield)\b"
        r"|\b(?:warranty|shield) (?:\w+ ){0,3}regist(?:er|ration)\w*"
        r"|\b(?:covered|coverage|covers?) (?:\w+ ){0,5}(?:warranty|shield)\b"
        r"|\b(?:warranty|shield) (?:\w+ ){0,3}(?:cover\w*|status|valid\w*|expir\w+)",
    ),
    (
        "Repairs",
        r"\bnot (?:working|turning on|starting|switching on)\b|\bstopped working\b|\bfault\w*|\bbreakdown\b"
        r"|\bbroken?\b|\berror\b|\bnois[ey]\w*|\bleak\w*|\btechnician\b|\brepair\w*",
    ),
    ("Billing", r"\brefund\w*"),  # any refund without a return or a fault
    (
        "Filters & Consumables",
        r"\bfilters?\b|\bcandles?\b|\bmembranes?\b|\bjars?\b|\bbrush(?:es)?\b|\bblades?\b|\bamc\b|\bspares?\b"
        r"|\bconsumables?\b",
    ),
    ("Installs & Demo", r"\binstall\w*|\bdemo\w*|\bmount\w*"),
    (
        "Product Advice",
        r"\bhow\b|\bwhich\b|\bwhat\b|\bcan (?:i|we)\b|\bshould (?:i|we)\b|\bsuitable\b|\brecommend\w*|\bsuggest\w*"
        r"|\badvice\b|\bus(?:e|ing|age)\b",
    ),
)
_COMPILED_RULES = tuple((team, re.compile(pattern)) for team, pattern in KEYWORD_RULES)


def majority_class(y) -> str:
    """Most frequent team; ties go to the first in config.TEAMS."""
    counts = pd.Series(y).value_counts()
    return max(config.TEAMS, key=lambda team: counts.get(team, 0))


def keyword_predict(text, fallback: str) -> tuple[np.ndarray, np.ndarray]:
    """(predicted team, whether a rule fired) per text."""
    matched = [next((team for team, rule in _COMPILED_RULES if rule.search(t)), None) for t in text]
    fired = np.array([team is not None for team in matched], dtype=bool)
    return np.array([team or fallback for team in matched], dtype=object), fired


def make_pipeline() -> Pipeline:
    features = ColumnTransformer(
        [
            ("word", TfidfVectorizer(ngram_range=WORD_NGRAMS, min_df=MIN_DF, sublinear_tf=True), TEXT),
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAMS, min_df=MIN_DF, sublinear_tf=True),
                TEXT,
            ),
            ("category", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=None, random_state=config.SEED)
    return Pipeline([("features", features), ("classifier", classifier)])


def fit_model(train_df: pd.DataFrame, target_col: str) -> Pipeline:
    """Fit vectorisers, encoder and classifier on train_df, using the allowed features only."""
    return make_pipeline().fit(train_df[list(ALLOWED_FEATURES)], train_df[target_col])


def predict_with_proba(model: Pipeline, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(predicted team, probabilities in config.TEAMS order)."""
    proba = align_proba(model.predict_proba(df[list(ALLOWED_FEATURES)]), model.classes_)
    return np.array(config.TEAMS, dtype=object)[proba.argmax(axis=1)], proba


def _accuracy(y_true, y_pred) -> float:
    return float(np.mean(y_true == y_pred))


def _validation_months() -> int:
    return len(pd.period_range(config.VALID_START, config.VALID_END, freq="M"))


def score(val_df: pd.DataFrame, pred, proba=None) -> dict:
    """Metrics for one baseline against the final team, plus agreement with the bot label."""
    truth = val_df[TARGET].to_numpy(dtype=object)
    pred = np.asarray(pred, dtype=object)
    metrics = evaluate(truth, pred, proba)
    low, point, high = bootstrap_ci(truth, pred, _accuracy)
    return {
        "evaluate": metrics,
        "accuracy_ci95": {"low": low, "point": point, "high": high},
        "agreement_with_team_label": _accuracy(val_df[BOT].to_numpy(dtype=object), pred),
        "cost_rs": metrics["cost_rs"],
        "cost_rs_per_month": metrics["cost_rs"] / _validation_months(),
    }


def compute_baselines(df: pd.DataFrame) -> dict:
    """Fit on the train window, score on the validation window. Aggregates only."""
    train, val = time_split(df)
    majority = majority_class(train[TARGET])
    keyword_pred, fired = keyword_predict(val[TEXT], majority)

    scores = {
        "majority": score(val, np.full(len(val), majority, dtype=object)),
        "bot": score(val, val[BOT]),
        "keywords": {**score(val, keyword_pred), "rule_fired_share": float(fired.mean())},
    }
    for name, target in (("tfidf_final", TARGET), ("tfidf_bot", BOT)):
        scores[name] = score(val, *predict_with_proba(fit_model(train, target), val))

    accuracy = {name: s["evaluate"]["accuracy"] for name, s in scores.items()}
    agreement = {name: s["agreement_with_team_label"] for name, s in scores.items()}
    monthly = {name: s["cost_rs_per_month"] for name, s in scores.items()}
    return {
        "baselines": scores,
        "comparisons": {
            "accuracy_points_tfidf_final_minus_bot": 100 * (accuracy["tfidf_final"] - accuracy["bot"]),
            "accuracy_points_tfidf_final_minus_tfidf_bot": 100 * (accuracy["tfidf_final"] - accuracy["tfidf_bot"]),
            "accuracy_points_keywords_minus_bot": 100 * (accuracy["keywords"] - accuracy["bot"]),
            "agreement_points_tfidf_bot_minus_tfidf_final": 100 * (agreement["tfidf_bot"] - agreement["tfidf_final"]),
            "cost_rs_per_month_bot_minus_tfidf_final": monthly["bot"] - monthly["tfidf_final"],
        },
        "meta": {
            "train_rows": len(train),
            "validation_rows": len(val),
            "validation_first_day": str(config.VALID_START),
            "validation_last_day": str(config.VALID_END),
            "validation_months": _validation_months(),
            "majority_team": majority,
            "features": list(ALLOWED_FEATURES),
            "model_settings": {
                "word_ngrams": list(WORD_NGRAMS),
                "char_wb_ngrams": list(CHAR_NGRAMS),
                "min_df": MIN_DF,
                "C": C,
                "max_iter": MAX_ITER,
                "seed": config.SEED,
            },
            "ci_level": CI_LEVEL,
            "cost_per_misroute_rs": config.COST_PER_MISROUTE,
            "bot_licence_rs_per_month": config.BOT_LICENCE_PER_YEAR / MONTHS_PER_YEAR,
        },
    }


def _pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def _rs(x: float) -> str:
    return f"{x:,.0f}"


def _findings(results: dict) -> list[str]:
    """Wording follows the values, so a rerun cannot leave a stale claim behind."""
    b, c, m = results["baselines"], results["comparisons"], results["meta"]
    final, bot, as_bot, rules = b["tfidf_final"], b["bot"], b["tfidf_bot"], b["keywords"]

    def points(key):
        value = c[key]
        return f"{abs(value):.2f} points {'above' if value > 0 else 'below' if value < 0 else 'from'}"

    def acc(s):
        return _pct(s["evaluate"]["accuracy"])

    overlap = (
        final["accuracy_ci95"]["low"] <= bot["accuracy_ci95"]["high"]
        and bot["accuracy_ci95"]["low"] <= final["accuracy_ci95"]["high"]
    )
    saving = c["cost_rs_per_month_bot_minus_tfidf_final"]
    return [
        f"TF-IDF trained on the final team reaches {acc(final)}, {points('accuracy_points_tfidf_final_minus_bot')} the "
        f"bot's {acc(bot)}; their {_pct(m['ci_level'], 0)} intervals {'overlap' if overlap else 'do not overlap'}.",
        f"The same pipeline trained on the bot's labels agrees with the bot on "
        f"{_pct(as_bot['agreement_with_team_label'])} of requests (against "
        f"{_pct(final['agreement_with_team_label'])}) but matches the final team on {acc(as_bot)}; the model trained on "
        f"the final team is {points('accuracy_points_tfidf_final_minus_tfidf_bot')} that.",
        f"The keyword rules reach {acc(rules)}, {points('accuracy_points_keywords_minus_bot')} the bot; a rule fires "
        f"on {_pct(rules['rule_fired_share'])} of requests and the rest go to {m['majority_team']}.",
        f"At Rs {m['cost_per_misroute_rs']:.2f} per misroute, the bot's errors cost Rs {_rs(bot['cost_rs_per_month'])} "
        f"per month on validation and the TF-IDF model's Rs {_rs(final['cost_rs_per_month'])}, a monthly "
        f"{'saving' if saving >= 0 else 'increase'} of Rs {_rs(abs(saving))}, before the bot's licence of "
        f"Rs {_rs(m['bot_licence_rs_per_month'])} per month.",
        "Some messages repeat word for word across months, so the TF-IDF scores may be optimistic and the intervals "
        "slightly too narrow.",
    ]


def render_markdown(results: dict) -> str:
    """Human-readable summary. Every number comes from results; the prose must stay digit-free."""
    m = results["meta"]
    header = [
        "Baseline",
        "Accuracy",
        f"{_pct(m['ci_level'], 0)} interval",
        "Macro F-score",
        "Agreement with bot label",
        "Misroutes",
        "Cost (Rs)",
        "Cost per month (Rs)",
    ]
    rows = [header, ["---"] + ["---:"] * (len(header) - 1)]
    for name, title in BASELINES.items():
        s = results["baselines"][name]
        ci = s["accuracy_ci95"]
        rows.append(
            [
                title,
                _pct(s["evaluate"]["accuracy"]),
                f"{_pct(ci['low'])} to {_pct(ci['high'])}",
                _pct(s["evaluate"]["macro_f1"]),
                _pct(s["agreement_with_team_label"]),
                f"{s['evaluate']['misroutes']:,}",
                _rs(s["cost_rs"]),
                _rs(s["cost_rs_per_month"]),
            ]
        )
    lines = [
        "# Baselines",
        "",
        f"Trained on {m['train_rows']:,} requests and scored against the final team on {m['validation_rows']:,} "
        f"validation requests ({m['validation_first_day']} to {m['validation_last_day']}, {m['validation_months']} "
        "months).",
        "Costs are for misroutes only and leave out the bot licence.",
        "",
        *["| " + " | ".join(cells) + " |" for cells in rows],
        "",
        "## Findings",
        "",
        *[f"- {sentence}" for sentence in _findings(results)],
    ]
    return "\n".join(lines) + "\n"


def build(out_dir: Path = config.REPORTS_DIR) -> dict:
    """Write baselines.json and baselines.md to out_dir and return the results."""
    results = compute_baselines(pd.read_parquet(config.TRAIN_CLEAN_PARQUET))
    markdown = render_markdown(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    (out_dir / JSON_NAME).write_text(payload, encoding="utf-8", newline="\n")
    (out_dir / MD_NAME).write_text(markdown, encoding="utf-8", newline="\n")
    return results


if __name__ == "__main__":
    build()
