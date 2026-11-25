"""M5b: Router. Contract: docs/contracts.md#m5b. Run: python -m src.router

Wraps the configuration chosen in M5a with confidence, triage, plain-English reasons and a clarifying question.
Nothing here changes the model's team. The triage threshold comes from policy costs and is not tuned.
The saved model embeds customer vocabulary (git-ignored) and is a pickle: load only a file this module wrote.
"""
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.audit import PRODUCT_PATTERNS, vague_mask
from src.baselines import ALLOWED_FEATURES, BOT, CATEGORICAL, TARGET, TEXT, round_floats
from src.clean import clean_text
from src.evaluate import align_proba, evaluate
from src.experiments import JSON_NAME, build_model, cue_features
from src.split import time_split

LIBRARIES = {  # meta key -> (name in messages, installed version)
    "sklearn_version": ("scikit-learn", sklearn.__version__),
    "joblib_version": ("joblib", joblib.__version__),
    "numpy_version": ("numpy", np.__version__),
}
TOP_TEAMS, MAX_PHRASES = 3, 3
LONGER_PHRASE_SHARE = 0.65  # a longer phrase is preferred when it has at least this share of its word's score
CLOSE_CALL = 0.10
BREAK_EVEN_FORMULA = "triaged requests x cost_clarify / (triaged misroutes x cost_per_misroute_reporting)"

# Function words that explain nothing to an agent. A negation is only quoted inside a longer phrase.
NEGATIONS = frozenset({"no", "not", "never", "nor", "none", "cannot", "without"})
STOPWORDS = frozenset(
    "a about after again all also am an and any are as at be because been before being but by can could did do does doing "
    "for from get got had has have having he her here hi hello him his how i if in into is it its just kindly me my "
    "now of on once only or our out over please pls sir madam she so some such than thank thanks that the their them then "
    "there these they this those to too under until up very was we were what when where which while who why will with "
    "would you your".split()
)
_APART = r"(?<![\w'’]){}(?![\w'’])"  # a whole phrase, not half of "don't"

CUE_REASONS = {  # M5a cue column -> sentence. Worded neutrally: a cue may point to another team.
    "payment_is_problem": "talks about a payment problem",
    "install_cue": "mentions an installation, demo or technician visit",
    "fault_cue": "mentions a fault or breakdown",
    "order_id_present": "includes an order number",
    "product_conflict": "names a different product from the one on the request",
}
EMPTY_REASON = "the message has no readable text, so this team is only a guess"
NO_PHRASE_REASON = "no single word stands out; the team is the best overall match"
POLICY3_REASON = "mentions a payment, but the problem may not be the payment itself (policy §3)"

POLICY3_OTHER_REASON = "mentions a payment, but the problem {} {}, not the payment itself (policy §3)"
AGENT_PHRASES = {  # for agents, in reasons; Billing has its own policy line
    "Repairs": "something has stopped working or is faulty",
    "Installs & Demo": "the customer is waiting for an installation or demo",
    "Returns & Replacement": "the customer wants to return or replace the product",
    "Warranty Claims": "it is about warranty or Shield cover",
    "Filters & Consumables": "the customer needs filters or spare parts",
    "Product Advice": "the customer has a question about using the product",
}

TEAM_PHRASES = {  # for customers, in the clarifying question
    "Repairs": "something has stopped working or is faulty",
    "Installs & Demo": "you're waiting for an installation or demo",
    "Returns & Replacement": "you want to return or replace it",
    "Billing": "there's a problem with a payment, invoice or refund",
    "Warranty Claims": "it's about warranty or Shield cover",
    "Filters & Consumables": "you need filters or spare parts",
    "Product Advice": "you have a question about using it",
}
PRODUCT_NAMES = {
    "Air Fryer": "air fryer",
    "Ceiling Fan": "fan",
    "Induction Cooktop": "cooktop",
    "Mixer Grinder": "mixer grinder",
    "Robot Vacuum": "vacuum",
    "Room Heater": "heater",
    "Water Purifier": "purifier",
}
# Wording only: neither number affects the team, the confidence or triage.
GENERAL_QUESTION_BELOW, GENERAL_TEAMS = 0.40, 4
QUESTION_MIN_PROB, QUESTION_TEAMS = 0.10, 3

POLICY3_CASES = (  # invented
    "I paid for installation but technician never came",
    "I have paid the full amount and the filter has not been delivered",
    "Paid by card last week, now the heater is not working",
    "I was charged twice for the same order",
    "Please convert my payment to EMI and send the GST invoice",
)


def _encodable(value: str) -> str:
    """Without lone surrogates, which are valid in JSON but cannot be stored as UTF-8."""
    return value.encode("utf-8", "replace").decode("utf-8")


def _category(value) -> str:
    """Optional field: anything missing or not text becomes "", which the encoder treats as unknown."""
    return _encodable(value).strip() if isinstance(value, str) else ""


def _to_frame(records: list[dict]) -> pd.DataFrame:
    for i, record in enumerate(records):
        if not isinstance(record.get("request_text"), str):
            raise ValueError(f"record {i}: request_text is required and must be a string")
    columns = {TEXT: [clean_text(_encodable(record["request_text"])) for record in records]}
    columns |= {name: [_category(record.get(name)) for record in records] for name in CATEGORICAL}
    return pd.DataFrame(columns)[list(ALLOWED_FEATURES)]


def _has_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _cue_rows(frame: pd.DataFrame) -> list[dict]:
    """M5a cues per row. They raise on an unknown product, so it is swapped out and its conflict cue dropped."""
    known = frame["product_family"].isin(list(PRODUCT_PATTERNS))
    product = frame["product_family"].where(known, next(iter(PRODUCT_PATTERNS)))
    cues = cue_features(frame.assign(product_family=product))
    cues["product_conflict"] = cues["product_conflict"].where(known, 0.0)
    return cues[["payment_only_mentioned", *CUE_REASONS]].to_dict("records")


def _usable(phrase: str) -> bool:
    words = phrase.split()
    if all(word in NEGATIONS for word in words):
        return False
    return not any(word in STOPWORDS or any(char.isdigit() for char in word) for word in words)


def _product_name(text: str, family: str) -> str | None:
    """The one product the text names, otherwise the request's own if the text names none. None if unclear."""
    named = [f for f, pattern in PRODUCT_PATTERNS.items() if re.search(rf"\b(?:{pattern})\b", text)]
    if not named and family in PRODUCT_NAMES:
        named = [family]
    return PRODUCT_NAMES[named[0]] if len(named) == 1 else None


def _either(parts: list[str], last: str = " or ") -> str:
    return f"{', '.join(parts[:-1])}{last}{parts[-1]}" if len(parts) > 1 else parts[0]


def _clarifying_question(proba, vague: bool, product: str | None) -> str:
    """proba in config.TEAMS order. Text without words counts as vague."""
    order = np.argsort(-proba, kind="stable")
    if vague or proba[order[0]] < GENERAL_QUESTION_BELOW:
        about = f" about your {product}" if product else ""
        phrases = [TEAM_PHRASES[config.TEAMS[j]] for j in order[:GENERAL_TEAMS]]
        return f"So we can send you to the right team{about}: is this because {_either(phrases, ', or ')}?"
    likely = [j for j in order if proba[j] >= QUESTION_MIN_PROB][:QUESTION_TEAMS]
    return f"Can you confirm whether {_either([TEAM_PHRASES[config.TEAMS[j]] for j in likely])}?"


def _reasons(phrases: list[str], cues: dict, has_text: bool, top: list[dict], triage: bool) -> list[str]:
    reasons = [] if has_text else [EMPTY_REASON]
    reasons += [f"mentions '{phrase}'" for phrase in phrases]
    if has_text and not phrases:
        reasons.append(NO_PHRASE_REASON)
    reasons += [sentence for name, sentence in CUE_REASONS.items() if cues[name]]
    team = top[0]["team"]
    if cues["payment_only_mentioned"]:  # M5a cue: mentions paying, without a payment-problem cue
        if team == "Billing":
            reasons.append(POLICY3_REASON)
        else:
            hedge = "may be that" if triage else "looks like"
            reasons.append(POLICY3_OTHER_REASON.format(hedge, AGENT_PHRASES[team]))
    if top[0]["prob"] - top[1]["prob"] <= CLOSE_CALL:
        reasons.append(f"close call between {team} and {top[1]['team']}")
    return reasons


def _meta_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meta.json")


def _model_version(df: pd.DataFrame, cfg: dict) -> str:
    """Same configuration, training rows and library versions give the same version."""
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(df[[*ALLOWED_FEATURES, TARGET]], index=False).to_numpy().tobytes())
    digest.update("|".join(version for _, version in LIBRARIES.values()).encode())
    return f"{cfg['name']}-{digest.hexdigest()[:12]}"


class Router:
    def __init__(self, model, meta: dict):
        self.model, self.meta = model, meta
        features = model.named_steps["features"]
        self._words = features.named_transformers_["word"].get_feature_names_out()
        self._word_usable = np.array([_usable(word) for word in self._words])
        self._word_slice = features.output_indices_["word"]
        self._class_row = {team: row for row, team in enumerate(model.classes_)}

    @classmethod
    def train(cls, df: pd.DataFrame, cfg: dict) -> "Router":
        """Fit on every row of df; the caller chooses the window."""
        if cfg["model_type"] != "logreg" or cfg["calibration"] is not None or cfg["feature_groups"]:
            raise ValueError("Router needs a plain logreg configuration: reasons are read from its coefficients")
        model = build_model(cfg).fit(df[cfg["features"]], df[TARGET])
        meta = {
            "model_version": _model_version(df, cfg),
            "training_window": {
                "first_day": str(df["created_at_ist"].min().date()),
                "last_day": str(df["created_at_ist"].max().date()),
                "rows": len(df),
            },
            "threshold": config.TRIAGE_THRESHOLD,
            "cost_inputs": {
                "cost_per_transfer": config.COST_PER_TRANSFER,
                "cost_per_misroute_contact": config.COST_PER_MISROUTE_CONTACT,
                "cost_per_misroute_train": config.COST_PER_MISROUTE_TRAIN,
                "cost_clarify": config.COST_CLARIFY,
            },
            **{key: version for key, (_, version) in LIBRARIES.items()},
            "config": cfg,
        }
        return cls(model, meta)

    def save(self, path) -> None:
        """Write the model to path and <stem>_meta.json next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_version": self.meta["model_version"], "model": self.model}, path)
        payload = json.dumps(self.meta, indent=2, sort_keys=True) + "\n"
        _meta_path(path).write_text(payload, encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path) -> "Router":
        """Only for files written by save: unpickling an untrusted file can run code. Versions are checked first."""
        path = Path(path)
        meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
        for key, (library, installed) in LIBRARIES.items():
            if meta[key] != installed:
                raise RuntimeError(
                    f"Model was built with {library} {meta[key]}; installed is {installed}. "
                    "Run the train step (see README) to rebuild."
                )
        saved = joblib.load(path)
        if saved["model_version"] != meta["model_version"]:
            raise RuntimeError("router model and metadata are from different builds")
        return cls(saved["model"], meta)

    def _needs_triage(self, confidence: float, has_text: bool) -> bool:
        costs = self.meta["cost_inputs"]
        return not has_text or (1 - confidence) * costs["cost_per_misroute_train"] > costs["cost_clarify"]

    def _phrases(self, word_row, text: str, team: str) -> list[str]:
        """Phrases of this text that pushed most toward the team: coefficient x tf-idf, word features only."""
        coef = self.model.named_steps["classifier"].coef_[self._class_row[team], self._word_slice]
        pushes = word_row.data * coef[word_row.indices]
        scores: dict[str, float] = {}  # strongest first
        for i in np.argsort(-pushes, kind="stable"):
            if pushes[i] <= 0:
                break
            column = word_row.indices[i]
            phrase = str(self._words[column])
            # A bigram can skip one-letter words or punctuation; quote only what is literally there.
            if self._word_usable[column] and re.search(_APART.format(re.escape(phrase)), text):
                scores[phrase] = float(pushes[i])
        chosen: list[str] = []
        for phrase, score in scores.items():
            # scores is strongest first, so next() gives the best of the longer phrases.
            words = set(phrase.split())
            longer = (
                p
                for p in scores
                if words < set(p.split())
                # "not working" always replaces "working", which alone says the opposite.
                and (scores[p] >= LONGER_PHRASE_SHARE * score or (set(p.split()) - words) & NEGATIONS)
            )
            phrase = next(longer, phrase)
            taken = {word for other in chosen for word in other.split()}
            if not taken & set(phrase.split()):  # also drops "working" once "not working" is chosen
                chosen.append(phrase)
            if len(chosen) == MAX_PHRASES:
                break
        return chosen

    def predict(self, records: list[dict]) -> list[dict]:
        if not records:
            return []
        frame = _to_frame(records)
        features = self.model.named_steps["features"].transform(frame).tocsr()
        proba = align_proba(self.model.named_steps["classifier"].predict_proba(features), self.model.classes_)
        words, cues = features[:, self._word_slice], _cue_rows(frame)
        vague, families = vague_mask(frame[TEXT]).tolist(), frame["product_family"].tolist()
        output = []
        for i, text in enumerate(frame[TEXT].tolist()):
            order = np.argsort(-proba[i], kind="stable")[:TOP_TEAMS]  # ties in config.TEAMS order, like argmax
            top = [{"team": config.TEAMS[j], "prob": float(proba[i, j])} for j in order]
            team, confidence, has_text = top[0]["team"], top[0]["prob"], _has_text(text)
            triage = self._needs_triage(confidence, has_text)
            question = _clarifying_question(proba[i], vague[i], _product_name(text, families[i])) if triage else None
            output.append(
                {
                    "team": team,
                    "confidence": confidence,
                    "top3": top,
                    "needs_triage": triage,
                    "reasons": _reasons(self._phrases(words[i], text, team), cues[i], has_text, top, triage),
                    "clarifying_question": question,
                    "model_version": self.meta["model_version"],
                }
            )
        return output


# --- Report ------------------------------------------------------------------
def to_records(df: pd.DataFrame) -> list[dict]:
    return df[["request_text", *CATEGORICAL]].to_dict("records")


def _share(part, whole) -> float | None:
    return float(part / whole) if whole else None


def evaluate_router(router: Router, val_df: pd.DataFrame) -> dict:
    """Aggregates on validation. Reporting only: nothing here feeds back into the router."""
    out = router.predict(to_records(val_df))
    truth = val_df[TARGET].to_numpy(dtype=object)
    pred = np.array([o["team"] for o in out], dtype=object)
    triaged = np.array([o["needs_triage"] for o in out])
    wrong = pred != truth
    n, n_triaged = len(val_df), int(triaged.sum())
    n_wrong, wrong_triaged = int(wrong.sum()), int((wrong & triaged).sum())
    wrong_sent = n_wrong - wrong_triaged
    bot_misroutes = int((val_df[BOT].to_numpy(dtype=object) != truth).sum())
    months = len(pd.period_range(val_df["created_at_ist"].min(), val_df["created_at_ist"].max(), freq="M"))
    confidence = np.array([o["confidence"] for o in out])
    vague = vague_mask(val_df[TEXT]).to_numpy()  # M2 rule
    repairs_sent = ~triaged & (pred == "Repairs")
    overall = evaluate(truth, pred)

    def monthly(misroutes, contacts):
        return (misroutes * config.COST_PER_MISROUTE + contacts * config.COST_CLARIFY) / months

    cost, bot_cost = monthly(wrong_sent, n_triaged), monthly(bot_misroutes, 0)
    return {
        "validation": {
            "rows": n,
            "first_day": str(val_df["created_at_ist"].min().date()),
            "last_day": str(val_df["created_at_ist"].max().date()),
            "months": months,
        },
        "triage": {
            "count": n_triaged,
            "rate": n_triaged / n,
            "sent_count": n - n_triaged,
            "mean_confidence": float(confidence[triaged].mean()) if n_triaged else None,
        },
        "vague": {"count": int(vague.sum()), "triaged_count": int((vague & triaged).sum())},
        "misroutes": {"all": n_wrong, "sent": wrong_sent, "triaged": wrong_triaged, "bot": bot_misroutes},
        "accuracy": {
            "all": overall["accuracy"],
            "sent": _share((~wrong & ~triaged).sum(), n - n_triaged),
            "triaged": _share((~wrong & triaged).sum(), n_triaged),
            "bot": 1 - bot_misroutes / n,
        },
        "repairs": {
            "sent_count": int(repairs_sent.sum()),
            "precision_sent": _share((~wrong & repairs_sent).sum(), repairs_sent.sum()),
            "precision_all": overall["per_class"]["Repairs"]["precision"],
        },
        "cost_per_month_rs": {
            "router": cost,
            "bot": bot_cost,
            "bot_minus_router": bot_cost - cost,
            "router_without_triage": monthly(n_wrong, 0),
            "router_if_clarification_fixes_nothing": monthly(n_wrong, n_triaged),
            "bot_licence": config.BOT_LICENCE_PER_YEAR / 12,
        },
        # Share of triaged misroutes that questions must fix before triage beats sending everything straight on.
        "break_even_clarification_rate": _share(
            n_triaged * config.COST_CLARIFY, wrong_triaged * config.COST_PER_MISROUTE
        ),
        "break_even_clarification_formula": BREAK_EVEN_FORMULA,
        "cost_per_misroute_reporting": config.COST_PER_MISROUTE,
        "policy3_cases": [
            {
                "text": text,
                "team": o["team"],
                "confidence": o["confidence"],
                "needs_triage": o["needs_triage"],
                "warning": POLICY3_REASON in o["reasons"],
            }
            for text, o in zip(POLICY3_CASES, router.predict([{"request_text": text} for text in POLICY3_CASES]))
        ],
        "router": router.meta,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _rs(x: float) -> str:
    return f"{x:,.0f}"


def render_markdown(results: dict) -> str:
    meta, window, inputs = results["router"], results["router"]["training_window"], results["router"]["cost_inputs"]
    val, triage, wrong, acc = results["validation"], results["triage"], results["misroutes"], results["accuracy"]
    repairs, cost = results["repairs"], results["cost_per_month_rs"]
    versions = ", ".join(f"{name} {meta[key]}" for key, (name, _) in LIBRARIES.items())
    lines = [
        "# Router",
        "",
        f"Model `{meta['model_version']}` ({versions}), fitted on {window['rows']:,} requests ({window['first_day']} to "
        f"{window['last_day']}). Scored on {val['rows']:,} validation requests ({val['first_day']} to {val['last_day']}). "
        "Reporting only: nothing was tuned on these numbers.",
        "",
        "## Triage",
        "",
        f"A request goes to triage when the expected misroute cost exceeds one clarifying contact: (1 - confidence) x "
        f"Rs {inputs['cost_per_misroute_train']:.2f} > Rs {inputs['cost_clarify']}, that is, confidence below "
        f"{meta['threshold']:.4f}. Messages without text always go to triage. The misroute cost is a train-window average. "
        "Triage never changes the team.",
        "",
        "| | Requests | Accuracy | Misroutes |",
        "| --- | ---: | ---: | ---: |",
        f"| Sent straight on | {triage['sent_count']:,} | {_pct(acc['sent'])} | {wrong['sent']} |",
        f"| Triaged | {triage['count']:,} | {_pct(acc['triaged'])} | {wrong['triaged']} |",
        f"| All | {val['rows']:,} | {_pct(acc['all'])} | {wrong['all']} |",
        f"| Bot, all | {val['rows']:,} | {_pct(acc['bot'])} | {wrong['bot']} |",
        "",
        f"- Triage rate: {_pct(triage['rate'])}. Mean confidence of triaged requests: {_pct(triage['mean_confidence'])}, "
        f"against {_pct(acc['triaged'])} accuracy.",
        f"- Vague requests (M2 rule): {results['vague']['count']} in validation, {results['vague']['triaged_count']} of them "
        "triaged.",
        f"- Repairs precision: {_pct(repairs['precision_sent'])} on the {repairs['sent_count']:,} requests sent straight to "
        f"Repairs, {_pct(repairs['precision_all'])} on all.",
        "",
        "## Cost per month, Rs",
        "",
        f"Misroutes sent straight on x Rs {results['cost_per_misroute_reporting']:.2f} + triaged requests x "
        f"Rs {inputs['cost_clarify']}, over {val['months']} months. **Assumption: a clarified request is never misrouted.** "
        "That is the best case; the other rows show what it is worth.",
        "",
        "| | Per month |",
        "| --- | ---: |",
        f"| Router with triage (headline, best case) | {_rs(cost['router'])} |",
        f"| Router with triage, clarification fixes nothing | {_rs(cost['router_if_clarification_fixes_nothing'])} |",
        f"| Router without triage | {_rs(cost['router_without_triage'])} |",
        f"| Bot, same requests | {_rs(cost['bot'])} |",
        f"| Bot licence, not included above | {_rs(cost['bot_licence'])} |",
        "",
        f"- Headline saving against the bot: Rs {_rs(cost['bot_minus_router'])} per month.",
        f"- Break-even clarification rate: {_pct(results['break_even_clarification_rate'])}. Triage beats no triage only if "
        f"clarification fixes more than this share of the triaged misroutes. Formula: "
        f"{results['break_even_clarification_formula']} = {triage['count']} x {inputs['cost_clarify']} / "
        f"({wrong['triaged']} x {results['cost_per_misroute_reporting']:.2f}).",
        "",
        "## Policy §3 check (invented messages)",
        "",
        "| Message | Team | Confidence | Triage | §3 warning |",
        "| --- | --- | ---: | --- | --- |",
        *[
            f"| {c['text']} | {c['team']} | {_pct(c['confidence'])} | {'yes' if c['needs_triage'] else 'no'} | "
            f"{'yes' if c['warning'] else 'no'} |"
            for c in results["policy3_cases"]
        ],
        "",
        "The team is whatever the model says; no rule overrides it. Channel, product and warranty were left out.",
        "",
        "## Caveats",
        "",
        "- The model is the M4 baseline (d), whose validation score had already been seen, so this is not an untouched "
        "estimate.",
        "- The cost per misroute used for reporting covers all labelled months, validation included, and assumes our "
        "misroutes bounce as often as the bot's did.",
        "- Triage depends on the probabilities being honest; they are not recalibrated here.",
        "- Reasons quote words with positive weight for the predicted team. The model also uses character patterns and the "
        "request fields, which are not shown, so reasons are a partial explanation.",
        "- Clarifying questions and reasons are fixed wording, untested with customers and agents.",
    ]
    return "\n".join(lines) + "\n"


def build(reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR / "train_window") -> dict:
    """Train on the train window, save the model, write router.json and router.md."""
    train_df, val_df = time_split(pd.read_parquet(config.TRAIN_CLEAN_PARQUET))
    cfg = json.loads((config.REPORTS_DIR / JSON_NAME).read_text(encoding="utf-8"))["chosen"]["config"]
    router = Router.train(train_df, cfg)
    router.save(models_dir / config.ROUTER_JOBLIB.name)
    results = round_floats(evaluate_router(router, val_df))
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (reports_dir / config.ROUTER_JSON.name).write_text(payload, encoding="utf-8", newline="\n")
    (reports_dir / config.ROUTER_MD.name).write_text(render_markdown(results), encoding="utf-8", newline="\n")
    return results


if __name__ == "__main__":
    build()
