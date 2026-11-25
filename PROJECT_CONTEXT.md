# Project context

Current-state snapshot, not history. Replace stale lines instead of appending; keep under ~120 lines.
Decisions and discarded ideas go in `PROGRESS.md`.

## 1. Project

Kestrel Home service-request router: classify each incoming request into one of 7 teams, replacing the vendor bot.
Deliverables: `predictions.csv`, FastAPI service + one HTML screen, evaluation evidence, memo to Ritu (Head of D2C
Operations), recording, submission form.

## 2. Non-negotiable rules

- Target = `final_team` (team that closed the request), NOT `team_label` (bot guess, ~77% agreement).
- Canonical team names = current 7: Installs & Demo, Filters & Consumables, Repairs, Billing, Returns & Replacement,
  Warranty Claims, Product Advice. Old names (Installations, Consumables) are mapped at load time.
- Billing only when the payment itself is the problem (policy §3). "I paid" is not Billing.
- Legacy Zoho `resolved_at` is UTC (policy §9); add 5h30m for IST.
- Costs: Rs 305/transfer, Rs 260/misroute contact, licence Rs 3.2 lakh/yr (policy §4).
- Time-based validation: train Apr 2025–Mar 2026, val Apr–Jun 2026. Test = Jul–Sep 2026.
- No paid API key needed; `data/raw` never committed (policy §10); seed 42.

Full rules and module workflow: `CLAUDE.md`. Constants: `src/config.py`.

## 3. Module status

| Module | Status | Main output path |
|---|---|---|
| M0 setup | VERIFIED | `src/config.py` |
| M1 clean | VERIFIED | `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet` |
| M2 audit | VERIFIED | `reports/audit_metrics.json`, `reports/data_audit.md` |
| M3 split + evaluation | VERIFIED | `src/split.py`, `src/evaluate.py` |
| M4 baselines | VERIFIED | `reports/baselines.json`, `reports/baselines.md` |
| M5a model selection | VERIFIED | `reports/experiments.json`, `reports/experiments.md` |
| M5b router | VERIFIED | `src/router.py`, `reports/router.json`, `reports/router.md` |
| M6 evaluation | VERIFIED | `reports/evaluation.json`, `reports/evaluation.md`, `reports/figures/` |
| M7 predictions | VERIFIED | `predictions.csv`, `reports/predictions_summary.json` + `.md`, `models/router.joblib` |
| M8 service | VERIFIED | `app/main.py`, `app/static/index.html`, `README.md` |
| M9 | not started | |
| M10 | not started | |

## 4. Verified interfaces

One line per module; schemas, signatures and rules are in docs/contracts.md#mN (M0 has no section; see the file).

- M0: `src/config.py` — paths, `TEAMS`, `TEAM_NAME_MAP`, costs, `SEED`, split dates.
- M1: the two parquet files; `src/clean.py: clean_text`. Target `final_team_canon`; raw `team_label` has old names, do not use.
- M2: `reports/audit_metrics.json`; `src/audit.py: vague_mask, product_conflict_mask`.
- M3: `time_split`; `evaluate, coverage_curve, align_proba, bootstrap_ci`; pass sklearn probabilities through `align_proba`.
- M4: `reports/baselines.json`; `src/baselines.py: make_pipeline, fit_model, predict_with_proba`.
- M5a: `reports/experiments.json` (`chosen.config`); `src/experiments.py: build_model, cue_features`.
- M5b: `src/router.py: Router` (`train`, `save`, `load`, `predict`), `config.TRIAGE_THRESHOLD`. Load only our own joblib file.
- M6: `reports/evaluation.json`.
- M7: `src/predictions.py: load_final` (M8 loads the final model with it), `predictions.csv`, `reports/predictions_summary.json`.
- M8: `app/main.py: create_app`; `GET /health`, `POST /predict`, `GET /`.

Model files (git-ignored): `models/router.joblib` is the final model (M7, all labelled rows); M5b's train-window model is in
`models/train_window/`. Run `bash scripts/run_all.sh <target>`; every target is implemented. `train` fits and saves the final
model; `predict` writes `predictions.csv` and the summary; `serve` starts the service on 127.0.0.1:8000 (M8, no LLM).

## 5. Key numbers

Accuracies are on validation against `final_team`. Full list: `PROGRESS.md`, "Detail moved from PROJECT_CONTEXT".

- M1: train 10,822 rows, test 2,178 (all `crm`); negative durations 1,140 before the UTC fix, 0 after.
- M2: bot agrees with `final_team` on 77.17%; costs Rs 122,171 per month (licence excluded); 8.1% of requests vague.
- M3: train 8,687 / validation 2,135 rows; `COST_PER_MISROUTE` Rs 698.06; bot misroute cost Rs 114,994 per month.
- M4: bot 76.67%, `tfidf_final` 83.89% (82.30 to 85.39), lead 7.21 points (5.76 to 8.81).
- M5a: chosen `logreg_c1_none` (same model as `tfidf_final`), inner-val 84.78%, real-val 83.89%.
- M5b: auto-routed 81.55% at 96.32%; triaged 18.45% at 28.93%; Rs 49,039 per month best case, bot 115,878 on the same rows.
- M6: expected hidden-test accuracy 80.74 to 86.01%; 80.52% if labels are the bot's; lead over the bot 6.17 to 8.10 by month.
- M7 (test): triaged 395 (18.14%), vague 177 (8.13%); largest gap to Apr-Jun 2026 is Repairs, +4.41 points (447 auto-routed).

## 6. Open questions / known risks

- The router's headline cost is a best case; triage pays only if clarification fixes over 52.41% of triaged misroutes. How
  often a question settles the team is not in the data: ask operations.
- `README.md` covers setup and running the service only; the rest is to write.
- M7 volumes are not headcount: Repairs is over-predicted and triaged requests are first guesses. No correction.
- The final model has no held-out score.
- The service has no login: local use only. Answers quote words of the request, so treat them as customer data.
- M5a's rule was amended after a real-val result; its choice (d) was already scored in M4, so 83.89% is not an untouched
  estimate. Cue features add at most 0.09 points to it on inner-val (report-only). See `PROGRESS.md`, M5a notes.
- Ritu's 90% bar is stated against the bot's labels (~77% right); restate it against `final_team`.
- Ritu says "eighteen months" of labels; there are 15 (the other 3 months are the unlabelled test set).
- `transfers` disagrees with the teams: 351 rows have transfers although first = final team; 126 rows changed team
  with 0 transfers. M2 cost uses `transfers` as recorded; Rs 7,178 per month of it is on rows the bot got right.
- Some requests cannot be routed without asking ("please call me about my purifier").
- Repeated text: 663 train rows repeat an earlier message and 258 test messages appear in train (688 and 260 on
  `text_clean`). Scores may look optimistic.
- Tuning on Apr–Jun 2026 would inflate the validation score; tune with time-based folds inside the train window.
- Farhan (Finance) wants the monthly run cost in writing before the switch; Ritu wants busiest teams for headcount.
- `httpx` is pinned for the test client; Starlette warns it will be replaced by `httpx2`.
- M2 tests check the headline lines of `data_audit.md` but not its table cells; read table values from the JSON.
- Request text contains order and registration numbers; M1 does not mask them. Decide in the feature module.
- M1 review findings are open (code unchanged since the review; list in `PROGRESS.md`). Relevant downstream: no test
  compares `team_label_canon`, `first_team_canon`, `misrouted` or test row order with raw, and running pytest rewrites
  `data/processed`, so rerun `clean` on known-good code after a failing test run.
- `CLAUDE.md` rule 2 bans old names after loading, but M1 keeps raw `team_label` by design; rule needs amending.
- Plain `python` on the dev machine is 3.10, too old for the pins; use `.venv` or `py -3.13`.
- Repo folder is spelled `kestral-routing`; the project name is `kestrel-routing`.
