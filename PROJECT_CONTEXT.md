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
| M5a model selection | BUILT, awaiting verification | `reports/experiments.json`, `reports/experiments.md` |
| M5b | not started | |
| M6 | not started | |
| M7 | not started | |
| M8 | not started | |
| M9 | not started | |
| M10 | not started | |

## 4. Verified interfaces

One line per VERIFIED module: `MN: <output path> — schema in docs/contracts.md#mN`.

- M0: `src/config.py` — paths, `TEAMS`, `TEAM_NAME_MAP`, costs, `SEED`, split dates (no contract section; see file).
- M1: `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet`; `src/clean.py: clean_text` — schema in
  docs/contracts.md#m1. Target column is `final_team_canon`; raw `team_label` still holds old names, do not use it.
- M2: `reports/audit_metrics.json` — keys in docs/contracts.md#m2 (`reports/data_audit.md` is rendered from it).
- M3: `src/split.py: time_split`; `src/evaluate.py: evaluate, coverage_curve, align_proba, bootstrap_ci`;
  `config.COST_PER_MISROUTE`, `config.COST_PER_MISROUTE_SIMPLE` — signatures, rules and caveats in docs/contracts.md#m3.
  Pass sklearn `predict_proba` output through `align_proba`; pass `confidence` when rules override predictions.
- M4: `reports/baselines.json` (`reports/baselines.md` is rendered from it); `src/baselines.py: make_pipeline, fit_model,
  predict_with_proba` — schema and signatures in docs/contracts.md#m4.
- M5a–M10: none verified yet. Do not consume `reports/experiments.json` until M5a is verified.

Run: `bash scripts/run_all.sh <target>`. Only `clean`, `audit`, `baselines`, `experiments` and `test` are
implemented; other targets exit 2.

## 5. Key numbers

Row counts, negative durations, ASCII `text_clean` and test `source` are covered by M1 tests, the M2 block by M2 tests.
The rest was measured on raw files and is not asserted by a test.

- Train: 10,822 rows, 2025-04-01 to 2026-06-30 (4,320 `legacy_zoho`, 6,502 `crm`). Test: 2,178 rows, all `crm`.
- Bot Billing precision: 64.7% of the 1,656 requests it sent to Billing were closed there.
- Transfers in train: 3,902 in total.
- Mojibake: 480 rows, all `legacy_zoho`. M1 repairs them; `text_clean` is pure ASCII in both files.
- Negative durations: 1,140 rows before the UTC fix, all `legacy_zoho`; 0 after adding 5h30m to those rows only.

M2 (`reports/audit_metrics.json`, all of train):

- Months covered: 15 (2025-04 to 2026-06).
- Bot baseline: agrees with `final_team` on 77.17% (8,351 / 10,822); 2,471 misrouted.
- Bot cost: Rs 122,171 per month (transfers + misroute contacts, licence excluded); Rs 114,994 from misrouted rows only.
- Vague requests: 8.1% (878 rows, rule-based estimate). Product named in text conflicts with `product_family`: 1,744
  rows (16.1%).

M3 (covered by M3 tests unless noted):

- Split: train 8,687 rows, validation 2,135 rows.
- `COST_PER_MISROUTE` Rs 698.06 (1.436 transfers each); `COST_PER_MISROUTE_SIMPLE` Rs 565 (1 transfer each).
- Bot misroute cost Rs 114,994 per month (2,471 x Rs 698.06 / 15 months, licence excluded). Same as M2's misrouted-only
  figure; the match was checked by hand, the M2 value is not asserted in M3 tests.
- `cost_rs` from `evaluate` is per evaluated set, not per month.

M4 (`reports/baselines.json`; validation accuracy against `final_team`, 95% bootstrap interval):

- Bot label 76.67% (74.89 to 78.36). Keyword rules 65.95% (64.07 to 67.96).
- TF-IDF trained on the final team 83.89% (82.30 to 85.39). Same model trained on the bot label 76.21% (74.47 to 77.99).
- Best baseline: `tfidf_final`, 7.21 points above the bot (paired interval 5.76 to 8.81); the number to beat in later modules.
- The training target alone is worth 7.68 points: trained on bot labels, the model copies the bot (95.64% agreement).
- Keyword rules changed once after validation had been seen (`technician`, train-count justified); see `PROGRESS.md`.
- Seen / unseen text: bot 80.28% / 76.27%, `tfidf_final` 84.98% / 83.77%.

M5a (`reports/experiments.json`, not verified): amended rule chose `logreg_c1_none` = (d), inner-val 84.78%, real-val 83.89%.
Original rule: `complement_nb_a0.1`, 85.01% / 82.58%, failed criterion 4.

## 6. Open questions / known risks

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
- FastAPI's test client may need `httpx`, which is not in `requirements.txt`.
- M2 tests check the headline lines of `data_audit.md` but not its table cells; read table values from the JSON.
- Request text contains order and registration numbers; M1 does not mask them. Decide in the feature module.
- M1 review findings are open (code unchanged since the review; list in `PROGRESS.md`). Relevant downstream: no test
  compares `team_label_canon`, `first_team_canon`, `misrouted` or test row order with raw, and running pytest rewrites
  `data/processed`, so rerun `clean` on known-good code after a failing test run.
- `CLAUDE.md` rule 2 bans old names after loading, but M1 keeps raw `team_label` by design; rule needs amending.
- Plain `python` on the dev machine is 3.10, too old for the pins; use `.venv` or `py -3.13`.
- Repo folder is spelled `kestral-routing`; the project name is `kestrel-routing`.
