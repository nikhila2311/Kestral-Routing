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
| M3 | not started | |
| M4 | not started | |
| M5 | not started | |
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
- M3–M10: none verified yet.

Run: `bash scripts/run_all.sh <target>`. Only `clean`, `audit` and `test` are implemented; other
targets exit 2.

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

Model results: none yet.

## 6. Open questions / known risks

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
