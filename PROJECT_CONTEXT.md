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
| M1 | not started | |
| M2 | not started | |
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
- M1–M10: none yet.

Run: `bash scripts/run_all.sh <target>`. Only `test` is implemented; other targets exit 2.

## 5. Key numbers

Measured on the raw files during setup; not yet covered by a module test.

- Train: 10,822 rows, 2025-04-01 to 2026-06-30 (4,320 `legacy_zoho`, 6,502 `crm`). Test: 2,178 rows, all `crm`.
- Bot baseline: `team_label` = `final_team` on 77.17% (8,351 / 10,822), over all of train.
- Bot Billing precision: 64.7% of the 1,656 requests it sent to Billing were closed there.
- Transfers in train: 3,902 in total.
- Mojibake: 480 rows, all `legacy_zoho`. ftfy clears all 480, but 231 stray "é" remain, so fold accents after it.
- UTC check: 1,140 Zoho rows resolve before creation as stored; 0 after adding 5h30m.

Module results: none yet.

## 6. Open questions / known risks

- Ritu's 90% bar is stated against the bot's labels (~77% right); restate it against `final_team`.
- Ritu says "eighteen months" of labels; there are 15 (the other 3 months are the unlabelled test set).
- `transfers` disagrees with the teams: 351 rows have transfers although first = final team; 126 rows changed team
  with 0 transfers. Decide how to count before computing cost.
- Some requests cannot be routed without asking ("please call me about my purifier").
- Repeated text: 663 train rows repeat an earlier message; 258 test messages appear verbatim in train. Scores may
  look optimistic.
- Tuning on Apr–Jun 2026 would inflate the validation score; tune with time-based folds inside the train window.
- Farhan (Finance) wants the monthly run cost in writing before the switch; Ritu wants busiest teams for headcount.
- Parquet output needs `pyarrow`, which is not in `requirements.txt` yet. FastAPI's test client may need `httpx`.
- Plain `python` on the dev machine is 3.10, too old for the pins; use `.venv` or `py -3.13`.
- Repo folder is spelled `kestral-routing`; the project name is `kestrel-routing`.
