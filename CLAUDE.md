# kestrel-routing

**At the start of every session read PROJECT_CONTEXT.md first. Only open other files when the current module's
contract lists them as inputs.**

Classifier that routes Kestrel Home service requests to the right team, replacing the vendor routing bot.
Every step in this repo must follow the rules below. Sources: `data/raw/ops-policy.pdf` (v4.1, cited as §),
`data/raw/README.txt`, `data/raw/email-thread.txt`, `data/raw/teams.csv`.

## Rules (non-negotiable)

1. **Target is `final_team` from `resolution_log.csv`, NOT `team_label`.** `team_label` is the bot's guess at
   creation and agrees with `final_team` on only ~77% of requests (measured: 77.17%, 8,351 / 10,822). Use bot
   labels only as a baseline to beat: never as target, never as a feature (test has no `team_label`).
   `first_team`, `transfers` and `resolved_at` are only known after the fact, so they are not features either.
2. **Team renames on 15 Jan 2026 (§5):** Installations -> Installs & Demo, Consumables -> Filters & Consumables.
   Responsibilities did not change. Canonical names everywhere are the CURRENT 7 listed below; map old names to
   them at load time. Nothing downstream of loading, including predictions, may contain an old name.
3. **Billing only when the problem IS the payment** (§3: invoice, GST, double charge, refund of a payment, EMI
   conversion; `teams.csv` adds coupons). "I paid" / "paid for installation" is NOT Billing by itself.
4. **`legacy_zoho` rows (§9):** `request_text` can be mojibake, so repair it (ftfy). `resolved_at` is stored in UTC,
   not IST, so add 5h30m before any duration calculation. `created_at_ist` is IST for all rows. `crm` rows need
   neither fix.
5. **Costs (§4):** Rs 305 per transfer; Rs 260 extra customer contact per misroute; bot licence Rs 3.2 lakh
   (320,000) per year. (§4 also lists a technician visit at Rs 540.)
6. **Validation must be time-based.** Data is chronological and test is the most recent 3 months. Train on earlier,
   validate on later months: train Apr 2025-Mar 2026, validation Apr-Jun 2026, test Jul-Sep 2026. No random splits
   or shuffled K-fold. Fit vectorisers, encoders, thresholds and other
   preprocessing on the training window only.
7. **Must run on a clean machine with no paid API key.** Local models only; no hosted LLM or other paid service.
   Inference must not depend on network access.
8. **Data handling (§10):** customer and operational data must not be published. `data/raw/` is git-ignored; of the
   data, **only `predictions.csv` (repo root) is committed**. Do not paste customer request text or row-level
   extracts into reports, test fixtures, notebook outputs or app examples; use invented text there. Commit only
   aggregates. Trained models can embed customer text (vocabularies), so they are git-ignored and rebuilt locally.

9. **Fixed random seed 42 everywhere** (`src/config.py: SEED`).

## Workflow (modules M1-M10)

- Work is split into modules M1-M10. Implement **only the module the user asks for**.
- Each module has inputs, outputs (exact paths + schema), acceptance criteria and a test file
  `tests/test_mN_<name>.py`, recorded in `docs/contracts.md`. A module is done only when its tests pass.
- A module may only read inputs listed in its contract: raw data or outputs of VERIFIED modules.
- **Never weaken tests to make them pass.**
- Constants (paths, teams, name map, costs, seed, split dates) come from `src/config.py`; do not redefine them.
- When a module changes state, update `PROJECT_CONTEXT.md` (snapshot), `PROGRESS.md` (decisions) and the status in
  `docs/contracts.md`.
- Run steps with `bash scripts/run_all.sh <target>`: clean, audit, baselines, train, evaluate, predict, serve, test.
  `clean` means data cleaning. Unbuilt targets exit non-zero.

## Canonical teams (use these exact strings)

Installs & Demo · Repairs · Filters & Consumables · Billing · Returns & Replacement · Warranty Claims · Product Advice

## Data (checked against the pack)

| File | Rows | Notes |
|---|---|---|
| `train.csv` | 10,822 | 2025-04-01 to 2026-06-30, sorted by time. `legacy_zoho` 4,320 (until 30 Sep 2025), `crm` 6,502 |
| `resolution_log.csv` | 10,822 | joins 1:1 to train on `request_id` |
| `test_unlabelled.csv` | 2,178 | 2026-07-01 to 2026-09-30, all `crm`, no `team_label` |
| `sample_submission.csv` | 2,178 | `request_id,team`, same order as test; `predictions.csv` must match this format |

- No nulls in any of these four files; `request_id` is unique.
- Old and new team names split cleanly at 15 Jan 2026 by `created_at_ist`.
- Mojibake: 480 `legacy_zoho` rows contain non-ASCII text; `crm` and test have none. Some damage is double-encoded,
  and some repairs to an accented letter in an ordinary English word, so fold accents after ftfy.
- UTC check: 1,140 `legacy_zoho` rows resolve before creation as stored; 0 do after adding 5h30m.
- `team_label` equals `first_team` on every row.
- Repeated text: 663 train rows duplicate an earlier `request_text`; 258 test texts appear verbatim in train.
  Time-based validation scores should be read with this in mind.

## Open questions (raise, don't silently resolve)

- The client email says "eighteen months of labelled requests"; train covers 15 (the 18 months include test).
- The client's "90%+" bar is stated as matching the labels, which are the bot's output (~77% right). The bar
  should be restated against `final_team`.
- `transfers` is not fully consistent with the team columns: 351 rows have `first_team` = `final_team` but
  transfers >= 1, and 126 rows changed team with transfers = 0. State how costs are counted when using it.
- Some requests are unroutable without asking ("please call me about my purifier", per the service desk manager).
- Finance wants the monthly run cost in writing; operations wants to know the busiest teams for headcount.

## Layout

- `data/raw/`: data pack, git-ignored, read-only (never edit in place)
- `src/`: pipeline code · `notebooks/`: exploration · `reports/`: write-ups and figures (aggregates only)
- `app/`: FastAPI service · `tests/`: pytest · `scripts/run_all.sh`: task runner
- `docs/contracts.md`: module contracts · `PROGRESS.md`: module status and decisions
- `predictions.csv`: final output, committed

## Setup (Python 3.12+; developed on 3.13)

```
py -3.13 -m venv .venv             # Windows; macOS/Linux: python3 -m venv .venv (needs 3.12+)
.venv\Scripts\activate            # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then copy the data pack files into `data/raw/` (they are not in git). Run commands from the repo root.
