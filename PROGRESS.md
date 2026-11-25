# Progress

| Module | Status | Outputs | Key numbers | Decisions | Discarded |
|---|---|---|---|---|---|
| M0 setup | VERIFIED | `src/config.py`, `tests/conftest.py`, `scripts/run_all.sh`, `docs/contracts.md` | 9 tests pass; bot label = `final_team` on 77.17% of train | Split: train Apr 2025-Mar 2026, validation Apr-Jun 2026 (confirmed) | Makefile (`make` not available on the dev machine) |
| M1 clean | VERIFIED | `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet` | 10,822 / 2,178 rows; negative resolution_hours 1,140 -> 0; 12 tests pass | ftfy before lowercasing; fold Latin accents (`the`); pyarrow added; raw `team_label` kept, use `*_canon`; `transfers` as recorded; IDs in text not masked | Raw `first_team`, `final_team`, `resolved_at` in output; marker check on lowercased text alone (cannot fail) |
| M2 audit | VERIFIED | `reports/audit_metrics.json`, `reports/data_audit.md` | 48 tests pass; agreement 77.17%; 15 months; Rs 122,171/month (Rs 114,994 from misrouted rows only); 1,744 product conflicts (16.1%); 878 vague (8.1%, estimate) | Cost formula as specified, caveats in `cost_inputs`; text counts on `text_clean` with `_raw` variants; tests build into a temp folder; months, vague and conflict rules in M2 notes | Vague by length or "call back" (that phrase is mostly a suffix on specific requests); product pattern needing "air fryer" |
| M3 | not started | | | | |
| M4 | not started | | | | |
| M5 | not started | | | | |
| M6 | not started | | | | |
| M7 | not started | | | | |
| M8 | not started | | | | |
| M9 | not started | | | | |
| M10 | not started | | | | |

## M1 notes

**Verification.** Marked VERIFIED by the project owner after the review below; the reviewer had advised waiting for
H1-H3. 12 M1 tests pass. Checked by hand: in both outputs, row order and passthrough columns match raw and there are no
nulls; `text_clean` in train equals `clean_text(request_text)`; bot label = `final_team` on 8,351 rows, 2,471 misrouted.
35 deliberate code breakages were run in a scratch copy: 13 caught, 22 not (some overlap or do not affect this data).

**Tried and discarded**

- Removing the `clean_text` loop was tried in a scratch copy only. Pass 1 is already final on all 13,000 rows, and later
  passes can damage text (`€ß™` -> `ۧtm`), but without the loop 891 of 200,000 adversarial strings were not idempotent.
  The loop is still in the code (H3).
- The first versions of two breakages were invalid (files left by an earlier breakage; shift keyed on the wrong date) and
  were replaced by corrected ones. The tests missed both corrected ones.

**Review issues.** Fixed in code or tests: none; both files are unchanged since the review. No reason for skipping was
given, so the issues are recorded as open, not rejected.

| ID | Issue | Status |
|---|---|---|
| H1 | Most output columns and test row order are never compared with raw | Open |
| H2 | Tests overwrite `data/processed`; stale or broken outputs can remain | Open |
| H3 | `clean_text` loop can corrupt text; its comment about nested entities is wrong | Open |
| M1 | Unknown `source` is silently treated as IST | Open |
| M2 | Tests cannot tell a source-based shift from a date-based one | Open |
| M3 | Lenient timestamp parsing; blank `resolved_at` is not rejected | Open |
| M4 | Input guards are untested | Open |
| M5 | Idempotence test is nearly tautological | Open |
| M6 | Interface and contract differed | Partly fixed: contract wording and `build()` keys; helper names unchanged |
| M7 | `IST_OFFSET`, `LEGACY_SOURCE` are not in `src/config.py` | Open |
| L1 | Invisible characters in source literals | Open |
| L2 | Zero-width-only text passes the blank check | Open |
| L3 | A failing test can print customer text | Open |
| L4 | Redundant summary fields and `print` in `build()` | Open |
| L5 | One assert tests raw data, not `clean.py` | Open |
| L6 | Speculative dash and NFC handling | Open |
| L7 | Raw `team_label` keeps old names (`CLAUDE.md` rule 2) | Open; kept by design, rule needs amending |
| - | No test that files are written only to `data/processed` | Open |

## M2 notes

**Verification.** Marked VERIFIED by the project owner after the review and fixes below. 48 M2 tests pass (69 in all).
During the review, agreement, cost, months, volumes and confusion were also recomputed from the raw files and matched.

**Decisions**

- **Months = 15**, the full range `config.TRAIN_START` to `config.VALID_END`, so an empty month still counts in the cost
  divisor.
- **Vague rule:** no words of `text_clean` are left after removing the listed filler and product words; digits and
  punctuation are ignored. 878 rows (8.1%), an estimate.
- **Conflict rule:** `text_clean` names at least one other product family and not the row's own (whole-word match).
  1,744 rows (16.1%).
- **Cost:** the headline uses `transfers` as recorded. `bot_cost_misrouted_only_rs_per_month` leaves out the 353 transfers
  on correctly labelled rows. The 126 misroutes with no recorded transfer count only the contact cost.

**Background**

- **Product conflicts: 1,744, not the 1,507 in the approved plan.** The plan's 1,507 came from a pattern that matched
  "air fryer" but not a bare "fryer". With the rule as worded, synonyms change nothing (1,744 with or without them).
- The bot agrees with the final team on about 20% of vague rows (measured once while planning; not in the JSON).
- The vague word lists were chosen from this data, so the count moves with them. 28 of the 89 words never occur in train
  and 47 are used by vague rows; no word is needed only by Apr-Jun 2026 rows.

**Tried and discarded**

- Counting months with `nunique()` on `created_month`: skips an empty month and would inflate the cost.
- `bot_cost_per_month(df)`: a second copy of the cost computation, removed.
- A count test that called the module's own rule functions: circular, replaced by independent word lists.
- 3-word runs in the leak test: 7 matches with train `text_clean` on the pre-fix report (cause not checked), 4-word runs
  gave 0.
- A content hash alone to show files are unchanged: misses a rewrite with the same bytes, so modification time is
  compared too.
- One test sentence written as invented text turned out to equal a real message; replaced.

**Review issues** (numbers from the M2 review). Skipped items were not in the owner's fix list; no other reason was given.
After the fixes, 17 deliberate code breakages were run in a scratch copy: 13 caught, 4 not (items 1 and 3). "Caught"
ignores `test_committed_reports_current`, which passes again once the reports are regenerated.

| # | Issue | Status |
|---|---|---|
| 1 | Markdown numbers checked for existence, not position | Partly fixed (light version): headline lines only; three table breakages still missed |
| 2 | Leak test only caught whole messages | Fixed: 4-word runs |
| 3 | Conflict and vague counts were circular | Fixed; dropping the unused pattern `mixies?` is still missed |
| 4 | Which files are written was untested | Fixed |
| 5 | Months not tied to config dates | Fixed |
| 6 | Old-name guard and confusion total untested | Fixed |
| 7 | Inaccurate report wording | Fixed |
| 8 | Cost computed twice | Fixed |
| 9 | Words and patterns that match nothing | Skipped |
| 10 | "Chosen by looking at this data" is only partly true | Skipped |
| 11 | Report says cleaned text "is what models see" | Skipped |
| 12 | JSON written before markdown was rendered | Fixed |
| 13 | `REPORTS_DIR` default is bound at import | Skipped |
| 14 | `recomputed` docstring overclaims independence from M1 | Skipped |
| 15 | Test called a helper that is not in the contract | That call went with item 8; the item 6 test calls `team_volume`, also not in the contract |
| 16 | Cost literal scan is redundant | Skipped |
| 17 | Seed hardcoded in a test | Fixed |
