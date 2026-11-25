# Progress

| Module | Status | Outputs | Key numbers | Decisions | Discarded |
|---|---|---|---|---|---|
| M0 setup | VERIFIED | `src/config.py`, `tests/conftest.py`, `scripts/run_all.sh`, `docs/contracts.md` | 9 tests pass; bot label = `final_team` on 77.17% of train | Split: train Apr 2025-Mar 2026, validation Apr-Jun 2026 (confirmed) | Makefile (`make` not available on the dev machine) |
| M1 clean | VERIFIED | `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet` | 10,822 / 2,178 rows; negative resolution_hours 1,140 -> 0; 12 tests pass | ftfy before lowercasing; fold Latin accents (`the`); pyarrow added; raw `team_label` kept, use `*_canon`; `transfers` as recorded; IDs in text not masked | Raw `first_team`, `final_team`, `resolved_at` in output; marker check on lowercased text alone (cannot fail) |
| M2 | not started | | | | |
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
