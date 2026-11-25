# Progress

| Module | Status | Outputs | Key numbers | Decisions | Discarded |
|---|---|---|---|---|---|
| M0 setup | VERIFIED | `src/config.py`, `tests/conftest.py`, `scripts/run_all.sh`, `docs/contracts.md` | 9 tests pass; bot label = `final_team` on 77.17% of train | Split: train Apr 2025-Mar 2026, validation Apr-Jun 2026 (confirmed) | Makefile (`make` not available on the dev machine) |
| M1 clean | VERIFIED | `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet` | 10,822 / 2,178 rows; negative resolution_hours 1,140 -> 0; 12 tests pass | ftfy before lowercasing; fold Latin accents (`the`); pyarrow added; raw `team_label` kept, use `*_canon`; `transfers` as recorded; IDs in text not masked | Raw `first_team`, `final_team`, `resolved_at` in output; marker check on lowercased text alone (cannot fail) |
| M2 audit | VERIFIED | `reports/audit_metrics.json`, `reports/data_audit.md` | 48 tests pass; agreement 77.17%; 15 months; Rs 122,171/month (Rs 114,994 from misrouted rows only); 1,744 product conflicts (16.1%); 878 vague (8.1%, estimate) | Cost formula as specified, caveats in `cost_inputs`; text counts on `text_clean` with `_raw` variants; tests build into a temp folder; months, vague and conflict rules in M2 notes | Vague by length or "call back" (that phrase is mostly a suffix on specific requests); product pattern needing "air fryer" |
| M3 split + evaluation | VERIFIED | `src/split.py`, `src/evaluate.py`, cost constants in `src/config.py` | 8,687 / 2,135 rows; Rs 698.06 per misroute (simple Rs 565); bot misroutes Rs 114,994/month; 64 tests pass | Average transfers from misrouted rows only; literal counts in config; macro-F1 over all 7 teams; `align_proba` and `confidence` added; error messages give counts only | Rejecting `y_pred` that is not the row maximum of `proba` (blocks M5 rules); average over all 3,902 transfers (Rs 741.6) |
| M4 baselines | VERIFIED | `reports/baselines.json`, `reports/baselines.md`, `src/baselines.py` | Validation accuracy: bot 76.67%, rules 65.95%, (d) 83.89%, (e) 76.21%; (d) minus bot 7.21 points (paired 5.76 to 8.81); 57 tests pass | Train window is the one reference set for repeated text; JSON rounded to 6 decimals; `technician` removed from Repairs | "Likely optimistic" verdict on repeated text; exact-text check of the committed reports |
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

## M3 notes

**Verification.** Marked VERIFIED by the project owner after the review and fixes below. 64 M3 tests pass (133 in all).
Cross-check with M2, bot labels on all of train: agreement 77.17%, 2,471 misroutes, `cost_rs` / 15 = Rs 114,994 per
month (misrouted rows only), all equal to M2. 38 deliberate code breakages were run in a scratch copy: first run 36
caught. One miss was an invalid breakage (it edited a comment) and was replaced; the other (`n_boot` default) was a real
gap and now has a test. Final run: 38 of 38 caught. That run came before the privacy fix and was not repeated; instead,
putting value-printing messages back made 7 of the 10 new test cases fail (the other 3 cover paths not broken).

**Decisions** (project owner, plan review)

- **Average transfers** come from misrouted rows only: (3902 - 353) / 2471 = 1.436. The 126 misroutes with 0 transfers
  stay in.
- **Literal counts in config**, because `src/audit.py` imports config; a test ties them to the parquet and the JSON.
- **All 15 months** in the average: accepted as a reporting constant, not for choosing thresholds.
- **Q5, changed from the plan.** The plan had `evaluate` raise when `y_pred` is not a row maximum of `proba`, to catch
  misordered columns. The owner dropped that check: it would reject M5's rule-overridden predictions whenever `proba` is
  passed, so they could get no coverage curve. Instead, `align_proba` handles sklearn's alphabetical columns, and an
  optional `confidence` (default: row maximum) feeds the curve. Cost: a misordered `proba` array is not detected, so
  later modules must call `align_proba`.
- Bootstrap interval is not clamped to the point estimate.

**Beyond the spec:** `confidence` without `proba` gives a coverage curve and no log loss, instead of being ignored.
`time_split` raises on rows outside both windows.

**Review issues** (gaps found in the acceptance run)

| # | Issue | Status |
|---|---|---|
| 1 | Privacy: error messages printed offending values, so a text column passed by mistake would print customer text | Fixed: counts only, not truncated snippets, because even 20 characters can be customer text. Also fixed: numpy's own error quoted text passed as `proba` or `confidence`; it is replaced and kept out of the traceback |
| 2 | Split sizes not pinned, only their sum | Fixed: 8,687 and 2,135 asserted |
| 3 | Criterion 6 is tested with 300 resamples, not the default 1,000 | Skipped by the owner: the default is already covered by the call-count and interval-width tests |

## M4 notes

**Verification.** Marked VERIFIED by the project owner after the review and fixes below, with M2, M3, L3 and L4 deferred.
57 M4 tests pass (190 in all). Best baseline: `tfidf_final` (d).

| Baseline | Validation accuracy | 95% interval |
|---|---:|---:|
| (b) bot label | 76.67% | 74.89 to 78.36 |
| (c) keyword rules | 65.95% | 64.07 to 67.96 |
| (d) TF-IDF trained on the final team | 83.89% | 82.30 to 85.39 |
| (e) TF-IDF trained on the bot label | 76.21% | 74.47 to 77.99 |

(d) is the highest baseline but below the client's 90% bar, which is stated against the bot's labels.

**Finding, (d) against (e).** Same pipeline and features, only the training target differs. Trained on the final team it
reaches 83.89% on validation; trained on the bot's labels it reaches 76.21%, no better than the bot itself (76.67%), and
agrees with the bot on 95.64% of requests (against 80.52%). A model trained on the bot's labels mostly learns to copy
the bot, errors included; the 7.68 points come from the target alone. This supports `CLAUDE.md` rule 1.

**`technician` rule (review item M1).** Justified by train-window counts only: of 139 train requests with the word, 134
closed at Installs & Demo, 2 at Product Advice, 2 at Filters & Consumables and 1 at Repairs, while the rule sent 135 to
Repairs. Removing it moved the rules from 64.59% to 66.09% on train. **Validation had already been seen when this change
was made:** the review measured the rule on both windows. Validation went from 64.36% (761 misroutes) to 65.95% (727), so
the rules' validation score is not fully blind. No other rule was changed. (d), (e) and the bot are unaffected.

**Decisions**

- Reading `test_clean.parquet` is limited to `text_clean` overlap counts, as authorised in the contract.
- Repeated text: validation and test are both compared with the train window (9.98% and 10.15% seen). The earlier 11.94%
  for test used all labelled data as its reference.
- Rs 698.06 is kept as the headline cost; the report adds the M3 caveats and the Rs 565 sensitivity row.
- The paired interval resamples both baselines on the same rows with M3's `bootstrap_ci`.
- Exact-value tests now use the unrounded results; another test requires the JSON to equal them rounded to 6 decimals.

**Tried and discarded**

- The "likely optimistic / pessimistic" verdict and its tests: it compared a re-weighted accuracy with the interval, and at
  these seen shares could not realistically fall outside it. Seen and unseen accuracies are reported instead.
- Exact-text comparison of the committed JSON: replaced by a 1e-4 tolerance (one validation row moves accuracy by 4.7e-4)
  because model floats may differ between machines. The markdown must still equal the rendering of the committed JSON.

**Review issues.** 17 deliberate code breakages were run in memory after the fixes, ignoring
`test_committed_reports_current`: 15 caught, 2 not (items M2 and M3).

| ID | Issue | Status |
|---|---|---|
| H1 | No M4 contract | Fixed |
| H2 | Tests did not enforce (d) = (e) except the target | Fixed: parameters compared with sub-steps, settings pinned. Both already came from one `make_pipeline()` |
| H3 | No test that (c) ignores the bot label | Fixed, together with L2 |
| M1 | `technician` in the Repairs rule was wrong | Fixed (see above) |
| M2 | Source of the keyword fallback team is untested | Deferred |
| M3 | Only the bot's interval is recomputed | Deferred |
| M4 | Repeated-text verdict could not fail; two reference sets | Fixed |
| M5 | M3 cost caveats missing | Fixed |
| M6 | Committed reports compared as exact text | Fixed |
| L1 | Interval overlap used for paired predictions | Fixed: paired interval added, overlap wording kept |
| L2 | Forbidden-column test was weak | Fixed: scramble test added, old test kept |
| L3 | Privacy check skips test text | Deferred |
| L4 | Report prose and duplicate fields add complexity | Deferred |

Deferred by the project owner: test strictness and tidiness only, no effect on results.
