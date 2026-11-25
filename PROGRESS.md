# Progress

| Module | Status | Outputs | Key numbers | Decisions | Discarded |
|---|---|---|---|---|---|
| M0 setup | VERIFIED | `src/config.py`, `tests/conftest.py`, `scripts/run_all.sh`, `docs/contracts.md` | 9 tests pass; bot label = `final_team` on 77.17% of train | Split: train Apr 2025-Mar 2026, validation Apr-Jun 2026 (confirmed) | Makefile (`make` not available on the dev machine) |
| M1 clean | VERIFIED | `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet` | 10,822 / 2,178 rows; negative resolution_hours 1,140 -> 0; 12 tests pass | ftfy before lowercasing; fold Latin accents (`the`); pyarrow added; raw `team_label` kept, use `*_canon`; `transfers` as recorded; IDs in text not masked | Raw `first_team`, `final_team`, `resolved_at` in output; marker check on lowercased text alone (cannot fail) |
| M2 audit | VERIFIED | `reports/audit_metrics.json`, `reports/data_audit.md` | 48 tests pass; agreement 77.17%; 15 months; Rs 122,171/month (Rs 114,994 from misrouted rows only); 1,744 product conflicts (16.1%); 878 vague (8.1%, estimate) | Cost formula as specified, caveats in `cost_inputs`; text counts on `text_clean` with `_raw` variants; tests build into a temp folder; months, vague and conflict rules in M2 notes | Vague by length or "call back" (that phrase is mostly a suffix on specific requests); product pattern needing "air fryer" |
| M3 split + evaluation | VERIFIED | `src/split.py`, `src/evaluate.py`, cost constants in `src/config.py` | 8,687 / 2,135 rows; Rs 698.06 per misroute (simple Rs 565); bot misroutes Rs 114,994/month; 64 tests pass | Average transfers from misrouted rows only; literal counts in config; macro-F1 over all 7 teams; `align_proba` and `confidence` added; error messages give counts only | Rejecting `y_pred` that is not the row maximum of `proba` (blocks M5 rules); average over all 3,902 transfers (Rs 741.6) |
| M4 baselines | VERIFIED | `reports/baselines.json`, `reports/baselines.md`, `src/baselines.py` | Validation accuracy: bot 76.67%, rules 65.95%, (d) 83.89%, (e) 76.21%; (d) minus bot 7.21 points (paired 5.76 to 8.81); 57 tests pass | Train window is the one reference set for repeated text; JSON rounded to 6 decimals; `technician` removed from Repairs | "Likely optimistic" verdict on repeated text; exact-text check of the committed reports |
| M5a model selection | VERIFIED | `reports/experiments.json`, `reports/experiments.md`, `src/experiments.py` | Original rule: `complement_nb_a0.1`, real-val 82.58%, failed criterion 4 (82.89%). Amended rule: `logreg_c1_none` = (d), real-val 83.89% (bot 76.67%) | Q1-Q8 as recommended; rule amended after a real-val result, see M5a notes | Added features, balanced class weights, calibrated LinearSVC (more complex), ComplementNB (log loss) |
| M5b router | VERIFIED | `src/router.py`, `reports/router.json`, `reports/router.md`, `models/` (git-ignored) | Auto-routed 81.55% at 96.32%; triaged 18.45% at 28.93%; Rs 49,039/month best case, 80,044 no triage, 114,191 fixes nothing, bot 115,878; break-even 52.41%; 101 M5b tests pass | Q1-Q5 as recommended, see M5b notes; scikit-learn, joblib, numpy pinned; wording revised twice, model unchanged | Threshold from `COST_PER_MISROUTE` (covers validation); character n-grams in reasons; question template per team pair |
| M6 evaluation | VERIFIED | `src/evaluation.py`, `reports/evaluation.json`, `reports/evaluation.md`, `reports/figures/*.png` | Validation 83.89% (82.30 to 85.39), bot 76.67%; expected hidden test 80.74% to 86.01%, or 80.52% if scored on bot labels; 38 M6 tests pass | Q1-Q7 as recommended, see M6 notes | Loading `models/router.joblib`; a combined cost total with the licence |
| M7 predictions | VERIFIED | `predictions.csv`, `reports/predictions_summary.json`, `reports/predictions_summary.md`, `models/router.joblib` (git-ignored), `src/predictions.py` | 2,178 rows; triaged 18.14%; vague 8.13%; largest gap Repairs +4.41 points; 25 M7 tests pass | Q1-Q9 as recommended, see M7 notes | Correcting volumes for the validation bias; lookup override for repeated texts; row-level confidence in the output |
| M8 service | VERIFIED | `app/main.py`, `app/static/index.html`, `README.md` | 56 M8 tests pass; 20 calls under 2 s | No LLM or external API; Q2 changed (unknown product accepted with a reason), others as recommended, see M8 notes | `/options` endpoint; default 422 messages |
| M9 documents | BUILT, not yet verified | `memo.md` | 6 M9 tests; memo 435 words | Q1-Q8 as recommended, see M9 notes | Number check by existence anywhere in the JSON |
| M10 integration + README | BUILT, not yet verified | `scripts/integration_test.ps1`, `scripts/integration_test.sh`, `pytest.ini`, `README.md`, `tests/test_m10_integration.py` | RESULT_LINE | Tests HEAD, not the working copy; temp folder deleted (customer data); M6 evaluate rerun (stale phrase lists, M8 notes) | Comparing `predictions.csv` with git diff (line-ending filters can hide a byte change) |

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
| L7 | Raw `team_label` keeps old names | Open; kept by design |
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
the bot, errors included; the 7.68 points come from the target alone. This supports `final_team` as target.

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

## M5a notes

**Status: rule amended by the project owner; VERIFIED.** The original rule chose `complement_nb_a0.1`, which
scores 82.58% on real validation, below the criterion 4 floor of 82.89% (M4 (d) 83.89% minus 1.0 point), and beats the bot
by 5.90 points (paired 4.31 to 7.54). Owner: accepted as a finding, evidence kept as `original_rule` in both reports. The
amended rule chooses `logreg_c1_none`, which is (d): 83.89%, 7.21 points above the bot (paired 5.76 to 8.81).

| Candidate | Inner-val accuracy | Inner-val log loss | Real-val accuracy | Real-val log loss |
|---|---:|---:|---:|---:|
| `complement_nb_a0.1` (original choice, failed criterion 4) | 85.01% | 0.6120 | 82.58% | 0.8115 |
| `logreg_c1_none` = (d) (amended choice) | 84.78% | 0.4550 | 83.89% | 0.4887 |
| `linear_svc_c0.1_isotonic` (highest) | 85.47% | 0.4892 | not scored | not scored |

**What went wrong.** Inner-val accuracy barely separates the candidates: the original choice beat (d) by 0.23 points (5 of 2,168
requests). It also has the worst inner-val log loss of the 15 candidates without added features, which the original rule
only used as a tie-break; (d) has the best of all 22. Four of the original choice's five largest real-val confusions send
other teams' requests to Billing.

**Amended rule** (inner-val numbers only). Within 1.0 point of the highest accuracy (18 of 22 left); drop log loss more than
0.1 above the best (8 left); lowest complexity (5 left); lowest log loss: `logreg_c1_none`. Reason: M5b triage depends on
calibrated probabilities; the original rule ignored calibration although it was a known requirement.

**Disclosure**

- The amendment was made after seeing one real-val result, and knowing which candidate it would select.
- As first proposed, the amendment kept accuracy before log loss in the tie-break. Checked on the stored numbers before
  coding, that picks `logreg_c10_none` (1 request of 2,168 ahead on inner-val; log loss 0.053 above the best, so within the
  limit), not the expected `logreg_c1_none`. The owner chose to put log loss first. `logreg_c10_none` and the LinearSVC
  candidates were never scored on real validation.
- The chosen model's real-val score was already seen in M4, so it is not an untouched estimate. It is (d), so criterion 4
  cannot fail for it.
- The three contenders (highest, original choice, amended choice) span 0.69 points on inner-val, within noise. Paired 95%
  intervals against `logreg_c1_none`, in points, measured once and not in the JSON: `linear_svc_c0.1_isotonic` -0.05 to
  1.38, `complement_nb_a0.1` -0.69 to 1.15. Exception: `linear_svc_c0.1_sigmoid` 0.14 to 1.15, which excludes zero, with no
  allowance for picking among 22 candidates.
- The amended choice has lower Repairs precision than the original one on real validation (77.49% against 83.33%): its five
  largest confusions all send other teams' requests to Repairs.

**Decisions** (project owner, plan review): complexity = added feature groups + 1 if calibrated; the base for the added
features is picked with the same rule; groups together and alone; extra grids for LinearSVC C and ComplementNB alpha;
contiguous unshuffled calibration folds; IVR x product conflict only; order id = `ko` + 7 digits (`sr` ids excluded); ids
in text are not masked; `experiments` run target.

**`technician` caveat (Q8).** Cue words come from policy section 3, `teams.csv` and the spec, and their counts were checked
on inner-train only. "technician" is the exception: it comes from the M4 finding, whose counts covered the whole train
window (inner-val included) and which was made after validation had been seen. It may flatter the `install` group and the
candidate with all groups. Neither was chosen, so it does not affect the chosen configuration.

**Discarded** (inner-val)

- Added features, against their ComplementNB base: at most +0.05 points alone, -0.46 with all 6 groups. All 7 cue candidates
  came within 1.0 point of the highest accuracy and none met the log loss limit. The selection tried them on
  ComplementNB only.
- Supplementary, report-only (owner request, inner-val only): the same 7 cue candidates on `logreg_c1_none` range from -0.42
  points (all 6 groups) to +0.09 (product conflict, 2 of 2,168 requests); log loss moves by at most 0.0024. The selection is
  unchanged: validation had already been consulted, so the selection is frozen. None were scored on real validation.
- `class_weight="balanced"`: lower accuracy and macro F-score than none at every C.
- Calibrated LinearSVC: up to 0.69 points above the chosen one, within the margin and one step more complex.

**Limits.** ComplementNB alpha 0.1 and LinearSVC C 0.1 are the lowest values tried, so the best may lie beyond the grid.

## M5b notes

**Decisions** (project owner, plan review). Q1: M5a verified. Q2: triage threshold from the train window
(`COST_PER_MISROUTE_TRAIN` 699.49, threshold 0.6283) instead of 698.06 (0.6275), which covers validation months; 698.06 stays
for reporting. Q3: bot cost on the same validation rows (498 misroutes, Rs 115,878 per month); the 15-month Rs 114,994 is not
comparable. Q4: "clarified requests are never misrouted" is the headline, with no-triage and fixes-nothing beside it. Q5: cue
reasons in neutral wording. Addition: scikit-learn, joblib and numpy pinned; `load` refuses a version mismatch.

**Findings**

- Triage only beats no triage if clarification fixes 52.41% of triaged misroutes. With none fixed, the router costs about as
  much as the bot (Rs 114,191 against Rs 115,878 per month).
- Triaged requests are nearly unroutable from the text: 28.93% right at a mean confidence of 32.96%. All 173 validation messages
  the M2 rule calls vague are triaged.
- Invented §3 cases: none of the three "paid" messages about another problem went to Billing; both payment problems did.

**Wording changes** (project owner, after seeing invented example outputs). Reasons and questions only: team, confidence, top3,
triage and model version were identical before and after on all 2,135 validation rows, and a test freezes 500 of them.

| Change | Why |
|---|---|
| One question builder from a plain phrase per team replaced 10 pair templates | The top-two pair asked a vague request about a payment nobody mentioned |
| General question when vague or confidence < 0.40; otherwise teams with probability >= 0.10, at most 3 | A two-way question hid a likely third team ("want replacement") |
| At most 3 phrases, no lone "not" or function word, no word in two phrases | 'not' alone and 'grinder stopped' told an agent nothing |
| A longer phrase wins with at least 65% of its word's score, or always if it adds a negation | 80% was tried first: "not working" (69%) and "stopped working" (68%) lost to 'working', which says the opposite |
| §3 line also when the team is not Billing | Explains why "I paid" did not go to Billing |
| Agent wording in reasons, customer wording only in the question | "you're waiting" is wrong when an agent reads it |
| "may be that" instead of "looks like" when triaged | A 36% guess should not sound certain |

**Open.** README: `Router.load`'s message says "see README", which does not exist yet; until M7 the rebuild step is
`bash scripts/run_all.sh router`. How often clarification works is unknown (ask operations). Wording is untested with
customers and agents.

**Verification.** Marked VERIFIED by the project owner. 101 M5b tests pass. The full suite (333 passed, 11 min 2 s) was last run
before the wording changes. Deliberate breakages in a scratch copy: all 20 of the first build and 9 of 10 for the wording
were caught; the miss (the longer-phrase rule switched off) now has a test that catches it.

**Discarded.** Character n-grams in reasons (unreadable); bigrams that skip a word or punctuation (not literally in the text);
checking the committed report text exactly (as in M4).

## M6 notes

**Decisions** (project owner, plan review). Q1: refit the router in memory and require the `router.json` model version. Q2: both
overlapping and exclusive failure categories. Q3: quoted phrases only if in at least 10 different train texts. Q4: expected range
= CI half-widths x square root of two, low end minus the 0.89-point drift. Q5: bot-label case from agreement, 80.52%. Q6: test
`text_clean` read for shares only. Q7: licence on its own line. Addition: one sentence on the lead over the bot by month.

**Findings**

- The requested sentence said the lead stays roughly constant. Measured leads are 6.17, 7.36 and 8.10 points (Apr, May, Jun): the
  paired intervals all overlap, so a constant lead is compatible with the data, but the point estimates rise every month. The
  report says both, and the wording is built from the values.
- The router's accuracy falls each month (85.08%, 84.33%, 82.24%) and so does the bot's (78.91%, 76.98%, 74.15%). Why is not known.
- The bot is better on precision for Installs & Demo, Returns & Replacement, Warranty Claims and Product Advice, and on recall
  for Filters & Consumables (validation, every request counted).
- Vague requests and close calls are 230 of the 344 misroutes (counted once each). IVR is a weak driver (17.44% misrouted
  against 16.11% overall).
- Confidence between 0.2 and 0.5 is too high (for example 46.05% mean against 30.77% right, 39 requests).
- If clarification fixes nothing, the router saves only Rs 1,687 per month against the bot, licence left out.

**Verification.** Marked VERIFIED with the M7 plan approval (Q9). 12 deliberate breakages in a scratch copy were all caught (4 of them by the
build's own consistency check). JSON key order once changed the markdown; the renderer now uses `meta.teams`.

**Open.** 83.89% is not an untouched estimate (M5a). The drift allowance is rough: the inner model saw fewer months.

## M7 notes

**Decisions** (project owner, plan approved without changes). Q1: the M5b default folder is `models/train_window/`, and
`load_final` refuses a model not fitted up to 2026-06-30. Q2: auto-routed and triaged volumes shown separately, with the
validation bias per team (M4 `baselines.json`); no correction. Q3: `predictions.csv text eol=lf`. Q4: the build never runs git;
`predictions.csv` staged once. Q5: byte identity promised on one machine with the pinned versions. Q6: triaged rows get the
first choice. Q7: 83.89% is stated as the train-window figure. Q8: limits are not changed if criterion 6 fails (it passed).
Q9: M6 marked VERIFIED, as the commit message already said.

**Findings**

- Repairs is predicted for 27.73% of test against 23.33% of Apr-Jun closing teams (+4.41 points). The same configuration
  over-predicted it on validation by 3.93 points (582 against 498), so most of the gap may be bias rather than volume.
- 157 of the 604 Repairs predictions are triaged first guesses.
- All 177 vague test requests are triaged. Triage share by month: 16.98%, 18.62%, 18.83%.
- `models/router.joblib` held the M5b train-window model before this module; `train` replaced it.

**Verification.** Marked VERIFIED by the project owner after two fixes: auto-routed-only volume added to the summary
(triaged as one separate count; predictions unchanged), and the key-number detail below.

**Open.** Byte identity across machines is not tested. No deliberate breakages were run for M7. `predict` fails on
Windows while `predictions.csv` is open in Excel (the old file is kept).

## Interfaces and key numbers (at the close of M7)

### Interfaces

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
- M5a: `reports/experiments.json` (`reports/experiments.md` is rendered from it), chosen configuration in `chosen.config`;
  `src/experiments.py: build_model, cue_features` — schema in docs/contracts.md#m5a.
- M5b: `src/router.py: Router` (`train`, `save`, `load`, `predict`); `models/train_window/router.joblib` +
  `router_meta.json` (git-ignored, train window only; since M7 `models/router.joblib` is the final model); `reports/router.json` (`reports/router.md` is rendered from it);
  `config.TRIAGE_THRESHOLD` — interface and schema in docs/contracts.md#m5b. Load only our own joblib file.
- M6: `reports/evaluation.json` (`reports/evaluation.md` is rendered from it) — schema in docs/contracts.md#m6.

### Key numbers

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

M4 (`reports/baselines.json`; validation accuracy against `final_team`, 95% bootstrap interval):

- Bot label 76.67% (74.89 to 78.36). Keyword rules 65.95% (64.07 to 67.96).
- TF-IDF trained on the final team 83.89% (82.30 to 85.39). Same model trained on the bot label 76.21% (74.47 to 77.99).
- Best baseline: `tfidf_final`, 7.21 points above the bot (paired interval 5.76 to 8.81); the number to beat in later modules.
- The training target alone is worth 7.68 points: trained on bot labels, the model copies the bot (95.64% agreement).
- Seen / unseen text: bot 80.28% / 76.27%, `tfidf_final` 84.98% / 83.77%.

M5a (`reports/experiments.json`): amended rule chose `logreg_c1_none` = (d), inner-val 84.78%, real-val 83.89%.
Original rule: `complement_nb_a0.1`, 85.01% / 82.58%, failed criterion 4.

M5b (`reports/router.json`, validation; M5b tests recompute the report, the values are not hard-coded):

- Auto-routed 81.55% (1,741) at 96.32% accuracy. Triaged 18.45% (394): accuracy 28.93%, mean confidence 32.96%.
- All 173 of 173 vague requests (M2 rule) are triaged. Threshold 0.6283, from the train-window misroute cost, not tuned.
- Rs per month: 49,039 best case (clarification always works), 80,044 no triage, 114,191 clarification fixes nothing, bot
  115,878 on the same rows (licence 26,667 not included). Break-even clarification rate 52.41%.
- Repairs precision 97.35% on auto-routed, 77.49% on all.

M6 (`reports/evaluation.json`, validation; headline numbers equal the M4/M5a/M5b values):

- Expected hidden-test accuracy 83.89%, range 80.74 to 86.01%; 80.52% (78.20 to 83.03) if labels are the bot's.
- Accuracy Apr, May, Jun: router 85.08, 84.33, 82.24%; bot 78.91, 76.98, 74.15%; lead 6.17, 7.36, 8.10 (intervals overlap).
- Vague ceiling 93.61%. 230 of 344 misroutes are vague or close calls. Clarification fixing nothing: Rs 1,687/month under bot.

M7 (`reports/predictions_summary.json`, test):

- Triaged 395 of 2,178 (18.14%); vague 177 (8.13%), all triaged. Largest gap to the Apr-Jun 2026 closing teams:
  Repairs, +4.41 points (604 predicted, 447 auto-routed); over-predicted by 3.93 points on validation.

## M8 notes

**Decision (project owner): no LLM, no external API.** No per-request AI cost (Farhan, Finance); no evidence that an LLM would
be more accurate; policy section 10 restricts sharing customer data.

**Decisions** (project owner, plan review). Q1: field values in config, copied from the final model's categories, not from the
data. Q2, changed from the plan: strict 422 for channel and warranty only; an unknown product is accepted and the reason
"product not recognised; routed without the product" is added (first wording "on the text only" was wrong: channel and
warranty still count). Q3: server fills one placeholder in the page. Q4: `load_final`.
Q5: a test counts example texts equal to a real message (0; the spec's purifier text is exempt). Q6: close-call text chosen by
trying 18 invented texts on the final model. Q7: triaged team labelled as a first guess. Q8: `/docs`, `/redoc` off. Q9: fixed
500, class-only log, `no-store`. Q10: odd text is accepted and triaged. Q11: window as "first to last". Q12: tests fail if the
model file is missing. Q13: README covers setup and running only.

**Findings**

- FastAPI passed a lone surrogate in `request_text` through as valid text; it is now refused with 422, as planned.
- For "stopped working", a reason could be the lone word 'working' next to 'grinder stopped'. Fixed in `src/router.py`
  (`_merge_phrases`): chosen phrases next to each other become one quoted span. Team, confidence, top3, triage and question are
  identical before and after on all 4,313 validation and test rows, for the final and the train-window model; the M5b freeze
  test is unchanged. `test_phrases_are_the_strongest_pushes` now takes the unmerged phrases from `_phrases` and also checks that
  the reasons are their merge. The train-window model already chose 'stopped working' for this text.
- `reports/router.json` and `.md` do not change. The phrase lists in `reports/evaluation.json` and `.md` (M6) do, and were stale
  until `bash scripts/run_all.sh evaluate` was run in M10; no other M6 value changed (M6 `test_committed_reports_current` failed until then).
- Examples: repair 98.9%, installation 80.9%, paid / technician 99.1% Installs & Demo, purifier 32.9% triaged, close call
  Product Advice against Repairs, triaged.

**Verification.** Marked VERIFIED by the project owner. 56 M8 tests pass. The page was opened in headless Chrome: examples render, and markup typed into the
request is shown as text.

**Discarded.** An `/options` endpoint; reading the dropdown values from the data; FastAPI's default 422 (repeats the input).

**Open.** Starlette warns that `httpx` in its test client is deprecated in favour of `httpx2`: harmless, left as pinned.
The page was not tried in other browsers.

## M9 notes

**Decisions** (project owner, plan review). Q1: files in the repo root. Q2: proposed switch-off conditions, on closed requests:
router at least 3 points above the bot, at least 93% of straight-on requests right, Meenal agrees. Q3: switch judged at two
weeks, the question at two months (about 60 flagged requests in two weeks, about 43 with a wrong first guess). Q4: server Rs
2,000-4,000 a month and one day's retraining a quarter are estimates, not measurements. Q5: derived numbers by formula. Q6: the
cooktop text is typed in the recording. Q7: Claude chat and Claude Code. Q8: allowlist.

**Corrections to the spec** (accepted)

- Rs 35,834 a month assumes no question is asked; with the question the saving runs from Rs 1,687 to Rs 66,839.
- The 52.4% break-even is over flagged requests whose first guess was wrong, not over all flagged requests.
- Bot labels overstate Repairs by 23%, not 25%; Product Advice, understated by 15%, was missing.
- The copy of the bot is below the router, but level with the bot (76.2% against 76.7%).

**Wording.** "No outside AI service" is stated about the router, not about how the repo was built. "Every vague request" is by
the M2 word-list rule, on Apr-Jun 2026.

**Findings.** The cooktop text is also the "Close call" example button. On the final model it is triaged: Product Advice first,
Repairs second.

**Discarded.** "Appears in some JSON": most two-digit numbers do, so a wrong number would pass.
