# Module contracts

A module may only read the inputs listed in its contract: raw data or outputs of VERIFIED modules.
Each contract states inputs, outputs (exact paths + schema), acceptance criteria and its test file
(`tests/test_mN_<name>.py`). Workflow rules are in `CLAUDE.md`.

## M1

status: VERIFIED

Data cleaning. Code `src/clean.py`, tests `tests/test_m1_clean.py`, run `bash scripts/run_all.sh clean`.

**Inputs:** `data/raw/train.csv`, `data/raw/test_unlabelled.csv`, `data/raw/resolution_log.csv`

**Interface**

- `clean_text(s: str) -> str` (pure, idempotent): ftfy, NFKD, drop combining marks U+0300-U+036F, NFC, dashes to `-`,
  lowercase, collapse whitespace; repeated until stable, at most 10 passes. NFKD also changes compatibility characters
  (`½`, `™`) and non-Latin accents (`й`); `ø`, `ß` and zero-width spaces are kept.
- `build() -> dict` writes both outputs. Keys: `train_rows`, `test_rows`, `negative_before`, `negative_after`.

**Output `data/processed/train_clean.parquet`** (10,822 rows, raw order, no nulls)

Types as read by pandas 3: string = `str`, datetime = `datetime64[us]`, int = `int64`, float = `float64`.

| Column | Type | Notes |
|---|---|---|
| request_id | string | unique |
| created_at_ist | datetime | timezone-naive IST |
| channel, product_family, warranty_status, request_text, source | string | as in raw; `source` is `legacy_zoho` or `crm` |
| team_label | string | raw bot label, keeps old names, traceability only; use `team_label_canon` |
| text_clean | string | `clean_text(request_text)`, never empty |
| team_label_canon, final_team_canon, first_team_canon | string | current 7 names only; target = `final_team_canon` |
| transfers | int | as recorded, not reconciled with the team columns |
| resolved_at_ist | datetime | timezone-naive IST; `legacy_zoho` shifted +5h30m |
| resolution_hours | float | `resolved_at_ist - created_at_ist` in hours, never negative |
| misrouted | bool | `team_label_canon != final_team_canon` |
| created_month | string | `YYYY-MM`, from `created_at_ist` |

**Output `data/processed/test_clean.parquet`** (2,178 rows, raw order, no nulls): `request_id`, `created_at_ist`, `channel`,
`product_family`, `warranty_status`, `request_text`, `source`, `text_clean`, `created_month`; types as above.

**Acceptance criteria**

1. Row counts 10,822 / 2,178; `request_id` unique in both; exactly one resolution row per train id.
2. `*_canon` columns contain only the 7 current names.
3. `text_clean` has no mojibake markers (either case) and no empty strings.
4. `clean_text("thÃ© mcb")` is readable; `clean_text` is idempotent.
5. Negative `resolution_hours` after the timezone fix = 0 (1,140 before).
6. Test set is all `crm`, created Jul-Sep 2026.

## M2

status: VERIFIED

Data audit. Code `src/audit.py`, tests `tests/test_m2_audit.py`, run `bash scripts/run_all.sh audit` (run `clean` first:
M1's tests rewrite the parquet). No models, no train/validation split.

**Inputs:** `data/processed/train_clean.parquet`, `data/processed/test_clean.parquet` (M1)

**Interface:** `build(out_dir=REPORTS_DIR) -> dict` writes both outputs; `compute_metrics(train, test) -> dict` and
`render_markdown(metrics) -> str` are pure; rules are `product_conflict_mask(df)` and `vague_mask(text)`.

**Output `reports/audit_metrics.json`** (aggregates only, all of train, full precision, sorted keys)

22 top-level keys, no others. Shares are 0-1 floats, counts are ints.

| Key | Content |
|---|---|
| bot_vs_final_agreement | share, `team_label_canon` = `final_team_canon` |
| agreement_by_month | `YYYY-MM` -> share; a month without rows is absent |
| agreement_by_source | source -> share |
| agreement_by_channel | channel -> share |
| confusion_bot_to_final | bot -> final -> count, zero-filled; total = train rows |
| transfers_distribution | transfers (string key) -> rows |
| transfer_consistency | `misrouted_with_0_transfers`, `correct_with_transfers` (row counts) |
| final_volume_by_team | team -> rows, zero-filled |
| bot_label_volume_by_team | team -> rows, zero-filled |
| final_team_volume_by_month | `YYYY-MM` -> team -> rows; every month in `months_covered`, zero-filled |
| product_text_conflict_count | rows naming another product family and not their own |
| product_text_conflict_rule | rule text |
| product_text_conflict_patterns | product family -> pattern |
| duplicate_text_count | train rows repeating an earlier `text_clean` |
| duplicate_text_count_raw | same, on `request_text` |
| test_text_overlap_count | test rows whose `text_clean` appears in train |
| test_text_overlap_count_raw | same, on `request_text` |
| vague_request_estimate | `count`, `rule`, `filler_words`, `product_words`, `final_team_distribution` |
| bot_cost_rs_per_month | (transfers x transfer cost + misroutes x contact cost) / months; no licence |
| bot_cost_misrouted_only_rs_per_month | same formula, counting only transfers on misrouted rows |
| cost_inputs | `rs_per_transfer`, `rs_per_misroute`, `transfers`, `misroutes`, `transfers_on_correct_rows`, `transfers_on_correct_rows_rs_per_month`, `licence_rs_per_year`, `licence_rs_per_month` |
| months_covered | `first`, `last`, `count`: full range `config.TRAIN_START` to `config.VALID_END` |

**Output `reports/data_audit.md`**: rendered from the JSON only.

**Acceptance criteria**

1. All keys exist in the JSON.
2. `bot_vs_final_agreement` and `bot_cost_rs_per_month` recomputed from the parquet equal the JSON values.
3. Agreement is between 0.70 and 0.85.
4. Both volume-by-team dicts sum to the train row count; only the 7 canonical names appear.
5. Every number in `data_audit.md` appears in `audit_metrics.json` (allowing rounding).
6. Costs come from `src/config.py`, not hardcoded.

## M3

status: VERIFIED

Time-based split and evaluation harness. Code `src/split.py`, `src/evaluate.py`, cost constants in `src/config.py`, tests
`tests/test_m3_harness.py`. Imported by later modules: no files written, no run target, no models.

**Inputs:** none at run time (callers pass data). `data/processed/train_clean.parquet` (M1) and
`reports/audit_metrics.json` `cost_inputs` (M2) were used to derive `COST_PER_MISROUTE` and are read by the tests.

**Interface**

- `time_split(df) -> (train_df, val_df)`: by `created_at_ist` and the config dates (inclusive days); keeps row order and
  index, returns copies. `ValueError` if any row is in neither window. On `train_clean`: 8,687 / 2,135 rows.
- `evaluate(y_true, y_pred, proba=None, labels=config.TEAMS, confidence=None) -> dict`, JSON-safe, keys below.
- `coverage_curve(y_true, y_pred, proba, thresholds=0.30..0.90 step 0.05, confidence=None) -> list[dict]`
- `align_proba(proba, classes_, labels=config.TEAMS) -> ndarray`: reorders columns from `classes_` to `labels` order.
  `ValueError` unless both hold the same names, without duplicates, one per column.
- `bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=config.SEED) -> (low, point, high)`: 95% percentile
  interval; resamples (true, predicted) pairs; `metric_fn(y_true, y_pred) -> float`.
- `config.COST_PER_MISROUTE` = 305 x 3549 / 2471 + 260 = Rs 698.06 (1.436 transfers per misrouted request);
  `config.COST_PER_MISROUTE_SIMPLE` = Rs 565 (1 transfer), for sensitivity.

| `evaluate` key | Content |
|---|---|
| accuracy | share correct |
| macro_f1 | mean F1 over all `labels`; a label in neither `y_true` nor `y_pred` counts as 0 |
| per_class | label -> `precision`, `recall`, `f1` (0 where undefined), `support` |
| confusion_matrix | list of lists in `labels` order, rows = true, columns = predicted |
| n, misroutes | row count, wrong predictions |
| cost_rs | misroutes x `COST_PER_MISROUTE` |
| log_loss | only with `proba`; scores `proba` (clipped at 1e-15), not `y_pred` |
| coverage_curve | with `proba` or `confidence`: list of `threshold`, `coverage` (share with confidence >= threshold), `accuracy_on_covered` (of `y_pred`; None if nothing is covered) |

**Rules**

- Inputs are matched by position, not by pandas index.
- **`proba` columns must follow `labels`; an array in another order cannot be detected. Pass sklearn output through
  `align_proba`** (`classes_` is alphabetical).
- `y_pred` may differ from the row maximum of `proba`. `confidence` defaults to `proba.max(axis=1)`; for rule-overridden
  predictions (M5) pass it explicitly, otherwise the curve uses the model's confidence in a team that was not predicted.
- `ValueError` for: unknown or missing labels, empty or unequal inputs, duplicate `labels`, `proba` with the wrong shape,
  values outside [0, 1], NaN or rows not summing to 1 (tolerance 1e-6), `confidence` with the wrong shape or range,
  non-numeric `proba` or `confidence`.
- `ValueError` messages raised by M3 give counts and shapes, never the offending values (they could be customer text).

**Caveats (state them when quoting results)**

- `cost_rs` is for the evaluated set, not per month, and excludes the bot licence.
- It assumes our misroutes bounce as often as the bot's did. The average covers all 15 labelled months, including
  validation, so it is a reporting constant: a module that picks a threshold by cost must derive a train-window value.
- The 126 bot misroutes with 0 recorded transfers are in the average; transfers on correctly labelled rows are not.
- Bootstrap intervals are slightly too narrow because repeated texts are resampled as independent rows. A percentile
  interval is not guaranteed to contain the point estimate (macro-F1 on small sets); it is not clamped.

**Acceptance criteria**

1. Split: no row in both; max train date < min val date; val months exactly 2026-04 to 2026-06; row counts add up
   (8,687 + 2,135 = 10,822).
2. Hand-made 10-row example gives the exact accuracy, macro-F1, misroutes and cost.
3. Perfect predictions: accuracy 1.0, misroutes 0, cost 0.
4. Unknown label in `y_pred` raises `ValueError`.
5. `proba` rows summing to 1 is validated; coverage at threshold 0 is 1.0; coverage does not increase with threshold.
6. `bootstrap_ci`: low <= point <= high; same seed gives the same result.
7. `COST_PER_MISROUTE` recomputed from the parquet matches config (about Rs 698).
8. `align_proba` reorders an alphabetical `classes_`; `evaluate` accepts `y_pred` that differs from argmax when
   `confidence` is given.
9. A long text passed as a label, probability or confidence does not appear in the error or its traceback.

## M4

status: VERIFIED

Baselines. Code `src/baselines.py`, tests `tests/test_m4_baselines.py`, run `bash scripts/run_all.sh baselines` (run
`clean` first). Fit on the train window, scored on the validation window against `final_team_canon`.

**Inputs**

- `data/processed/train_clean.parquet` (M1).
- `data/processed/test_clean.parquet` (M1): column `text_clean` only, for overlap counts only. Never fitted on or scored.
- Code from M3: `time_split`, `evaluate`, `align_proba`, `bootstrap_ci`, cost constants in `src/config.py`.

**Baselines** (JSON name: title)

- `majority`: (a) most frequent `final_team_canon` in the train window; ties go to the first in `config.TEAMS`.
- `bot`: (b) `team_label_canon` as given.
- `keywords`: (c) `KEYWORD_RULES`, first match wins, train majority team when no rule fires.
- `tfidf_final`, `tfidf_bot`: (d) and (e), the same pipeline trained on `final_team_canon` / `team_label_canon`.

Features of (d) and (e): `text_clean`, `channel`, `product_family`, `warranty_status`. Pipeline: `TfidfVectorizer` word
1-2 grams + `TfidfVectorizer` char_wb 3-5 grams (both `min_df=2`, `sublinear_tf=True`) + `OneHotEncoder`, then
`LogisticRegression(C=1.0, max_iter=2000, class_weight=None, random_state=SEED)`. Settings are not tuned.

**Interface**

- `build(out_dir=config.REPORTS_DIR) -> dict`: writes both outputs, returns the results rounded as in the JSON.
- `compute_baselines(df, test_text) -> dict`: full precision; `df` is all of `train_clean`, `test_text` the test `text_clean`.
- `render_markdown(results) -> str`: pure.
- `make_pipeline() -> Pipeline`: unfitted; the only place where (d) and (e) are defined.
- `fit_model(train_df, target_col) -> Pipeline`; `predict_with_proba(model, df) -> (teams, proba)`, `proba` in
  `config.TEAMS` order through `align_proba`.
- `keyword_predict(text, fallback) -> (teams, fired)`; `majority_class(y) -> str`.
- `score(val_df, pred, proba=None) -> dict`; `paired_difference(truth, pred, reference) -> dict`;
  `cost_sensitivity(scores) -> dict`; `repeated_text(train, val, preds, test_text) -> dict`.
- `round_floats(node)`: floats rounded to `JSON_DECIMALS` = 6.

**Output `reports/baselines.json`** (aggregates only, floats rounded to 6 decimals, sorted keys)

6 top-level keys, no others. Shares are 0-1; keys containing `points` are percentage points.

| Key | Content |
|---|---|
| baselines | the 5 names above -> `evaluate` (M3 keys; `log_loss` and `coverage_curve` only for (d) and (e)), `accuracy_ci95` {low, point, high}, `agreement_with_team_label`, `cost_rs`, `cost_rs_per_month`; `keywords` also has `rule_fired_share` |
| comparisons | `accuracy_points_tfidf_final_minus_bot`, `accuracy_points_tfidf_final_minus_tfidf_bot`, `accuracy_points_keywords_minus_bot`, `agreement_points_tfidf_bot_minus_tfidf_final`, `cost_rs_per_month_bot_minus_tfidf_final` |
| paired_accuracy_points_tfidf_final_minus_bot | {low, point, high}: 95% bootstrap interval of the accuracy difference, both baselines resampled on the same rows |
| cost_sensitivity | `average` (`COST_PER_MISROUTE`) and `simple` (`COST_PER_MISROUTE_SIMPLE`) -> `cost_per_misroute_rs`, `cost_rs_per_month` (name -> Rs), `saving_rs_per_month_bot_minus_tfidf_final` |
| repeated_text | see below |
| meta | `train_rows`, `validation_rows`, `validation_first_day`, `validation_last_day`, `validation_months`, `majority_team`, `features`, `model_settings` {word_ngrams, char_wb_ngrams, min_df, C, max_iter, seed}, `ci_level`, `cost_per_misroute_rs`, `bot_licence_rs_per_month` |

`repeated_text`: exact `text_clean` matches. The reference set is the train window for validation and for test.

| Key | Content |
|---|---|
| validation_seen_share, validation_seen_rows, validation_unseen_rows | validation texts that are / are not in the train window |
| test_seen_share | share of test texts in the train window |
| accuracy | `bot` and `tfidf_final` -> `seen`, `unseen` accuracy on validation (None if the group is empty) |
| tfidf_final_accuracy_at_test_share | (d) seen / unseen accuracy weighted by `test_seen_share` (None if a group is empty); JSON only |

**Output `reports/baselines.md`**: rendered from the rounded results only. States the M3 cost caveats, the cost sensitivity
rows, the paired interval and the seen / unseen accuracies.

**Caveats:** costs are for misroutes only and leave out the bot licence; the other cost and interval caveats are in M3.

**Acceptance criteria**

1. All 5 baselines and all keys exist; costs, comparisons, sensitivity and the paired interval can be recomputed; the JSON
   equals the results rounded to 6 decimals.
2. Bot accuracy equals the parquet value; agreement of the bot with itself is 1.0.
3. (d) has higher validation accuracy than the bot.
4. (d) has higher accuracy than (e); (e) agrees with the bot label more than (d) does.
5. (a) has the lowest accuracy and its team comes from the train window.
6. Only the allowed features are used and everything is fitted on the train window only. Scrambling the bot label, first
   team, transfers, misrouted and resolution columns leaves the predictions of (a), (c) and (d) unchanged. (d) and (e)
   have identical parameters, sub-steps included, and settings match the values above.
7. `build` is deterministic on one machine. The committed JSON matches a fresh run within 1e-4 and the committed markdown
   equals the rendering of the committed JSON.
8. Every number in `baselines.md` is in the results (allowing rounding), no request text (4-word runs) and no old team
   names are in the reports.

## M5a

status: BUILT, rule amended, awaiting owner verification (see `PROGRESS.md`)

Model selection. Code `src/experiments.py`, tests `tests/test_m5a_experiments.py`, run
`bash scripts/run_all.sh experiments` (run `clean` first). No Router class, API, saved model, thresholds or predictions.

**Inputs**

- `data/processed/train_clean.parquet` (M1). `reports/baselines.json` (M4) is read by the tests only.
- Code: M2 `product_conflict_mask`; M3 `time_split`, `evaluate`, `align_proba` (through M4 `predict_with_proba`),
  `bootstrap_ci`; M4 `make_pipeline`, `predict_with_proba`, `paired_difference`, `round_floats` and constants; the dates
  `INNER_TRAIN_END`, `INNER_VAL_START` in `src/config.py`.
- `ops-policy.pdf` section 3 and `teams.csv`: read once to write the cue patterns, not at run time.

**Protocol**

- Inner split of the train window: inner-train Apr-Dec 2025 (6,519 rows), inner-val Jan-Mar 2026 (2,168). All choices use
  inner-val numbers. Real validation is scored for the chosen configuration, the reference (d) and the original rule's
  choice (evidence), each refitted on the whole train window.
- Candidates (22): LogisticRegression C {0.3, 1, 3, 10} x class_weight {none, balanced} (`logreg_c1_none` is (d));
  LinearSVC C {0.1, 1} x {sigmoid, isotonic}; ComplementNB alpha {0.1, 0.3, 1}; then 6 feature groups together and each
  alone on the best of these, picked with the same rule. All share the M4 features.
- Original rule, on the stored rounded inner-val numbers: among candidates within 1.0 point of the highest accuracy, lowest
  complexity (added feature groups + 1 if calibrated), then accuracy, log loss, list order. No chaining. It chose
  `complement_nb_a0.1`, which failed criterion 4 (82.58% against 82.89%); kept as `original_rule`.
- Amended rule (used): within 1.0 point of the highest accuracy, drop candidates whose log loss is more than 0.1 above the
  best of all candidates, then lowest complexity, then log loss, accuracy, list order. It chooses `logreg_c1_none` = (d).
  It was amended after a real-val result had been seen; the disclosure is in `experiments.md`. The base of the feature
  candidates is still the original rule's, so no candidate was added.
- Supplementary, report-only: the 7 cue candidates on `logreg_c1_none`, inner-val only, never given to a selection rule and
  never scored on real validation.
- Calibration: contiguous `KFold(5, shuffle=False)`, `ensemble=False`, whole pipeline inside each fold.

**Interface**

- `build(out_dir=config.REPORTS_DIR) -> dict`, `compute_experiments(df) -> dict`, `run_selection(train_df) -> dict`,
  `render_markdown(results) -> str` (pure).
- `inner_split(train_df) -> (inner_train, inner_val)`: `ValueError` on rows outside the train window.
- `cue_features(df) -> DataFrame` of 0/1 columns, from the 4 allowed columns only; `FEATURE_GROUPS` maps group -> columns.
  `ValueError` on an unknown `product_family` (from M2).
- `build_model(cfg) -> estimator` (unfitted); `fit_predict(cfg, fit_df, score_df) -> (teams, proba)`.
- `base_candidates()`, `feature_candidates(base)`, `complexity(cfg)`, `select(rows)` (amended), `select_original(rows)`,
  `supplementary(train_df, base_row)`, `score_inner(...)`,
  `score_final(cfg, train_df, val_df)`, `top_confusions(matrix, k=5)`.

**Output `reports/experiments.json`** (aggregates only, floats rounded to 6 decimals, sorted keys)

| Key | Content |
|---|---|
| candidates | list: `name`, `config`, `complexity`, `inner_val` {accuracy, macro_f1, log_loss}, `log_loss_above_best`, `points_minus_chosen` |
| chosen | amended choice: `name`, `config` (below), `top_candidate`, `points_below_top`, `reason` |
| discarded | name -> reason under the amended rule, for every candidate not chosen |
| selection | `rule` (amended), `amendment_reason`, `margin_points`, `log_loss_limit`, `accuracy_floor`, `best_log_loss`, `log_loss_ceiling`, `within_margin_count`, `within_log_loss_limit_count`, `simplest_count`, `accuracy_first_choice` (choice with the tie-break as first proposed), `feature_base`, `candidate_count` |
| cue_candidates | `count`, `within_margin` (names), `within_margin_count`, `within_log_loss_limit_count` |
| supplementary | `base`, `base_inner_val`, `count`, `candidates` (`name`, `feature_groups`, `inner_val`, `accuracy_points_minus_base`, `log_loss_minus_base`), `note` |
| original_rule | `rule`, `chosen`, `discarded`, `real_val` (same keys as below), `criterion_4` {`criterion`, `result`, `allowed_drop_points`, `threshold_accuracy`, `points_minus_reference`} |
| inner_split | rows, first and last day of inner-train and inner-val |
| reference | `logreg_c1_none` |
| real_val | amended choice and reference names only -> `accuracy`, `accuracy_ci95`, `macro_f1`, `log_loss`, `paired_accuracy_points_minus_bot`, `per_team_recall`, `repairs_precision`, `top_confusions` (5) |
| bot_real_val_accuracy | share |
| meta | `train_rows`, `validation_rows`, `validation_first_day`, `validation_last_day`, `ci_level`, `seed`, `feature_group_columns` |

`config` (consumed by `build_model`): `name`, `model_type` (`logreg`, `linear_svc`, `complement_nb`), `params` (classifier
arguments), `calibration` (null or {method, folds}), `features` (the 4 columns), `feature_groups`, `text` {word_ngrams,
char_wb_ngrams, min_df}.

**Output `reports/experiments.md`**: rendered from the results only, plus fixed caveats (including the `technician` cue).

**Acceptance criteria**

1. Inner split is chronological and contains no Apr-Jun 2026 row.
2. Every candidate has inner-val metrics; real-val metrics exist only for the chosen configuration and (d), plus the
   original choice inside `original_rule`.
3. The chosen configuration is what the amended rule picks from the stored numbers (`logreg_c1_none`); `original_rule` is
   present and records the failure.
4. Chosen real-val accuracy >= M4 (d) - 1.0 point, and above the bot. Applies to the amended choice, which is (d) itself,
   so the first half cannot fail. The original choice failed it.
5. Only the 4 allowed columns are used (scrambling the others changes nothing).
6. Deterministic with `config.SEED`; JSON floats rounded to 6 decimals.
7. Every number in `experiments.md` is in `experiments.json`.

## M5b

status: not started

## M6

status: not started

## M7

status: not started

## M8

status: not started

## M9

status: not started

## M10

status: not started
