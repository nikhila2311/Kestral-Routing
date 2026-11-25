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

status: IMPLEMENTED (tests pass; not yet VERIFIED by the project owner)

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
  values outside [0, 1], NaN or rows not summing to 1 (tolerance 1e-6), `confidence` with the wrong shape or range.

**Caveats (state them when quoting results)**

- `cost_rs` is for the evaluated set, not per month, and excludes the bot licence.
- It assumes our misroutes bounce as often as the bot's did. The average covers all 15 labelled months, including
  validation, so it is a reporting constant: a module that picks a threshold by cost must derive a train-window value.
- The 126 bot misroutes with 0 recorded transfers are in the average; transfers on correctly labelled rows are not.
- Bootstrap intervals are slightly too narrow because repeated texts are resampled as independent rows. A percentile
  interval is not guaranteed to contain the point estimate (macro-F1 on small sets); it is not clamped.

**Acceptance criteria**

1. Split: no row in both; max train date < min val date; val months exactly 2026-04 to 2026-06; row counts add up.
2. Hand-made 10-row example gives the exact accuracy, macro-F1, misroutes and cost.
3. Perfect predictions: accuracy 1.0, misroutes 0, cost 0.
4. Unknown label in `y_pred` raises `ValueError`.
5. `proba` rows summing to 1 is validated; coverage at threshold 0 is 1.0; coverage does not increase with threshold.
6. `bootstrap_ci`: low <= point <= high; same seed gives the same result.
7. `COST_PER_MISROUTE` recomputed from the parquet matches config (about Rs 698).
8. `align_proba` reorders an alphabetical `classes_`; `evaluate` accepts `y_pred` that differs from argmax when
   `confidence` is given.

## M4

status: not started

## M5

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
