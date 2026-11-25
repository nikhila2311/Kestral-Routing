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

status: not started

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
