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

status: not started

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
