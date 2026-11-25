"""Project constants. Single source of truth: other code imports these and does not redefine them."""
from datetime import date
from pathlib import Path

SEED = 42

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"  # git-ignored with the rest of data/
MODELS_DIR = ROOT / "models"  # git-ignored
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test_unlabelled.csv"
RESOLUTION_LOG_CSV = RAW_DIR / "resolution_log.csv"
TEAMS_CSV = RAW_DIR / "teams.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"
OPS_POLICY_PDF = RAW_DIR / "ops-policy.pdf"
EMAIL_THREAD_TXT = RAW_DIR / "email-thread.txt"
README_TXT = RAW_DIR / "README.txt"

RAW_FILES = (
    TRAIN_CSV,
    TEST_CSV,
    RESOLUTION_LOG_CSV,
    TEAMS_CSV,
    SAMPLE_SUBMISSION_CSV,
    OPS_POLICY_PDF,
    EMAIL_THREAD_TXT,
    README_TXT,
)

PREDICTIONS_CSV = ROOT / "predictions.csv"

# --- M1 outputs ------------------------------------------------------------
TRAIN_CLEAN_PARQUET = PROCESSED_DIR / "train_clean.parquet"
TEST_CLEAN_PARQUET = PROCESSED_DIR / "test_clean.parquet"

_RAW_COLUMNS = (
    "request_id",
    "created_at_ist",
    "channel",
    "product_family",
    "warranty_status",
    "request_text",
    "source",
)
TRAIN_CLEAN_COLUMNS = _RAW_COLUMNS + (
    "team_label",
    "text_clean",
    "team_label_canon",
    "final_team_canon",
    "first_team_canon",
    "transfers",
    "resolved_at_ist",
    "resolution_hours",
    "misrouted",
    "created_month",
)
TEST_CLEAN_COLUMNS = _RAW_COLUMNS + ("text_clean", "created_month")

# --- M2 outputs ------------------------------------------------------------
AUDIT_METRICS_JSON = REPORTS_DIR / "audit_metrics.json"
DATA_AUDIT_MD = REPORTS_DIR / "data_audit.md"

# --- Teams -----------------------------------------------------------------
TEAMS = (
    "Installs & Demo",
    "Repairs",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
)

# Old name -> current name, applied at load time (ops-policy section 5).
TEAM_NAME_MAP = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}

# --- Costs in Rs (ops-policy section 4) ------------------------------------
COST_PER_TRANSFER = 305
COST_PER_MISROUTE_CONTACT = 260
BOT_LICENCE_PER_YEAR = 320_000

# Expected cost of one misrouted request = transfer cost x average transfers per misrouted request + extra contact.
# Average = 3549 / 2471 = 1.436, from reports/audit_metrics.json `cost_inputs`: transfers 3902 minus 353 on correctly
# labelled rows, over 2471 misroutes (all 15 labelled months; includes 126 misroutes with 0 recorded transfers).
# Literal counts because src/audit.py imports this file; tests/test_m3_harness.py checks them against the data.
# Reporting only: to choose a threshold, derive the average from the training window instead (CLAUDE.md rule 6).
COST_PER_MISROUTE = COST_PER_TRANSFER * 3549 / 2471 + COST_PER_MISROUTE_CONTACT  # about Rs 698.06
# Sensitivity: assumes exactly 1 transfer per misroute (Rs 565).
COST_PER_MISROUTE_SIMPLE = COST_PER_TRANSFER + COST_PER_MISROUTE_CONTACT

# --- Dates (IST, all bounds inclusive calendar days) -----------------------
TEAM_RENAME_DATE = date(2026, 1, 15)

# Time-based split. Validation is the last 3 months of labelled data, the same length as test.
TRAIN_START = date(2025, 4, 1)
TRAIN_END = date(2026, 3, 31)
VALID_START = date(2026, 4, 1)
VALID_END = date(2026, 6, 30)
TEST_START = date(2026, 7, 1)
TEST_END = date(2026, 9, 30)
