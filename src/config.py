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

# --- Dates (IST, all bounds inclusive calendar days) -----------------------
TEAM_RENAME_DATE = date(2026, 1, 15)

# Time-based split. Validation is the last 3 months of labelled data, the same length as test.
TRAIN_START = date(2025, 4, 1)
TRAIN_END = date(2026, 3, 31)
VALID_START = date(2026, 4, 1)
VALID_END = date(2026, 6, 30)
TEST_START = date(2026, 7, 1)
TEST_END = date(2026, 9, 30)
