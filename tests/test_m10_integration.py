"""M10: integration scripts, README, test markers and what git tracks. The end-to-end run itself is scripts/integration_test.*"""
import re
import subprocess

import pytest

STEPS = ["clean", "audit", "baselines", "experiments", "router", "evaluate", "train", "predict"]
MODULES = ["src.clean", "src.audit", "src.baselines", "src.experiments", "src.router", "src.evaluation", "src.predictions"]
SCRIPTS = ["integration_test.ps1", "integration_test.sh"]


def git(root_dir, *args):
    done = subprocess.run(["git", *args], cwd=root_dir, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


@pytest.fixture(scope="module")
def readme(root_dir):
    return (root_dir / "README.md").read_text(encoding="utf-8")


# --- Criterion 1: git ----------------------------------------------------------------------------
def test_no_data_or_model_file_is_tracked(root_dir):
    tracked = git(root_dir, "ls-files")
    assert not [name for name in tracked if name.startswith(("data/", "models/"))]
    assert not [name for name in tracked if name.endswith((".parquet", ".joblib", ".pkl", ".xlsx"))]
    assert [name for name in tracked if name.endswith(".csv")] == ["predictions.csv"]


def test_predictions_are_stored_without_line_ending_conversion(root_dir):
    raw = git(root_dir, "hash-object", "--no-filters", "predictions.csv")
    assert raw == git(root_dir, "hash-object", "predictions.csv")  # no CRLF in the file on disk


# --- Criterion 2: quick and slow tests -----------------------------------------------------------
def test_slow_marker_is_registered_and_strict(pytestconfig):
    assert any(line.startswith("slow:") for line in pytestconfig.getini("markers"))
    assert "--strict-markers" in (pytestconfig.rootpath / "pytest.ini").read_text(encoding="utf-8")


def test_every_module_keeps_quick_tests(root_dir):
    for path in sorted((root_dir / "tests").glob("test_m*.py")):
        text = path.read_text(encoding="utf-8")
        slow = len(re.findall(r"^@pytest\.mark\.slow$", text, re.MULTILINE))
        assert len(re.findall(r"^def test_", text, re.MULTILINE)) > slow, path.name
        assert not re.search(r"^pytestmark", text, re.MULTILINE), path.name


# --- Criterion 3: integration scripts ------------------------------------------------------------
@pytest.mark.parametrize("name", SCRIPTS)
def test_script_has_every_step(root_dir, name):
    text = (root_dir / "scripts" / name).read_text(encoding="utf-8")
    for part in ("clone", "-m venv", "requirements.txt", "not slow", "/health", "/predict", "hash-object --no-filters"):
        assert part in text or part.replace(" ", '", "') in text, part
    assert "PASS" in text and "FAIL" in text and "SKIP" in text


def test_scripts_run_the_whole_pipeline_in_order(root_dir):
    shell = (root_dir / "scripts" / "integration_test.sh").read_text(encoding="utf-8")
    assert f"for target in {' '.join(STEPS)}; do" in shell
    windows = (root_dir / "scripts" / "integration_test.ps1").read_text(encoding="utf-8")
    assert re.findall(r'@\("(\w+)", "src\.', windows) == STEPS


def test_shell_script_has_lf_endings(root_dir):
    assert b"\r" not in (root_dir / "scripts" / "integration_test.sh").read_bytes()


# --- Criterion 4: README -------------------------------------------------------------------------
def test_readme_covers_a_clean_machine(readme):
    for part in (
        "Python 3.12", "data/raw/", "§10", "requirements.txt", "No API key", "http://127.0.0.1:8000", "Invoke-RestMethod",
        "curl", '-m "not slow"', "integration_test.ps1", "integration_test.sh", f"run_all.sh {' '.join(STEPS)}",
    ):  # fmt: skip
        assert part in readme, part
    for module in MODULES:
        assert module in readme, module


def test_readme_deliverables_exist_and_are_tracked(readme, root_dir):
    for name in ("predictions.csv", "memo.md", "reports/evaluation.md"):
        assert f"`{name}`" in readme
        assert git(root_dir, "ls-files", name) == [name]


def test_readme_names_every_raw_file(readme, raw_dir):
    missing = [path.name for path in raw_dir.iterdir() if f"`{path.name}`" not in readme]
    assert not missing
