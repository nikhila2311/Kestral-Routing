"""M0: project setup."""
import csv
import shutil
import subprocess

import pytest

EXPECTED_TEAMS = {
    "Installs & Demo",
    "Repairs",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
}

EXPECTED_RAW_FILES = {
    "README.txt",
    "email-thread.txt",
    "ops-policy.pdf",
    "resolution_log.csv",
    "sample_submission.csv",
    "teams.csv",
    "test_unlabelled.csv",
    "train.csv",
}


def test_config_imports(config, root_dir):
    assert config.ROOT == root_dir
    assert config.SEED == 42


def test_seven_current_teams(config):
    assert len(config.TEAMS) == 7
    assert set(config.TEAMS) == EXPECTED_TEAMS


def test_name_map_covers_both_old_names(config):
    assert config.TEAM_NAME_MAP == {
        "Installations": "Installs & Demo",
        "Consumables": "Filters & Consumables",
    }
    assert not set(config.TEAM_NAME_MAP) & set(config.TEAMS)


def test_every_team_in_teams_csv_maps_to_a_current_team(config):
    with open(config.TEAMS_CSV, encoding="utf-8", newline="") as f:
        names = [row["team"] for row in csv.DictReader(f)]
    assert {config.TEAM_NAME_MAP.get(n, n) for n in names} == EXPECTED_TEAMS


def test_costs(config):
    assert config.COST_PER_TRANSFER == 305
    assert config.COST_PER_MISROUTE_CONTACT == 260
    assert config.BOT_LICENCE_PER_YEAR == 320_000


def test_split_dates_are_chronological_and_contiguous(config):
    assert config.TRAIN_START < config.TRAIN_END < config.VALID_START < config.VALID_END
    assert config.VALID_END < config.TEST_START < config.TEST_END
    assert (config.VALID_START - config.TRAIN_END).days == 1
    assert (config.TEST_START - config.VALID_END).days == 1


def test_raw_has_all_eight_files(config, raw_dir):
    assert {p.name for p in config.RAW_FILES} == EXPECTED_RAW_FILES
    missing = [n for n in sorted(EXPECTED_RAW_FILES) if not (raw_dir / n).is_file()]
    assert not missing, f"missing from data/raw: {missing}"
    empty = [n for n in sorted(EXPECTED_RAW_FILES) if (raw_dir / n).stat().st_size == 0]
    assert not empty, f"empty files in data/raw: {empty}"


def test_data_raw_is_in_gitignore(root_dir):
    lines = (root_dir / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/raw/" in [line.strip() for line in lines]


def test_git_ignores_raw_files_but_not_predictions(config, root_dir):
    if shutil.which("git") is None or not (root_dir / ".git").exists():
        pytest.skip("not a git checkout")

    def git(*args):
        return subprocess.run(["git", *args], cwd=root_dir, capture_output=True, text=True)

    for path in config.RAW_FILES:
        rel = path.relative_to(root_dir).as_posix()
        assert git("check-ignore", "-q", rel).returncode == 0, f"{rel} is not git-ignored"
    assert git("check-ignore", "-q", "predictions.csv").returncode == 1
    assert git("ls-files", "data").stdout.strip() == "", "files under data/ are tracked by git"
