"""Shared fixtures for all module tests."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def config():
    from src import config as cfg

    return cfg


@pytest.fixture(scope="session")
def root_dir():
    return ROOT


@pytest.fixture(scope="session")
def raw_dir(config):
    return config.RAW_DIR


@pytest.fixture(scope="session")
def processed_dir(config):
    return config.PROCESSED_DIR


@pytest.fixture(scope="session")
def reports_dir(config):
    return config.REPORTS_DIR
