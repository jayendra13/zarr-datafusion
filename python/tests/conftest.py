"""Pytest configuration and shared fixtures."""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYNTHETIC_V2 = DATA_DIR / "synthetic_v2.zarr"
SYNTHETIC_V3 = DATA_DIR / "synthetic_v3.zarr"
ALL_SYNTHETIC_STORES = [SYNTHETIC_V2, DATA_DIR / "synthetic_v2_blosc.zarr",
                        SYNTHETIC_V3, DATA_DIR / "synthetic_v3_blosc.zarr"]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests")
