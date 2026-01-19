"""Pytest configuration and shared fixtures for integration tests."""

import os
from pathlib import Path

import pytest

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYNTHETIC_V2 = DATA_DIR / "synthetic_v2.zarr"
SYNTHETIC_V2_BLOSC = DATA_DIR / "synthetic_v2_blosc.zarr"
SYNTHETIC_V3 = DATA_DIR / "synthetic_v3.zarr"
SYNTHETIC_V3_BLOSC = DATA_DIR / "synthetic_v3_blosc.zarr"

ALL_SYNTHETIC_STORES = [
    SYNTHETIC_V2,
    SYNTHETIC_V2_BLOSC,
    SYNTHETIC_V3,
    SYNTHETIC_V3_BLOSC,
]


@pytest.fixture(params=[str(p) for p in ALL_SYNTHETIC_STORES if p.exists()])
def zarr_store_path(request):
    """Parametrized fixture providing paths to all available Zarr test stores."""
    return request.param


@pytest.fixture
def synthetic_v3_path():
    """Fixture providing path to synthetic v3 Zarr store."""
    path = str(SYNTHETIC_V3)
    if not SYNTHETIC_V3.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


@pytest.fixture
def synthetic_v2_path():
    """Fixture providing path to synthetic v2 Zarr store."""
    path = str(SYNTHETIC_V2)
    if not SYNTHETIC_V2.exists():
        pytest.skip(f"Test data not found: {path}")
    return path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
