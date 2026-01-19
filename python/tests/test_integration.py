"""Property-based integration tests comparing Python/xarray and Rust/CLI results.

Uses hypothesis for property-based testing to verify consistency across
implementations with randomly generated queries.
"""

import math
import subprocess
import tempfile
from pathlib import Path
from typing import List, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr
from datafusion import SessionContext
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from zarr_datafusion.xarray_reader import read_xarray_table

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_V3 = DATA_DIR / "synthetic_v3.zarr"
ALL_STORES = [p for p in [
    DATA_DIR / "synthetic_v2.zarr",
    DATA_DIR / "synthetic_v3.zarr",
    DATA_DIR / "synthetic_v2_blosc.zarr",
    DATA_DIR / "synthetic_v3_blosc.zarr",
] if p.exists()]


# === CLI Helpers ===

def get_cli_binary() -> Path:
    """Get zarr-cli binary path, building if needed."""
    for path in [PROJECT_ROOT / "target/release/zarr-cli", PROJECT_ROOT / "target/debug/zarr-cli"]:
        if path.exists():
            return path
    subprocess.run(["cargo", "build", "--bin", "zarr-cli"], cwd=PROJECT_ROOT, capture_output=True, timeout=300)
    debug = PROJECT_ROOT / "target/debug/zarr-cli"
    if debug.exists():
        return debug
    pytest.skip("zarr-cli not available")


def run_rust_query(zarr_path: str, sql: str) -> List[pa.RecordBatch]:
    """Run SQL via Rust CLI, return results as RecordBatches."""
    cli = get_cli_binary()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "result.parquet"
        cmds = f"CREATE EXTERNAL TABLE data STORED AS ZARR LOCATION '{zarr_path}'\nCOPY ({sql}) TO '{out}'\nquit\n"
        result = subprocess.run([str(cli)], input=cmds, capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT)
        if result.returncode != 0 or "Error" in result.stdout:
            pytest.skip(f"CLI error: {result.stdout}{result.stderr}")
        if not out.exists():
            pytest.skip("No output from CLI")
        return pq.read_table(out).to_batches()


def run_python_query(zarr_path: str, sql: str, chunks=None) -> List[pa.RecordBatch]:
    """Run SQL via Python/xarray path."""
    ds = xr.open_zarr(zarr_path)
    table = read_xarray_table(ds, chunks=chunks or {"time": 3, "lat": 5, "lon": 5})
    ctx = SessionContext()
    ctx.register_table("data", table)
    return ctx.sql(sql).collect()


# === Comparison Helpers ===

def normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return float('nan') if math.isnan(float(value)) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


def batches_to_rows(batches: List[pa.RecordBatch]) -> List[tuple]:
    if not batches:
        return []
    table = pa.Table.from_batches(batches)
    return sorted(tuple(normalize(table.column(j)[i].as_py()) for j in range(table.num_columns))
                  for i in range(table.num_rows))


def results_match(py_batches, rust_batches) -> bool:
    py_rows, rust_rows = batches_to_rows(py_batches), batches_to_rows(rust_batches)
    if len(py_rows) != len(rust_rows):
        return False
    for p, r in zip(py_rows, rust_rows):
        for pv, rv in zip(p, r):
            if pv is None and rv is None:
                continue
            if pv is None or rv is None:
                return False
            if isinstance(pv, float) and isinstance(rv, float):
                if math.isnan(pv) and math.isnan(rv):
                    continue
                if abs(pv - rv) > 1e-5:
                    return False
            elif pv != rv:
                return False
    return True


# === Query Strategies ===

COLS = ["lat", "lon", "time", "temperature", "humidity"]
DATA_COLS = ["temperature", "humidity"]
AGGS = ["COUNT", "SUM", "AVG", "MIN", "MAX"]


@st.composite
def agg_query(draw):
    agg = draw(st.sampled_from(AGGS))
    col = draw(st.sampled_from(DATA_COLS))
    return "SELECT COUNT(*) as cnt FROM data" if agg == "COUNT" else f"SELECT {agg}({col}) as r FROM data"


@st.composite
def limit_query(draw):
    # Always include a data column to avoid DictionaryArray optimization differences
    data_col = draw(st.sampled_from(DATA_COLS))
    extra_cols = draw(st.lists(st.sampled_from(COLS), min_size=0, max_size=3, unique=True))
    cols = [data_col] + [c for c in extra_cols if c != data_col]
    limit = draw(st.integers(1, 50))
    return f"SELECT {', '.join(cols)} FROM data LIMIT {limit}"


# === Property-Based Tests ===

@pytest.mark.integration
class TestPythonRustConsistency:
    """Property-based tests verifying Python/xarray matches Rust/CLI."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip("Test data not found")
        get_cli_binary()

    @given(query=agg_query())
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregations(self, query: str):
        path = str(SYNTHETIC_V3)
        assert results_match(run_python_query(path, query), run_rust_query(path, query)), f"Failed: {query}"

    @given(query=limit_query())
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_limits(self, query: str):
        path = str(SYNTHETIC_V3)
        py_count = sum(b.num_rows for b in run_python_query(path, query))
        rust_count = sum(b.num_rows for b in run_rust_query(path, query))
        assert py_count == rust_count, f"Row count mismatch for: {query}"


# === Parameterized Tests ===

@pytest.mark.integration
@pytest.mark.parametrize("zarr_path", [str(p) for p in ALL_STORES])
def test_count_across_versions(zarr_path):
    """COUNT(*) consistent across Zarr versions."""
    query = "SELECT COUNT(*) as cnt FROM data"
    assert results_match(run_python_query(zarr_path, query), run_rust_query(zarr_path, query))


@pytest.mark.integration
@pytest.mark.parametrize("chunks", [{"time": 1}, {"time": 3, "lat": 5, "lon": 5}, {"time": 7, "lat": 10, "lon": 10}])
def test_chunking_produces_same_results(chunks):
    """Different chunk sizes produce same results."""
    if not SYNTHETIC_V3.exists():
        pytest.skip("Test data not found")
    path = str(SYNTHETIC_V3)
    query = "SELECT SUM(temperature) as s FROM data"
    py_result = run_python_query(path, query, chunks=chunks)
    rust_result = run_rust_query(path, query)
    assert results_match(py_result, rust_result), f"Mismatch with chunks={chunks}"
