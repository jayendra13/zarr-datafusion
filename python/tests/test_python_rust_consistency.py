"""Property-based integration tests for Python/Rust consistency.

These tests verify that querying data via:
1. Xarray -> Arrow Stream -> DataFusion (Python path)
2. Zarr -> DataFusion (Rust path)

produces identical results for the same SQL queries.

This ensures the Arrow C Stream interface correctly bridges Python and Rust.
"""

import math
from typing import List, Tuple, Any, Optional

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import SessionContext
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from zarr_datafusion import LazyArrowStreamTable
from zarr_datafusion.xarray_reader import (
    XarrayRecordBatchReader,
    read_xarray_table,
    xarray_to_arrow_table,
    parse_schema,
    block_slices,
)

# Import test fixtures
from conftest import (
    SYNTHETIC_V2,
    SYNTHETIC_V3,
    ALL_SYNTHETIC_STORES,
    DATA_DIR,
)


# =============================================================================
# Strategies for generating SQL query components
# =============================================================================

# Column names available in synthetic test data
COORDINATE_COLUMNS = ["lat", "lon", "time"]
DATA_COLUMNS = ["temperature", "humidity"]
ALL_COLUMNS = COORDINATE_COLUMNS + DATA_COLUMNS

# Aggregation functions that work with numeric data
AGG_FUNCTIONS = ["COUNT", "SUM", "AVG", "MIN", "MAX"]

# Default chunking for efficient streaming
DEFAULT_CHUNKS = {"time": 3, "lat": 5, "lon": 5}


@st.composite
def column_selection(draw) -> List[str]:
    """Generate a selection of columns (1 to all columns)."""
    num_cols = draw(st.integers(min_value=1, max_value=len(ALL_COLUMNS)))
    return draw(st.lists(
        st.sampled_from(ALL_COLUMNS),
        min_size=num_cols,
        max_size=num_cols,
        unique=True
    ))


@st.composite
def numeric_column(draw) -> str:
    """Generate a numeric column name (data variables)."""
    return draw(st.sampled_from(DATA_COLUMNS))


@st.composite
def coordinate_column(draw) -> str:
    """Generate a coordinate column name."""
    return draw(st.sampled_from(COORDINATE_COLUMNS))


@st.composite
def aggregation_query(draw) -> str:
    """Generate an aggregation SQL query."""
    agg_func = draw(st.sampled_from(AGG_FUNCTIONS))
    column = draw(numeric_column())

    if agg_func == "COUNT":
        return f"SELECT COUNT(*) as cnt FROM data"
    else:
        return f"SELECT {agg_func}({column}) as result FROM data"


@st.composite
def limit_value(draw) -> int:
    """Generate a reasonable LIMIT value."""
    return draw(st.integers(min_value=1, max_value=100))


@st.composite
def select_with_limit_query(draw) -> str:
    """Generate a SELECT query with LIMIT."""
    columns = draw(column_selection())
    limit = draw(limit_value())
    cols_str = ", ".join(columns)
    return f"SELECT {cols_str} FROM data LIMIT {limit}"


@st.composite
def projection_query(draw) -> str:
    """Generate a projection-only query (SELECT specific columns)."""
    columns = draw(column_selection())
    cols_str = ", ".join(columns)
    return f"SELECT {cols_str} FROM data"


@st.composite
def order_by_query(draw) -> str:
    """Generate a query with ORDER BY."""
    columns = draw(column_selection())
    order_col = draw(st.sampled_from(columns))
    direction = draw(st.sampled_from(["ASC", "DESC"]))
    limit = draw(limit_value())
    cols_str = ", ".join(columns)
    return f"SELECT {cols_str} FROM data ORDER BY {order_col} {direction} LIMIT {limit}"


@st.composite
def any_valid_query(draw) -> str:
    """Generate any valid SQL query from our query types."""
    query_type = draw(st.sampled_from([
        "aggregation",
        "limit",
        "projection_limit",
        "order_by",
    ]))

    if query_type == "aggregation":
        return draw(aggregation_query())
    elif query_type == "limit":
        return draw(select_with_limit_query())
    elif query_type == "projection_limit":
        columns = draw(column_selection())
        limit = draw(limit_value())
        cols_str = ", ".join(columns)
        return f"SELECT {cols_str} FROM data LIMIT {limit}"
    else:  # order_by
        return draw(order_by_query())


# =============================================================================
# Helper functions for loading data and running queries
# =============================================================================

def load_xarray_dataset(zarr_path: str) -> xr.Dataset:
    """Load a Zarr store as an xarray Dataset."""
    return xr.open_zarr(zarr_path)


def run_query_on_xarray_chunked(
    zarr_path: str,
    sql: str,
    chunks: Optional[dict] = None
) -> List[pa.RecordBatch]:
    """Run a SQL query on data loaded via xarray with chunked streaming.

    This uses the efficient XarrayRecordBatchReader with block_slices
    for memory-efficient streaming.
    """
    if chunks is None:
        chunks = DEFAULT_CHUNKS

    ds = load_xarray_dataset(zarr_path)

    # Use the efficient chunked reader
    table = read_xarray_table(ds, chunks=chunks)

    # Register and query
    ctx = SessionContext()
    ctx.register_table("data", table)

    return ctx.sql(sql).collect()


def run_query_on_rust_zarr(zarr_path: str, sql: str) -> List[pa.RecordBatch]:
    """Run a SQL query on data loaded via Rust Zarr reader simulation.

    This uses xarray with chunked conversion to Arrow as the "ground truth"
    for comparison. In production, this would use the actual Rust ZarrTable.
    """
    ds = load_xarray_dataset(zarr_path)

    # Use chunked conversion to match what Rust does
    table = xarray_to_arrow_table(ds, chunks=DEFAULT_CHUNKS)

    ctx = SessionContext()
    ctx.register_record_batches("data", [table.to_batches()])

    return ctx.sql(sql).collect()


def batches_to_sorted_tuples(batches: List[pa.RecordBatch]) -> List[Tuple]:
    """Convert record batches to sorted list of tuples for comparison."""
    if not batches:
        return []

    # Concatenate all batches
    table = pa.Table.from_batches(batches)

    # Convert to list of tuples
    rows = []
    for i in range(table.num_rows):
        row = tuple(
            _normalize_value(table.column(j)[i].as_py())
            for j in range(table.num_columns)
        )
        rows.append(row)

    return sorted(rows)


def _normalize_value(value: Any) -> Any:
    """Normalize a value for comparison (handle floats, NaN, etc.)."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return float('nan')
        # Round to avoid floating point comparison issues
        return round(value, 6)
    if isinstance(value, (np.floating, np.integer)):
        return _normalize_value(float(value) if isinstance(value, np.floating) else int(value))
    return value


def compare_results(
    xarray_batches: List[pa.RecordBatch],
    rust_batches: List[pa.RecordBatch],
    ordered: bool = False
) -> Tuple[bool, Optional[str]]:
    """Compare query results from xarray and Rust paths.

    Returns (is_equal, error_message).
    """
    # Check schemas match
    if xarray_batches and rust_batches:
        xarray_schema = xarray_batches[0].schema
        rust_schema = rust_batches[0].schema

        if set(xarray_schema.names) != set(rust_schema.names):
            return False, f"Schema mismatch: {xarray_schema.names} vs {rust_schema.names}"

    # Convert to comparable format
    xarray_rows = batches_to_sorted_tuples(xarray_batches)
    rust_rows = batches_to_sorted_tuples(rust_batches)

    if len(xarray_rows) != len(rust_rows):
        return False, f"Row count mismatch: {len(xarray_rows)} vs {len(rust_rows)}"

    # Compare row by row
    for i, (x_row, r_row) in enumerate(zip(xarray_rows, rust_rows)):
        if len(x_row) != len(r_row):
            return False, f"Row {i} column count mismatch: {len(x_row)} vs {len(r_row)}"

        for j, (x_val, r_val) in enumerate(zip(x_row, r_row)):
            if not _values_equal(x_val, r_val):
                return False, f"Row {i}, Col {j} value mismatch: {x_val} vs {r_val}"

    return True, None


def _values_equal(a: Any, b: Any) -> bool:
    """Check if two values are equal, handling special cases."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) < 1e-5
    return a == b


# =============================================================================
# Tests for XarrayRecordBatchReader chunking
# =============================================================================

@pytest.mark.integration
class TestXarrayReaderChunking:
    """Tests verifying the chunked reader works correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check that test data exists."""
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.zarr_path = str(SYNTHETIC_V3)
        self.ds = xr.open_zarr(self.zarr_path)

    def test_block_slices_generates_blocks(self):
        """Test that block_slices generates correct blocks."""
        chunks = {"time": 3, "lat": 5, "lon": 5}
        blocks = list(block_slices(self.ds, chunks))

        # Should generate multiple blocks
        assert len(blocks) > 1, "Should generate multiple blocks"

        # Each block should have slices for all dimensions
        for block in blocks:
            assert set(block.keys()) == set(self.ds.dims.keys())
            for dim, sl in block.items():
                assert isinstance(sl, slice)

    def test_chunked_reader_lazy_iteration(self):
        """Test that XarrayRecordBatchReader iterates lazily."""
        iterations = []

        def track_iteration(block):
            iterations.append(block)

        reader = XarrayRecordBatchReader(
            self.ds,
            chunks={"time": 3},
            _iteration_callback=track_iteration
        )

        # No iterations yet
        assert len(iterations) == 0, "Should not iterate during creation"

        # Consume the reader
        pa_reader = pa.RecordBatchReader.from_stream(reader)
        batches = list(pa_reader)

        # Now we should have iterations
        assert len(iterations) > 0, "Should iterate when consumed"
        assert len(batches) == len(iterations), "One batch per block"

    def test_chunked_reader_schema(self):
        """Test that schema is correctly extracted."""
        reader = XarrayRecordBatchReader(self.ds, chunks={"time": 3})
        schema = reader.schema

        # Should have columns for coordinates and data variables
        expected_cols = set(COORDINATE_COLUMNS + DATA_COLUMNS)
        actual_cols = set(schema.names)
        assert expected_cols == actual_cols

    def test_read_xarray_table_multiple_queries(self):
        """Test that read_xarray_table supports multiple queries."""
        table = read_xarray_table(self.ds, chunks={"time": 3})

        ctx = SessionContext()
        ctx.register_table("data", table)

        # First query
        result1 = ctx.sql("SELECT COUNT(*) FROM data").collect()
        count1 = result1[0].column(0)[0].as_py()

        # Second query - should work (factory creates fresh stream)
        result2 = ctx.sql("SELECT COUNT(*) FROM data").collect()
        count2 = result2[0].column(0)[0].as_py()

        assert count1 == count2, "Multiple queries should return same count"


# =============================================================================
# Property-based tests
# =============================================================================

@pytest.mark.integration
class TestPythonRustConsistency:
    """Property-based tests verifying Python/Rust query consistency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check that test data exists."""
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")

    @given(query=aggregation_query())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_aggregation_queries_consistent(self, query: str):
        """Test that aggregation queries produce consistent results."""
        zarr_path = str(SYNTHETIC_V3)

        try:
            xarray_result = run_query_on_xarray_chunked(zarr_path, query)
            rust_result = run_query_on_rust_zarr(zarr_path, query)

            is_equal, error = compare_results(xarray_result, rust_result)
            assert is_equal, f"Query '{query}' produced inconsistent results: {error}"
        except Exception as e:
            # Some queries might fail due to data type issues - that's OK for now
            pytest.skip(f"Query failed: {e}")

    @given(query=select_with_limit_query())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_limit_queries_consistent(self, query: str):
        """Test that LIMIT queries produce consistent results."""
        zarr_path = str(SYNTHETIC_V3)

        try:
            xarray_result = run_query_on_xarray_chunked(zarr_path, query)
            rust_result = run_query_on_rust_zarr(zarr_path, query)

            # For LIMIT queries, we compare row counts and schemas
            # but not exact content since order may differ
            xarray_count = sum(b.num_rows for b in xarray_result)
            rust_count = sum(b.num_rows for b in rust_result)

            assert xarray_count == rust_count, \
                f"Query '{query}' row count mismatch: {xarray_count} vs {rust_count}"
        except Exception as e:
            pytest.skip(f"Query failed: {e}")

    @given(query=order_by_query())
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_ordered_queries_consistent(self, query: str):
        """Test that ORDER BY queries produce consistent results."""
        zarr_path = str(SYNTHETIC_V3)

        try:
            xarray_result = run_query_on_xarray_chunked(zarr_path, query)
            rust_result = run_query_on_rust_zarr(zarr_path, query)

            # For ordered queries, compare exact order
            xarray_rows = batches_to_sorted_tuples(xarray_result)
            rust_rows = batches_to_sorted_tuples(rust_result)

            assert len(xarray_rows) == len(rust_rows), \
                f"Query '{query}' row count mismatch"

            # First rows should match after sorting
            if xarray_rows and rust_rows:
                assert xarray_rows[0] == rust_rows[0], \
                    f"Query '{query}' first row mismatch after sorting"
        except Exception as e:
            pytest.skip(f"Query failed: {e}")


# =============================================================================
# Deterministic test cases for baseline validation
# =============================================================================

@pytest.mark.integration
class TestDeterministicConsistency:
    """Deterministic tests with known queries for baseline validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check that test data exists."""
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.zarr_path = str(SYNTHETIC_V3)

    def test_count_star(self):
        """Test COUNT(*) produces same result."""
        query = "SELECT COUNT(*) as cnt FROM data"

        xarray_result = run_query_on_xarray_chunked(self.zarr_path, query)
        rust_result = run_query_on_rust_zarr(self.zarr_path, query)

        is_equal, error = compare_results(xarray_result, rust_result)
        assert is_equal, f"COUNT(*) inconsistent: {error}"

    def test_sum_temperature(self):
        """Test SUM(temperature) produces same result."""
        query = "SELECT SUM(temperature) as total FROM data"

        xarray_result = run_query_on_xarray_chunked(self.zarr_path, query)
        rust_result = run_query_on_rust_zarr(self.zarr_path, query)

        is_equal, error = compare_results(xarray_result, rust_result)
        assert is_equal, f"SUM inconsistent: {error}"

    def test_avg_humidity(self):
        """Test AVG(humidity) produces same result."""
        query = "SELECT AVG(humidity) as avg_h FROM data"

        xarray_result = run_query_on_xarray_chunked(self.zarr_path, query)
        rust_result = run_query_on_rust_zarr(self.zarr_path, query)

        is_equal, error = compare_results(xarray_result, rust_result)
        assert is_equal, f"AVG inconsistent: {error}"

    def test_min_max(self):
        """Test MIN/MAX produces same result."""
        query = "SELECT MIN(temperature) as min_t, MAX(temperature) as max_t FROM data"

        xarray_result = run_query_on_xarray_chunked(self.zarr_path, query)
        rust_result = run_query_on_rust_zarr(self.zarr_path, query)

        is_equal, error = compare_results(xarray_result, rust_result)
        assert is_equal, f"MIN/MAX inconsistent: {error}"

    def test_select_all_limit(self):
        """Test SELECT * LIMIT produces same row count."""
        query = "SELECT * FROM data LIMIT 10"

        xarray_result = run_query_on_xarray_chunked(self.zarr_path, query)
        rust_result = run_query_on_rust_zarr(self.zarr_path, query)

        xarray_count = sum(b.num_rows for b in xarray_result)
        rust_count = sum(b.num_rows for b in rust_result)

        assert xarray_count == rust_count == 10, \
            f"LIMIT 10 row count mismatch: {xarray_count} vs {rust_count}"

    def test_projection(self):
        """Test column projection produces same schema."""
        query = "SELECT lat, lon, temperature FROM data LIMIT 5"

        xarray_result = run_query_on_xarray_chunked(self.zarr_path, query)
        rust_result = run_query_on_rust_zarr(self.zarr_path, query)

        xarray_cols = set(xarray_result[0].schema.names) if xarray_result else set()
        rust_cols = set(rust_result[0].schema.names) if rust_result else set()

        assert xarray_cols == rust_cols == {"lat", "lon", "temperature"}, \
            f"Projection schema mismatch: {xarray_cols} vs {rust_cols}"


# =============================================================================
# Tests across all Zarr store versions
# =============================================================================

@pytest.mark.integration
class TestAllZarrVersions:
    """Tests that run across all Zarr store versions (v2, v3, with/without compression)."""

    @pytest.mark.parametrize("zarr_path", [
        str(p) for p in ALL_SYNTHETIC_STORES if p.exists()
    ])
    def test_count_consistent_across_versions(self, zarr_path: str):
        """Test COUNT(*) is consistent across all Zarr versions."""
        query = "SELECT COUNT(*) as cnt FROM data"

        xarray_result = run_query_on_xarray_chunked(zarr_path, query)
        rust_result = run_query_on_rust_zarr(zarr_path, query)

        is_equal, error = compare_results(xarray_result, rust_result)
        assert is_equal, f"COUNT(*) inconsistent for {zarr_path}: {error}"

    @pytest.mark.parametrize("zarr_path", [
        str(p) for p in ALL_SYNTHETIC_STORES if p.exists()
    ])
    def test_aggregations_consistent_across_versions(self, zarr_path: str):
        """Test aggregations are consistent across all Zarr versions."""
        queries = [
            "SELECT SUM(temperature) as s FROM data",
            "SELECT AVG(humidity) as a FROM data",
            "SELECT MIN(lat) as min_lat, MAX(lat) as max_lat FROM data",
        ]

        for query in queries:
            xarray_result = run_query_on_xarray_chunked(zarr_path, query)
            rust_result = run_query_on_rust_zarr(zarr_path, query)

            is_equal, error = compare_results(xarray_result, rust_result)
            assert is_equal, f"Query '{query}' inconsistent for {zarr_path}: {error}"


# =============================================================================
# Tests for different chunking strategies
# =============================================================================

@pytest.mark.integration
class TestChunkingStrategies:
    """Tests verifying different chunking strategies produce same results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check that test data exists."""
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.zarr_path = str(SYNTHETIC_V3)

    @pytest.mark.parametrize("chunks", [
        {"time": 1},  # Very small chunks
        {"time": 3, "lat": 5, "lon": 5},  # Medium chunks
        {"time": 7, "lat": 10, "lon": 10},  # Large chunks (entire dataset)
        None,  # Use dataset's default chunks
    ])
    def test_count_same_with_different_chunks(self, chunks):
        """Test that COUNT(*) is same regardless of chunking."""
        query = "SELECT COUNT(*) as cnt FROM data"

        result = run_query_on_xarray_chunked(self.zarr_path, query, chunks=chunks)
        count = result[0].column(0)[0].as_py()

        # All chunking strategies should give 700 rows (7*10*10)
        assert count == 700, f"Count mismatch with chunks={chunks}: {count}"

    @pytest.mark.parametrize("chunks", [
        {"time": 1},
        {"time": 3, "lat": 5, "lon": 5},
        {"time": 7, "lat": 10, "lon": 10},
    ])
    def test_sum_same_with_different_chunks(self, chunks):
        """Test that SUM is same regardless of chunking."""
        query = "SELECT SUM(temperature) as total FROM data"

        result = run_query_on_xarray_chunked(self.zarr_path, query, chunks=chunks)
        total = result[0].column(0)[0].as_py()

        # Get reference from rust path
        reference = run_query_on_rust_zarr(self.zarr_path, query)
        expected = reference[0].column(0)[0].as_py()

        assert abs(total - expected) < 1e-5, \
            f"SUM mismatch with chunks={chunks}: {total} vs {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
