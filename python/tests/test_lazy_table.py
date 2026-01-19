"""Tests for LazyArrowStreamTable."""

import pytest
import pyarrow as pa
from datafusion import SessionContext

from zarr_datafusion import LazyArrowStreamTable


def test_lazy_table_creation():
    """Test that LazyArrowStreamTable can be created."""
    schema = pa.schema([
        ("x", pa.int64()),
        ("y", pa.float64()),
    ])

    def make_stream():
        data = {"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}
        return pa.Table.from_pydict(data).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    assert table is not None


def test_lazy_table_schema():
    """Test that schema is correctly returned."""
    schema = pa.schema([
        ("x", pa.int64()),
        ("y", pa.float64()),
    ])

    def make_stream():
        data = {"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}
        return pa.Table.from_pydict(data).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    returned_schema = table.schema()

    assert returned_schema.equals(schema)


def test_lazy_table_repr():
    """Test string representation."""
    schema = pa.schema([("x", pa.int64())])

    def make_stream():
        return pa.Table.from_pydict({"x": [1]}).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    repr_str = repr(table)

    assert "LazyArrowStreamTable" in repr_str


def test_lazy_evaluation():
    """Test that data is not loaded until query execution."""
    call_count = [0]

    schema = pa.schema([("x", pa.int64())])

    def make_stream():
        call_count[0] += 1
        return pa.Table.from_pydict({"x": [1, 2, 3]}).to_reader()

    # Create table - factory should NOT be called
    table = LazyArrowStreamTable(make_stream, schema)
    assert call_count[0] == 0, "Factory called during table creation"

    # Register with DataFusion - factory should NOT be called
    ctx = SessionContext()
    ctx.register_table("data", table)
    assert call_count[0] == 0, "Factory called during registration"

    # Execute query - factory should be called NOW
    result = ctx.sql("SELECT * FROM data").collect()
    assert call_count[0] == 1, "Factory should be called during execution"

    # Verify results
    assert len(result) > 0
    total_rows = sum(batch.num_rows for batch in result)
    assert total_rows == 3


def test_multiple_queries():
    """Test that the same table can be queried multiple times."""
    call_count = [0]

    schema = pa.schema([("x", pa.int64())])

    def make_stream():
        call_count[0] += 1
        return pa.Table.from_pydict({"x": [1, 2, 3]}).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)

    # First query
    result1 = ctx.sql("SELECT * FROM data").collect()
    assert call_count[0] == 1

    # Second query - factory should be called again
    result2 = ctx.sql("SELECT * FROM data LIMIT 1").collect()
    assert call_count[0] == 2, "Factory should be called for each query"


def test_aggregation_query():
    """Test aggregation queries work correctly."""
    schema = pa.schema([
        ("x", pa.int64()),
        ("y", pa.float64()),
    ])

    def make_stream():
        data = {"x": [1, 2, 3, 4, 5], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}
        return pa.Table.from_pydict(data).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)

    # Sum query
    result = ctx.sql("SELECT SUM(x) as sum_x, AVG(y) as avg_y FROM data").collect()
    assert len(result) == 1

    batch = result[0]
    assert batch.num_rows == 1

    # Verify results
    sum_x = batch.column("sum_x")[0].as_py()
    avg_y = batch.column("avg_y")[0].as_py()

    assert sum_x == 15  # 1+2+3+4+5
    assert avg_y == 30.0  # (10+20+30+40+50)/5


def test_filter_query():
    """Test filter queries work correctly."""
    schema = pa.schema([("x", pa.int64())])

    def make_stream():
        return pa.Table.from_pydict({"x": [1, 2, 3, 4, 5]}).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)

    result = ctx.sql("SELECT x FROM data WHERE x > 3").collect()

    total_rows = sum(batch.num_rows for batch in result)
    assert total_rows == 2  # 4 and 5


def test_projection_query():
    """Test projection (column selection) works correctly."""
    schema = pa.schema([
        ("a", pa.int64()),
        ("b", pa.int64()),
        ("c", pa.int64()),
    ])

    def make_stream():
        data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        return pa.Table.from_pydict(data).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)

    result = ctx.sql("SELECT a, c FROM data").collect()

    assert len(result) > 0
    batch = result[0]
    assert batch.num_columns == 2
    assert batch.schema.names == ["a", "c"]


def test_error_invalid_schema():
    """Test error handling for invalid schema."""
    with pytest.raises(TypeError):
        # Pass invalid schema type
        LazyArrowStreamTable(lambda: None, "not a schema")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
