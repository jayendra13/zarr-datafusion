"""Tests for LazyArrowStreamTable."""

import pyarrow as pa
from datafusion import SessionContext
from zarr_datafusion import LazyArrowStreamTable


def test_lazy_evaluation():
    """Verify data is not loaded until query execution."""
    call_count = [0]
    schema = pa.schema([("x", pa.int64())])

    def make_stream():
        call_count[0] += 1
        return pa.Table.from_pydict({"x": [1, 2, 3]}).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)
    assert call_count[0] == 0, "Factory called before query"

    result = ctx.sql("SELECT * FROM data").collect()
    assert call_count[0] == 1, "Factory not called during query"
    assert sum(b.num_rows for b in result) == 3


def test_multiple_queries():
    """Verify table can be queried multiple times (factory called each time)."""
    call_count = [0]
    schema = pa.schema([("x", pa.int64())])

    def make_stream():
        call_count[0] += 1
        return pa.Table.from_pydict({"x": [1, 2, 3]}).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)

    ctx.sql("SELECT * FROM data").collect()
    ctx.sql("SELECT * FROM data LIMIT 1").collect()
    assert call_count[0] == 2


def test_aggregation_and_filter():
    """Verify SQL operations work correctly."""
    schema = pa.schema([("x", pa.int64()), ("y", pa.float64())])

    def make_stream():
        return pa.Table.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}).to_reader()

    table = LazyArrowStreamTable(make_stream, schema)
    ctx = SessionContext()
    ctx.register_table("data", table)

    # Aggregation
    result = ctx.sql("SELECT SUM(x), AVG(y) FROM data").collect()
    assert result[0].column(0)[0].as_py() == 15
    assert result[0].column(1)[0].as_py() == 30.0

    # Filter
    result = ctx.sql("SELECT x FROM data WHERE x > 3").collect()
    assert sum(b.num_rows for b in result) == 2
