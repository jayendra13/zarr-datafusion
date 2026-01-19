"""Tests for the xarray_reader module."""

import pytest
import pyarrow as pa

# Skip all tests if xarray is not installed
xr = pytest.importorskip("xarray")

from zarr_datafusion.xarray_reader import (
    XarrayRecordBatchReader,
    read_xarray_lazy,
    read_xarray_table,
    xarray_to_arrow_table,
    block_slices,
    parse_schema,
    pivot,
)

from conftest import SYNTHETIC_V3


class TestBlockSlices:
    """Tests for the block_slices function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_block_slices_with_chunks(self):
        """Test block_slices generates correct number of blocks."""
        chunks = {"time": 3, "lat": 5, "lon": 5}
        blocks = list(block_slices(self.ds, chunks))

        # time: ceil(7/3) = 3 blocks
        # lat: ceil(10/5) = 2 blocks
        # lon: ceil(10/5) = 2 blocks
        # Total: 3 * 2 * 2 = 12 blocks
        assert len(blocks) == 12

    def test_block_slices_single_chunk(self):
        """Test block_slices with chunk size >= dimension size."""
        chunks = {"time": 100, "lat": 100, "lon": 100}  # Larger than data
        blocks = list(block_slices(self.ds, chunks))

        # Should be single block covering entire dataset
        assert len(blocks) == 1

    def test_block_slices_returns_slices(self):
        """Test that block_slices returns proper slice objects."""
        chunks = {"time": 2}
        blocks = list(block_slices(self.ds, chunks))

        for block in blocks:
            for dim, sl in block.items():
                assert isinstance(sl, slice)
                # Should be able to use with isel
                chunk_ds = self.ds.isel({dim: sl})
                assert chunk_ds is not None


class TestParseSchema:
    """Tests for the parse_schema function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_parse_schema_columns(self):
        """Test that schema has correct columns."""
        schema = parse_schema(self.ds)

        expected_columns = {"time", "lat", "lon", "temperature", "humidity"}
        actual_columns = set(schema.names)

        assert expected_columns == actual_columns

    def test_parse_schema_types(self):
        """Test that schema has correct types."""
        schema = parse_schema(self.ds)

        # Check that we have proper Arrow types
        for field in schema:
            assert field.type is not None
            # All fields should be numeric for this test data
            assert pa.types.is_floating(field.type) or pa.types.is_integer(field.type) or pa.types.is_timestamp(field.type)


class TestPivot:
    """Tests for the pivot function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_pivot_returns_dataframe(self):
        """Test that pivot returns a pandas DataFrame."""
        import pandas as pd

        # Take a small slice
        small_ds = self.ds.isel(time=slice(0, 2), lat=slice(0, 2), lon=slice(0, 2))
        df = pivot(small_ds)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2 * 2 * 2  # 8 rows

    def test_pivot_has_all_columns(self):
        """Test that pivot includes coordinates and data variables."""
        small_ds = self.ds.isel(time=slice(0, 1), lat=slice(0, 1), lon=slice(0, 1))
        df = pivot(small_ds)

        expected_cols = {"time", "lat", "lon", "temperature", "humidity"}
        actual_cols = set(df.columns)

        assert expected_cols == actual_cols


class TestXarrayRecordBatchReader:
    """Tests for XarrayRecordBatchReader."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_reader_schema(self):
        """Test that reader has correct schema."""
        reader = XarrayRecordBatchReader(self.ds, chunks={"time": 3})

        assert reader.schema is not None
        assert set(reader.schema.names) == {"time", "lat", "lon", "temperature", "humidity"}

    def test_reader_lazy_creation(self):
        """Test that reader doesn't iterate on creation."""
        iterations = []

        def track(block):
            iterations.append(block)

        reader = XarrayRecordBatchReader(
            self.ds, chunks={"time": 3}, _iteration_callback=track
        )

        assert len(iterations) == 0, "Should not iterate during creation"

    def test_reader_iterates_on_consume(self):
        """Test that reader iterates when consumed."""
        iterations = []

        def track(block):
            iterations.append(block)

        reader = XarrayRecordBatchReader(
            self.ds, chunks={"time": 3}, _iteration_callback=track
        )

        # Consume via PyArrow
        pa_reader = pa.RecordBatchReader.from_stream(reader)
        _ = list(pa_reader)

        assert len(iterations) > 0, "Should iterate when consumed"

    def test_reader_single_consumption(self):
        """Test that reader can only be consumed once."""
        reader = XarrayRecordBatchReader(self.ds, chunks={"time": 3})

        # First consumption
        pa_reader = pa.RecordBatchReader.from_stream(reader)
        _ = list(pa_reader)

        # Second consumption should fail
        with pytest.raises(RuntimeError, match="already consumed"):
            pa.RecordBatchReader.from_stream(reader)

    def test_reader_batch_count(self):
        """Test that reader produces expected number of batches."""
        # With time=3, we get ceil(7/3) = 3 time chunks
        # Other dims are not chunked, so 3 batches total
        reader = XarrayRecordBatchReader(self.ds, chunks={"time": 3})

        pa_reader = pa.RecordBatchReader.from_stream(reader)
        batches = list(pa_reader)

        assert len(batches) == 3

    def test_reader_total_rows(self):
        """Test that reader produces all rows."""
        reader = XarrayRecordBatchReader(self.ds, chunks={"time": 3})

        pa_reader = pa.RecordBatchReader.from_stream(reader)
        batches = list(pa_reader)

        total_rows = sum(b.num_rows for b in batches)
        expected_rows = 7 * 10 * 10  # time * lat * lon

        assert total_rows == expected_rows


class TestReadXarrayLazy:
    """Tests for read_xarray_lazy function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_returns_reader(self):
        """Test that read_xarray_lazy returns XarrayRecordBatchReader."""
        reader = read_xarray_lazy(self.ds, chunks={"time": 3})

        assert isinstance(reader, XarrayRecordBatchReader)

    def test_can_consume(self):
        """Test that returned reader can be consumed."""
        reader = read_xarray_lazy(self.ds, chunks={"time": 3})

        pa_reader = pa.RecordBatchReader.from_stream(reader)
        batches = list(pa_reader)

        assert len(batches) > 0


class TestReadXarrayTable:
    """Tests for read_xarray_table function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_returns_lazy_table(self):
        """Test that read_xarray_table returns LazyArrowStreamTable."""
        from zarr_datafusion import LazyArrowStreamTable

        table = read_xarray_table(self.ds, chunks={"time": 3})

        assert isinstance(table, LazyArrowStreamTable)

    def test_can_register_with_datafusion(self):
        """Test that table can be registered with DataFusion."""
        from datafusion import SessionContext

        table = read_xarray_table(self.ds, chunks={"time": 3})

        ctx = SessionContext()
        ctx.register_table("data", table)

        # Should be queryable
        result = ctx.sql("SELECT COUNT(*) FROM data").collect()
        assert len(result) > 0

    def test_multiple_queries(self):
        """Test that table supports multiple queries."""
        from datafusion import SessionContext

        table = read_xarray_table(self.ds, chunks={"time": 3})

        ctx = SessionContext()
        ctx.register_table("data", table)

        # First query
        result1 = ctx.sql("SELECT COUNT(*) as cnt FROM data").collect()
        count1 = result1[0].column(0)[0].as_py()

        # Second query
        result2 = ctx.sql("SELECT COUNT(*) as cnt FROM data").collect()
        count2 = result2[0].column(0)[0].as_py()

        assert count1 == count2


class TestXarrayToArrowTable:
    """Tests for xarray_to_arrow_table function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SYNTHETIC_V3.exists():
            pytest.skip(f"Test data not found: {SYNTHETIC_V3}")
        self.ds = xr.open_zarr(str(SYNTHETIC_V3))

    def test_returns_table(self):
        """Test that xarray_to_arrow_table returns PyArrow Table."""
        table = xarray_to_arrow_table(self.ds, chunks={"time": 3})

        assert isinstance(table, pa.Table)

    def test_table_row_count(self):
        """Test that table has correct row count."""
        table = xarray_to_arrow_table(self.ds, chunks={"time": 3})

        expected_rows = 7 * 10 * 10  # time * lat * lon
        assert table.num_rows == expected_rows

    def test_table_columns(self):
        """Test that table has correct columns."""
        table = xarray_to_arrow_table(self.ds, chunks={"time": 3})

        expected_cols = {"time", "lat", "lon", "temperature", "humidity"}
        actual_cols = set(table.column_names)

        assert expected_cols == actual_cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
