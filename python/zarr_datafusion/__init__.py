"""zarr-datafusion: Query Zarr and Xarray data with SQL via DataFusion.

This package provides Python bindings for zarr-datafusion, enabling SQL queries
on Zarr stores and Xarray datasets through Apache DataFusion.

The key feature is lazy evaluation: data is not read until query execution time,
enabling efficient streaming from large datasets like those served by tile servers.

Example with raw Arrow streams:
    >>> import pyarrow as pa
    >>> from datafusion import SessionContext
    >>> from zarr_datafusion import LazyArrowStreamTable
    >>>
    >>> # Create a factory that returns Arrow streams
    >>> def make_stream():
    ...     data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    ...     return pa.Table.from_pydict(data).to_reader()
    >>>
    >>> # Get schema from sample
    >>> sample = make_stream()
    >>> schema = sample.schema
    >>>
    >>> # Create lazy table - NO DATA LOADED YET
    >>> table = LazyArrowStreamTable(make_stream, schema)
    >>>
    >>> # Register with DataFusion
    >>> ctx = SessionContext()
    >>> ctx.register_table("data", table)
    >>>
    >>> # Data only loaded HERE during collect()
    >>> result = ctx.sql("SELECT * FROM data").collect()

Example with xarray:
    >>> import xarray as xr
    >>> from datafusion import SessionContext
    >>> from zarr_datafusion import read_xarray_table
    >>>
    >>> # Open xarray dataset (lazy)
    >>> ds = xr.open_zarr("data.zarr")
    >>>
    >>> # Create lazy table with chunked streaming
    >>> table = read_xarray_table(ds, chunks={'time': 100})
    >>>
    >>> # Register and query
    >>> ctx = SessionContext()
    >>> ctx.register_table("data", table)
    >>> result = ctx.sql("SELECT AVG(temperature) FROM data").collect()
"""

from zarr_datafusion._native import LazyArrowStreamTable

# Lazy imports for xarray functionality (optional dependency)
def __getattr__(name):
    if name in (
        "XarrayRecordBatchReader",
        "read_xarray_lazy",
        "read_xarray_table",
        "xarray_to_arrow_table",
        "block_slices",
        "parse_schema",
        "pivot",
    ):
        from zarr_datafusion import xarray_reader
        return getattr(xarray_reader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = [
    "LazyArrowStreamTable",
    # Xarray utilities (lazy imports)
    "XarrayRecordBatchReader",
    "read_xarray_lazy",
    "read_xarray_table",
    "xarray_to_arrow_table",
    "block_slices",
    "parse_schema",
    "pivot",
]
