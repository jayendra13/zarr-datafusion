"""zarr-datafusion: Query Zarr and Xarray data with SQL via DataFusion.

This package provides Python bindings for zarr-datafusion, enabling SQL queries
on Zarr stores and Xarray datasets through Apache DataFusion.

The key feature is lazy evaluation: data is not read until query execution time,
enabling efficient streaming from large datasets like those served by tile servers.

Example:
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
"""

from zarr_datafusion._native import LazyArrowStreamTable

__version__ = "0.1.0"
__all__ = ["LazyArrowStreamTable"]
