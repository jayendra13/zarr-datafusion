"""zarr-datafusion: Query Zarr and Xarray data with SQL via DataFusion.

Example with xarray:
    >>> import xarray as xr
    >>> from datafusion import SessionContext
    >>> from zarr_datafusion import read_xarray_table
    >>>
    >>> ds = xr.open_zarr("data.zarr")
    >>> table = read_xarray_table(ds, chunks={'time': 100})
    >>>
    >>> ctx = SessionContext()
    >>> ctx.register_table("data", table)
    >>> result = ctx.sql("SELECT AVG(temperature) FROM data").collect()
"""

from zarr_datafusion._native import LazyArrowStreamTable


def __getattr__(name):
    if name == "read_xarray_table":
        from zarr_datafusion.xarray_reader import read_xarray_table
        return read_xarray_table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = ["LazyArrowStreamTable", "read_xarray_table"]
