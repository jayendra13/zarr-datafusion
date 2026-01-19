"""Lazy Arrow stream reader for xarray Datasets.

This module provides utilities for efficiently streaming xarray data to
DataFusion via the Arrow C Data Interface. The key features are:

1. Lazy evaluation: data is only read when queries are executed
2. Chunked streaming: data is processed in memory-efficient blocks
3. Zero-copy where possible via Arrow's PyCapsule interface

The implementation follows the pattern from xarray-sql.
"""

from __future__ import annotations

import itertools
import typing as t

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

# Type aliases
Block = t.Dict[t.Hashable, slice]
Chunks = t.Optional[t.Dict[str, int]]


# =============================================================================
# Block slicing utilities (adapted from xarray-sql)
# =============================================================================

def _get_chunk_slicer(
    dim: t.Hashable, chunk_index: t.Mapping, chunk_bounds: t.Mapping
) -> slice:
    """Get a slice for a specific chunk along a dimension."""
    if dim in chunk_index:
        which_chunk = chunk_index[dim]
        return slice(
            chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1]
        )
    return slice(None)


def block_slices(ds: xr.Dataset, chunks: Chunks = None) -> t.Iterator[Block]:
    """Compute block slices for a chunked Dataset.

    This generates slice dictionaries that can be used with ds.isel() to
    extract each chunk of the dataset without loading all data at once.

    Args:
        ds: An xarray Dataset (must be chunked or chunks must be provided)
        chunks: Optional chunk specification like {'time': 240, 'lat': 10}

    Yields:
        Block dictionaries mapping dimension names to slices

    Example:
        >>> ds = xr.open_zarr('data.zarr', chunks={'time': 100})
        >>> for block in block_slices(ds):
        ...     chunk_ds = ds.isel(block)
        ...     # Process chunk_ds
    """
    if chunks is not None:
        # Create a temporary chunked copy to get chunk structure
        for_chunking = ds.copy(data=None, deep=False).chunk(chunks)
        chunk_dict = for_chunking.chunks
        del for_chunking
    else:
        chunk_dict = ds.chunks

    if not chunk_dict:
        # Not chunked - yield single block covering entire dataset
        yield {dim: slice(None) for dim in ds.dims}
        return

    # chunks is Dict[str, Tuple[int, ...]] from xarray
    chunk_bounds = {
        dim: np.cumsum((0,) + tuple(c))
        for dim, c in chunk_dict.items()
    }
    ichunk = {dim: range(len(tuple(c))) for dim, c in chunk_dict.items()}
    ick, icv = zip(*ichunk.items())  # Makes same order of keys and values
    chunk_idxs = (dict(zip(ick, i)) for i in itertools.product(*icv))
    blocks = (
        {
            dim: _get_chunk_slicer(dim, chunk_index, chunk_bounds)
            for dim in ds.dims
        }
        for chunk_index in chunk_idxs
    )
    yield from blocks


def explode(ds: xr.Dataset, chunks: Chunks = None) -> t.Iterator[xr.Dataset]:
    """Explode a dataset into its chunks.

    Args:
        ds: An xarray Dataset
        chunks: Optional chunk specification

    Yields:
        Dataset chunks
    """
    yield from (ds.isel(b) for b in block_slices(ds, chunks=chunks))


# =============================================================================
# Schema and conversion utilities
# =============================================================================

def pivot(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an xarray Dataset to a pandas DataFrame.

    This flattens the multidimensional data into a 2D tabular format,
    with coordinates becoming columns alongside data variables.
    """
    return ds.to_dataframe().reset_index()


def parse_schema(ds: xr.Dataset) -> pa.Schema:
    """Extract a PyArrow Schema from an xarray Dataset.

    This extracts the schema without loading any data, by inspecting
    the coordinate and data variable dtypes.

    Args:
        ds: An xarray Dataset

    Returns:
        A PyArrow Schema with columns for coordinates and data variables
    """
    columns = []

    # Add dimension coordinates
    for coord_name, coord_var in ds.coords.items():
        if coord_name in ds.dims:
            pa_type = pa.from_numpy_dtype(coord_var.dtype)
            columns.append(pa.field(str(coord_name), pa_type))

    # Add data variables
    for var_name, var in ds.data_vars.items():
        pa_type = pa.from_numpy_dtype(var.dtype)
        columns.append(pa.field(str(var_name), pa_type))

    return pa.schema(columns)


# =============================================================================
# XarrayRecordBatchReader - Lazy Arrow streaming
# =============================================================================

class XarrayRecordBatchReader:
    """A lazy Arrow stream reader for xarray Datasets.

    Implements the Arrow PyCapsule Interface (__arrow_c_stream__) to enable
    zero-copy, lazy streaming of xarray data to DataFusion and other Arrow
    consumers.

    The key property is that xarray blocks are only converted to Arrow
    RecordBatches when the consumer calls get_next (e.g., during DataFusion's
    collect()), NOT when the reader is created or registered.

    Example:
        >>> import xarray as xr
        >>> from zarr_datafusion.xarray_reader import XarrayRecordBatchReader
        >>> ds = xr.open_zarr('data.zarr')
        >>> reader = XarrayRecordBatchReader(ds, chunks={'time': 240})
        >>> # At this point, NO data has been read from xarray
        >>> # Data is only read when consumed:
        >>> import pyarrow as pa
        >>> pa_reader = pa.RecordBatchReader.from_stream(reader)
        >>> for batch in pa_reader:
        ...     print(batch.num_rows)  # Data read here
    """

    def __init__(
        self,
        ds: xr.Dataset,
        chunks: Chunks = None,
        *,
        _iteration_callback: t.Optional[t.Callable[[Block], None]] = None,
    ):
        """Initialize the lazy reader.

        Args:
            ds: An xarray Dataset. All data_vars must share the same dimensions.
            chunks: Xarray-like chunks specification (e.g., {'time': 240}).
                   If not provided, uses the Dataset's existing chunks.
            _iteration_callback: Internal callback for testing. Called with
                each block dict just before it's converted to Arrow.
        """
        self._ds = ds
        self._chunks = chunks
        self._schema = parse_schema(ds)
        self._iteration_callback = _iteration_callback
        self._consumed = False

        # Validate that all data variables have the same dimensions
        if len(ds.data_vars) > 0:
            fst = next(iter(ds.data_vars.values())).dims
            if not all(da.dims == fst for da in ds.data_vars.values()):
                raise ValueError(
                    "All data variables must have the same dimensions. "
                    "Please filter data_vars in the Dataset."
                )

    @property
    def schema(self) -> pa.Schema:
        """The Arrow schema for this stream."""
        return self._schema

    def _generate_batches(self) -> t.Iterator[pa.RecordBatch]:
        """Generate RecordBatches lazily from xarray blocks.

        This generator is only consumed when the Arrow stream's get_next
        is called, ensuring true lazy evaluation.
        """
        for block in block_slices(self._ds, self._chunks):
            # Call the iteration callback if provided (for testing)
            if self._iteration_callback is not None:
                self._iteration_callback(block)

            # Convert this block to a RecordBatch
            chunk_ds = self._ds.isel(block)
            df = pivot(chunk_ds)
            yield pa.RecordBatch.from_pandas(df, schema=self._schema)

    def __arrow_c_stream__(
        self, requested_schema: t.Optional[object] = None
    ) -> object:
        """Export as Arrow C Stream via PyCapsule.

        This method is called by Arrow consumers (like DataFusion) to get
        a C-level stream interface. The actual data iteration only begins
        when the consumer calls get_next on the stream.

        Args:
            requested_schema: Optional schema for type casting (passed to PyArrow)

        Returns:
            PyCapsule containing ArrowArrayStream pointer

        Raises:
            RuntimeError: If the stream has already been consumed
        """
        if self._consumed:
            raise RuntimeError(
                "Stream already consumed. XarrayRecordBatchReader can only "
                "be iterated once. Create a new reader for additional iterations."
            )
        self._consumed = True

        # Create a PyArrow RecordBatchReader from our generator
        # The generator is NOT consumed here - only when get_next is called
        reader = pa.RecordBatchReader.from_batches(
            self._schema, self._generate_batches()
        )

        # Delegate to PyArrow's __arrow_c_stream__ implementation
        return reader.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(
        self, requested_schema: t.Optional[object] = None
    ) -> object:
        """Export the schema as Arrow C Schema via PyCapsule.

        This allows consumers to inspect the schema without consuming the stream.
        """
        return self._schema.__arrow_c_schema__()


# =============================================================================
# High-level API functions
# =============================================================================

def read_xarray_lazy(
    ds: xr.Dataset, chunks: Chunks = None
) -> XarrayRecordBatchReader:
    """Create a lazy Arrow stream reader from an xarray Dataset.

    This is the recommended way to create a single-use stream from xarray data.
    Data is only read when the stream is consumed (e.g., during collect()).

    Args:
        ds: An xarray Dataset. All data_vars must share the same dimensions.
        chunks: Xarray-like chunks specification (e.g., {'time': 240}).

    Returns:
        An XarrayRecordBatchReader implementing __arrow_c_stream__

    Example:
        >>> import xarray as xr
        >>> from zarr_datafusion.xarray_reader import read_xarray_lazy
        >>> ds = xr.open_zarr('data.zarr')
        >>> reader = read_xarray_lazy(ds, chunks={'time': 240})
        >>> # Use with PyArrow or DataFusion
    """
    return XarrayRecordBatchReader(ds, chunks)


def read_xarray_table(
    ds: xr.Dataset,
    chunks: Chunks = None,
    *,
    _iteration_callback: t.Optional[t.Callable[[Block], None]] = None,
):
    """Create a lazy DataFusion table from an xarray Dataset.

    This is the simplest way to register xarray data with DataFusion.
    Data is only read when queries are executed (during collect()),
    not during registration. The table can be queried multiple times.

    Args:
        ds: An xarray Dataset. All data_vars must share the same dimensions.
        chunks: Xarray-like chunks specification (e.g., {'time': 240}).
        _iteration_callback: Internal callback for testing.

    Returns:
        A LazyArrowStreamTable ready for registration with DataFusion

    Example:
        >>> from datafusion import SessionContext
        >>> import xarray as xr
        >>> from zarr_datafusion.xarray_reader import read_xarray_table
        >>>
        >>> ds = xr.open_zarr('data.zarr')
        >>> table = read_xarray_table(ds, chunks={'time': 240})
        >>>
        >>> ctx = SessionContext()
        >>> ctx.register_table("data", table)
        >>>
        >>> # Data is only read here, during collect()
        >>> result = ctx.sql("SELECT AVG(temperature) FROM data").collect()
        >>> # Can query again - each query creates a fresh stream
        >>> result2 = ctx.sql("SELECT * FROM data LIMIT 10").collect()
    """
    from zarr_datafusion import LazyArrowStreamTable

    # Get schema from dataset without creating a stream
    schema = parse_schema(ds)

    # Create a factory function that produces fresh streams on each call
    def make_stream() -> XarrayRecordBatchReader:
        return XarrayRecordBatchReader(
            ds, chunks, _iteration_callback=_iteration_callback
        )

    return LazyArrowStreamTable(make_stream, schema)


# =============================================================================
# Utility for non-streaming reads (for comparison/testing)
# =============================================================================

def xarray_to_arrow_table(ds: xr.Dataset, chunks: Chunks = None) -> pa.Table:
    """Convert an xarray Dataset to a PyArrow Table.

    This is a convenience function that fully materializes the dataset.
    For large datasets, prefer read_xarray_lazy() or read_xarray_table().

    Args:
        ds: An xarray Dataset
        chunks: Optional chunk specification for processing

    Returns:
        A PyArrow Table containing all data
    """
    schema = parse_schema(ds)
    batches = []

    for block in block_slices(ds, chunks):
        chunk_ds = ds.isel(block)
        df = pivot(chunk_ds)
        batch = pa.RecordBatch.from_pandas(df, schema=schema)
        batches.append(batch)

    return pa.Table.from_batches(batches, schema=schema)
