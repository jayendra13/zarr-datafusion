//! Python bindings for zarr-datafusion via PyO3.
//!
//! This module provides Python interoperability for zarr-datafusion, enabling:
//!
//! - `LazyArrowStreamTable`: A DataFusion TableProvider that wraps Python objects
//!   implementing `__arrow_c_stream__` (like xarray Datasets via pyarrow).
//!
//! The key feature is **lazy evaluation**: data is not read from Python until
//! query execution time (during `collect()`), not at registration time.
//!
//! # Example
//!
//! ```python
//! from datafusion import SessionContext
//! from zarr_datafusion import LazyArrowStreamTable
//! import xarray as xr
//! import pyarrow as pa
//!
//! # Xarray dataset
//! ds = xr.tutorial.open_dataset('air_temperature')
//!
//! # Create a factory that returns Arrow streams
//! def make_stream():
//!     # Convert xarray to Arrow table/stream
//!     df = ds.to_dataframe().reset_index()
//!     return pa.Table.from_pandas(df).to_reader()
//!
//! # Get schema
//! sample = make_stream()
//! schema = sample.schema
//!
//! # Create lazy table - NO DATA LOADED
//! table = LazyArrowStreamTable(make_stream, schema)
//!
//! # Register with DataFusion
//! ctx = SessionContext()
//! ctx.register_table("air", table)
//!
//! # Data only loaded HERE during collect()
//! result = ctx.sql("SELECT AVG(air) FROM air").collect()
//! ```

#[cfg(feature = "python")]
pub mod bindings {
    use std::ffi::{c_void, CString};
    use std::fmt::Debug;
    use std::sync::Arc;

    use arrow::array::RecordBatch;
    use arrow::datatypes::SchemaRef;
    use arrow::ffi_stream::ArrowArrayStreamReader;
    use arrow_pyarrow::FromPyArrow;
    use datafusion::catalog::streaming::StreamingTable;
    use datafusion::datasource::TableProvider;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::memory::MemoryStream;
    use datafusion::physical_plan::streaming::PartitionStream;
    use datafusion::physical_plan::SendableRecordBatchStream;
    use datafusion_ffi::table_provider::FFI_TableProvider;
    use pyo3::prelude::*;
    use pyo3::types::PyCapsule;
    use tokio::runtime::Handle;
    use tracing::{debug, warn};

    /// A partition stream that wraps a Python factory function that creates streams.
    ///
    /// The factory is called lazily on each `execute()` invocation, allowing
    /// the same table to be queried multiple times.
    struct PyArrowStreamPartition {
        schema: SchemaRef,
        /// A Python callable (factory) that returns a fresh stream implementing `__arrow_c_stream__`.
        /// Called on each execute() to create a new stream.
        stream_factory: Py<PyAny>,
    }

    impl PyArrowStreamPartition {
        fn new(stream_factory: Py<PyAny>, schema: SchemaRef) -> Self {
            Self {
                schema,
                stream_factory,
            }
        }
    }

    impl Debug for PyArrowStreamPartition {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PyArrowStreamPartition")
                .field("schema", &self.schema)
                .finish()
        }
    }

    impl PartitionStream for PyArrowStreamPartition {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            // Call the factory to get a fresh stream for this execution
            let batches: Vec<RecordBatch> = Python::with_gil(|py| {
                // Call the factory to get a fresh stream
                let stream_result = self.stream_factory.call0(py);

                match stream_result {
                    Ok(stream_obj) => {
                        let bound = stream_obj.bind(py);

                        match ArrowArrayStreamReader::from_pyarrow_bound(bound) {
                            Ok(reader) => {
                                // Collect batches, propagating errors as warnings
                                // In streaming context, we can't easily return errors,
                                // so we log and skip failed batches
                                reader
                                    .filter_map(|result| match result {
                                        Ok(batch) => Some(batch),
                                        Err(e) => {
                                            warn!("Failed to read batch from Python stream: {e}");
                                            None
                                        }
                                    })
                                    .collect()
                            }
                            Err(e) => {
                                warn!("Failed to create stream reader from Python object: {e}");
                                vec![]
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to call Python stream factory: {e}");
                        vec![]
                    }
                }
            });

            debug!(
                num_batches = batches.len(),
                total_rows = batches.iter().map(|b| b.num_rows()).sum::<usize>(),
                "PyArrowStreamPartition executed"
            );

            Box::pin(
                MemoryStream::try_new(batches, Arc::clone(&self.schema), None)
                    .expect("MemoryStream creation should not fail with valid schema"),
            )
        }
    }

    /// A lazy table provider that wraps a Python stream factory.
    ///
    /// This class implements the `__datafusion_table_provider__` protocol, allowing
    /// it to be registered with DataFusion's `SessionContext.register_table()`.
    ///
    /// Data is NOT read until query execution time - this enables true lazy evaluation.
    /// The factory function is called on each query execution to create a fresh stream,
    /// allowing the same table to be queried multiple times.
    ///
    /// # Example
    ///
    /// ```python
    /// from datafusion import SessionContext
    /// from zarr_datafusion import LazyArrowStreamTable
    /// import pyarrow as pa
    ///
    /// # Create a factory that produces Arrow streams
    /// def make_stream():
    ///     # Return any object implementing __arrow_c_stream__
    ///     data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    ///     return pa.Table.from_pydict(data).to_reader()
    ///
    /// # Get schema from a sample stream
    /// sample = make_stream()
    /// schema = sample.schema
    ///
    /// # Wrap factory in lazy table - NO DATA LOADED
    /// table = LazyArrowStreamTable(make_stream, schema)
    ///
    /// # Register with DataFusion - STILL NO DATA LOADED
    /// ctx = SessionContext()
    /// ctx.register_table("data", table)
    ///
    /// # Data only loaded HERE during collect()
    /// # Each query creates a fresh stream via the factory
    /// result = ctx.sql("SELECT * FROM data").collect()
    /// result2 = ctx.sql("SELECT * FROM data LIMIT 10").collect()  # Works!
    /// ```
    #[pyclass(name = "LazyArrowStreamTable")]
    pub struct LazyArrowStreamTable {
        /// The underlying StreamingTable
        table: Arc<StreamingTable>,
    }

    #[pymethods]
    impl LazyArrowStreamTable {
        /// Create a new LazyArrowStreamTable from a stream factory function.
        ///
        /// Args:
        ///     stream_factory: A callable that returns a Python object implementing
        ///             the Arrow PyCapsule interface (`__arrow_c_stream__`).
        ///             Called on each query execution to create a fresh stream.
        ///     schema: A PyArrow Schema for the table. Required since the factory
        ///             hasn't been called yet.
        ///
        /// Raises:
        ///     TypeError: If the schema is not a valid PyArrow Schema.
        #[new]
        fn new(stream_factory: &Bound<'_, PyAny>, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
            // Convert the PyArrow schema to Arrow schema
            use arrow::datatypes::Schema;
            use arrow_pyarrow::FromPyArrow;

            let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
            })?;
            let schema_ref = Arc::new(arrow_schema);

            // Create the partition stream with the factory
            let partition =
                PyArrowStreamPartition::new(stream_factory.clone().unbind(), schema_ref.clone());

            // Create the StreamingTable
            let table =
                StreamingTable::try_new(schema_ref, vec![Arc::new(partition)]).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create StreamingTable: {e}"
                    ))
                })?;

            Ok(Self {
                table: Arc::new(table),
            })
        }

        /// Returns a PyCapsule implementing the DataFusion TableProvider FFI.
        ///
        /// This method is called by DataFusion's `register_table()` to get a
        /// foreign table provider that can be used in queries.
        fn __datafusion_table_provider__<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<Bound<'py, PyCapsule>> {
            // Create the FFI table provider
            let provider: Arc<dyn TableProvider + Send> = self.table.clone();

            // Try to get the current tokio runtime handle (available when called from DataFusion context)
            let runtime = Handle::try_current().ok();

            // Create FFI wrapper
            let ffi_provider = FFI_TableProvider::new(
                provider,
                false, // can_support_pushdown_filters
                runtime,
            );

            // Create the capsule name
            let name = CString::new("datafusion_table_provider").unwrap();

            // Create the PyCapsule with a destructor closure
            // The PyCapsule takes ownership of the FFI_TableProvider
            PyCapsule::new_with_destructor(
                py,
                ffi_provider,
                Some(name),
                |_provider: FFI_TableProvider, _context: *mut c_void| {
                    // The FFI_TableProvider will be dropped automatically
                },
            )
        }

        /// Get the schema of the table as a PyArrow Schema.
        fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            use arrow_pyarrow::ToPyArrow;
            self.table
                .schema()
                .to_pyarrow(py)
                .map(|bound| bound.unbind())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
        }

        fn __repr__(&self) -> String {
            format!("LazyArrowStreamTable(schema={:?})", self.table.schema())
        }
    }

    /// Python module initialization
    #[pymodule]
    pub fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<LazyArrowStreamTable>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use bindings::*;
