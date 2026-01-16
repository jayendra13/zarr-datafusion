//! User-Defined Table Functions for Zarr-DataFusion
//!
//! Provides custom table functions that extend DataFusion's SQL capabilities
//! with Zarr-specific functionality.

use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{TableFunctionImpl, TableProvider};
use datafusion::common::{Result, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::Expr;
use datafusion::prelude::SessionContext;

use crate::datasource::zarr::ZarrTable;
use crate::reader::schema_inference::ZarrStoreMeta;

/// Register all Zarr-specific table functions with the session context
pub fn register_zarr_functions(ctx: &SessionContext) {
    ctx.register_udtf(
        "zarr_describe",
        Arc::new(ZarrDescribeFunc::new(ctx.clone())),
    );
}

/// Table function that provides extended DESCRIBE output for Zarr tables
///
/// Usage: `SELECT * FROM zarr_describe('table_name')`
///
/// Returns columns:
/// - column_name: Name of the column
/// - type: "coord", "data_var", or empty for non-Zarr tables
/// - dimension: Dimension name for coords, "(dim1: size1, ...)" for data vars
/// - size: Number of elements
/// - chunks: Chunk sizes as "(c1, c2, ...)"
pub struct ZarrDescribeFunc {
    ctx: SessionContext,
}

impl std::fmt::Debug for ZarrDescribeFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrDescribeFunc").finish()
    }
}

impl ZarrDescribeFunc {
    pub fn new(ctx: SessionContext) -> Self {
        Self { ctx }
    }
}

impl TableFunctionImpl for ZarrDescribeFunc {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        // Extract table name from first argument
        let table_name = extract_string_literal(exprs.first().ok_or_else(|| {
            DataFusionError::Plan("zarr_describe requires a table name argument".into())
        })?)?;

        // Look up the table provider
        let provider = futures::executor::block_on(self.ctx.table_provider(&table_name))?;

        let schema = provider.schema();

        // Try to get Zarr-specific metadata
        let zarr_table = provider.as_any().downcast_ref::<ZarrTable>();
        let is_zarr_table = zarr_table.is_some();
        let store_meta = zarr_table.and_then(|t| t.store_meta());

        // Build the describe RecordBatch (with Zarr-specific columns only for Zarr tables)
        let batch = build_describe_batch(&schema, store_meta, is_zarr_table)?;

        // Return as MemTable
        Ok(Arc::new(MemTable::try_new(
            batch.schema(),
            vec![vec![batch]],
        )?))
    }
}

/// Extract a string literal from an expression
fn extract_string_literal(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Ok(s.clone()),
        Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Ok(s.clone()),
        _ => Err(DataFusionError::Plan(
            "zarr_describe expects a string literal table name".into(),
        )),
    }
}

/// Build a RecordBatch with extended describe information
/// For Zarr tables, includes type, dimension, size, chunks columns
/// For non-Zarr tables, only includes column_name, data_type, is_nullable
fn build_describe_batch(
    schema: &SchemaRef,
    meta: Option<&ZarrStoreMeta>,
    is_zarr_table: bool,
) -> Result<RecordBatch> {
    // Build basic column arrays (common to all tables)
    let mut column_names: Vec<String> = Vec::new();
    let mut data_types: Vec<String> = Vec::new();
    let mut is_nullables: Vec<String> = Vec::new();

    for field in schema.fields() {
        column_names.push(field.name().clone());
        data_types.push(format!("{:?}", field.data_type()));
        is_nullables.push(if field.is_nullable() { "YES" } else { "NO" }.to_string());
    }

    let column_name_array = Arc::new(StringArray::from(column_names));
    let data_type_array = Arc::new(StringArray::from(data_types));
    let is_nullable_array = Arc::new(StringArray::from(is_nullables));

    // For non-Zarr tables, return only basic columns
    if !is_zarr_table {
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("column_name", DataType::Utf8, false),
            Field::new("data_type", DataType::Utf8, false),
            Field::new("is_nullable", DataType::Utf8, false),
        ]));

        return RecordBatch::try_new(
            output_schema,
            vec![column_name_array, data_type_array, is_nullable_array],
        )
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None));
    }

    // For Zarr tables, include extended metadata columns
    use std::collections::HashMap;

    // Build lookup maps from metadata
    let coord_map: HashMap<&str, &crate::reader::schema_inference::ZarrArrayMeta> = meta
        .map(|m| m.coords.iter().map(|c| (c.name.as_str(), c)).collect())
        .unwrap_or_default();

    let data_var_map: HashMap<&str, &crate::reader::schema_inference::ZarrArrayMeta> = meta
        .map(|m| m.data_vars.iter().map(|v| (v.name.as_str(), v)).collect())
        .unwrap_or_default();

    // Build coordinate size lookup for dimension strings
    let coord_sizes: HashMap<&str, u64> = meta
        .map(|m| m.coords.iter().map(|c| (c.name.as_str(), c.shape[0])).collect())
        .unwrap_or_default();

    // Build Zarr-specific arrays
    let mut types: Vec<Option<String>> = Vec::new();
    let mut dimensions: Vec<Option<String>> = Vec::new();
    let mut sizes: Vec<Option<String>> = Vec::new();
    let mut chunks_col: Vec<Option<String>> = Vec::new();

    for field in schema.fields() {
        let name = field.name();

        // Determine type, dimension, size, and chunks based on Zarr metadata
        if let Some(coord) = coord_map.get(name.as_str()) {
            types.push(Some("coord".to_string()));
            dimensions.push(Some(format!("({})", name)));
            sizes.push(Some(coord.shape[0].to_string()));
            // Coords typically have simple chunks
            chunks_col.push(coord.chunks.as_ref().map(|c| {
                format!(
                    "({})",
                    c.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }));
        } else if let Some(data_var) = data_var_map.get(name.as_str()) {
            types.push(Some("data_var".to_string()));
            // Use the variable's actual dimensions, not all coordinates
            let dims_str = if let Some(dims) = &data_var.dimensions {
                // Variable has explicit dimension names - use them with sizes
                format!(
                    "({})",
                    dims.iter()
                        .map(|d| {
                            let size = coord_sizes.get(d.as_str()).copied().unwrap_or(0);
                            format!("{}: {}", d, size)
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                // Fallback: infer from shape length matching coordinate count
                // This is less accurate but handles legacy data
                let all_coords: Vec<_> = meta
                    .map(|m| m.coords.iter().collect())
                    .unwrap_or_default();
                if data_var.shape.len() == all_coords.len() {
                    format!(
                        "({})",
                        all_coords
                            .iter()
                            .map(|c| format!("{}: {}", c.name, c.shape[0]))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    // Shape doesn't match all coords - just show shape
                    format!(
                        "(shape: {:?})",
                        data_var.shape
                    )
                }
            };
            dimensions.push(Some(dims_str));
            // Size is total elements (product of shape)
            let total: u64 = data_var.shape.iter().product();
            sizes.push(Some(total.to_string()));
            // Show chunk sizes like "(160, 145, 144)"
            chunks_col.push(data_var.chunks.as_ref().map(|c| {
                format!(
                    "({})",
                    c.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }));
        } else {
            // No Zarr metadata or unknown column
            types.push(None);
            dimensions.push(None);
            sizes.push(None);
            chunks_col.push(None);
        }
    }

    // Create Zarr-specific Arrow arrays
    let type_array = Arc::new(StringArray::from(types));
    let dimension_array = Arc::new(StringArray::from(dimensions));
    let size_array = Arc::new(StringArray::from(sizes));
    let chunks_array = Arc::new(StringArray::from(chunks_col));

    // Create full schema for Zarr tables
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("column_name", DataType::Utf8, false),
        Field::new("data_type", DataType::Utf8, false),
        Field::new("is_nullable", DataType::Utf8, false),
        Field::new("type", DataType::Utf8, true),
        Field::new("dimension", DataType::Utf8, true),
        Field::new("size", DataType::Utf8, true),
        Field::new("chunks", DataType::Utf8, true),
    ]));

    RecordBatch::try_new(
        output_schema,
        vec![
            column_name_array,
            data_type_array,
            is_nullable_array,
            type_array,
            dimension_array,
            size_array,
            chunks_array,
        ],
    )
    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}
