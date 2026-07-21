//! Admission and grid derivation: from the physical plan feeding the sink, decide
//! whether it is a well-defined write and derive the target grid's *structure*.
//!
//! Phase 3 of docs/zarr-write-roundtrip-plan.md. The governing question (§5.8) is
//! *what is the admissible shape of the query feeding the sink?* — and the answer
//! must be **expressed in terms of `optimizer::cardinality`, not reinvented**.
//! Admission and derivation are the same act: a query is admissible *because* we
//! could derive its target grid; a query that cannot be lowered is exactly the one
//! that is rejected. So this is a fallible constructor, not a boolean gate.
//!
//! This module derives the grid *structure* — which source axes form the target
//! grid, and the data variables to write — with **no store I/O**. Materialising the
//! grid's coordinate *values* (loading the source coord arrays and gathering them
//! by any subset filter) is a downstream step that *consumes* a [`WriteShape`]; it
//! does not re-derive it, so there is still one implementation of "what is a legal
//! write" (the Q1 concern in the plan discussion).
//!
//! ## The rule that closes the §5.4 corruption hole
//!
//! A Zarr chunk is written whole, and the sink scatters rows by grid index. If a
//! source coordinate axis is neither **projected** (kept as an output coordinate)
//! nor **reduced** (grouped away by an aggregate), then many source rows collapse
//! onto the same target cell and the last write silently wins. So:
//!
//! - **Pure projection** (no `GROUP BY`): *every* source axis must appear as an
//!   output coordinate column. A missing axis is rejected as an ambiguous collapse.
//! - **Reduce** (`GROUP BY` on coordinate axes): the target grid is the group-key
//!   axes; the remaining axes are legitimately reduced. Admissible exactly when the
//!   aggregate is of pushable shape — which is `recognize()`'s existing decision,
//!   reused rather than duplicated.

use std::fmt;
use std::sync::Arc;

use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::optimizer::cardinality::pushdown::recognize;
use crate::physical_plan::zarr_exec::ZarrExec;

use super::skeleton::WriteDataType;

/// A data variable the write will produce: its output name and element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteVar {
    pub name: String,
    pub data_type: WriteDataType,
}

/// The structural shape of a write derived from the plan feeding the sink.
///
/// Carries the target grid's axes (in dimension order — a subset/reorder of the
/// source axes) and the data variables to write. Coordinate *values* are not here:
/// materialising them is a downstream step that consumes this shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteShape {
    /// Target grid axis names, in source dimension order.
    pub grid_axes: Vec<String>,
    /// Source-axis index (into the store's coordinate list) for each grid axis,
    /// in the same order as `grid_axes`.
    pub grid_axis_source: Vec<usize>,
    /// Data variables to write.
    pub data_vars: Vec<WriteVar>,
    /// True when the write reduces the source cube (the plan has a `GROUP BY`);
    /// false for a pure projection.
    pub is_reduce: bool,
}

/// Why a plan is not an admissible write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// No single-source `ZarrExec` reachable — e.g. a join, or a non-Zarr source.
    NotSingleZarrScan,
    /// The scan carries no store metadata, so axes cannot be resolved.
    NoStoreMeta,
    /// A pure projection dropped a source axis (neither projected nor reduced),
    /// which would collapse rows onto shared cells.
    DroppedAxis(String),
    /// The plan has an aggregate, but not of a pushable shape (`recognize` declined
    /// — e.g. a computed aggregate argument, `DISTINCT`, or a non-coordinate key).
    UnpushableAggregate,
    /// A global aggregate (no `GROUP BY`) yields a scalar, not a grid to write.
    ScalarAggregate,
    /// An output data column has a type with no writable Zarr mapping.
    UnwritableColumn { name: String, why: String },
    /// The output has no data variables to write.
    NoDataVariables,
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSingleZarrScan => write!(
                f,
                "not a single-source Zarr scan (a write must read one Zarr store; \
                 joins and non-Zarr sources are not admissible)"
            ),
            Self::NoStoreMeta => write!(f, "the Zarr scan carries no store metadata"),
            Self::DroppedAxis(a) => write!(
                f,
                "coordinate axis '{a}' is neither selected nor grouped: without it \
                 in the output, multiple source rows collapse onto the same target \
                 cell (last write wins). Select it, or reduce it with GROUP BY."
            ),
            Self::UnpushableAggregate => write!(
                f,
                "the aggregate is not of a writable shape (a computed argument, \
                 DISTINCT, or a non-coordinate GROUP BY key)"
            ),
            Self::ScalarAggregate => write!(
                f,
                "a global aggregate produces a single scalar, not a grid to write; \
                 add a GROUP BY on coordinate axes"
            ),
            Self::UnwritableColumn { name, why } => {
                write!(f, "output column '{name}' cannot be written: {why}")
            }
            Self::NoDataVariables => write!(
                f,
                "the query outputs no data variables to write (only coordinates)"
            ),
        }
    }
}

impl std::error::Error for RejectReason {}

/// Walk down single-child plan nodes to the `ZarrExec`, capturing an
/// `AggregateExec` on the way. Returns `None` (not admissible) at any node with a
/// number of children other than one — which is what rejects joins and other
/// multi-input plans. Mirrors `cardinality::rule::descend_to_zarr`, but the
/// aggregate is optional so a pure projection is reachable too.
pub(crate) fn find_scan(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<(Option<&AggregateExec>, &ZarrExec)> {
    fn go<'a>(
        node: &'a Arc<dyn ExecutionPlan>,
        agg: Option<&'a AggregateExec>,
    ) -> Option<(Option<&'a AggregateExec>, &'a ZarrExec)> {
        if let Some(zarr) = node.downcast_ref::<ZarrExec>() {
            return Some((agg, zarr));
        }
        let agg = node.downcast_ref::<AggregateExec>().or(agg);
        let children = node.children();
        if children.len() != 1 {
            return None;
        }
        go(children[0], agg)
    }
    go(plan, None)
}

/// Derive the write's structural shape from the physical plan feeding the sink, or
/// reject it. See the module docs for the admission rule.
pub fn derive_write_shape(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<WriteShape, RejectReason> {
    let (maybe_agg, zarr) = find_scan(plan).ok_or(RejectReason::NotSingleZarrScan)?;
    let meta = zarr.store_meta().ok_or(RejectReason::NoStoreMeta)?;

    // Source axes, in dimension order, with a name -> index lookup.
    let axis_names: Vec<&str> = meta.coords.iter().map(|c| c.name.as_str()).collect();
    let axis_index = |name: &str| axis_names.iter().position(|a| *a == name);

    let out = plan.schema();

    // Every output column that is not a source coordinate is a data variable to
    // write. Its element type is the widened Arrow type (gap 5). Output columns
    // that *are* coordinates are grid axes, handled below.
    let mut data_vars = Vec::new();
    for field in out.fields() {
        if axis_index(field.name()).is_some() {
            continue; // a coordinate column, not a data variable
        }
        let data_type = WriteDataType::from_arrow(field.data_type()).map_err(|e| {
            RejectReason::UnwritableColumn {
                name: field.name().clone(),
                why: e.to_string(),
            }
        })?;
        data_vars.push(WriteVar {
            name: field.name().clone(),
            data_type,
        });
    }
    if data_vars.is_empty() {
        return Err(RejectReason::NoDataVariables);
    }

    // Grid axes: group-key axes for a reduce, all projected axes for a projection.
    let (grid_source, is_reduce) = match maybe_agg {
        Some(agg) => {
            let cand = recognize(agg, zarr).ok_or(RejectReason::UnpushableAggregate)?;
            if cand.group_keys.is_empty() {
                return Err(RejectReason::ScalarAggregate);
            }
            // recognize() has already established every key is a coordinate axis.
            let mut idx: Vec<usize> = cand.group_keys.iter().map(|k| k.axis()).collect();
            idx.sort_unstable();
            idx.dedup();
            (idx, true)
        }
        None => {
            // Pure projection: every source axis must be present as an output
            // coordinate, or rows collapse (§5.4).
            let projected: std::collections::HashSet<&str> = out
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .filter(|n| axis_index(n).is_some())
                .collect();
            for (i, name) in axis_names.iter().enumerate() {
                if !projected.contains(name) {
                    return Err(RejectReason::DroppedAxis(axis_names[i].to_string()));
                }
            }
            ((0..axis_names.len()).collect(), false)
        }
    };

    let grid_axes = grid_source.iter().map(|&i| axis_names[i].to_string()).collect();

    Ok(WriteShape {
        grid_axes,
        grid_axis_source: grid_source,
        data_vars,
        is_reduce,
    })
}
