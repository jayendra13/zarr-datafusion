//! Recognizing aggregate-pushdown opportunities (Phase 7.2, observe-only).
//!
//! An `AGG(data_var) [GROUP BY coord]` directly over a `ZarrExec` can, in principle,
//! be folded chunk-by-chunk during the scan instead of materializing the flattened
//! selection — *if* the number of output groups is known to fit in memory. This
//! module implements the **recognizer**: it inspects a physical `AggregateExec` over
//! a `ZarrExec`, decides whether the aggregates and grouping are of a pushable shape,
//! and (in Phase 7.2) only *reports* the opportunity. The rewrite lands in 7.3.
//!
//! Pushable shape (this phase):
//! - aggregates are `SUM`/`COUNT`/`AVG`/`MIN`/`MAX`, none `DISTINCT`;
//! - each aggregate argument is a **data variable** (or `COUNT(*)`/`COUNT(coord)`);
//! - `GROUP BY` is empty, or on **coordinate columns** (each mapped to a cube axis).
//!   Grouping by a periodic function of a coordinate (e.g. month-of-time) is deferred
//!   to Phase 7.5.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::expressions::Column;

use super::{group_cardinality, AxisSet, GroupKey, ProductSet};
use crate::physical_plan::zarr_exec::ZarrExec;
use crate::reader::schema_inference::ZarrStoreMeta;

/// A pushable aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

/// One aggregate to push: its function and the data-variable column it reads
/// (`None` for `COUNT(*)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggSpec {
    pub kind: AggKind,
    pub column: Option<String>,
}

/// A recognized aggregate-pushdown opportunity over a `ZarrExec` scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushdownCandidate {
    pub aggs: Vec<AggSpec>,
    pub group_keys: Vec<GroupKey>,
    /// Coordinate names behind `group_keys`, in order — for logging/EXPLAIN.
    pub group_names: Vec<String>,
}

/// Decide whether `agg` directly over `zarr` is a pushable-shape aggregate.
///
/// Returns `None` (decline) when the store metadata is absent, any aggregate is
/// `DISTINCT` or an unsupported function, an aggregate reads a coordinate (other
/// than `COUNT`), or a `GROUP BY` key isn't a bare coordinate column. Pure and
/// side-effect-free — the caller decides what to do with a recognized candidate.
pub fn recognize(agg: &AggregateExec, zarr: &ZarrExec) -> Option<PushdownCandidate> {
    let meta = zarr.store_meta()?;
    let coord_axis: HashMap<&str, usize> = meta
        .coords
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();
    let data_vars: HashSet<&str> = meta.data_vars.iter().map(|d| d.name.as_str()).collect();

    // --- aggregate functions ---
    let mut aggs = Vec::new();
    for a in agg.aggr_expr() {
        if a.is_distinct() {
            return None;
        }
        let kind = match a.fun().name() {
            "sum" => AggKind::Sum,
            "count" => AggKind::Count,
            "avg" => AggKind::Avg,
            "min" => AggKind::Min,
            "max" => AggKind::Max,
            _ => return None,
        };
        let column = column_name(&a.expressions());
        let ok = match (&kind, &column) {
            // COUNT(*) / COUNT(coord) / COUNT(data_var) all just count rows.
            (AggKind::Count, _) => true,
            // Other aggregates must read a data variable, not a coordinate.
            (_, Some(name)) => data_vars.contains(name.as_str()),
            (_, None) => false,
        };
        if !ok {
            return None;
        }
        aggs.push(AggSpec { kind, column });
    }
    if aggs.is_empty() {
        return None;
    }

    // --- grouping ---
    let mut group_keys = Vec::new();
    let mut group_names = Vec::new();
    for (_, name) in agg.group_expr().expr() {
        // A coordinate `GROUP BY` names the coordinate in its output alias. Anything
        // else (periodic function, data variable, expression) isn't handled here.
        let axis = *coord_axis.get(name.as_str())?;
        group_keys.push(GroupKey::Axis(axis));
        group_names.push(name.clone());
    }

    Some(PushdownCandidate {
        aggs,
        group_keys,
        group_names,
    })
}

/// The whole cube as an unfiltered selection — its group count is the *maximum*
/// possible, so `group_cardinality(universe, keys) <= budget` guarantees any
/// filtered subset fits too. Metadata-only (no coordinate reads).
pub fn universe(meta: &ZarrStoreMeta) -> ProductSet {
    let axes = meta
        .coords
        .iter()
        .map(|c| AxisSet::interval(0, c.shape.first().copied().unwrap_or(0)))
        .collect();
    ProductSet::single(axes)
}

/// Upper bound on the output group count for `cand` over `zarr`'s cube — the group
/// count of the full (unfiltered) cube under the candidate's keys.
pub fn max_group_count(cand: &PushdownCandidate, meta: &ZarrStoreMeta) -> u128 {
    group_cardinality(&universe(meta), &cand.group_keys)
}

/// The column an aggregate reads, searching *through* wrapper expressions such as
/// the `CAST` that `AVG` inserts (`AVG(temperature)` lowers to
/// `avg(CAST(temperature AS Float64))`). Returns the first `Column` found in tree
/// order — sufficient for the single-data-column aggregates we push.
fn column_name(exprs: &[Arc<dyn PhysicalExpr>]) -> Option<String> {
    exprs.iter().find_map(find_column)
}

fn find_column(e: &Arc<dyn PhysicalExpr>) -> Option<String> {
    if let Some(c) = e.downcast_ref::<Column>() {
        return Some(c.name().to_string());
    }
    e.children().into_iter().find_map(find_column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::schema_inference::{ZarrArrayMeta, ZarrStoreMeta};

    fn coord(name: &str, size: u64) -> ZarrArrayMeta {
        ZarrArrayMeta {
            name: name.into(),
            data_type: "int64".into(),
            shape: vec![size],
            chunks: Some(vec![size]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: Some(vec![name.into()]),
        }
    }

    fn meta() -> ZarrStoreMeta {
        // Cube axes: [time, lat, lon]; one data variable `temperature`.
        ZarrStoreMeta {
            total_rows: 0,
            coords: vec![coord("time", 12), coord("lat", 5), coord("lon", 7)],
            data_vars: vec![ZarrArrayMeta {
                name: "temperature".into(),
                data_type: "float32".into(),
                shape: vec![12, 5, 7],
                chunks: Some(vec![1, 5, 7]),
                coord_min_max: None,
                cf_time_attrs: None,
                dimensions: Some(vec!["time".into(), "lat".into(), "lon".into()]),
            }],
        }
    }

    #[test]
    fn universe_group_count_matches_axis_extents() {
        let m = meta();
        // GROUP BY lat -> 5 groups; GROUP BY lat, lon -> 35; global -> 1.
        assert_eq!(group_cardinality(&universe(&m), &[GroupKey::Axis(1)]), 5);
        assert_eq!(
            group_cardinality(&universe(&m), &[GroupKey::Axis(1), GroupKey::Axis(2)]),
            35
        );
        assert_eq!(group_cardinality(&universe(&m), &[]), 1);
    }
}
