//! Phase 2.6 checkpoint: the exact-cardinality oracle matches reality.
//!
//! Drives the real filter-lowering pipeline (resolve pushed-down filters against
//! actual coordinate values -> lower into a `ProductSet` -> count) and asserts the
//! resulting cardinality equals an INDEPENDENT brute-force count of matching
//! coordinate tuples, and equals the scan's own `calculate_filtered_rows`. That
//! proves the plan-time oracle agrees with the executor's row-count formula on
//! live inputs.

use datafusion::scalar::ScalarValue;

use zarr_datafusion::optimizer::cardinality::predicate::selection_from_filters;
use zarr_datafusion::optimizer::cardinality::IndexSet;
use zarr_datafusion::reader::filter::{
    calculate_coord_ranges, calculate_filtered_rows, CoordFilterKind, CoordFilters, CoordValuesRef,
};

fn f64s(scalars: &[f64]) -> Vec<ScalarValue> {
    scalars
        .iter()
        .map(|&v| ScalarValue::Float64(Some(v)))
        .collect()
}

fn closed_range(low: f64, high: f64) -> CoordFilterKind {
    CoordFilterKind::Range {
        low: Some(ScalarValue::Float64(Some(low))),
        high: Some(ScalarValue::Float64(Some(high))),
        low_inclusive: true,
        high_inclusive: true,
    }
}

#[test]
fn cardinality_matches_reality_range_and_inlist() {
    let names = vec!["time".to_string(), "lat".to_string(), "lon".to_string()];
    let time: Vec<i64> = (0..50).map(|i| i * 10).collect();
    let lat: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let lon: Vec<f64> = (0..20).map(|i| i as f64).collect();

    // lat BETWEEN 2 AND 5, lon IN (3, 7, 11), time unfiltered.
    let mut filters = CoordFilters::new();
    filters.push("lat", closed_range(2.0, 5.0));
    filters.push("lon", CoordFilterKind::InList(f64s(&[3.0, 7.0, 11.0])));

    let values = vec![
        CoordValuesRef::Int64(&time),
        CoordValuesRef::Float64(&lat),
        CoordValuesRef::Float64(&lon),
    ];

    let ps = selection_from_filters(&filters, &names, &values).expect("non-empty selection");

    // Independent reality: count matching tuples directly from the raw arrays.
    let lat_hits = lat.iter().filter(|&&v| (2.0..=5.0).contains(&v)).count();
    let lon_hits = lon.iter().filter(|v| [3.0, 7.0, 11.0].contains(v)).count();
    let expected = (time.len() * lat_hits * lon_hits) as u128;
    assert_eq!(expected, 50 * 4 * 3);
    assert_eq!(ps.cardinality(), expected);

    // And equals the scan's own row-count formula.
    let sels = calculate_coord_ranges(&filters, &names, &values).unwrap();
    assert_eq!(ps.cardinality(), calculate_filtered_rows(&sels) as u128);
}

#[test]
fn cardinality_matches_reality_and_on_same_axis() {
    let names = vec!["time".to_string(), "lat".to_string(), "lon".to_string()];
    let time: Vec<i64> = (0..7).collect();
    let lat: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let lon: Vec<f64> = (0..5).map(|i| i as f64).collect();

    // lat IN (1,2,3,4,5) AND lat BETWEEN 3 AND 9  ->  {3,4,5} (intersection).
    let mut filters = CoordFilters::new();
    filters.push(
        "lat",
        CoordFilterKind::InList(f64s(&[1.0, 2.0, 3.0, 4.0, 5.0])),
    );
    filters.push("lat", closed_range(3.0, 9.0));

    let values = vec![
        CoordValuesRef::Int64(&time),
        CoordValuesRef::Float64(&lat),
        CoordValuesRef::Float64(&lon),
    ];
    let ps = selection_from_filters(&filters, &names, &values).expect("non-empty selection");

    let lat_hits = lat
        .iter()
        .filter(|&&v| [1.0, 2.0, 3.0, 4.0, 5.0].contains(&v) && (3.0..=9.0).contains(&v))
        .count();
    assert_eq!(lat_hits, 3);
    assert_eq!(
        ps.cardinality(),
        (time.len() * lat_hits * lon.len()) as u128
    );
    assert_eq!(ps.cardinality(), 7 * 3 * 5);
}

#[test]
fn contradictory_filter_is_none() {
    let names = vec!["lat".to_string()];
    let lat: Vec<f64> = (0..10).map(|i| i as f64).collect();

    let mut filters = CoordFilters::new();
    filters.push(
        "lat",
        CoordFilterKind::Eq(ScalarValue::Float64(Some(100.0))),
    ); // no such value

    let values = vec![CoordValuesRef::Float64(&lat)];
    assert!(selection_from_filters(&filters, &names, &values).is_none());
}
