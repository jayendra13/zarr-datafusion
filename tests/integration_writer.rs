//! Round-trip test for the Zarr v3 writer POC.
//!
//! Write a tiny dataset with the POC writer → load it with the existing
//! reader via DataFusion → assert coords + data values match what we wrote.

mod common;

use std::path::PathBuf;

use arrow::array::{Array, Float32Array, Float64Array, Int64Array};
use arrow::datatypes::DataType;
use common::execute_query_single;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;
use zarr_datafusion::writer::{write_zarr_v3, CoordSpec, DataVarSpec, WriteValues};

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("zarr_writer_roundtrip_{name}_{pid}_{nanos}.zarr"));
    p
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn writer_roundtrip_small_dataset() {
    // Layout (mirrors synthetic data shape used by the rest of the suite):
    //   lat(3) × lon(2) × time(2)  → temperature, humidity
    let path = tmp_path("small");
    let _guard = TempDir(path.clone());

    let coords = vec![
        CoordSpec {
            name: "lat".into(),
            values: WriteValues::F64(vec![10.0, 20.0, 30.0]),
        },
        CoordSpec {
            name: "lon".into(),
            values: WriteValues::F64(vec![100.0, 110.0]),
        },
        CoordSpec {
            name: "time".into(),
            values: WriteValues::I64(vec![1_000_000, 2_000_000]),
        },
    ];

    // Row-major: index = ((lat*lon_n)+lon)*time_n + time
    // Pick distinct values per cell so we can verify positionally.
    let mut temperature: Vec<f32> = Vec::with_capacity(3 * 2 * 2);
    let mut humidity: Vec<f64> = Vec::with_capacity(3 * 2 * 2);
    for lat_i in 0..3 {
        for lon_i in 0..2 {
            for time_i in 0..2 {
                temperature.push((lat_i * 100 + lon_i * 10 + time_i) as f32);
                humidity.push((lat_i * 100 + lon_i * 10 + time_i) as f64 + 0.5);
            }
        }
    }

    let data_vars = vec![
        DataVarSpec {
            name: "temperature".into(),
            values: WriteValues::F32(temperature.clone()),
            shape: vec![3, 2, 2],
        },
        DataVarSpec {
            name: "humidity".into(),
            values: WriteValues::F64(humidity.clone()),
            shape: vec![3, 2, 2],
        },
    ];

    write_zarr_v3(&path, &coords, &data_vars).expect("write_zarr_v3 failed");

    // === Read back via existing reader ===
    let path_str = path.to_str().unwrap();
    let (schema, _meta) = infer_schema_with_meta(path_str).expect("schema inference failed");
    let schema = std::sync::Arc::new(schema);

    // Sanity: schema should include all 5 columns
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(names.contains(&"lat"), "schema missing lat: {names:?}");
    assert!(names.contains(&"lon"), "schema missing lon: {names:?}");
    assert!(names.contains(&"time"), "schema missing time: {names:?}");
    assert!(names.contains(&"temperature"));
    assert!(names.contains(&"humidity"));

    let ctx = common::create_test_context();
    ctx.register_table(
        "t",
        std::sync::Arc::new(ZarrTable::new(schema.clone(), path_str)),
    )
    .unwrap();

    // Verify total row count = product of coord lengths
    let batch = execute_query_single(&ctx, "SELECT COUNT(*) AS n FROM t").await;
    let n = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 3 * 2 * 2, "expected 12 rows, got {n}");

    // Verify a specific cell: lat=20, lon=110, time=2_000_000
    // Expected position-derived values:
    //   lat_i=1, lon_i=1, time_i=1 → temperature = 1*100 + 1*10 + 1 = 111.0
    //   humidity = 111.5
    let batch = execute_query_single(
        &ctx,
        "SELECT temperature, humidity FROM t \
         WHERE lat = 20.0 AND lon = 110.0 AND time = 2000000",
    )
    .await;
    assert_eq!(batch.num_rows(), 1, "expected exactly 1 row");

    let temp_col = batch.column(0);
    let hum_col = batch.column(1);

    let temp = match temp_col.data_type() {
        DataType::Float32 => temp_col
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0) as f64,
        DataType::Float64 => temp_col
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        other => panic!("unexpected temperature dtype {other:?}"),
    };
    assert!((temp - 111.0).abs() < 1e-5, "temperature: got {temp}");

    let hum = hum_col
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("humidity must be f64")
        .value(0);
    assert!((hum - 111.5).abs() < 1e-9, "humidity: got {hum}");
}
