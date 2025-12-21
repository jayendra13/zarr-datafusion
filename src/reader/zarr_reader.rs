
use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch},
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use datafusion::{
    common::DataFusionError, error::Result, execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};
use futures::stream;
use std::sync::Arc;
use zarrs::{array::Array, array_subset::ArraySubset, filesystem::FilesystemStore};

// Column indices
const COL_TIMESTAMP: usize = 0;
const COL_LAT: usize = 1;
const COL_LON: usize = 2;
const COL_TEMPERATURE: usize = 3;
const COL_HUMIDITY: usize = 4;

pub fn zarr_weather_schema() -> Schema {
    Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("lat", DataType::Int64, false),
        Field::new("lon", DataType::Int64, false),
        Field::new("temperature", DataType::Int64, false),
        Field::new("humidity", DataType::Int64, false),
    ])
}

fn zarr_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
) -> Result<SendableRecordBatchStream> {
    let store = Arc::new(FilesystemStore::new(store_path).map_err(zarr_err)?);

    // Determine which columns we need
    let projected_indices = projection
        .clone()
        .unwrap_or_else(|| (0..schema.fields().len()).collect());

    let need_timestamp = projected_indices.contains(&COL_TIMESTAMP);
    let need_lat = projected_indices.contains(&COL_LAT);
    let need_lon = projected_indices.contains(&COL_LON);
    let need_temp = projected_indices.contains(&COL_TEMPERATURE);
    let need_humid = projected_indices.contains(&COL_HUMIDITY);

    // Open arrays to get shapes (metadata only, no data read yet)
    let time_array = Array::open(store.clone(), "/time").map_err(zarr_err)?;
    let lat_array = Array::open(store.clone(), "/lat").map_err(zarr_err)?;
    let lon_array = Array::open(store.clone(), "/lon").map_err(zarr_err)?;

    let ntime = time_array.shape()[0] as usize;
    let nlat = lat_array.shape()[0] as usize;
    let nlon = lon_array.shape()[0] as usize;
    let total_rows = ntime * nlat * nlon;

    // Only read coordinate values if needed
    let time_vals: Option<Vec<i64>> = if need_timestamp {
        let (vals, _) = time_array
            .retrieve_array_subset_ndarray::<i64>(&ArraySubset::new_with_shape(vec![
                ntime as u64,
            ]))
            .map_err(zarr_err)?
            .into_raw_vec_and_offset();
        Some(vals)
    } else {
        None
    };

    let lat_vals: Option<Vec<i64>> = if need_lat {
        let (vals, _) = lat_array
            .retrieve_array_subset_ndarray::<i64>(&ArraySubset::new_with_shape(vec![nlat as u64]))
            .map_err(zarr_err)?
            .into_raw_vec_and_offset();
        Some(vals)
    } else {
        None
    };

    let lon_vals: Option<Vec<i64>> = if need_lon {
        let (vals, _) = lon_array
            .retrieve_array_subset_ndarray::<i64>(&ArraySubset::new_with_shape(vec![nlon as u64]))
            .map_err(zarr_err)?
            .into_raw_vec_and_offset();
        Some(vals)
    } else {
        None
    };

    // Only read data arrays if needed
    let temp_vals: Option<Vec<i64>> = if need_temp {
        let temp_array = Array::open(store.clone(), "/temperature").map_err(zarr_err)?;
        let shape = temp_array.shape().to_vec();
        let (vals, _) = temp_array
            .retrieve_array_subset_ndarray::<i64>(&ArraySubset::new_with_shape(shape))
            .map_err(zarr_err)?
            .into_raw_vec_and_offset();
        Some(vals)
    } else {
        None
    };

    let humid_vals: Option<Vec<i64>> = if need_humid {
        let humid_array = Array::open(store.clone(), "/humidity").map_err(zarr_err)?;
        let shape = humid_array.shape().to_vec();
        let (vals, _) = humid_array
            .retrieve_array_subset_ndarray::<i64>(&ArraySubset::new_with_shape(shape))
            .map_err(zarr_err)?
            .into_raw_vec_and_offset();
        Some(vals)
    } else {
        None
    };

    // Build only the arrays we need, in schema order
    let mut all_arrays: Vec<Option<ArrayRef>> = vec![None; 5];

    if let Some(ref time_vals) = time_vals {
        let mut vals = Vec::with_capacity(total_rows);
        for t in 0..ntime {
            for _ in 0..nlat {
                for _ in 0..nlon {
                    vals.push(time_vals[t]);
                }
            }
        }
        all_arrays[COL_TIMESTAMP] = Some(Arc::new(Int64Array::from(vals)));
    }

    if let Some(ref lat_vals) = lat_vals {
        let mut vals = Vec::with_capacity(total_rows);
        for _ in 0..ntime {
            for i in 0..nlat {
                for _ in 0..nlon {
                    vals.push(lat_vals[i]);
                }
            }
        }
        all_arrays[COL_LAT] = Some(Arc::new(Int64Array::from(vals)));
    }

    if let Some(ref lon_vals) = lon_vals {
        let mut vals = Vec::with_capacity(total_rows);
        for _ in 0..ntime {
            for _ in 0..nlat {
                for j in 0..nlon {
                    vals.push(lon_vals[j]);
                }
            }
        }
        all_arrays[COL_LON] = Some(Arc::new(Int64Array::from(vals)));
    }

    if let Some(temp_vals) = temp_vals {
        all_arrays[COL_TEMPERATURE] = Some(Arc::new(Int64Array::from(temp_vals)));
    }

    if let Some(humid_vals) = humid_vals {
        all_arrays[COL_HUMIDITY] = Some(Arc::new(Int64Array::from(humid_vals)));
    }

    // Build projected schema and collect only projected arrays
    let projected_schema = Arc::new(Schema::new(
        projected_indices
            .iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
    ));

    let arrays: Vec<ArrayRef> = projected_indices
        .iter()
        .map(|&i| all_arrays[i].clone().expect("projected column not built"))
        .collect();

    let batch = RecordBatch::try_new(projected_schema.clone(), arrays)?;
    let stream = stream::iter(vec![Ok(batch)]);

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}
