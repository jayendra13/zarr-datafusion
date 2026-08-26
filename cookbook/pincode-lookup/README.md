# Pincode lookup — point extraction and feature fusion from a cloud grid

Two queries that turn a **list of postcodes into model-ready meteorology**, read
straight from the public ARCO-ERA5 store on GCS. No download, no conversion, no
staging step — the `CREATE EXTERNAL TABLE` points at the bucket and the query
runs.

| file | what it does | out | read |
|------|--------------|-----|------|
| [`pincode_temperature.sql`](pincode_temperature.sql) | 8 pincodes × 24 hourly steps → mean/max/min temperature | 8 rows | ~56 MB |
| [`met_feature_fusion.sql`](met_feature_fusion.sql) | 6 variables → 6 derived features, per city | 6 rows | ~394 MB |

```bash
zarr-cli cookbook/pincode-lookup/pincode_temperature.sql
zarr-cli cookbook/pincode-lookup/met_feature_fusion.sql
```

## The trick: snap the lookup, not the grid

A postcode centroid never lands exactly on the 0.25° grid, and coordinate
matching is **exact equality** — there is no nearest-neighbour mode. So the
rounding happens on the lookup side of the join:

```sql
JOIN pins p
  ON e.latitude  = ROUND(p.lat/0.25)*0.25
 AND e.longitude = ROUND(p.lon/0.25)*0.25
```

Multiples of 0.25 are exact in f32/f64, so this stays a plain equality join. All
eight cities are east of the prime meridian; for a western-hemisphere point wrap
with `((lon + 360) % 360)` first, since ERA5 longitude is 0–360.

## What it costs

ARCO-ERA5 `chunk-1` stores **one timestep per chunk over the full lat×lon
plane**, so the pincode filter does not cut chunk reads — the `time` predicate
does. Adding pincodes is free; adding days is not. Both queries read a single
day (24 steps).

## Feature fusion

`met_feature_fusion.sql` reads temperature, dewpoint, wind u/v, boundary-layer
height and precipitation in one statement and derives relative humidity (Magnus),
wind speed from the vector components, and the **ventilation coefficient** —
boundary-layer height × wind speed, a standard operational air-quality metric.

It is the `ORDER BY` on purpose. Ranking on mixing height alone is misleading:

```
city         BLH    wind   ventilation
Delhi        439    1.80        790     <- worst dispersion
Hyderabad    593    2.61       1550
Mumbai       431    3.89       1676
Chennai      578    3.32       1923
Bengaluru    741    2.83       2098
Kolkata      643    3.50       2251
```

Mumbai's lid sits *below* Delhi's, yet Mumbai ventilates at more than twice the
wind speed and washes out with 8.5 mm of rain. A low lid only matters if the air
is also still — which is the argument that needs six variables in one query
rather than one.

This is dispersion **potential**, not pollution: there is no PM2.5 in ERA5. It is
one day (2021-06-01), so it is an anecdote, not a climatology — widen the `WHERE`
for the multi-year version.
