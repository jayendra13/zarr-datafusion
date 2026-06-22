# ENSO Indices: ONI, RONI, SNI

Three indices used to measure the El Niño–Southern Oscillation (ENSO). All three
start from the **Niño‑3.4 sea‑surface‑temperature (SST) anomaly** — the SST
departure from a climatological average, area‑averaged over the box:

```
Niño‑3.4 box:  5°S – 5°N ,  170°W – 120°W
            =  latitude  ∈ [ -5.0,   5.0 ]
               longitude ∈ [ 190.0, 240.0 ]   (0–360° system)
```

## The formulas (easy form)

### ONI — Oceanic Niño Index
3‑month running mean of the Niño‑3.4 SST anomaly (anomaly vs a fixed 30‑yr
climatology, updated every 5 years).

```
ONI = mean₃ₘₒ( SST_Niño3.4 − climatology )
```

* El Niño if `ONI ≥ +0.5 °C`, La Niña if `≤ −0.5 °C`, Neutral in between
  (sustained ≥ 5 consecutive overlapping seasons).

### RONI — Relative Oceanic Niño Index
Same Niño‑3.4 anomaly, but first subtract the mean anomaly over the whole
tropical belt (20°S–20°N), then rescale so its variability matches the ONI.
Removes the global‑warming background drift. (NOAA/CPC's official index since
Feb 2026.)

```
RONI = (ONI − TropAvg) × σ_ONI / σ_(ONI − TropAvg)

  TropAvg              = mean SST anomaly over 20°S–20°N
  σ_ONI                = std-dev of ONI
  σ_(ONI − TropAvg)    = std-dev of the relative term
```

Same ±0.5 °C thresholds apply (the σ ratio puts it back on the ONI scale).

### SNI — Standardized Niño Index
The Niño anomaly expressed in units of its standard deviation (a z‑score),
instead of in °C.

```
SNI = (SST_Niño3.4 − mean) / σ        (σ = std-dev for that calendar month)
```

Dimensionless; typically ranges ≈ −2.5 … +2.5.

**In one line:** ONI = raw anomaly · RONI = anomaly minus tropical‑mean warming
(rescaled) · SNI = anomaly divided by σ.

Sources: [CPC RONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/),
[RONI PDF](https://www.weather.gov/media/climateservices/RONI.pdf),
[CPC ONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php),
[NOAA Climate.gov — ENSO indexes](https://www.climate.gov/news-features/blogs/enso/why-are-there-so-many-enso-indexes-instead-just-one).

---

## Compute‑saving simplification: "noon of the 15th"

ONI is the cheapest of the three to compute, so it is our target. A true monthly
mean averages **every hour of the month** (744 hourly global snapshots for
December). For the ERA5 local store each hourly step is a full global chunk
(721×1440) that has to be decompressed in full, so a month costs ~744 chunk
reads.

To save compute we approximate each monthly mean by a **single instantaneous
sample — noon (12:00 UTC) on the 15th** of that month. One chunk per month
instead of ~744 (≈ 250× fewer reads), and the 15th-at-noon is a reasonable
mid-month proxy.

```
Monthly Niño‑3.4 SST  ≈  spatial mean of SST over the Niño‑3.4 box
                          at  YYYY‑MM‑15 12:00:00 UTC
DJF index             ≈  average of the Dec / Jan / Feb noon‑15th samples
```

### Season labelling
We compute **DJF 2025 = Dec 2025 + Jan 2026 + Feb 2026** (the winter whose
December falls in 2025). All three noon‑15th snapshots are present in the local
store.

---

## Local data & query gotchas

Store: `data/era5_sst_local.zarr` (ARCO‑ERA5, `sea_surface_temperature`, Kelvin).

1. **Kelvin → Celsius:** subtract `273.15`.
2. **Sparse mirror → use `=`, not `BETWEEN`, on `time`.** Range pushdown uses
   binary search over the time axis and can silently return 0 rows when chunks
   are absent. Equality (`time = '…'`) is a linear scan and is safe — and it is
   exactly what the noon‑15th sampling needs (it touches only that one chunk).
3. **`IN` / `OR` do not push down** (only `Eq`, `Gt/GtEq/Lt/LtEq`, `Between`,
   `And`). Combine the three monthly samples with `UNION ALL`, so each branch is
   a single `=` and only 3 chunks are read.
4. **Date functions crash on the dict‑encoded `time` column.** Wrap in
   `arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))')` if you need
   `date_part`/`date_trunc`. (Not needed below — we filter on literal
   timestamps.)

---

## SQL — DJF 2025 Niño‑3.4 SST (the raw season value)

Each `UNION ALL` branch is one month's spatial mean over the Niño‑3.4 box at
noon on the 15th; the outer `AVG` gives equal weight to the three months.

```sql
CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'data/era5_sst_local.zarr';

SELECT AVG(monthly_sst_c) AS nino34_djf2025_c
FROM (
    -- December 2025
    SELECT AVG(sea_surface_temperature - 273.15) AS monthly_sst_c
    FROM era5
    WHERE time      =  '2025-12-15 12:00:00'
      AND latitude  BETWEEN  -5.0 AND   5.0
      AND longitude BETWEEN 190.0 AND 240.0

    UNION ALL
    -- January 2026
    SELECT AVG(sea_surface_temperature - 273.15)
    FROM era5
    WHERE time      =  '2026-01-15 12:00:00'
      AND latitude  BETWEEN  -5.0 AND   5.0
      AND longitude BETWEEN 190.0 AND 240.0

    UNION ALL
    -- February 2026
    SELECT AVG(sea_surface_temperature - 273.15)
    FROM era5
    WHERE time      =  '2026-02-15 12:00:00'
      AND latitude  BETWEEN  -5.0 AND   5.0
      AND longitude BETWEEN 190.0 AND 240.0
) AS djf;
```

> The box filters do **not** reduce bytes read (chunks are full global
> snapshots), but they restrict the cells that enter the average. `AVG` skips
> land `NaN`s automatically, so this is the ocean mean over the box.

## SQL — turning it into the ONI (anomaly)

ONI needs the anomaly, so subtract the climatological DJF value computed the
**same** noon‑15th way over a 30‑yr base period (1991–2020):

```
ONI(DJF 2025) = nino34_djf2025_c  −  climatology_djf_c
```

The climatology is the average of the Dec/Jan/Feb noon‑15th samples across the
30 base years (90 timestamps). Because `IN`/`OR`/ranges don't push down here, the
robust pattern is the same single‑`=` branch repeated per base‑year‑month and
`UNION ALL`‑ed, then averaged — e.g. one branch:

```sql
SELECT AVG(sea_surface_temperature - 273.15) AS monthly_sst_c
FROM era5
WHERE time      =  '1991-01-15 12:00:00'      -- (Jan of base year 1991)
  AND latitude  BETWEEN  -5.0 AND   5.0
  AND longitude BETWEEN 190.0 AND 240.0
```

That is a lot of branches to hand‑write, so generate the climatology query
programmatically (loop years 1991–2020 × months Dec/Jan/Feb, emit one `=` branch
each, wrap in `SELECT AVG(monthly_sst_c) FROM ( … )`). A few base‑year Jan/Feb
noon‑15th chunks are missing from the local mirror (25/30 present for Jan and
Feb; all 30 for Dec) — those simply drop out of the average.

Once both numbers are in hand:

```
ONI(DJF 2025) [°C] = nino34_djf2025_c − climatology_djf_c
   ONI ≥ +0.5  → El Niño
   ONI ≤ −0.5  → La Niña
   else        → Neutral
```

> Note: this is ERA5‑SST‑based and uses a one‑sample‑per‑month proxy, so it will
> differ slightly from NOAA's official ERSSTv5 ONI.
