# El Niño / La Niña — ONI from ERA5 with SQL, validated against NOAA

Compute the **Oceanic Niño Index (ONI)** — the headline ENSO indicator — for every
overlapping 3-month season from **1950 to present**, directly over public ERA5
sea-surface-temperature data using a single SQL query through `zarr-datafusion`,
then validate it against NOAA CPC's official ONI table.

**Headline result (916 seasons, 1950–2026, vs NOAA CPC):**

| | |
|---|---|
| MAE (anomaly) | **0.21 °C** overall · **0.14 °C** in the satellite era (1979+) |
| Pearson r | **0.958** |
| ENSO-phase agreement | **quadratic-weighted κ = 0.85**, with **zero La Niña↔El Niño sign flips** |

The residual gap is fully explained by dataset differences (ERA5 vs NOAA's ERSSTv5),
a single-sample monthly proxy, and pre-1979 reanalysis uncertainty.

---

## Files

| File | What it is |
|------|-----------|
| `oni_all_seasons.sql` | The query: ONI + ENSO phase per season, 1950→now, over ARCO-ERA5 on GCS |
| `oni_compare.py` | Comparison script: our values vs NOAA, regression + classification metrics |
| `oni_noaa_reference.txt` | Vendored NOAA CPC reference (`oni.ascii.txt`, `SEAS YR TOTAL ANOM`) |
| `oni_computed.txt` | Our 917-season output (raw `zarr-cli` table dump) |
| `oni_comparison.csv` | Per-season merged comparison with residuals and both classes |

---

## What ONI is, and how the query computes it

ONI = **3-month running mean of monthly SST anomalies** in the **Niño-3.4 box**
(5°S–5°N, 170°W–120°W → longitude 190–240 in 0–360°), where the anomaly is the
departure from a **1991–2020 monthly climatology**.

`oni_all_seasons.sql` does this end to end:

1. **Sample** one value per month (noon of the 15th) inside the Niño-3.4 box —
   `WHERE` touches **coordinates only** (`latitude`, `longitude`, and `time` via
   `extract()`), so the engine pushes the selection into the scan.
2. **Monthly spatial mean** per `(year, month)`; NaN fill from absent chunks is
   dropped post-aggregate with `HAVING`.
3. **Climatology** = per-month mean over 1991–2020.
4. **Anomaly** on a contiguous month index `t = yr*12 + (mo-1)`.
5. **ONI** = centred 3-month moving average via a self-join on `t-1, t, t+1`
   (all three must exist), so one row is emitted per centre month → DJF…NDJ.
6. **ENSO phase** from the ±0.5 °C thresholds:
   `≥ +0.5 → El Niño`, `≤ −0.5 → La Niña`, else `Neutral`.

Season labelling follows NOAA's **centre-month-year** convention
(DJF 1998 = Dec 1997, Jan 1998, Feb 1998).

---

## Reproduce

**1 — produce our values** (remote scan of public ARCO-ERA5; ~40 s on a
same-region GCP VM, ~20 min over a home connection):

```bash
zarr-cli cookbook/el-nino-oni/oni_all_seasons.sql > cookbook/el-nino-oni/oni_computed.txt
```

**2 — compare against NOAA** (offline, using the vendored reference):

```bash
uv run --with requests --with pandas --with numpy --with scikit-learn \
       --with scipy --with tabulate \
  scripts/oni_compare.py \
    --computed  cookbook/el-nino-oni/oni_computed.txt \
    --reference cookbook/el-nino-oni/oni_noaa_reference.txt \
    --out       cookbook/el-nino-oni/oni_comparison.csv
```

(Drop `--reference` to fetch the latest NOAA table live instead of the vendored copy.
`oni_compare.py` here is a copy of `scripts/oni_compare.py`.)

---

## Which metrics, and why

The two comparisons are different problems and need different metrics:

**Raw anomalies (continuous, °C):** **MAE** (interpretable), **mean error / bias**
(separates a systematic offset from scatter — important since NOAA is ERSSTv5, not
ERA5), and **Pearson r** (phase agreement). RMSE/R² secondary.

**ENSO class (3 ordinal classes):** the headline is **quadratic-weighted Cohen's κ**,
because the classes are ordered (La Niña < Neutral < El Niño) so a sign flip must be
penalised far more than a boundary slip, and κ corrects for chance under the heavy
Neutral imbalance. Report the **confusion matrix** + **macro-F1**; **do not** lead with
accuracy (Neutral dominates and inflates it).

---

## Findings

### Raw anomalies (regression)

| Metric | Value |
|--------|-------|
| N | 916 seasons |
| MAE | 0.21 °C |
| RMSE | 0.28 °C |
| Mean error (bias) | −0.14 °C (we run slightly **cold**) |
| MAE after de-bias | 0.19 °C |
| Pearson r | 0.958 |
| R² | 0.885 |

**The satellite-era split is the key result:**

| Era | N | MAE | bias |
|-----|---|-----|------|
| pre-1979 | 348 | 0.32 | −0.21 |
| 1979+ | 568 | **0.14** | −0.10 |

Error more than halves at 1979 — ERA5 gains satellite constraint over the tropical
Pacific. Post-1979 MAE (0.14 °C) is at the expected ERA5-vs-ERSSTv5 floor.

### ENSO class

**Quadratic-weighted κ = 0.85** (“almost perfect”). Confusion matrix
(rows = NOAA truth, cols = ours):

| truth \ pred | La Niña | Neutral | El Niño |
|---|---|---|---|
| **La Niña** | 239 | 13 | 0 |
| **Neutral** | 80 | 329 | 10 |
| **El Niño** | 0 | 51 | 194 |

**Zero sign flips** — every error is a one-step boundary slip across ±0.5. That is
exactly why weighted κ (0.85) > unweighted κ (0.74) > accuracy (0.83): the
disagreements are the harmless kind.

| class | precision | recall |
|---|---|---|
| La Niña | 0.75 | 0.95 |
| Neutral | 0.84 | 0.79 |
| El Niño | 0.95 | 0.79 |

The **−0.14 °C cold bias** explains the asymmetry: we **over-call** La Niña (recall
0.95) and **under-call** El Niño (recall 0.79; 51 weak El Niños slip to Neutral), yet
our El Niño calls are almost always right (precision 0.95). The worst residuals
(1963, 1969 — weak, pre-satellite events) are the same story.

---

## Caveats

- **Dataset:** NOAA's ONI is **ERSSTv5**; we use **ERA5**. A ~0.1–0.2 °C difference
  is expected and is most of the post-1979 error.
- **Monthly proxy:** we use a single **noon-of-the-15th** sample per month, not a true
  monthly mean — adds scatter and likely contributes the cold bias.
- **Base period:** a single fixed **1991–2020** climatology is used for the whole
  series; NOAA's official historical ONI uses *centred* 30-year base periods that shift
  every 5 years, so deep-past values differ slightly.
- **Pre-1979:** ERA5 is weakly constrained over the tropical Pacific before the
  satellite era — treat those values with more caution.

**Cheap improvements:** remove the −0.14 °C bias, or sample the true monthly mean
instead of noon-15th — either would recover most of the 51 missed El Niños.

---

## Reference

NOAA CPC — Oceanic Niño Index (ONI), ERSSTv5, 1991–2020 base:
- Table: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
- Data:  https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
