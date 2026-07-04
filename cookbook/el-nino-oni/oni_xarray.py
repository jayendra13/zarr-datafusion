"""
ONI all-seasons in pure xarray/pandas — the line-for-line equivalent of
oni_all_seasons.sql, used to (1) confirm the SQL result is correct and
(2) time the "classic" Python/xarray path against `zarr-cli`.

Same data (ARCO-ERA5 SST on GCS), same Niño-3.4 box, same noon-of-the-15th
monthly proxy, same NOAA rolling 30-year base periods, same centred 3-month
running mean. Output columns match the SQL: year, season, oni_c, enso_phase.

The data step (open store -> select box + noon-15th samples -> monthly spatial
mean) is done in xarray; the climatology / anomaly / running-mean step is done
in pandas, mirroring each SQL CTE one-to-one so the numbers line up exactly.

Cost is the same as the SQL: ERA5 SST is chunked (1, 721, 1440), i.e. one full
lat×lon plane per timestep, so the day-15/hour-12 selection still fetches ~1000
full planes (~4 GB) over the network. Expect ~25-35 min over a home connection.

Usage:
    uv run --with xarray --with zarr --with gcsfs --with dask --with numpy \
           --with pandas cookbook/el-nino-oni/oni_xarray.py

    # also diff against the zarr-cli output to prove they agree:
    uv run ... cookbook/el-nino-oni/oni_xarray.py \
        --compare cookbook/el-nino-oni/oni_computed.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Niño-3.4 box (5°S–5°N, 170°W–120°W -> 190–240 in 0–360°)
LAT_LO, LAT_HI = -5.0, 5.0
LON_LO, LON_HI = 190.0, 240.0
YEAR_LO, YEAR_HI = 1940, 2026          # widened: rolling base needs the deep past

# season-year block -> centred 30-yr base period (NOAA CPC schedule).
# (yr_lo, yr_hi, base_lo, base_hi) — identical to the SQL base_periods VALUES.
BASE_PERIODS = [
    (1940, 1955, 1936, 1965),   # 1936-1965 is effectively 1940-1965 in ERA5
    (1956, 1960, 1941, 1970),
    (1961, 1965, 1946, 1975),
    (1966, 1970, 1951, 1980),
    (1971, 1975, 1956, 1985),
    (1976, 1980, 1961, 1990),
    (1981, 1985, 1966, 1995),
    (1986, 1990, 1971, 2000),
    (1991, 1995, 1976, 2005),
    (1996, 2000, 1981, 2010),
    (2001, 2005, 1986, 2015),
    (2006, 2010, 1991, 2020),
    (2011, 2026, 1996, 2025),   # 1996-2025 base; held for 2016+ (later periods need future data)
]

# centre month -> season code (DJF centre = Jan = 1, ... NDJ = Dec = 12)
SEASON_LABEL = {1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM", 5: "AMJ", 6: "MJJ",
                7: "JJA", 8: "JAS", 9: "ASO", 10: "SON", 11: "OND", 12: "NDJ"}


def enso_phase(x):
    """±0.5 °C thresholds, matching the SQL CASE."""
    if x >= 0.5:
        return "El Niño"
    if x <= -0.5:
        return "La Niña"
    return "Neutral"


def load_monthly(store):
    """xarray half: store -> monthly box-mean SST (°C), DataFrame[yr, mo, sst_c].

    Mirrors the SQL `samples` + `monthly` CTEs: noon-of-the-15th sample per
    month inside the Niño-3.4 box, spatial mean per (year, month), drop months
    whose mean is not in [0, 50] (absent chunks read back as NaN).
    """
    ds = xr.open_zarr(store, chunks={}, storage_options={"token": "anon"})
    sst = ds["sea_surface_temperature"]

    # WHERE latitude/longitude BETWEEN ... — coordinate masks (lat is descending)
    lat = ds.latitude
    lon = ds.longitude
    sst = sst.sel(
        latitude=lat[(lat >= LAT_LO) & (lat <= LAT_HI)],
        longitude=lon[(lon >= LON_LO) & (lon <= LON_HI)],
    )

    # WHERE day=15 AND hour=12 AND year BETWEEN 1940 AND 2026
    t = sst["time"]
    keep = (
        (t.dt.day == 15)
        & (t.dt.hour == 12)
        & (t.dt.year >= YEAR_LO)
        & (t.dt.year <= YEAR_HI)
    )
    sst = sst.sel(time=t[keep])

    print(f"fetching {sst.sizes['time']} monthly planes "
          f"({sst.sizes['latitude']}×{sst.sizes['longitude']} box) ...", flush=True)
    sst_c = (sst - 273.15).load()      # the network read happens here

    # spatial mean per (year, month). skipna=False so an absent (all-NaN) plane
    # yields NaN, matching DataFusion's AVG over NaN; the [0,50] filter drops it.
    monthly = sst_c.mean(dim=["latitude", "longitude"], skipna=False)

    df = pd.DataFrame({
        "yr": monthly["time"].dt.year.values,
        "mo": monthly["time"].dt.month.values,
        "sst_c": monthly.values,
    })
    # HAVING AVG(sst_c) BETWEEN 0 AND 50
    df = df[(df["sst_c"] >= 0) & (df["sst_c"] <= 50)].reset_index(drop=True)
    return df


def compute_oni(monthly):
    """pandas half: monthly box-mean -> ONI per centre month.

    Mirrors the SQL `clim` / `anom` / `oni` CTEs and final SELECT.
    """
    bp = pd.DataFrame(BASE_PERIODS,
                      columns=["yr_lo", "yr_hi", "base_lo", "base_hi"])

    # clim: per base-period, per month, mean over [base_lo, base_hi]
    clim_rows = []
    for _, p in bp.iterrows():
        win = monthly[(monthly.yr >= p.base_lo) & (monthly.yr <= p.base_hi)]
        for mo, g in win.groupby("mo"):
            clim_rows.append({"yr_lo": p.yr_lo, "mo": mo,
                              "clim_sst": g.sst_c.mean()})
    clim = pd.DataFrame(clim_rows)

    # anom: each month vs its own base period (range-join yr -> exactly one block)
    def block_of(yr):
        hit = bp[(bp.yr_lo <= yr) & (yr <= bp.yr_hi)]
        return hit.yr_lo.iloc[0]

    anom = monthly.copy()
    anom["yr_lo"] = anom.yr.map(block_of)
    anom = anom.merge(clim, on=["yr_lo", "mo"], how="inner")
    anom["t"] = anom.yr * 12 + (anom.mo - 1)        # contiguous month counter
    anom["anom"] = anom.sst_c - anom.clim_sst

    # oni: centred 3-month running mean via self-join on t-1, t, t+1
    by_t = anom.set_index("t")["anom"]
    rows = []
    for _, c in anom.iterrows():
        tprev, tnext = c.t - 1, c.t + 1
        if tprev in by_t.index and tnext in by_t.index and c.yr >= 1950:
            oni_raw = (by_t[tprev] + c.anom + by_t[tnext]) / 3.0
            rows.append({"year": int(c.yr), "mo": int(c.mo),
                         "oni_raw": oni_raw, "oni_c": round(oni_raw, 2)})

    out = pd.DataFrame(rows).sort_values(["year", "mo"]).reset_index(drop=True)
    out["season"] = out.mo.map(SEASON_LABEL)
    # phase comes from the UNROUNDED value, exactly like the SQL CASE on oni_raw
    # (the displayed oni_c is ROUND(oni_raw, 2)); at a ±0.5 boundary the two differ.
    out["enso_phase"] = out.oni_raw.map(enso_phase)
    return out[["year", "season", "oni_c", "enso_phase"]]


def parse_box_table(path):
    """zarr-cli pretty-table dump -> DataFrame[year, season, oni_c, enso_phase]."""
    rows = [ln for ln in Path(path).read_text().splitlines()
            if ln.lstrip().startswith("|")]
    cells = lambda ln: [c.strip() for c in ln.strip().strip("|").split("|")]
    header = [h.lower() for h in cells(rows[0])]
    recs = []
    for ln in rows[1:]:
        d = dict(zip(header, cells(ln)))
        recs.append({"year": int(d["year"]), "season": d["season"],
                     "oni_c": float(d["oni_c"]), "enso_phase": d["enso_phase"]})
    return pd.DataFrame(recs)


def compare(ours, sql_path):
    """Row-by-row agreement check against the zarr-cli output."""
    sql = parse_box_table(sql_path)
    m = ours.merge(sql, on=["year", "season"], how="outer",
                   suffixes=("_py", "_sql"), indicator=True)
    only = m[m["_merge"] != "both"]
    both = m[m["_merge"] == "both"].copy()
    both["doni"] = (both.oni_c_py - both.oni_c_sql).abs()
    phase_mismatch = both[both.enso_phase_py != both.enso_phase_sql]

    print("\n=== agreement vs zarr-cli (oni_all_seasons.sql) ===")
    print(f"matched seasons : {len(both)}")
    print(f"py-only / sql-only rows : {len(only)}")
    print(f"max |Δ oni_c|   : {both.doni.max():.4f} °C")
    print(f"rows with Δ > 0.005 : {(both.doni > 0.005).sum()}")
    print(f"enso_phase mismatches : {len(phase_mismatch)}")
    if len(phase_mismatch):
        print(phase_mismatch[["year", "season", "oni_c_py", "oni_c_sql",
                              "enso_phase_py", "enso_phase_sql"]].to_string(index=False))
    big = both[both.doni > 0.005]
    if len(big):
        print("largest disagreements:")
        print(big.reindex(big.doni.sort_values(ascending=False).index)
              .head(10)[["year", "season", "oni_c_py", "oni_c_sql", "doni"]]
              .to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "oni_xarray.csv"))
    ap.add_argument("--compare", default=None,
                    help="zarr-cli table dump to check agreement against")
    ap.add_argument("--cache-monthly", default=None,
                    help="path to cache the monthly box-mean SST; reused on "
                         "re-runs to re-check the (instant) pandas half without "
                         "repeating the multi-GB remote scan")
    a = ap.parse_args()

    t0 = time.perf_counter()
    if a.cache_monthly and Path(a.cache_monthly).exists():
        print(f"using cached monthly means: {a.cache_monthly}", flush=True)
        monthly = pd.read_csv(a.cache_monthly)
    else:
        monthly = load_monthly(a.store)
        if a.cache_monthly:
            monthly.to_csv(a.cache_monthly, index=False)
    t_fetch = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = compute_oni(monthly)
    t_compute = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    out.to_csv(a.out, index=False)
    print(f"\n{len(out)} seasons, {out.year.min()}–{out.year.max()}  -> {a.out}")
    print(out.head(12).to_string(index=False))
    print("...")
    print(out.tail(12).to_string(index=False))

    print("\n=== timing (xarray/pandas path) ===")
    print(f"data fetch + monthly mean : {t_fetch:8.1f} s")
    print(f"climatology + ONI (pandas): {t_compute:8.1f} s")
    print(f"total wall                : {t_total:8.1f} s")

    if a.compare:
        compare(out, a.compare)


if __name__ == "__main__":
    main()
