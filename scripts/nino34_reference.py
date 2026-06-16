"""
Niño 3.4 reference validation script.

Two parts:
  1. Raw Niño 3.4 SST  — NOAA CPC sstoi.indices (ERSSTv5, monthly, ~50KB)
  2. ONI anomaly index — NOAA CPC oni.ascii.txt  (ERSSTv5, 3-month smooth, ~50KB)

These are the authoritative NOAA reference values that zarr-datafusion SQL
output will be compared against. Both derive from ERSSTv5 (not ERA5), so
ERA5-derived values will differ by ~0.1–0.2°C — that gap is expected.

Run:
    uv run --with requests --with pandas --with tabulate \
           scripts/nino34_reference.py
"""

import io
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# NOAA CPC data URLs
# ---------------------------------------------------------------------------

# Raw monthly SST for Niño 1+2, 3, 3.4, 4 regions (from ERSSTv5)
SSTOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"

# ONI: 3-month running mean of Niño 3.4 SSTA (from ERSSTv5, 1991-2020 base)
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Niño 3.4 SSTA monthly (pre-computed anomaly, 1991-2020 base)
NINO34_ANOM_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"


# ===========================================================================
# PART 1 — Raw Niño 3.4 SST (not anomaly, absolute °C)
# ===========================================================================

print("=" * 65)
print("PART 1 — Raw Niño 3.4 SST  (NOAA CPC sstoi.indices, ERSSTv5)")
print("=" * 65)
print(f"Source: {SSTOI_URL}")
print()

resp = requests.get(SSTOI_URL, timeout=30)
resp.raise_for_status()

# Column layout:  YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
# 0-indexed:       0   1    2      3    4     5    6     7     8       9
rows = []
for line in resp.text.splitlines():
    parts = line.split()
    if len(parts) < 10 or not parts[0].isdigit():
        continue
    rows.append({
        "year":    int(parts[0]),
        "month":   int(parts[1]),
        "nino34":  float(parts[8]),   # raw SST °C
        "anom34":  float(parts[9]),   # anomaly °C (from this file's internal baseline)
    })

df_sst = pd.DataFrame(rows)
df_sst["date"] = pd.to_datetime(df_sst[["year","month"]].assign(day=1))
print(f"Loaded {len(df_sst)} monthly records  "
      f"({df_sst['year'].min()}–{df_sst['year'].max()})")
print()

# --- Key spot-check months ---
SPOT_CHECKS = [
    (1997, 12, "Dec 1997", "~28–30°C (strong El Niño, warm pool)"),
    (1988, 12, "Dec 1988", "~25–27°C (strong La Niña, cool pool)"),
    (2015, 11, "Nov 2015", "~28–30°C (strong El Niño)"),
    (2010, 12, "Dec 2010", "~25–27°C (strong La Niña)"),
    (1982, 12, "Dec 1982", "~28–29°C (moderate El Niño)"),
    (1998,  3, "Mar 1998", "~28–30°C (El Niño peak fading)"),
]

print(f"  {'Label':<12}  {'Raw SST':>10}  {'Anom':>8}   Expected range")
print("  " + "-" * 58)
for yr, mo, label, expected in SPOT_CHECKS:
    row = df_sst[(df_sst.year == yr) & (df_sst.month == mo)]
    if row.empty:
        print(f"  {label:<12}  (no data)")
        continue
    sst  = row.iloc[0]["nino34"]
    anom = row.iloc[0]["anom34"]
    print(f"  {label:<12}  {sst:>8.3f}°C  {anom:>+7.3f}°C   {expected}")

print()

# --- 1991-2020 climatology from this data ---
clim = (
    df_sst[(df_sst.year >= 1991) & (df_sst.year <= 2020)]
    .groupby("month")["nino34"]
    .mean()
)
print("Niño 3.4 climatology 1991–2020 (raw SST °C):")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
for m in range(1, 13):
    print(f"  {month_names[m-1]}: {clim[m]:.3f}°C")
print()


# ===========================================================================
# PART 2 — ONI  (3-month running mean SSTA, official ENSO index)
# ===========================================================================

print("=" * 65)
print("PART 2 — ONI Reference Values  (NOAA CPC oni.ascii.txt, ERSSTv5)")
print("=" * 65)
print(f"Source: {ONI_URL}")
print()

resp2 = requests.get(ONI_URL, timeout=30)
resp2.raise_for_status()

SEASON_ORDER = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
oni_records = {}
for line in resp2.text.splitlines():
    parts = line.split()
    if len(parts) < 3 or parts[0] == "SEAS":
        continue
    seas, yr = parts[0], int(parts[1])
    # Format: SEAS YR TOTAL ANOM  (4 cols)
    anom = float(parts[3])
    oni_records[(yr, seas)] = anom

print(f"Loaded {len(oni_records)} ONI season records "
      f"({min(r[0] for r in oni_records)}–{max(r[0] for r in oni_records)}).")
print()

# --- Validation anchors ---
EVENTS = [
    # (year, season) — DJF year refers to Jan-Feb, so Dec 1982 peak → DJF 1983
    (1997, "NDJ", "1997-98 El Niño peak",  +2.4),   # NDJ 1997 = Nov97 Dec97 Jan98
    (2015, "NDJ", "2015-16 El Niño peak",  +2.6),   # NDJ 2015 = Nov15 Dec15 Jan16
    (1983, "DJF", "1982-83 El Niño peak",  +2.2),   # DJF 1983 = Dec82 Jan83 Feb83
    (2011, "DJF", "2010-11 La Niña peak",  -1.6),   # DJF 2011 = Dec10 Jan11 Feb11
    (1989, "DJF", "1988-89 La Niña peak",  -1.8),   # DJF 1989 = Dec88 Jan89 Feb89
]

print(f"  {'Season':<11} {'ONI (°C)':>10}  {'Textbook ref':>13}   Label")
print("  " + "-" * 55)
for yr, seas, label, textbook_ref in EVENTS:
    oni = oni_records.get((yr, seas), float("nan"))
    match = "✓" if abs(oni - textbook_ref) < 0.15 else "~"
    print(f"  {seas} {yr:<5}   {oni:>+8.2f}°C   {textbook_ref:>+8.2f}°C    {label}  {match}")

print()

# --- Recent 36 seasons ---
all_seasons = sorted(
    oni_records.keys(),
    key=lambda x: (x[0], SEASON_ORDER.index(x[1]) if x[1] in SEASON_ORDER else 0)
)
print("Recent ONI (last 36 seasons):")
print(f"  {'Season':<11} {'ONI':>8}   Phase")
print("  " + "-" * 35)
for yr, seas in all_seasons[-36:]:
    val = oni_records[(yr, seas)]
    phase = "El Niño" if val >= 0.5 else ("La Niña" if val <= -0.5 else "Neutral")
    bar = "+" * int(abs(val) / 0.1) if val > 0 else "-" * int(abs(val) / 0.1)
    bar = bar[:20]
    print(f"  {seas} {yr:<5}   {val:>+6.2f}°C   {phase:<9} {bar}")

print()
print("SUMMARY")
print("-------")
print("Use Part 1 raw SST values to validate zarr-datafusion SQL output.")
print("Use Part 2 ONI values after computing anomaly + 3-month smoothing.")
print("ERA5-derived values will differ by ~0.1–0.2°C from ERSSTv5 (expected).")
