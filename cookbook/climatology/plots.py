"""
Plots for the diurnal-climatology cookbook.

Reads the recipe output (lat, lon, hour, t2m_mean_c) and writes three PNGs
next to it. Fully offline and reproducible — no remote scan. The CSV is the
frozen result of `diurnal_climatology.sql` (72-timestep window over a CONUS
box); to regenerate it, wrap that query in

    COPY ( <the WITH ... SELECT ...> ) TO 'diurnal_climatology.csv' STORED AS CSV;

  1. diurnal_cycles.png    — the "average day" at a few representative cells,
                             on a LOCAL-SOLAR-time x-axis so the curves align at
                             noon and only their AMPLITUDE differs (marine flat,
                             continental big swing).
  2. diurnal_range_map.png — diurnal temperature range (max−min over the 24 h)
                             per grid cell: the nD cube flattened back to a map.
  3. peak_hour_map.png     — UTC hour of peak warmth per cell; sweeps later
                             east→west, i.e. the longitude axis as a timezone.

Usage:
    uv run --with pandas --with numpy --with matplotlib \
        cookbook/climatology/plots.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Representative cells (all on the 0.25° grid, ERA5 lon 0–360°E). Chosen to
# contrast marine vs continental diurnal amplitude across the CONUS box.
CELLS = [
    ("Pacific NW coast", 47.0, 236.0, "#2c7fb8"),   # 124°W — marine, near-flat
    ("Great Basin (arid)", 40.0, 243.0, "#d95f0e"),  # 117°W — big DTR
    ("Central Plains", 39.0, 262.0, "#756bb1"),      # 98°W  — continental
    ("Gulf coast (humid)", 30.0, 268.0, "#31a354"),  # 92°W  — warm, damped
]


def local_solar_hour(utc_hour, lon_deg_east):
    """UTC hour → local *solar* hour by longitude (0–360°E). Puts local noon at
    ~12 for every cell so the cycles align and only amplitude/shape differ."""
    return (utc_hour + lon_deg_east / 15.0) % 24.0


def plot_cycles(df, out):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, lat, lon, color in CELLS:
        cell = df[(df["lat"] == lat) & (df["lon"] == lon)].sort_values("hour")
        if cell.empty:
            print(f"  ! no data for {name} ({lat},{lon}) — skipped")
            continue
        lst = local_solar_hour(cell["hour"].to_numpy(), lon)
        order = np.argsort(lst)
        t = cell["t2m_mean_c"].to_numpy()[order]
        x = lst[order]
        dtr = t.max() - t.min()
        west = 360.0 - lon
        ax.plot(x, t, "-o", ms=4, color=color, lw=1.8,
                label=f"{name}  ({west:.0f}°W, DTR {dtr:.1f} °C)")

    ax.axvspan(11, 16, color="0.9", zorder=0)  # afternoon band
    ax.axvline(12, color="0.6", lw=0.7, ls="--")
    ax.set_title("The average day — diurnal 2 m-temperature cycle by location\n"
                 "(ARCO-ERA5, 2020-06-01→03; local solar time)")
    ax.set_xlabel("local solar hour")
    ax.set_ylabel("mean 2 m temperature (°C)")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.margins(y=0.08)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.text(0.995, 0.02, "shaded: local afternoon (11–16 h)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _grid(df, values):
    """Pivot a per-cell scalar Series (indexed like df) into a lat×lon grid for
    imshow. Returns (Z, extent) with latitude descending (ERA5-native)."""
    g = df.assign(v=values).pivot_table(index="lat", columns="lon", values="v")
    g = g.sort_index(ascending=False)  # lat descending, north at top
    lons, lats = g.columns.to_numpy(), g.index.to_numpy()
    west = 360.0 - lons  # 0–360°E → °W for readable ticks
    extent = [west.max(), west.min(), lats.min(), lats.max()]
    return g.to_numpy(), extent


def _per_cell(df):
    """Collapse the (lat,lon,hour) table to one row per cell with the diurnal
    range and the UTC hour of the daily maximum."""
    grp = df.groupby(["lat", "lon"])
    dtr = grp["t2m_mean_c"].max() - grp["t2m_mean_c"].min()
    peak = grp.apply(lambda c: c.loc[c["t2m_mean_c"].idxmax(), "hour"],
                     include_groups=False)
    cells = pd.DataFrame({"dtr": dtr, "peak_utc": peak}).reset_index()
    return cells


def plot_range_map(cells, out):
    Z, extent = _grid(cells, cells["dtr"].to_numpy())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(Z, extent=extent, aspect="auto", origin="upper",
                   cmap="inferno")
    ax.set_title("Diurnal temperature range — max − min over the 24 h\n"
                 "(ARCO-ERA5, 2020-06-01→03; the nD cube flattened back to a map)")
    ax.set_xlabel("longitude (°W)")
    ax.set_ylabel("latitude (°N)")
    for name, lat, lon, _ in CELLS:
        ax.plot(360.0 - lon, lat, "o", ms=5, mfc="none", mec="cyan", mew=1.4)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("DTR (°C)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_peak_hour_map(cells, out, dtr_floor=3.0):
    # Where the diurnal cycle is flat (ocean, DTR≈0) the argmax hour is pure
    # noise — mask those cells so only the real land timezone sweep shows.
    peak = cells["peak_utc"].where(cells["dtr"] >= dtr_floor)
    Z, extent = _grid(cells, peak.to_numpy())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("0.85")  # masked (low-DTR / ocean) cells → light grey
    im = ax.imshow(np.ma.masked_invalid(Z), extent=extent, aspect="auto",
                   origin="upper", cmap=cmap, vmin=16, vmax=23)
    ax.set_title("UTC hour of peak warmth — sweeps east→west\n"
                 f"(the longitude axis is a real timezone gradient; DTR<{dtr_floor:.0f}°C masked)")
    ax.set_xlabel("longitude (°W)")
    ax.set_ylabel("latitude (°N)")
    cb = fig.colorbar(im, ax=ax, pad=0.02, ticks=range(16, 24))
    cb.set_label("hour of daily max (UTC)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=here / "diurnal_climatology.csv.gz")
    ap.add_argument("--outdir", default=here)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print(f"loaded {len(df)} rows · "
          f"{df['lat'].nunique()} lat × {df['lon'].nunique()} lon × "
          f"{df['hour'].nunique()} hours")
    outdir = Path(a.outdir)
    cells = _per_cell(df)
    plot_cycles(df, outdir / "diurnal_cycles.png")
    plot_range_map(cells, outdir / "diurnal_range_map.png")
    plot_peak_hour_map(cells, outdir / "peak_hour_map.png")


if __name__ == "__main__":
    main()
