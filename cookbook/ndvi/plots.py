"""
Plots for the NDVI cookbook — display the per-pixel recipe output.

Reads the recipe output (x, y, ndvi) and writes three PNGs next to it. Fully
offline and reproducible — no remote scan. The CSV is the frozen result of
`ndvi.sql` (a 1024×1024 Sentinel-2 window); to regenerate it, wrap that query in

    COPY ( SELECT x, y, ROUND((b08-b04)/(b08+b04),4) AS ndvi
           FROM scene WHERE NOT isnan(b08-b04) ORDER BY y, x )
    TO 'ndvi.csv' STORED AS CSV;     -- then: gzip -9 ndvi.csv

  1. ndvi_map.png        — the NDVI raster: the flat (x, y, ndvi) table pivoted
                           back into the 2D scene, on a red→green vegetation ramp.
                           The whole point — a Zarr scene IS a table, and the
                           table IS the scene.
  2. ndvi_hist.png       — distribution of NDVI over the valid pixels, with the
                           mean and the usual land-cover thresholds marked.
  3. ndvi_landcover.png  — NDVI binned into land-cover classes (water/snow,
                           bare soil, sparse, moderate, dense vegetation): the
                           per-pixel expression turned into a classified map.

Both maps carry a LOCATOR INSET with country borders + coastline (cartopy) so the
~10 km window is placed geographically — it sits in Piedmont, NW Italy, near the
French/Alpine border. (No border crosses a 10 km tile, so the context belongs in
the inset, not on the raster.) cartopy/pyproj are optional: without them the maps
still render, just without the inset.

Usage:
    uv run --with pandas --with numpy --with matplotlib --with cartopy --with pyproj \
        cookbook/ndvi/plots.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

# Geographic locator is optional — degrade gracefully if cartopy/pyproj absent.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from pyproj import Transformer

    _HAVE_GEO = True
except ImportError:  # pragma: no cover
    _HAVE_GEO = False

UTM_EPSG = 32632          # scene CRS: WGS 84 / UTM zone 32N (from the store attrs)
TURIN = (7.686, 45.070)   # city marker for orientation in the locator

# Land-cover NDVI break points (standard remote-sensing convention) and the
# labels/colors for the classified map. Edges feed both the histogram guides and
# the discrete classification.
CLASS_EDGES = [-1.0, 0.0, 0.2, 0.4, 0.6, 1.0]
# NDVI < 0 is water/snow/cloud/built; this Alpine-foothills scene (May, near
# Turin) is mostly snow on the high peaks, so we label the class "water / snow".
CLASS_NAMES = ["water / snow", "bare soil", "sparse veg.",
               "moderate veg.", "dense veg."]
CLASS_COLORS = ["#2c7fb8", "#d9b382", "#c2e699", "#78c679", "#238443"]


def to_grid(df):
    """Pivot the flat (x, y, ndvi) table back into a 2D raster for imshow.
    Returns (Z, extent) with north (max y) at the top."""
    g = df.pivot_table(index="y", columns="x", values="ndvi")
    g = g.sort_index(ascending=False)          # northing descending → north on top
    xs, ys = g.columns.to_numpy(), g.index.to_numpy()
    # UTM metres → kilometres, relative to the window's SW corner, for readable ticks.
    x_km = (xs - xs.min()) / 1000.0
    y_km = (ys - ys.min()) / 1000.0
    extent = [x_km.min(), x_km.max(), y_km.min(), y_km.max()]
    return g.to_numpy(), extent


def scene_lonlat_bounds(df):
    """Convert the window's UTM (x, y) extent to lon/lat corners for the locator.
    Returns (lon_w, lon_e, lat_s, lat_n), or None if pyproj/cartopy are absent."""
    if not _HAVE_GEO:
        return None
    t = Transformer.from_crs(UTM_EPSG, 4326, always_xy=True)
    xs, ys = df["x"], df["y"]
    lon0, lat0 = t.transform(xs.min(), ys.min())
    lon1, lat1 = t.transform(xs.max(), ys.max())
    return min(lon0, lon1), max(lon0, lon1), min(lat0, lat1), max(lat0, lat1)


def add_locator(fig, bounds, rect=(0.135, 0.135, 0.24, 0.24)):
    """Draw a small regional map (country borders + coastline) with the scene
    footprint marked, so the ~10 km window is placed geographically. No-op when
    cartopy/pyproj are unavailable."""
    if not _HAVE_GEO or bounds is None:
        return
    lon_w, lon_e, lat_s, lat_n = bounds
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    ax.set_extent([5.0, 10.6, 43.3, 47.1], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#efece4")
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe2f3")
    ax.add_feature(cfeature.BORDERS, lw=0.7, edgecolor="0.3")
    ax.coastlines("50m", lw=0.5, color="0.3")
    # The footprint is tiny at this scale — draw the box AND a centroid marker.
    ax.add_patch(mpatches.Rectangle((lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
                                    fill=False, edgecolor="#c0392b", lw=1.4, zorder=5))
    ax.plot((lon_w + lon_e) / 2, (lat_s + lat_n) / 2, "s", ms=5,
            color="#c0392b", mec="white", mew=0.5, zorder=6)
    ax.plot(*TURIN, "o", ms=3, color="0.15", zorder=6)
    ax.text(TURIN[0] + 0.15, TURIN[1], "Turin", fontsize=6, va="center", zorder=6)
    ax.set_title("scene location", fontsize=7, pad=2)
    for s in ax.spines.values():
        s.set_edgecolor("0.4")


def plot_map(df, out):
    Z, extent = to_grid(df)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(Z, extent=extent, origin="upper", aspect="equal",
                   cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    ax.set_title("NDVI — the (x, y, ndvi) table flattened back to the scene\n"
                 "(Sentinel-2 L2A, 10 m, near Turin, Italy, 2025-05-05)")
    ax.set_xlabel("easting (km from window SW corner)")
    ax.set_ylabel("northing (km)")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("NDVI = (NIR − Red) / (NIR + Red)")
    add_locator(fig, scene_lonlat_bounds(df))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_hist(df, out):
    v = df["ndvi"].to_numpy()
    mean = float(np.nanmean(v))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(v, bins=120, range=(-0.3, 1.0), color="#3a8a4a", alpha=0.85)
    for edge in CLASS_EDGES[1:-1]:
        ax.axvline(edge, color="0.5", lw=0.8, ls="--")
    ax.axvline(mean, color="#c0392b", lw=1.6)
    ymax = ax.get_ylim()[1]
    ax.text(mean + 0.01, ymax * 0.55, f"mean = {mean:.4f}", color="#c0392b",
            fontsize=9, rotation=90, va="center", ha="left")
    # Class labels along the top, centered in each NDVI band.
    for lo, hi, name in zip(CLASS_EDGES[:-1], CLASS_EDGES[1:], CLASS_NAMES):
        ax.text((max(lo, -0.3) + min(hi, 1.0)) / 2, ymax * 0.96, name,
                ha="center", va="top", fontsize=8, color="0.35", rotation=0)
    ax.set_title("NDVI distribution over the valid pixels\n"
                 f"({len(v):,} pixels; dashed lines = land-cover thresholds)")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("pixel count")
    ax.set_xlim(-0.3, 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_landcover(df, out):
    Z, extent = to_grid(df)
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(CLASS_EDGES, cmap.N)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(Z, extent=extent, origin="upper", aspect="equal",
                   cmap=cmap, norm=norm)
    ax.set_title("NDVI land cover — the per-pixel expression, classified\n"
                 "(standard remote-sensing NDVI breaks)")
    ax.set_xlabel("easting (km from window SW corner)")
    ax.set_ylabel("northing (km)")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85,
                      ticks=[(a + b) / 2 for a, b in
                             zip(CLASS_EDGES[:-1], CLASS_EDGES[1:])])
    cb.ax.set_yticklabels(CLASS_NAMES)
    add_locator(fig, scene_lonlat_bounds(df))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=here / "ndvi.csv.gz")
    ap.add_argument("--outdir", default=here)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print(f"loaded {len(df):,} pixels · "
          f"{df['x'].nunique()} x × {df['y'].nunique()} y · "
          f"NDVI {df['ndvi'].min():.4f}…{df['ndvi'].max():.4f} "
          f"(mean {df['ndvi'].mean():.4f})")
    outdir = Path(a.outdir)
    plot_map(df, outdir / "ndvi_map.png")
    plot_hist(df, outdir / "ndvi_hist.png")
    plot_landcover(df, outdir / "ndvi_landcover.png")


if __name__ == "__main__":
    main()
