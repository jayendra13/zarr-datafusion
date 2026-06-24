"""
Plots for the El Niño / ONI cookbook, from the offline comparison CSV.

Reads cookbook/el-nino-oni/oni_comparison.csv (columns:
year, season, ours, noaa, resid, ours_cls, noaa_cls) and writes two PNGs
next to it. No remote scan needed — purely offline and reproducible.

  1. oni_timeseries.png  — ONI 1950→now, ours vs NOAA, with the +/-0.5 degC
                           El Niño / La Niña threshold bands.
  2. oni_residual.png    — residual (ours - noaa) over time, split pre-1979 vs
                           1979+, with the per-era mean bias.

Usage:
    uv run --with pandas --with matplotlib \
        cookbook/el-nino-oni/plots.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# centre-month index of each 3-month season (DJF centre = Jan = 1, ... NDJ = Dec = 12)
SEASON_ORDER = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
                "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
SEASON_MONTH = {s: i + 1 for i, s in enumerate(SEASON_ORDER)}

THRESHOLD = 0.5          # degC, El Niño / La Niña cutoff
SPLIT_YEAR = 1979        # modern satellite era


def load(path):
    df = pd.read_csv(path)
    df = df[df["season"].isin(SEASON_MONTH)].copy()
    # continuous time axis: decimal year at the centre month of each season
    df["t"] = df["year"] + (df["season"].map(SEASON_MONTH) - 1) / 12.0
    return df.sort_values("t").reset_index(drop=True)


def plot_timeseries(df, out):
    fig, ax = plt.subplots(figsize=(14, 5))

    # threshold bands
    ax.axhspan(THRESHOLD, df[["ours", "noaa"]].max().max() + 0.3,
               color="#d6604d", alpha=0.10, zorder=0)
    ax.axhspan(df[["ours", "noaa"]].min().min() - 0.3, -THRESHOLD,
               color="#4393c3", alpha=0.10, zorder=0)
    ax.axhline(THRESHOLD, color="#d6604d", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-THRESHOLD, color="#4393c3", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0, color="0.5", lw=0.6)

    ax.plot(df["t"], df["noaa"], color="0.25", lw=1.3, label="NOAA CPC (ERSSTv5)")
    ax.plot(df["t"], df["ours"], color="#c0392b", lw=1.1, alpha=0.85,
            label="zarr-datafusion (ERA5)")

    ax.set_title("Oceanic Niño Index (ONI), 1950–present — zarr-datafusion (ERA5) vs NOAA CPC")
    ax.set_xlabel("year")
    ax.set_ylabel("ONI anomaly (°C)")
    ax.margins(x=0.01)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.text(0.995, 0.04,
            "shaded: El Niño (≥ +0.5 °C) / La Niña (≤ −0.5 °C)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.4")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_residual(df, out):
    pre = df[df["year"] < SPLIT_YEAR]
    post = df[df["year"] >= SPLIT_YEAR]
    pre_bias = pre["resid"].mean()
    post_bias = post["resid"].mean()
    pre_mae = pre["resid"].abs().mean()
    post_mae = post["resid"].abs().mean()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(SPLIT_YEAR, color="0.4", lw=1.0, ls="--")
    ax.text(SPLIT_YEAR, 0.95, " satellite era →",
            transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9, color="0.4")

    ax.scatter(pre["t"], pre["resid"], s=10, color="#8c8c8c", alpha=0.7,
               label=f"pre-{SPLIT_YEAR}  (n={len(pre)}, MAE {pre_mae:.2f}, bias {pre_bias:+.2f} °C)")
    ax.scatter(post["t"], post["resid"], s=10, color="#2166ac", alpha=0.7,
               label=f"{SPLIT_YEAR}+  (n={len(post)}, MAE {post_mae:.2f}, bias {post_bias:+.2f} °C)")

    # per-era mean bias lines
    ax.hlines(pre_bias, pre["t"].min(), pre["t"].max(),
              color="#5a5a5a", lw=2)
    ax.hlines(post_bias, post["t"].min(), post["t"].max(),
              color="#08306b", lw=2)

    ax.set_title("ONI residual (ERA5 − NOAA) over time — error halves in the satellite era")
    ax.set_xlabel("year")
    ax.set_ylabel("residual: ours − NOAA (°C)")
    ax.margins(x=0.01)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=here / "oni_comparison.csv")
    ap.add_argument("--outdir", default=here)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    df = load(a.csv)
    print(f"loaded {len(df)} seasons, {df.year.min()}–{df.year.max()}")
    plot_timeseries(df, outdir / "oni_timeseries.png")
    plot_residual(df, outdir / "oni_residual.png")


if __name__ == "__main__":
    main()
