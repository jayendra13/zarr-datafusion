"""
Plots for the El Niño / ONI cookbook — both recipes overlaid vs NOAA.

Reads the two per-recipe comparison CSVs (columns:
year, season, ours, noaa, resid, ours_cls, noaa_cls):
  - oni_ersst_comparison.csv   (ERSST v5 — NOAA's own source, ~exact)
  - oni_comparison.csv         (ERA5 — independent, high-res reanalysis)

Writes three PNGs next to them. Offline and reproducible — no remote scan.

  1. oni_timeseries.png  — ONI 1950→now: NOAA, ERSST (overlays NOAA), ERA5.
  2. oni_residual.png    — residual (ours − NOAA) over time: ERSST pinned at ~0,
                           ERA5 scattered (its spread is pure dataset difference).
  3. oni_scatter.png     — ours vs NOAA: ERSST on the y=x diagonal, ERA5 around it.

Usage:
    uv run --with pandas --with matplotlib cookbook/el-nino-oni/plots.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SEASON_ORDER = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
                "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
SEASON_MONTH = {s: i + 1 for i, s in enumerate(SEASON_ORDER)}

THRESHOLD = 0.5  # degC, El Niño / La Niña cutoff

ERSST_C = "#2c8a4a"   # green
ERA5_C = "#c0392b"    # red
NOAA_C = "0.55"       # medium grey — drawn as a wide underlay so it stays
                      # visible as its own series even where ERSST traces it exactly


def load(path):
    df = pd.read_csv(path)
    df = df[df["season"].isin(SEASON_MONTH)].copy()
    df["t"] = df["year"] + (df["season"].map(SEASON_MONTH) - 1) / 12.0
    return df.sort_values("t").reset_index(drop=True)


def stats(df):
    r = df["resid"]
    return r.abs().mean(), r.mean()


def plot_timeseries(era5, ersst, out):
    fig, ax = plt.subplots(figsize=(14, 5))
    ylo = min(era5["ours"].min(), ersst["ours"].min(), era5["noaa"].min()) - 0.3
    yhi = max(era5["ours"].max(), ersst["ours"].max(), era5["noaa"].max()) + 0.3

    ax.axhspan(THRESHOLD, yhi, color="#d6604d", alpha=0.08, zorder=0)
    ax.axhspan(ylo, -THRESHOLD, color="#4393c3", alpha=0.08, zorder=0)
    ax.axhline(THRESHOLD, color="#d6604d", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-THRESHOLD, color="#4393c3", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="0.6", lw=0.6)

    mae_ersst, _ = stats(ersst)
    mae_era5, _ = stats(era5)
    ax.plot(era5["t"], era5["noaa"], color=NOAA_C, lw=3.5, alpha=0.9,
            solid_capstyle="round", label="NOAA CPC reference (ERSSTv5)", zorder=2)
    ax.plot(ersst["t"], ersst["ours"], color=ERSST_C, lw=1.1, alpha=0.95,
            label=f"zarr-datafusion · ERSST v5  (MAE {mae_ersst:.3f} — exact)", zorder=3)
    ax.plot(era5["t"], era5["ours"], color=ERA5_C, lw=1.0, alpha=0.75,
            label=f"zarr-datafusion · ERA5  (MAE {mae_era5:.2f} — independent)", zorder=1)

    ax.set_title("Oceanic Niño Index (ONI), 1950–present — two recipes vs NOAA CPC")
    ax.set_xlabel("year")
    ax.set_ylabel("ONI anomaly (°C)")
    ax.margins(x=0.01)
    ax.set_ylim(ylo, yhi)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.text(0.995, 0.04, "shaded: El Niño (≥ +0.5 °C) / La Niña (≤ −0.5 °C)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_residual(era5, ersst, out):
    mae_e, bias_e = stats(era5)
    mae_s, bias_s = stats(ersst)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.axhline(0, color="0.6", lw=0.8)
    ax.scatter(era5["t"], era5["resid"], s=10, color=ERA5_C, alpha=0.55,
               label=f"ERA5   (MAE {mae_e:.2f}, bias {bias_e:+.2f} °C) — dataset difference")
    ax.scatter(ersst["t"], ersst["resid"], s=10, color=ERSST_C, alpha=0.8,
               label=f"ERSST v5  (MAE {mae_s:.3f}, bias {bias_s:+.3f} °C) — reproduces NOAA")

    ax.set_title("ONI residual (ours − NOAA) — ERSST is exact; ERA5's spread is ERA5-vs-ERSST")
    ax.set_xlabel("year")
    ax.set_ylabel("residual: ours − NOAA (°C)")
    ax.margins(x=0.01)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_scatter(era5, ersst, out):
    r_e = era5["ours"].corr(era5["noaa"])
    r_s = ersst["ours"].corr(ersst["noaa"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim = [min(era5["noaa"].min(), era5["ours"].min()) - 0.2,
           max(era5["noaa"].max(), era5["ours"].max()) + 0.2]

    ax.plot(lim, lim, color="0.6", lw=1.0, ls="--", zorder=1, label="y = x")
    for v in (THRESHOLD, -THRESHOLD):
        ax.axhline(v, color="0.85", lw=0.7, zorder=0)
        ax.axvline(v, color="0.85", lw=0.7, zorder=0)
    ax.scatter(era5["noaa"], era5["ours"], s=9, color=ERA5_C, alpha=0.5,
               label=f"ERA5  (r = {r_e:.3f})", zorder=2)
    ax.scatter(ersst["noaa"], ersst["ours"], s=9, color=ERSST_C, alpha=0.7,
               label=f"ERSST v5  (r = {r_s:.3f})", zorder=3)

    ax.set_title("ours vs NOAA — ERSST on the diagonal, ERA5 around it")
    ax.set_xlabel("NOAA CPC ONI (°C)")
    ax.set_ylabel("zarr-datafusion ONI (°C)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-csv", default=here / "oni_comparison.csv")
    ap.add_argument("--ersst-csv", default=here / "oni_ersst_comparison.csv")
    ap.add_argument("--outdir", default=here)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    era5 = load(a.era5_csv)
    ersst = load(a.ersst_csv)
    print(f"loaded ERA5 {len(era5)} seasons, ERSST {len(ersst)} seasons")
    plot_timeseries(era5, ersst, outdir / "oni_timeseries.png")
    plot_residual(era5, ersst, outdir / "oni_residual.png")
    plot_scatter(era5, ersst, outdir / "oni_scatter.png")


if __name__ == "__main__":
    main()
