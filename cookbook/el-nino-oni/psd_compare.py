"""
Power-spectral-density comparison of two Niño-3.4 monthly-mean SST series.

Input: cookbook/el-nino-oni/nino34_monthly_sst.csv  (columns: time, sst_ersst, sst_era5)
  - sst_ersst : ERSST v5, an ACTUAL monthly mean
  - sst_era5  : ERA5, a single representative hour per month (noon of the 15th) -> a PROXY

The question a PSD answers here: does the cheap one-hour-per-month ERA5 proxy carry
the same variance-by-timescale as a true monthly mean? They should agree at the low
frequencies that matter for ENSO (the 2-7 year band) and at the annual cycle; any
*excess* ERA5 power at high frequency is the sampling noise of taking one hour instead
of averaging the month.

Welch's method needs EVENLY SPACED samples, so before anything spectral we:
  1. parse `time`, sort strictly ascending, drop duplicate months,
  2. reindex onto a complete month-start grid (MS) from first to last month,
  3. report any missing months and linearly interpolate them (spectral methods cannot
     take NaNs), so the sample spacing is exactly 1 month everywhere.

Usage:
    uv run --with pandas --with numpy --with scipy --with matplotlib \
        cookbook/el-nino-oni/psd_compare.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid

ERSST_C = "#2c8a4a"   # green  — true monthly mean
ERA5_C = "#c0392b"    # red    — noon-of-15th proxy

FS = 12.0             # samples per year -> frequency axis is in cycles/year
NPERSEG = 256         # Welch segment length (~21 yr); freq resolution ~0.047 cyc/yr

# ENSO interannual band: periods of 2-7 years -> frequency 1/7 .. 1/2 cycles/year
ENSO_LO_CPY, ENSO_HI_CPY = 1.0 / 7.0, 1.0 / 2.0


def load_ordered(path):
    """Load, order strictly by time, put on a gap-free monthly grid."""
    df = pd.read_csv(path, parse_dates=["time"])
    df = (
        df.dropna(subset=["time"])
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")           # <- guarantee ascending, even if the CSV wasn't
        .set_index("time")
    )

    full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing = full.difference(df.index)
    df = df.reindex(full)

    n_interp = int(df[["sst_ersst", "sst_era5"]].isna().any(axis=1).sum())
    df = df.interpolate(method="linear", limit_direction="both")

    # sanity: index is the contiguous month-start grid (one sample per calendar
    # month, no gaps) — Welch treats these as evenly spaced at fs=12/year
    assert df.index.equals(full), "grid is not a contiguous monthly series"
    assert not df[["sst_ersst", "sst_era5"]].isna().any().any(), "NaNs remain"

    print(f"loaded {len(df)} months: {df.index.min():%Y-%m} -> {df.index.max():%Y-%m}")
    if len(missing):
        print(f"  filled {len(missing)} missing month(s) by interpolation: "
              f"{', '.join(m.strftime('%Y-%m') for m in missing[:12])}"
              f"{' ...' if len(missing) > 12 else ''}")
    else:
        print("  no missing months — series was already contiguous")
    print(f"  rows needing interpolation on either column: {n_interp}")
    return df


def welch_psd(x):
    """Linear-detrended Welch PSD; returns (freq cyc/yr, power)."""
    f, pxx = signal.welch(
        np.asarray(x, float),
        fs=FS,
        window="hann",
        nperseg=min(NPERSEG, len(x)),
        detrend="linear",              # strip mean + warming trend before the FFT
        scaling="density",
    )
    return f, pxx


def band_power(f, pxx, lo, hi):
    """Integrate the PSD over [lo, hi] cycles/year (variance in that band)."""
    m = (f >= lo) & (f <= hi)
    return float(trapezoid(pxx[m], f[m]))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=here / "nino34_monthly_sst.csv")
    ap.add_argument("--out", default=here / "nino34_psd.png")
    a = ap.parse_args()

    df = load_ordered(a.csv)

    f_e, p_e = welch_psd(df["sst_ersst"])
    f_a, p_a = welch_psd(df["sst_era5"])

    # ---- numeric summary -----------------------------------------------------
    print("\nseries variance (detrended, degC^2):")
    for name, col in (("ERSST (true monthly mean)", "sst_ersst"),
                      ("ERA5  (noon-15th proxy)  ", "sst_era5")):
        v = float(signal.detrend(df[col].to_numpy(), type="linear").var())
        print(f"  {name}: {v:.4f}")
    resid = signal.detrend((df["sst_era5"] - df["sst_ersst"]).to_numpy(), type="linear")
    print(f"  ERA5 - ERSST difference    : {resid.var():.4f}  "
          f"(RMS {resid.std():.3f} degC) — proxy sampling noise")

    print("\nband-integrated power (degC^2):")
    print(f"  {'band':<26}{'ERSST':>10}{'ERA5':>10}{'ERA5/ERSST':>12}")
    bands = [
        ("ENSO 2-7 yr",        ENSO_LO_CPY, ENSO_HI_CPY),
        ("annual 0.9-1.1 c/yr", 0.9, 1.1),
        ("high-freq > 1.5 c/yr", 1.5, FS / 2),
    ]
    for label, lo, hi in bands:
        be, ba = band_power(f_e, p_e, lo, hi), band_power(f_a, p_a, lo, hi)
        ratio = ba / be if be else float("nan")
        print(f"  {label:<26}{be:>10.4f}{ba:>10.4f}{ratio:>12.2f}")

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # ENSO band shading + annual/semiannual guides
    ax.axvspan(ENSO_LO_CPY, ENSO_HI_CPY, color="#4393c3", alpha=0.10, zorder=0,
               label="ENSO band (2-7 yr)")
    for cyc, lab in ((1.0, "annual"), (2.0, "semi-annual")):
        ax.axvline(cyc, color="0.6", lw=0.8, ls="--", zorder=1)
        ax.text(cyc, 0.97, lab, transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=8, color="0.45")

    ax.loglog(f_e, p_e, color=ERSST_C, lw=1.6,
              label="ERSST v5 — true monthly mean", zorder=3)
    ax.loglog(f_a, p_a, color=ERA5_C, lw=1.6, alpha=0.85,
              label="ERA5 — noon-of-15th proxy", zorder=2)

    ax.set_xlabel("frequency (cycles / year)")
    ax.set_ylabel("PSD  (°C² · year)")
    ax.set_title("Niño-3.4 monthly SST — power spectral density: ERSST vs ERA5")
    ax.set_xlim(f_e[1], FS / 2)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower left", framealpha=0.9)

    # top axis: period in years (period = 1 / frequency); guard the 0 endpoint
    def _recip(v):
        v = np.asarray(v, float)
        return np.divide(1.0, v, out=np.full_like(v, np.inf), where=v != 0)

    secax = ax.secondary_xaxis("top", functions=(_recip, _recip))
    secax.set_xlabel("period (years)")
    secax.set_xticks([10, 5, 2, 1, 0.5])
    secax.set_xticklabels(["10", "5", "2", "1", "0.5"])

    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
