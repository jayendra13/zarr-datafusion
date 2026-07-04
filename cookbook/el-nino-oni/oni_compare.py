"""
Compare zarr-datafusion ONI output against NOAA CPC's official ONI table.

Reference: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
Machine-readable source used here: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
  columns: SEAS YR TOTAL ANOM   (ANOM = the ONI value, 1991-2020 base, ERSSTv5)

We compare two things, with the metrics each one actually calls for:

  1. RAW ANOMALIES (continuous, degC) -> regression error metrics
       MAE   primary, interpretable in degC
       ME    mean error = mean(ours - noaa): the SYSTEMATIC bias. Matters because
             NOAA's ONI is ERSSTv5 (not ERA5) and our monthly value is a single
             noon-of-the-15th sample, not a true monthly mean.
       r     Pearson correlation: phase/timing agreement, offset-independent
       RMSE, R2  secondary
       (MAE is also reported after removing the bias, to split offset from scatter)

  2. ENSO CLASS (El Nino / La Nina / Neutral from the +/-0.5 thresholds)
       confusion matrix          the diagnostic (boundary slip vs sign flip)
       quadratic-weighted kappa  HEADLINE: classes are ORDINAL, so a La Nina->El
                                 Nino error is penalised far more than a Neutral
                                 boundary slip; also corrects for chance under the
                                 heavy Neutral imbalance
       macro-F1 + per-class P/R  which phase we miss
     Accuracy is reported but deliberately NOT the headline (Neutral dominates).

Our values come from sql/oni_2025_all_seasons.sql. zarr-cli prints a pretty box
table (no CSV mode), so --computed accepts either that raw table dump or a plain
CSV with columns: year,season,oni_c.

Usage:
    # 1) produce our values once (slow remote scan):
    zarr-cli sql/oni_2025_all_seasons.sql > data/oni_computed.txt
    # 2) compare:
    uv run --with requests --with pandas --with numpy --with scikit-learn \
           --with scipy --with tabulate \
           scripts/oni_compare.py --computed data/oni_computed.txt

Options:
    --computed PATH   our values: zarr-cli table dump OR year,season,oni_c CSV
    --reference PATH  offline copy of oni.ascii.txt (else fetched live)
    --threshold T     El Nino/La Nina cutoff in degC (default 0.5)
    --out PATH        write merged per-season comparison CSV (default data/oni_comparison.csv)
"""

import argparse
import io
import sys

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tabulate import tabulate

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
SEASON_ORDER = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
                "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
CLASSES = ["La Nina", "Neutral", "El Nino"]   # ordinal: -1, 0, +1


def classify(x, t):
    """ONI value -> ENSO phase using the +/-t thresholds (NOAA single-season)."""
    if x >= t:
        return "El Nino"
    if x <= -t:
        return "La Nina"
    return "Neutral"


def load_reference(path=None):
    """NOAA CPC oni.ascii.txt -> DataFrame[year, season, noaa]."""
    if path:
        text = open(path, encoding="utf-8").read()
    else:
        print(f"Fetching NOAA reference: {ONI_URL}")
        r = requests.get(ONI_URL, timeout=30)
        r.raise_for_status()
        text = r.text
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 4 or p[0] == "SEAS" or not p[1].isdigit():
            continue
        rows.append({"season": p[0], "year": int(p[1]), "noaa": float(p[3])})
    return pd.DataFrame(rows)


def load_computed(path):
    """Our values from a zarr-cli box-table dump OR a year,season,oni_c CSV."""
    raw = open(path, encoding="utf-8").read()
    if "|" in raw and "+--" in raw:          # zarr-cli pretty table
        return _parse_box_table(raw)
    df = pd.read_csv(io.StringIO(raw))        # plain CSV
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    return df[["year", "season", "oni_c"]].rename(columns={"oni_c": "ours"})


def _parse_box_table(raw):
    rows = [ln for ln in raw.splitlines() if ln.lstrip().startswith("|")]
    if not rows:
        raise ValueError("no box-table rows found in computed dump")
    cells = lambda ln: [c.strip() for c in ln.strip().strip("|").split("|")]
    header = [h.lower() for h in cells(rows[0])]
    recs = []
    for ln in rows[1:]:
        d = dict(zip(header, cells(ln)))
        recs.append({"year": int(d["year"]),
                     "season": d["season"],
                     "ours": float(d["oni_c"])})
    return pd.DataFrame(recs)


def regression_report(m):
    ours, noaa = m["ours"].to_numpy(), m["noaa"].to_numpy()
    resid = ours - noaa
    bias = resid.mean()
    out = {
        "N (matched seasons)": len(m),
        "MAE  (degC)": mean_absolute_error(noaa, ours),
        "RMSE (degC)": np.sqrt(mean_squared_error(noaa, ours)),
        "ME / bias (degC)": bias,
        "MAE after de-bias": mean_absolute_error(noaa - noaa.mean(),
                                                 ours - ours.mean()),
        "Pearson r": pearsonr(ours, noaa)[0],
        "R^2": r2_score(noaa, ours),
        "max |resid| (degC)": np.abs(resid).max(),
    }
    return out, resid


def classification_block(m, t):
    y_true = m["noaa_cls"].to_numpy()
    y_pred = m["ours_cls"].to_numpy()
    idx = {c: i - 1 for i, c in enumerate(CLASSES)}   # La Nina=-1 .. El Nino=+1
    ord_true = np.array([idx[c] for c in y_true])
    ord_pred = np.array([idx[c] for c in y_pred])

    print("Confusion matrix  (rows = NOAA truth, cols = ours):")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_tbl = [[CLASSES[i]] + list(cm[i]) for i in range(len(CLASSES))]
    print(tabulate(cm_tbl, headers=["truth \\ pred"] + CLASSES, tablefmt="github"))
    print()

    headline = {
        "Quadratic-weighted kappa (HEADLINE)":
            cohen_kappa_score(ord_true, ord_pred, weights="quadratic"),
        "Cohen kappa (unweighted)": cohen_kappa_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, labels=CLASSES, average="macro"),
        "Accuracy (not headline)": (y_true == y_pred).mean(),
    }
    print("Class-agreement metrics:")
    print(tabulate([[k, f"{v:.3f}"] for k, v in headline.items()],
                   headers=["metric", "value"], tablefmt="github"))
    print()
    print("Per-class precision / recall / F1:")
    print(classification_report(y_true, y_pred, labels=CLASSES,
                                target_names=CLASSES, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--computed", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="data/oni_comparison.csv")
    ap.add_argument("--label", default="ERA5", help="dataset label for the report header")
    a = ap.parse_args()

    ours = load_computed(a.computed)
    ref = load_reference(a.reference)
    m = ours.merge(ref, on=["year", "season"], how="inner")
    if m.empty:
        sys.exit("No overlapping (year, season) between computed and reference.")
    m["sord"] = m["season"].map(lambda s: SEASON_ORDER.index(s)
                                if s in SEASON_ORDER else 99)
    m = m.sort_values(["year", "sord"]).reset_index(drop=True)
    m["resid"] = m["ours"] - m["noaa"]
    m["ours_cls"] = m["ours"].map(lambda x: classify(x, a.threshold))
    m["noaa_cls"] = m["noaa"].map(lambda x: classify(x, a.threshold))

    print("=" * 68)
    print(f"ONI comparison: zarr-datafusion ({a.label})  vs  NOAA CPC (ERSSTv5)")
    print(f"matched {len(m)} seasons, "
          f"{m.year.min()}-{m.year.max()}, threshold +/-{a.threshold} degC")
    print("=" * 68)
    print("\n[1] RAW ANOMALY VALUES  (regression)")
    print("-" * 40)
    reg, resid = regression_report(m)
    print(tabulate([[k, (f"{v:.3f}" if isinstance(v, float) else v)]
                    for k, v in reg.items()],
                   headers=["metric", "value"], tablefmt="github"))

    # MAE by era: ERA5 quality and reanalysis change pre/post the modern
    # satellite era (1979), so a split is informative.
    print("\nMAE by era:")
    eras = [("pre-1979", m.year < 1979), ("1979+", m.year >= 1979)]
    rows = [[lbl, int(mask.sum()),
             f"{mean_absolute_error(m.noaa[mask], m.ours[mask]):.3f}" if mask.any() else "-",
             f"{(m.ours[mask]-m.noaa[mask]).mean():+.3f}" if mask.any() else "-"]
            for lbl, mask in eras]
    print(tabulate(rows, headers=["era", "N", "MAE", "bias"], tablefmt="github"))

    print("\nLargest 10 residuals (|ours - noaa|):")
    top = m.reindex(m.resid.abs().sort_values(ascending=False).index).head(10)
    print(tabulate(
        top[["year", "season", "ours", "noaa", "resid", "ours_cls", "noaa_cls"]],
        headers=["yr", "seas", "ours", "noaa", "resid", "ours_cls", "noaa_cls"],
        tablefmt="github", showindex=False, floatfmt="+.2f"))

    print("\n[2] ENSO CLASS  (El Nino / La Nina / Neutral)")
    print("-" * 40)
    classification_block(m, a.threshold)

    m.drop(columns=["sord"]).to_csv(a.out, index=False)
    print(f"Wrote per-season comparison -> {a.out}")


if __name__ == "__main__":
    main()
