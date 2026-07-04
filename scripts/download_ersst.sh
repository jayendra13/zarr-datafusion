#!/usr/bin/env bash
# Download NOAA NCEI ERSST v5 monthly SST granules (NetCDF-4, uncompressed).
#
#   scripts/download_ersst.sh [START_YEAR] [END_YEAR]
#
# Defaults: 1854 .. current year. Files land in data/ersst_v5_nc/. Existing
# non-empty files are skipped, so re-running resumes an interrupted download.
# Months not yet published (recent/future) return 404 and are skipped.
#
# One granule per month, ~168 KB each; the full record is ~350 MB (~2,070 files).
# For the ONI cookbook only 1936..present is needed:
#   scripts/download_ersst.sh 1936
#
# Source: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
set -uo pipefail

BASE="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT_DIR:-$ROOT/data/ersst_v5_nc}"
START="${1:-1854}"
END="${2:-$(date +%Y)}"
JOBS="${JOBS:-10}"

mkdir -p "$OUT"

# Clear any zero-byte leftovers from prior 404s so they get retried.
find "$OUT" -name 'ersst.v5.*.nc' -type f -size 0 -delete 2>/dev/null || true

echo "ERSST v5: ${START}..${END} -> $OUT (parallel=$JOBS)"

# Build the list of YYYYMM granules still missing.
need=()
for y in $(seq "$START" "$END"); do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    f="ersst.v5.${y}${m}.nc"
    [ -s "$OUT/$f" ] && continue
    need+=("$f")
  done
done
echo "${#need[@]} granules to fetch (existing skipped)"
[ "${#need[@]}" -eq 0 ] && { echo "nothing to do"; exit 0; }

# Parallel fetch. -f => skip HTTP errors (unpublished months 404); non-zero exit
# from those is expected, so don't abort the batch.
printf '%s\n' "${need[@]}" | xargs -P "$JOBS" -I{} \
  curl -fsS --retry 3 --retry-delay 2 -o "$OUT/{}" "$BASE/{}" || true

# Drop any zero-byte files left by 404s so a rerun retries just those.
find "$OUT" -name 'ersst.v5.*.nc' -type f -size 0 -delete 2>/dev/null || true

n=$(find "$OUT" -name 'ersst.v5.*.nc' -type f | wc -l)
sz=$(du -sh "$OUT" 2>/dev/null | cut -f1)
echo "Done. $n granules in $OUT ($sz)"
