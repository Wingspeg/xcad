#!/usr/bin/env bash
# Second-dataset screening for R1-1 named experiment (2).  Zero GPU: byte-range
# extraction of Alibaba cluster-trace-gpu-v2026 partitions + CPU statistics.
# See docs/reviews/2026-11-16_second_dataset_screening.md for the criteria.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=$REPO/.venv/bin/python
SRC=$REPO/xcad_code/src/data
RAW=$REPO/datasets/v2026
RPT=$REPO/xcad_code/outputs/reports
POD=asi_opensource_pod_hourly.zip

mkdir -p "$RAW/raw" "$RPT"

# ---------------------------------------------------------------- 1. fetch
# one dense day (24 hourly partitions, 2.2 GB, ~60 s at 40 MB/s)
"$PY" "$SRC/ali_trace_slice.py" index "$POD"
"$PY" "$SRC/ali_trace_slice.py" get "$POD" --pattern '/day=180/hour='
"$PY" "$SRC/ali_trace_slice.py" get "$POD" --pattern '/day=90/hour='
# the recommended grid: one snapshot per day, days 165..184
"$PY" "$SRC/ali_trace_slice.py" get "$POD" --pattern '/hour=00/' \
  --days 165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184

# ------------------------------------------------- 2. criteria (i)(iii)(iv)
"$PY" "$SRC/screen_v2026.py" "$RAW/daily20"            > "$RPT/screen_v2026_inventory_daily20.log"
"$PY" "$SRC/screen_v2026.py" "$RAW/spaced7"          > "$RPT/screen_v2026_inventory_spaced7.log"

# ------------------------------------------ 3. criterion (ii): channel gap
"$PY" "$SRC/screen_v2026_conflict.py" "$RAW/daily20"   > "$RPT/screen_v2026_conflict_daily20.log"
"$PY" "$SRC/screen_v2026_conflict.py" "$RAW/day180"    > "$RPT/screen_v2026_conflict_day180.log"
"$PY" "$SRC/screen_v2026_conflict.py" "$RAW/day90"     > "$RPT/screen_v2026_conflict_day90.log"

# ------------------------------ 4. criterion (ii): ceiling of any static model
"$PY" "$SRC/screen_v2026_prior.py" "$RAW/day180" 6 12 18 23 > "$RPT/screen_v2026_prior_day180.log"
"$PY" "$SRC/screen_v2026_prior.py" "$RAW/day90"  6 12 18 23 > "$RPT/screen_v2026_prior_day90.log"
"$PY" "$SRC/screen_v2026_prior.py" "$RAW/daily20" 5 10 15 19 > "$RPT/screen_v2026_prior_daily20.log"

echo "ALL_DONE"
