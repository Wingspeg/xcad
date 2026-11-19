#!/usr/bin/env bash
# Re-run philly (fast) and ali_v2020 (cached sensor dedup) after the gpu_types
# dimension fix.  ali is written to a staging dir first: the validated bundle is
# only replaced when the new run exits 0, so a failure can never leave a
# half-written set of parquets where the good one used to be.
set -uo pipefail
cd "$HOME/xcad/xcad_code"
R="$HOME/xcad/datasets/clean"
LOG="$HOME/xcad/xcad_code/outputs/reports"
PY=.venv/bin/python

echo "== philly $(date -u) =="
$PY -u src/data/clean_traces.py --dataset philly \
    --raw "$R/../philly/trace-data" --out "$R/philly" 2>&1 | tail -20
echo "philly rc=${PIPESTATUS[0]}"

echo "== ali staging $(date -u) =="
rm -rf "$R/_new"
mkdir -p "$R/_new/ali_v2020"
$PY -u src/data/clean_traces.py --dataset ali_v2020 \
    --raw "$HOME/xcad/datasets/alibaba-cluster-trace-gpu-v2020/raw" \
    --out "$R/_new/ali_v2020" 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "ali rc=$rc"

if [ "$rc" -eq 0 ] && [ -f "$R/_new/ali_v2020/manifest.json" ]; then
    mv "$R/ali_v2020" "$R/_old_ali_v2020"
    mv "$R/_new/ali_v2020" "$R/ali_v2020"
    rm -rf "$R/_old_ali_v2020" "$R/_new"
    echo "ali bundle swapped $(date -u)"
else
    rm -rf "$R/_new"
    echo "ali run failed -- kept the previous bundle"
fi

echo "== validate $(date -u) =="
$PY -u src/data/validate_clean.py --root "$R" --datasets ali_v2020 philly eagle 2>&1 | tail -30
echo "validate rc=${PIPESTATUS[0]}"
echo "DONE $(date -u)"
