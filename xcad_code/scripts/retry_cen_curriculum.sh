#!/bin/bash
# CEN 的 h=3 蒸馏臂在 12:28 挂在 loss.backward(): cuDNN CUDNN_STATUS_NOT_INITIALIZED。
# 这里只做一件事：换显存分配策略重跑那一条链（curriculum -> test-only），别的都不动。
set -u
BL=~/xcad/baselines
OUT=~/xcad/outputs/reports/baselines_idspace
mkdir -p "$OUT"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$(ls -d "$BL"/.venv/lib/python3.10/site-packages/nvidia/*/lib | tr "\n" ":")${LD_LIBRARY_PATH:-}"
CEN=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
     --start-history-len 2 --train-history-len 3 --test-history-len 3 --dilate-len 1
     --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
     --self-loop --layer-norm --entity-prediction --lr 0.001 --ft_lr 0.001 --norm_weight 0.1)
( cd "$BL/CEN/src" || exit 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader --loop=20s > "$OUT/cen_mem.csv" &
  MON=$!
  echo "[v2_cen_curriculum_r2] start=$(date -u +%H:%M:%S)"
  "$BL/.venv/bin/python3" main.py "${CEN[@]}" -d xcad_v2 --test 0 --n-epochs 30 --evaluate-every 1 > "$OUT/v2_cen_curriculum_r2.log" 2>&1
  echo "[v2_cen_curriculum_r2] exit=$? done=$(date -u +%H:%M:%S)"
  kill $MON 2>/dev/null
  echo "[v2_cen_test_r2] start=$(date -u +%H:%M:%S)"
  "$BL/.venv/bin/python3" main.py "${CEN[@]}" -d xcad_v2 --test 2 > "$OUT/v2_cen_test_r2.log" 2>&1
  echo "[v2_cen_test_r2] exit=$? done=$(date -u +%H:%M:%S)" )
tr "\r" "\n" < "$OUT/v2_cen_test_r2.log" | grep -aE "MRR \(filter_ent\)|Hits \(filter_ent\) @ 1:|r1_suits|Per-GPU|id-space|Traceback" | head -6
echo "CEN_RETRY_DONE $(date -u +%H:%M:%S)"
