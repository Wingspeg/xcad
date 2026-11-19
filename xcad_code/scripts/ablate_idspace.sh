#!/usr/bin/env bash
# R1-1 归因隔离 · 论文配置消融（新编号数据 xcad_v2）
#
# 目的: 把"性能提升来自哪里"拆成 2x2 的四格 —— 是否有 点1(异质关系基 message passing)
#       x 是否有 点2(algorithm->GPU 兼容性头)。6 月日志只有旧编号数据上的
#       base / het / compat_v2 / joint, 且其中多数没有 per-GPU macro 行;
#       rerun_idspace_paper.sh 只补了 base 与 joint 两格。这里补中间两格。
#
# 前置: rerun_idspace_paper.sh 已跑完(单进程占 11.6-12.5GB, 不能并发, 3090 只有 24GB)。
# 成本: 论文配置单臂 ~15min(实测 base_xcad 30 epoch ≈ 13min + 末次 test eval)。
set -uo pipefail

SRC_DIR="$HOME/xcad/xcad_code/xcad_model/src"
VENV="$HOME/xcad/xcad_code/.venv/bin/python"
OUT_DIR="$SRC_DIR/ablate_idspace"
RUN_OUT="$SRC_DIR/ablate_idspace_run.out"

mkdir -p "$OUT_DIR"
: > "$RUN_OUT"

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1
       --weight 1 --discount 1 --angle 10 --seed 42)

HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)

run() {
  local name="$1"; shift
  echo "[$name] start=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
  cd "$SRC_DIR" || exit 1
  "$VENV" main.py -d xcad_v2 "$@" > "$OUT_DIR/$name.log" 2>&1
  echo "[$name] done =$(date -u +%H:%M:%S)" >> "$RUN_OUT"
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# 只有中间两格; base / het+compat 由 rerun_idspace_paper.sh 提供
run compat_only "${PAPER[@]}" "${COMPAT[@]}"
run het_only    "${PAPER[@]}" "${HET[@]}"

echo "[all] finished=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
