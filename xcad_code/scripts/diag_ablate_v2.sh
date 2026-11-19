#!/bin/bash
# diag_ablate_v2.sh — 四个消融格在 xcad_v2(修好编号)上的 test-only 同源重跑
#
# 为什么要重跑: 论文表 4 的四行来自 6 月/11-12 的训练日志, 每行的"总指标 / r1 / 逐类 /
# 同口径"分属不同打印点, 且旧日志的编号口径混杂。这里用现成 checkpoint 各跑一次
# --test(约 30 秒/臂, 不训练), 让每一行的所有格子都出自同一次评测。
#
# checkpoint 都是 2026-11-12 论文配置(h=6, dp0.4, seed42)训好的, 名字由参数唯一决定。
# 严格串行; 跑前确认 GPU 空闲。
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/outputs/reports/baselines_idspace/ablate_v2
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42)
HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  "$PY" main.py --test -d xcad_v2 "${PAPER[@]}" "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "stat name|Traceback|FileNotFound|覆盖不全" | head -2
}

run a1_base                                       # 仅骨架(不加任何机制)
run a2_compat   "${COMPAT[@]}"                        # 仅点2
run a3_het      "${HET[@]}"                           # 仅点1
run a4_joint    "${HET[@]}" "${COMPAT[@]}"            # 完整 xCAD

echo "ALL_DONE $(date -u +%H:%M:%S)"
