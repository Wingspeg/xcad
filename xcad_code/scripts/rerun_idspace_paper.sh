#!/bin/bash
# rerun_idspace_paper.sh — 论文配置下, baseline / Ours 各跑两套编号数据(seed 42)
#
# 顺序很重要, 不要并发: 单进程占 11.6GB, 两个并发直接 OOM(2026-11-12 已踩)。
#
# 目的:
#   1) 旧数据臂(xcad)先证明"改了 device 处理 + gpu_base 判定之后, GPU 上仍能复现 6 月数字";
#   2) 新数据臂(xcad_v2)给出全局四段编号下同一配置的指标。
#
# 复现基准(6 月日志, test, filter_ent):
#   base_h6_macro_s42 : MRR 0.047811 | r1 MRR 0.684149 H@1 0.425144 | Per-GPU macro 0.2000
#   joint_s42         : MRR 0.065265 | r1 MRR 0.925953 H@1 0.883998 | Per-GPU macro 0.5079
#
# GPU 探针(小配置, 2026-11-12)已经给出的方向:
#   compat: xcad -> MRR 0.060591 / r1 0.896036 / macro 0.4964
#           xcad_v2 -> MRR 0.060378 / r1 0.896938 / macro 0.4940   (差异 <0.3%)
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/rerun_idspace
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
JOINT=(--hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  "$PY" main.py "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "\[id-space\]|覆盖不全|OutOfMemory|Traceback" | head -4
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/  gpu_after: /"
}

# 1/2 baseline: 旧编号 -> 新编号
run base_xcad    -d xcad    "${PAPER[@]}"
run base_xcad_v2 -d xcad_v2 "${PAPER[@]}"
# 3/4 Ours(点1+点2): 旧编号 -> 新编号
run joint_xcad    -d xcad    "${PAPER[@]}" "${JOINT[@]}"
run joint_xcad_v2 -d xcad_v2 "${PAPER[@]}" "${JOINT[@]}"

echo "ALL_DONE $(date -u +%H:%M:%S)"
