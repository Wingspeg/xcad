#!/bin/bash
# diag_base_topseg.sh — 复用已有 checkpoint 的 test-only 诊断(约 35 秒/臂, 不训练)
#
# 要回答的问题:
#   ours 的 xCAD-base(仅骨架)臂在修好编号的 xcad_v2 上 r1 H@1 从 legacy 的 0.443224
#   掉到 0.011659、per-GPU macro 从 20.00 掉到 6.21。这是"模型真的不会"还是
#   "答案指向了别的编号段(候选集里根本不看 GPU 段)"?
#   main.py 里的 seg_probe 已经在记录(纯记录, 不影响任何数值):
#     "无约束 top-1 落在 GPU 段的比例" + "GPU 段内(filtered) H@1 / macro5"
#   11-12 那批日志早于这段探针, 所以必须拿现成 checkpoint 重跑一次 test-only 才有读数。
#
# 严格串行: 跑之前确认 GPU 空闲(单臂 11.6GB, 并发直接 OOM)。
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/outputs/reports/baselines_idspace
mkdir -p "$OUT"

BL=~/xcad/xcad_code
export LD_LIBRARY_PATH="$(ls -d "$BL"/.venv/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  "$PY" main.py --test "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Sanity Check: stat name|覆盖不全|Traceback|OutOfMemory|FileNotFound" | head -3
}

# 1) 新编号(xcad_v2)下的骨架臂 —— 本次要的答案
run diag_base_v2      -d xcad_v2 "${PAPER[@]}"
# 2) 旧编号(xcad)下的骨架臂 —— 同一把尺子的对照, 看 legacy 撞号时 top-1 落点比例
run diag_base_legacy  -d xcad    "${PAPER[@]}"
# 3) 新编号下的完整方法(joint) —— 证明修好段后段内口径仍然成立, 不被骨架的崩坏牵连
run diag_joint_v2     -d xcad_v2 "${PAPER[@]}" --hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5

echo "ALL_DONE $(date -u +%H:%M:%S)"
