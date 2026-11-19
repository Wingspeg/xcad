#!/bin/bash
# run_mit_arms_long_long.sh —— MIT Supercloud 第二份数据上的三臂（串行，单臂预算 15 分钟）
#
# 为什么要这三臂：R1-1 具名实验②要的是"复发 cohort 可构造"的场景，MIT 是唯一的
# 国外候选（见 docs/reviews/2026-11-16_second_dataset_screening.md §12/§14）。
# 论文配置里的 --hetero/--compat 两件事都建立在"tail 段是 6 个 GPU 型号"上，
# MIT 只有 1 个关系(placement)、卡型方差为 0，所以：
#   base          : PAPER 全套（GRU 关系演化开）—— 与 xcad_v2 的 base 臂同超参
#   hetero        : per-relation basis 分解（通用，不依赖 GPU 段）
#   static_rel    : 具名消融①（--no-rel-evolution），即把关系 GRU 换成静态关系嵌入
# 三臂都带 --probe-rel 0，让 main.py 走 generic_strata：同一批 test 行、同一套
# filtered 口径，逐层附 entity_memory / tail_prior 闭式基线（GPU 段专属报表跳过）。
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/mit_arms_long
mkdir -p "$OUT"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 300 --evaluate-every 20 --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42 -d mit --probe-rel 0)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  "$PY" main.py "${PAPER[@]}" --dump-preds "$OUT/$tag.pt" "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Traceback|OutOfMemory|Error" | head -3
}

run base
run hetero   --hetero --n-msg-basis 2
run static   --no-rel-evolution

echo "ALL_DONE $(date -u +%H:%M:%S)"
