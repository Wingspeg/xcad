#!/bin/bash
# test_idspace_strata.sh — 只做 test, 复用 rerun_idspace_paper.sh 训练出的 4 个 checkpoint,
# 让打了分层补丁的 main.py 输出 "按 head 复现性分层" 的 r1 表(R1-1 的决定性实验)。
#
# 为什么能复用: main.py 在 --test True 且 ../models/<model_name> 存在时直接加载
# best-stat checkpoint, 评测路径与训练末尾那次 test 完全一致(train+valid 作 history,
# test_history_len=20), 所以分层数字和总表是同一批预测, 只是多做了记账。
#
# 前置: 4 臂训练队列(rerun_idspace_paper.sh)必须已经全部结束, 否则 GPU 会被抢占;
#       main.py 必须已经是带 stratified_report 补丁的版本。
#
# 每臂 ~2 分钟(6 个 test 快照), 顺序跑, 不要并发。
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/strata_idspace
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42 --test)
JOINT=(--hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  "$PY" main.py "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Load Model name|覆盖不全|Traceback|OutOfMemory" | head -3
}

run base_xcad    -d xcad    "${PAPER[@]}"
run base_xcad_v2 -d xcad_v2 "${PAPER[@]}"
run joint_xcad    -d xcad    "${PAPER[@]}" "${JOINT[@]}"
run joint_xcad_v2 -d xcad_v2 "${PAPER[@]}" "${JOINT[@]}"

echo "ALL_DONE $(date -u +%H:%M:%S)"
