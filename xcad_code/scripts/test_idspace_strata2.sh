#!/bin/bash
# test_idspace_strata2.sh — 与 test_idspace_strata.sh 同一件事, 但:
#   (1) main.py 已加入 "compat argmax vs 模型段内 top-1 一致性" 探针(R1-1 最硬的因果隔离证据);
#   (2) 覆盖 6 臂 = 论文配置 2x2 的四角 + 新编号的两格中间消融(het_only / compat_only)。
#
# 只做 test: --test True 路径加载 ../models/<model_name> 里 best-epoch checkpoint,
# 评测流程与训练末尾那次 test 完全一致(history = train+valid 55 个快照, test 窗口 20)。
# 前置: 训练队列(rerun_idspace_paper.sh + ablate_idspace.sh)必须已结束, 单进程占 ~12GB。
# 成本: 每臂实测 30-35 秒(6 个 test 快照), 顺序跑。
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/strata_idspace2
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
HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  "$PY" main.py "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Load Model name|一致性自检|覆盖不全|Traceback|out of range|CUDA out of memory" | head -4
}

run base_xcad      -d xcad    "${PAPER[@]}"
run base_xcad_v2   -d xcad_v2 "${PAPER[@]}"
run joint_xcad     -d xcad    "${PAPER[@]}" "${HET[@]}" "${COMPAT[@]}"
run joint_xcad_v2  -d xcad_v2 "${PAPER[@]}" "${HET[@]}" "${COMPAT[@]}"
run compat_only_v2 -d xcad_v2 "${PAPER[@]}" "${COMPAT[@]}"
run het_only_v2    -d xcad_v2 "${PAPER[@]}" "${HET[@]}"

echo "ALL_DONE $(date -u +%H:%M:%S)"
