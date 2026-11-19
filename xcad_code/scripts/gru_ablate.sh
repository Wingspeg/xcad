#!/usr/bin/env bash
# R1-1 具名消融① —— "the GRU evolution is replaced by per-snapshot static message passing"
#
# 背景: 评审意见点名的两个实验中, 此前只做了第二个的替代版(自然复发分层),
#       第一个(具名消融①)由本脚本补上。
#       2x2 里的 --hetero 切换的是**消息基**(n_msg_basis), 不是 rrgcn.py:241 的
#       relation GRUCell, 所以"关系随时间演化"这条通道至今没有被直接否证。
#
# 做法: main.py 新增 --no-rel-evolution (默认关闭)。开启时 forward 里跳过
#       relation_cell_1, 每个快照都用静态 emb_rel 做消息传递, 且 decoder 的
#       r_emb 也退回 emb_rel —— 即 GRU 演化整体移除, 节点侧的时间门控保留。
#       不加 flag 时代码路径逐位不变, 由 step 0 的 --test 复现来证明。
#
# 成本: step0 ~35s(纯打分) / step1 ~55s(微型训练, 冒烟) / step2-3 论文配置各 ~15min
#       单臂占 11.6-12.5GB, 3090 24GB 必须串行。
set -uo pipefail

SRC_DIR="$HOME/xcad/xcad_code/xcad_model/src"
VENV="$HOME/xcad/xcad_code/.venv/bin/python"
OUT_DIR="$SRC_DIR/gru_ablate"
RUN_OUT="$SRC_DIR/gru_ablate_run.out"

mkdir -p "$OUT_DIR"
: > "$RUN_OUT"

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1
       --weight 1 --discount 1 --angle 10 --seed 42)

MICRO=(--gpu 0 --n-hidden 32 --n-layers 1 --n-bases 8
       --train-history-len 1 --test-history-len 1
       --n-epochs 2 --entity-prediction --dilate-len 1
       --weight 1 --discount 1 --angle 10 --seed 42)

HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)
NOREV=(--no-rel-evolution)

run() {
  local name="$1"; shift
  echo "[$name] start=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
  cd "$SRC_DIR" || exit 1
  "$VENV" main.py -d xcad_v2 "$@" > "$OUT_DIR/$name.log" 2>&1
  echo "[$name] exit=$? done =$(date -u +%H:%M:%S)" >> "$RUN_OUT"
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# ---- step 0: 回归 —— 复用已有微型 checkpoint, 不加 flag 必须复现 0.854343 ----
run inert_check "${MICRO[@]}" "${COMPAT[@]}" --test
grep -a "段内(filtered) H@1" "$OUT_DIR/inert_check.log" >> "$RUN_OUT"

# ---- step 1: 冒烟 —— 微型配置开 flag, 看 GRU 移除是否让纯属性头照常工作 ----
run micro_norev "${MICRO[@]}" "${COMPAT[@]}" "${NOREV[@]}"

# ---- step 2: Ours 去掉 GRU  (对照 joint_xcad_v2: r1 H@1 0.884843 / MRR 0.065962) ----
run joint_norev "${PAPER[@]}" "${HET[@]}" "${COMPAT[@]}" "${NOREV[@]}"

# ---- step 3: het-only 去掉 GRU (对照 het_only: r1 H@1 0.425144 = 恒答 T4) ----
run het_only_norev "${PAPER[@]}" "${HET[@]}" "${NOREV[@]}"

echo "[all] finished=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
