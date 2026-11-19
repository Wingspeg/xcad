#!/usr/bin/env bash
# R1-1 反驳探针（两条, 都只复用 checkpoint, 不训练）
#
# 探针 A "把历史喂到嘴边": RE-GCN 测试期只聚合最后 test_history_len 个快照, 论文配置是 20,
#   而 xCAD 的 history 一共 49 个快照 —— 于是"复发 head 在窗口内不可达"有一半是我们自己
#   造成的。这里用同一批 checkpoint 把窗口改成 60(=全量), 其它一切不动。
#   他的解释若成立: 复发 cohort 的 generic TKG 应当回升到接近纯记忆上界(99.90%)。
#   我们的结论若不倒: 它仍然贴在类别先验上。
#
# 探针 B "冲突单元格": 每臂都 --dump-preds 落盘逐边预测, 交给 r1_conflict.py 离线切格子
#   (复发 / 构造性 cohort n_hist>=2,3 / 历史与属性冲突 / 历史对而属性错的最强格),
#   并把"模型在冲突边里跟历史还是跟属性"直接计数。
#
# 成本: 每臂每次 ~35s, 5 臂 x 2 窗口 = 10 次 ≈ 6 分钟。等 gru 队列跑完自动开始。
set -uo pipefail

SRC_DIR="$HOME/xcad/xcad_code/xcad_model/src"
VENV="$HOME/xcad/xcad_code/.venv/bin/python"
OUT_DIR="$SRC_DIR/rebuttal_probe"
RUN_OUT="$SRC_DIR/rebuttal_probe_run.out"
GATE="$SRC_DIR/gru_micro_run.out"

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
NOREV=(--no-rel-evolution)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

wait_for_gate() {
  local waited=0
  while ! grep -aq "^\[all\] finished=" "$GATE" 2>/dev/null; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge 7200 ]; then echo "[wait] gate timeout" >> "$RUN_OUT"; return 1; fi
  done
  echo "[wait] gru queue done, waited=${waited}s ($(date -u +%H:%M:%S))" >> "$RUN_OUT"
}

run() {
  local name="$1" win="$2"; shift 2
  echo "[$name w=$win] start=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
  cd "$SRC_DIR" || exit 1
  "$VENV" main.py -d xcad_v2 "${PAPER[@]}" --test-history-len "$win" --test \
        --dump-preds "$OUT_DIR/${name}_w${win}.pt" "$@" \
        > "$OUT_DIR/${name}_w${win}.log" 2>&1
  local rc=$?
  echo "[$name w=$win] exit=$rc done=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
  grep -a "段内(filtered) H@1\|复发 head\|分层器 FULL & r1" "$OUT_DIR/${name}_w${win}.log" \
        | head -4 >> "$RUN_OUT" 2>&1
}

wait_for_gate || exit 1

for win in 20 60; do
  run joint        "$win" "${HET[@]}" "${COMPAT[@]}"
  run compat_only  "$win" "${COMPAT[@]}"
  run het_only     "$win" "${HET[@]}"
  run base         "$win"
  run joint_norev  "$win" "${HET[@]}" "${COMPAT[@]}" "${NOREV[@]}"
done

cd "$SRC_DIR" || exit 1
"$VENV" r1_conflict.py \
  "joint_w20=$OUT_DIR/joint_w20.pt" "joint_w60=$OUT_DIR/joint_w60.pt" \
  "joint_norev_w20=$OUT_DIR/joint_norev_w20.pt" "joint_norev_w60=$OUT_DIR/joint_norev_w60.pt" \
  "compat_only_w20=$OUT_DIR/compat_only_w20.pt" "compat_only_w60=$OUT_DIR/compat_only_w60.pt" \
  "het_only_w20=$OUT_DIR/het_only_w20.pt" "het_only_w60=$OUT_DIR/het_only_w60.pt" \
  "base_w20=$OUT_DIR/base_w20.pt" "base_w60=$OUT_DIR/base_w60.pt" \
  > "$OUT_DIR/conflict_tables.txt" 2>&1
echo "[conflict] exit=$? $(date -u +%H:%M:%S)" >> "$RUN_OUT"

echo "[all] finished=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
