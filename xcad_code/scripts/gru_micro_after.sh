#!/usr/bin/env bash
# gru_ablate.sh 的收尾两步(配置订正版), 由主队列结束后自动触发。
#
# 为什么要单独跑: 主队列里 step0/step1 都因为 MICRO 数组写错了参数而 exit=1 ——
#   1) inert_check: 传了 --weight 1 --discount 1, argparse 把它们转成 float,
#      checkpoint 名变成 weight:1.0-discount:1.0, 与已存在的 weight:1-discount:1 不符
#      -> main.py 判定"不存在", 走 --test 分支直接退出 1(不是代码缺陷, 是取名规则)。
#   2) micro_norev: 微型配置默认 --evaluate-every 20 而 n_epochs=2, 训练期一次都不 eval
#      -> 从不 torch.save -> 末尾 test(mode='test') 去 load 不存在的文件而 FileNotFoundError。
#      (flag 本身没问题: Epoch 1 已正常训练完成, 报错发生在保存/加载环节。)
#
# 目标数字: 不加 flag 的微型臂必须复现 src/probe_check/compat_xcad_v2.log 的
#          "GPU 段内(filtered) H@1 = 0.854343"  —— 这是本轮 main.py/rrgcn.py 打补丁
#          在 GPU 路径上空操作的回归证据。
set -uo pipefail

SRC_DIR="$HOME/xcad/xcad_code/xcad_model/src"
VENV="$HOME/xcad/xcad_code/.venv/bin/python"
OUT_DIR="$SRC_DIR/gru_ablate"
MAIN_OUT="$SRC_DIR/gru_ablate_run.out"
RUN_OUT="$SRC_DIR/gru_micro_run.out"

: > "$RUN_OUT"

MICRO=(--gpu 0 --n-hidden 32 --n-layers 1 --n-bases 8
       --train-history-len 1 --test-history-len 1
       --n-epochs 2 --entity-prediction --seed 42)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)
NOREV=(--no-rel-evolution)

wait_for_main() {
  local waited=0
  while ! grep -aq "^\[all\] finished=" "$MAIN_OUT"; do
    sleep 30; waited=$((waited + 30))
    if [ "$waited" -ge 5400 ]; then echo "[wait] timeout after ${waited}s" >> "$RUN_OUT"; return 1; fi
  done
  echo "[wait] main queue finished, waited=${waited}s ($(date -u +%H:%M:%S))" >> "$RUN_OUT"
}

run() {
  local name="$1"; shift
  echo "[$name] start=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
  cd "$SRC_DIR" || exit 1
  "$VENV" main.py -d xcad_v2 "$@" > "$OUT_DIR/$name.log" 2>&1
  echo "[$name] exit=$? done =$(date -u +%H:%M:%S)" >> "$RUN_OUT"
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

wait_for_main || exit 1

# 回归: 不加 flag, 复用已有微型 checkpoint, 期望 段内 H@1 = 0.854343
run inert_fixed "${MICRO[@]}" "${COMPAT[@]}" --test
grep -a "段内(filtered) H@1" "$OUT_DIR/inert_fixed.log" >> "$RUN_OUT" 2>&1

# 冒烟: 加 flag 的微型臂从头训 2 epoch(带 --evaluate-every 1 才会保存 checkpoint)
run micro_norev_fixed "${MICRO[@]}" "${COMPAT[@]}" "${NOREV[@]}" --evaluate-every 1
grep -a "段内(filtered) H@1" "$OUT_DIR/micro_norev_fixed.log" >> "$RUN_OUT" 2>&1

echo "[all] finished=$(date -u +%H:%M:%S)" >> "$RUN_OUT"
