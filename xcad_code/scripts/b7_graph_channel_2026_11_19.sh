#!/bin/bash
# @File    : b7_graph_channel_2026_11_19.sh
# B7 / R2-4 的 hybrid 需要"图通道"的逐边分数 ⇒ 实装了 --dump-seg-scores(过滤前 GPU 段 6 列)。
# 本脚本做两件事:
#   A0  回归门: 不带任何 dump 旗标的 test-only ⇒ 三个已发表读数必须逐位不动
#       (证明新探针在旗标关闭时对数值路径是零影响, 不靠"应该没影响")
#   A1..A3  三个种子的 test-only + 分数导出(同进程再打一次 valid, 供混合器拟合)
# 全部走 --test 只读 checkpoint, 不训练、不新增臂、不覆盖任何 model 文件。
# 配置数组逐字抄自 mem_probe_2026_11_19.sh / night_2026_11_19.sh(P0 门禁用的同一条命令)。
# 判线口径见 docs/reviews/2026-11-19_b7_mlp_hybrid.md §4(G-SCORE/G-ALIGN)。
set -uo pipefail

PY=~/xcad/xcad_code/.venv/bin/python
SRC=~/xcad/xcad_code/xcad_model/src
OUT=~/xcad/outputs/reports/b7_mlp_hybrid_2026_11_19
LOG=$OUT/logs
KEEP=$OUT/scores

BASE=(--gpu 0 --encoder uvrgcn --decoder convtranse
      --n-hidden 200 --n-layers 2 --n-bases 100
      --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
      --train-history-len 6 --test-history-len 20
      --n-epochs 30 --evaluate-every 5 --lr 0.001
      --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10)
V2=(-d xcad_v2)
HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG" "$KEEP"
cd "$SRC" || exit 1

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used:-99999}" -ge 2000 ]; then echo "GPU 被占 ${used} MiB, 不插队"; exit 8; fi

rc=0
run() {   # run <tag> <args...>
  local tag=$1; shift
  local t0=$(date +%s)
  echo "=== [$tag] start $(date -u '+%F %T UTC') ==="
  timeout 900 "$PY" main.py --test "$@" > "$LOG/$tag.log" 2>&1
  local r=$?
  local t1=$(date +%s)
  echo "    [$tag] rc=$r  墙钟=$((t1-t0))s"
  if [ $r -ne 0 ]; then
    echo "    ---- tail ----"; tail -15 "$LOG/$tag.log"
    rc=1
  fi
  tr '\r' '\n' < "$LOG/$tag.log" \
    | grep -a "MRR (filter_ent)\|r1_suits  \|Per-GPU Hits@1 Macro\|段内(filtered)\|段内 per-GPU macro5\|G-SCORE\|dump-seg-scores\|落在 GPU 段的比例\|Traceback" \
    | tail -14
  return $r
}

# ── A0 回归门(旗标关闭) ──
run A0_regress_s42 "${BASE[@]}" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" --seed 42

# ── A1..A3 分数导出 ──
for s in 42 123 2024; do
  run "A_dump_s${s}" "${BASE[@]}" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" --seed "$s" \
      --dump-seg-scores "$KEEP/seg_s${s}" --dump-preds "$KEEP/preds_s${s}.pkl" || rc=1
done

echo "=== 产物 ==="
ls -la "$KEEP" 2>&1 | tail -12
echo "RUN rc=$rc"
