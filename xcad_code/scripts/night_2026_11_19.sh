#!/bin/bash
# night_2026_11_19.sh — 无人值守串行 GPU 队列(RTX 3090, 一次一次只跑一臂)
#
# 队列来源: docs/reviews/2026-11-18_overnight_plan.md §1(用户裁定"默认六块")
#   P0 门禁   : GPU 空闲 + 旗舰 checkpoint 备份 + 回归臂必须逐位复现 0.065962 / 0.884843 / 0.5085
#              —— 回归不过, 后面一律不跑(否则整晚数字作废)
#   B1 标签支持度分层 : 1 次 test-only --dump-preds, CPU 侧分层分析另脚本
#   B2 归一化口径     : 3 臂(种子 42/123/2024)用 --feat-dir 变体目录
#   B3 外部基线补种   : RE-GCN ×2 / CEN ×2 / RETIA ×2(每臂跑完 tar checkpoint 防覆盖)
#   B4 代价采样       : 随 B2/B3 挂显存轮询, 无新臂
#   B7 静态分类器     : CPU 侧, 另脚本并行
#
# 硬线: 06:30 UTC(= 14:30 北京时间)之后不再开新臂; 每臂 timeout 按类型给;
#       每臂跑完立刻检查日志里有没有 final test 的指标行, 没有就标 NO_METRICS(不猜)。
set -u

cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
BL=~/xcad/baselines
OUT=~/xcad/outputs/reports/night_2026_11_19
MK=~/xcad/xcad_code/scripts
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM_TIMEOUT=${ARM_TIMEOUT:-1800}
DEADLINE=$(date -d '2026-11-19 06:30 UTC' +%s)
QUEUE="$OUT/queue.out"

# 论文配置(h=6 是默认; 臂可以用 --train-history-len 覆盖, argparse 后者赢)
BASE=(--gpu 0 --encoder uvrgcn --decoder convtranse
      --n-hidden 200 --n-layers 2 --n-bases 100
      --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
      --train-history-len 6 --test-history-len 20
      --n-epochs 30 --evaluate-every 5 --lr 0.001
      --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10)
V2=(-d xcad_v2)
HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)

# 三个基线的超参(逐字抄自 rerun_baselines_idspace.sh, 不含 -d / --test / --n-epochs)
REGCN=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
       --train-history-len 3 --test-history-len 3
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --weight 0.5 --task-weight 0.7 --angle 10 --discount 1.0 --self-loop --layer-norm
       --weight-decay 1e-4 --lr 0.001 --entity-prediction)
CEN=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
     --start-history-len 2 --train-history-len 3 --test-history-len 3 --dilate-len 1
     --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
     --self-loop --layer-norm --entity-prediction --lr 0.001 --ft_lr 0.001 --norm_weight 0.1)
RETIA=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
       --train-history-len 2 --test-history-len 2
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --self-loop --layer-norm --task-weight 0.7 --norm_weight 1.0
       --weight 1 --discount 1 --angle 10 --lr 0.001)

ts()  { date -u '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$QUEUE"; }

# 每次启动都把上一轮的清单清掉
: > "$OUT/manifest.txt"; : > "$OUT/problems.txt"; : > "$OUT/skipped.txt"

# gpu_idle —— 有别人的进程占卡就等, 最多 3 次 5 分钟; 仍然占着就放弃整晚
gpu_idle() {
  local used i
  for i in 1 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "${used:-99999}" -lt 2000 ]; then return 0; fi
    log "GPU 被占用 ${used} MiB, 等 5 分钟 (${i}/3)"
    sleep 300
  done
  return 1
}

# run_ours <tag> <seed> <dataset 参数...> -- <extra flags>
run_ours() {
  local tag="$1" seed="$2"; shift 2
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "$tag SKIPPED(过 06:30 UTC 硬线, 不开新臂)"; echo "SKIPPED $tag" >> "$OUT/skipped.txt"; return 99
  fi
  gpu_idle || { log "$tag ABORTED(GPU 一直被占)"; return 98; }
  local tmo="$ARM_TIMEOUT"
  log "$tag start  seed=$seed timeout=${tmo}s args=[$*]"
  local t0=$(date +%s)
  # shellcheck disable=SC2048
  timeout "$tmo" "$PY" main.py "${BASE[@]}" --seed "$seed" $* > "$OUT/$tag.log" 2>&1
  local rc=$?
  local has=$(tr '\r' '\n' < "$OUT/$tag.log" | grep -ac "Hits (filter_ent) @ 1:")
  log "$tag done   rc=$rc 用时=$(( $(date +%s) - t0 ))s  final-test块=$has"
  if [ "$has" -lt 1 ]; then
    log "$tag NO_METRICS !! $(tr '\r' '\n' < "$OUT/$tag.log" | grep -aE 'Traceback|Error|not exist|覆盖不全' | head -2)"
    echo "NO_METRICS $tag rc=$rc" >> "$OUT/problems.txt"
  fi
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Sanity Check: stat name" | head -1 | tee -a "$QUEUE"
  echo "$tag|$seed|$*" >> "$OUT/manifest.txt"
  return $rc
}

# run_base <tag> <RE-GCN|CEN|RETIA> <args...>
run_base() {
  local tag="$1"; shift
  local proj="$1"; shift
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "$tag SKIPPED(过 06:30 UTC 硬线)"; echo "SKIPPED $tag" >> "$OUT/skipped.txt"; return 99
  fi
  gpu_idle || { log "$tag ABORTED(GPU 一直被占)"; return 98; }
  log "$tag start  proj=$proj args=[$*]"
  local t0=$(date +%s)
  ( cd "$BL/$proj/src" || exit 1
    export LD_LIBRARY_PATH="$(ls -d "$BL"/.venv/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
    "$BL/.venv/bin/python3" main.py "$@" > "$OUT/$tag.log" 2>&1 )
  local rc=$?
  log "$tag done   rc=$rc 用时=$(( $(date +%s) - t0 ))s"
  # 归档 checkpoint(基线没有 --stat-suffix, 第二次会覆盖第一次)
  if [ -d "$BL/$proj/models/xcad_v2" ]; then
    tar -C "$BL/$proj/models" -czf "$OUT/${tag}_checkpoint.tar.gz" "xcad_v2" 2>/dev/null
    log "$tag checkpoint archived -> $OUT/${tag}_checkpoint.tar.gz"
  fi
  echo "$tag|baseline|$proj|$*" >> "$OUT/manifest.txt"
  return $rc
}

# macro_of <log> —— 只从 final test 块读 Per-GPU macro; 读不到返回空串(不猜)
macro_of() {
  tr '\r' '\n' < "$1" | awk '/Load Model name/{buf=""} {buf=buf"\n"$0} END{print buf}' \
    | grep -a "Per-GPU Hits@1 Macro (unweighted)" | tail -1 | grep -oE '[0-9.]+$' || true
}

log "==== night_2026_11_19 队列启动 | ARM_TIMEOUT=$ARM_TIMEOUT 硬线=$(date -u -d @$DEADLINE '+%H:%M') UTC ===="

# ── P0 门禁 ────────────────────────────────────────────────────────────────
gpu_idle || { log "FATAL: GPU 不空闲, 整晚作废"; exit 9; }

if [ ! -f "$OUT/backup_models.tar" ]; then
  tar -C ~/xcad/xcad_code/xcad_model -cf "$OUT/backup_models.tar" \
      --ignore-failed-read \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed42" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed123" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed2024" \
      > /dev/null 2>&1
  log "P0 旗舰 checkpoint 已备份 -> $OUT/backup_models.tar ($(du -h "$OUT/backup_models.tar" | cut -f1))"
fi

# (a) 回归门禁: 旗舰 joint seed42 test-only, 必须逐位复现 11-17 主表
timeout "$ARM_TIMEOUT" "$PY" main.py --test "${BASE[@]}" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" --seed 42 \
    > "$OUT/p0_regression.log" 2>&1
reg_mrr=$(tr '\r' '\n' < "$OUT/p0_regression.log" | grep -a "MRR (filter_ent)" | tail -1 | grep -oE '[0-9.]+$')
reg_r1=$(tr '\r' '\n' < "$OUT/p0_regression.log" | grep -a "r1_suits" | tail -1 | grep -oE 'H@1=[0-9.]+' | head -1 | cut -d= -f2)
reg_macro=$(macro_of "$OUT/p0_regression.log")
log "P0 回归: MRR=$reg_mrr r1_H@1=$reg_r1 macro=$reg_macro (期望 0.065962 / 0.884843 / 0.5085)"
if [ "$reg_mrr" != "0.065962" ] || [ "$reg_r1" != "0.884843" ]; then
  log "FATAL: 回归没逐位复现 —— checkpoint 被换或数据口径变了, 整晚作废"; exit 7
fi
log "P0 通过, 开跑 B1–B4"

# ── B1 标签支持度分层(1 次 test-only --dump-preds) ──────────────────────────
log "---- B1 dump-preds for stratification ----"
timeout 120 "$PY" main.py --test "${BASE[@]}" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" --seed 42 \
    --dump-preds "$OUT/b1_preds.pkl" \
    > "$OUT/b1_dump.log" 2>&1
log "B1 done, preds -> $OUT/b1_preds.pkl"

# ── B2 归一化口径(3 臂, --feat-dir 变体目录) ────────────────────────────────
log "---- B2 train-caliber variant (3 seeds) ----"
VARIANT_FEAT_DIR="../data/xcad_v2_tc"
for s in 42 123 2024; do
  run_ours "b2_tc_s$s" "$s" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
      --feat-dir "$VARIANT_FEAT_DIR" --stat-suffix=-tc
done

# ── B3 外部基线补到三次运行(6 臂) ───────────────────────────────────────────
log "---- B3 external baselines ×2 more runs ----"
# RE-GCN ×2(第 1 次已在 11-17 跑过)
run_base "b3_regcn_r2" RE-GCN "${REGCN[@]}" -d xcad_v2 --n-epochs 150 --evaluate-every 5
run_base "b3_regcn_r3" RE-GCN "${REGCN[@]}" -d xcad_v2 --n-epochs 150 --evaluate-every 5
# CEN ×2(pretrain + curriculum + test = 3 步, 但只算 1 次运行)
run_base "b3_cen_r2_pretrain"   CEN "${CEN[@]}" -d xcad_v2 --test -1 --n-epochs 30 --evaluate-every 1
run_base "b3_cen_r2_curriculum" CEN "${CEN[@]}" -d xcad_v2 --test  0 --n-epochs 30 --evaluate-every 1
run_base "b3_cen_r2_test"       CEN "${CEN[@]}" -d xcad_v2 --test  2
run_base "b3_cen_r3_pretrain"   CEN "${CEN[@]}" -d xcad_v2 --test -1 --n-epochs 30 --evaluate-every 1
run_base "b3_cen_r3_curriculum" CEN "${CEN[@]}" -d xcad_v2 --test  0 --n-epochs 30 --evaluate-every 1
run_base "b3_cen_r3_test"       CEN "${CEN[@]}" -d xcad_v2 --test  2
# RETIA ×2(cur-train 30 epoch)
run_base "b3_retia_r2" RETIA "${RETIA[@]}" -d xcad_v2 --cur-train --n-epochs 30 --evaluate-every 5
run_base "b3_retia_r3" RETIA "${RETIA[@]}" -d xcad_v2 --cur-train --n-epochs 30 --evaluate-every 5

# ── B4 代价采样(无新臂, 从 B2/B3 日志提取) ──────────────────────────────────
log "---- B4 cost sampling (piggybacks on B2/B3) ----"
# 显存峰值与训练时长已在每臂的 log 里, 后续用 CPU 脚本提取

# ── 汇总 ────────────────────────────────────────────────────────────────────
log "==== GPU 队列结束 $(ts) ===="
log "问题臂: $(wc -l < "$OUT/problems.txt" 2>/dev/null || echo 0) 个 | 跳过臂: $(wc -l < "$OUT/skipped.txt" 2>/dev/null || echo 0) 个"
log "读数在 $OUT/manifest.txt, 问题在 $OUT/problems.txt"
