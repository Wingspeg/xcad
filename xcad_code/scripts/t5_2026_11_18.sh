#!/bin/bash
# t5_2026_11_18.sh — 论文表 5（兼容性头 λ/β 超参敏感性）补到真三种子
#
# 为什么只有这 6 臂：
#   表 5 是 xcad_v2 上**唯一还没重建的论文表**（手稿五个数 51.5/51.5/51.4/51.9±1.6/
#   51.6±0.8 全是撞号编号下的 legacy 读数；主表/表4 已在 11-17~11-18 换到 v2）。
#   不补 ⇒ 同一篇新稿里表 5 与其余表格不同源，这是硬伤（id_space 缺陷记录 §7）。
#   λ=5 与 β=0.5 两格本来就是同一个配置（主配置），seed {42,123,2024} 都跑过，
#   读数在主表里；所以只缺 {λ1, λ10, β0} × {seed 123, 2024} = **6 臂**。
#   单价实测 856–863 s/臂（11-18 夜 s3 那四臂），串行约 1.5 小时。
#
# 口径与主表逐字一致：-d xcad_v2、h=6、dp0.4、n-epochs 30、只取 final test 块。
# stat/checkpoint 名不编码 λ/β ⇒ 每臂必须带 --stat-suffix，否则覆盖旗舰 checkpoint。
#
# 门禁（0 GPU，不过就不开卡）：
#   G0 三份旗舰日志（λ5β0.5 @ seed 42/123/2024）的 final-test macro5 必须
#      = 0.5085 / 0.5069 / 0.5069，且日志里的 Namespace 确实是 lambda=5.0/beta=0.5
#      —— 复现不了说明我读的尺子不对，补出来的六个臂不能和 λ=5 那格并排。
#   G1 GPU 空闲 + 旗舰 checkpoint 先打包备份。
#
# 判线（跑之前就写死，不看完数字再改）：
#   L1 不回退线：任一臂 macro5 不得比旗舰 0.5085 低超过 0.02（v2020 实测 run-to-run
#      噪声 = 0.0172，见项目记忆 §30）；越过 ⇒ 该超参取值不是"不敏感"，正文那句要改。
#   L2 结论线：三种子均值跨度 ≤ 0.02 ⇒ "λ/β 不敏感"这句话在 v2 上活着，可以照写；
#      跨度 > 0.02 ⇒ 表 5 必须报 sd，且"取自 λ=5/β=0.5 是因为不敏感"要重述。
#   L3 β 方向：手稿说"β=0 方差略大故取 0.5"。若 v2 三种子下 β=0 的均值反而 ≥ β=0.5
#      ⇒ 那句**没有 v2 证据**，必须删掉或换理由（11-18 单种子读到 β=0 更低 49.60<50.85）。
set -u

cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
MK=~/xcad/xcad_code/scripts
OUT=~/xcad/outputs/reports/t5_2026_11_18
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=(--gpu 0 --encoder uvrgcn --decoder convtranse
      --n-hidden 200 --n-layers 2 --n-bases 100
      --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
      --train-history-len 6 --test-history-len 20
      --n-epochs 30 --evaluate-every 5 --lr 0.001
      --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10)
V2=(-d xcad_v2)
HET=(--hetero --n-msg-basis 2)

ts()  { date -u '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$OUT/queue.out"; }

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

macro_of() {
  tr '\r' '\n' < "$1" | awk '/Load Model name/{buf=""} {buf=buf"\n"$0} END{print buf}' \
    | grep -a "Per-GPU Hits@1 Macro (unweighted)" | tail -1 | grep -oE '[0-9.]+$' || true
}

run_arm() {
  local tag="$1" seed="$2"; shift 2
  gpu_idle || { log "$tag ABORTED(GPU 一直被占)"; return 98; }
  log "$tag start  seed=$seed args=[$*]"
  local t0=$(date +%s)
  # shellcheck disable=SC2048
  timeout 1800 "$PY" main.py "${BASE[@]}" --seed "$seed" $* > "$OUT/$tag.log" 2>&1
  local rc=$?
  local has=$(tr '\r' '\n' < "$OUT/$tag.log" | grep -ac "Hits (filter_ent) @ 1:")
  local mac=$(macro_of "$OUT/$tag.log")
  log "$tag done   rc=$rc 用时=$(( $(date +%s) - t0 ))s final-test块=$has macro5=${mac:-无}"
  if [ "$has" -lt 1 ]; then
    log "$tag NO_METRICS !! $(tr '\r' '\n' < "$OUT/$tag.log" | grep -aE 'Traceback|Error|not exist|覆盖不全' | head -2)"
    echo "NO_METRICS $tag rc=$rc" >> "$OUT/problems.txt"
  else
    echo "$tag|$seed|$mac" >> "$OUT/readings.txt"
  fi
  echo "$tag|$seed|$*" >> "$OUT/manifest.txt"
  return $rc
}

: > "$OUT/manifest.txt"; : > "$OUT/problems.txt"; : > "$OUT/readings.txt"

log "==== t5 队列启动（表 5 补三种子，6 臂串行）===="

# ── G0 门禁：尺子必须和主表同源 ────────────────────────────────────────────
"$PY" "$MK/gate_t5.py" 2>&1 | tee -a "$OUT/queue.out"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log "FATAL: G0 没过 —— 旗舰 macro5 没复现或 λ/β 不是 5.0/0.5，不开 GPU"
  exit 7
fi

# ── G1 GPU 空闲 + 旗舰 checkpoint 备份 ─────────────────────────────────────
gpu_idle || { log "FATAL: GPU 不空闲，本轮作废"; exit 9; }
if [ ! -f "$OUT/backup_models.tar" ]; then
  tar -C ~/xcad/xcad_code/xcad_model -cf "$OUT/backup_models.tar" --ignore-failed-read \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed42" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed123" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed2024" \
      > /dev/null 2>&1
  log "G1 旗舰 checkpoint 已备份 -> $OUT/backup_models.tar ($(du -h "$OUT/backup_models.tar" | cut -f1))"
fi

for S in 123 2024; do
  run_arm "t5_lam1_s$S"  "$S" "${V2[@]}" "${HET[@]}" \
      --compat --compat-lambda 1.0  --compat-aux-weight 0.5 --stat-suffix=-lam1
  run_arm "t5_lam10_s$S" "$S" "${V2[@]}" "${HET[@]}" \
      --compat --compat-lambda 10.0 --compat-aux-weight 0.5 --stat-suffix=-lam10
  run_arm "t5_beta0_s$S" "$S" "${V2[@]}" "${HET[@]}" \
      --compat --compat-lambda 5.0  --compat-aux-weight 0.0 --stat-suffix=-beta0
done

log "==== 6 臂跑完，出汇总 ===="
"$PY" "$MK/summarize_night_logs.py" "$OUT" 2>&1 | tail -3 | tee -a "$OUT/queue.out"
log "seed42 的三格在昨晚目录: night_2026_11_18/{s3_joint_lam1,s3_joint_lam10,s3_joint_beta0}.log"
log "λ=5/β=0.5（同一个主配置）三种子读数见 baselines_idspace/v2_joint_s{123,2024}.log 与昨晚 p0_regression"
log "==== t5 队列结束 ===="
