#!/bin/bash
# night_2026_11_18.sh — 无人值守串行 GPU 队列(RTX 3090, 一次只跑一臂)
#
# 为什么是这些臂: docs/reviews/2026-11-17_idspace_main_table.md §7 留了四件未决的事,
# 其中三件必须靠 GPU 才能收口, 另加两处"表要变成真三种子"的补种。全部只在
# 修好编号的 data/xcad_v2(以及 maas)上跑, 口径与 11-17 主表逐字一致
# (论文配置 h=6/dp0.4/seed 由臂指定, 只取 final test 块)。
#
#   P0 门禁   : GPU 空闲 + 旗舰 checkpoint 备份 + --stat-suffix/--train-history-len
#              覆盖生效 + 【回归臂必须逐位复现 0.065962 / 0.884843 / 0.5085】
#              —— 回归不过, 后面一律不跑(否则整晚数字作废)
#   P1 骨架行 : 仅骨架(base) 的 train-history-len ∈ {3,1,9,12} —— §7 第 2 项,
#              判"答案跑出候选段"是不是 h=6 特有的毛病
#   P2 完整模型: joint 的 h ∈ {3,1,9,12} —— 把 §5.6 图 5 换到修正编号上重画
#   P3 λ/β    : joint 的 --compat-lambda ∈ {1,10}/可学习 与 --compat-aux-weight 0
#              —— 表 5 的超参敏感性, 顺便试稀有类(P100/V100M32=0)能不能救
#              注意 stat 名不编码 λ/β, 全部带 --stat-suffix 防止覆盖旗舰
#   P4 消融补种: base/compat/het @ h=6 × seed {123,2024} —— 表 4 变真三种子
#   P5 maas   : 旗舰 --base-prior --head-prior --prior-ts × seed {44,45} —— 0.7364 补第三种子
#   P6 自动加种: 任一臂 macro5 > 0.5285(= 旗舰 0.5085 + v2020 噪声 0.02)则用 seed 123 复跑
#   P7 汇总   : summarize_night_logs.py 出 tsv/md
#
# 硬线: 23:10 UTC(= 07:10 北京时间)之后不再开新臂; 每臂 timeout 1800s;
#       每臂跑完立刻检查日志里有没有 final test 的指标行, 没有就标 NO_METRICS(不猜)。
set -u

cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/outputs/reports/night_2026_11_18
MK=~/xcad/xcad_code/scripts
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM_TIMEOUT=${ARM_TIMEOUT:-1800}
DEADLINE=$(date -d '2026-11-17 23:10 UTC' +%s)
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

ts()  { date -u '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$QUEUE"; }

# 每次启动都把上一轮的清单清掉, 否则 P6 会读到旧 manifest 重跑老臂
: > "$OUT/manifest.txt"; : > "$OUT/problems.txt"; : > "$OUT/skipped.txt"; : > "$OUT/promote.txt"

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

# train <tag> <seed> <dataset 参数...> -- <extra flags>
# manifest 里存的就是这套参数, 便于 P6 换个 seed 复跑
run_arm() {
  local tag="$1" seed="$2"; shift 2
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "$tag SKIPPED(过 23:10 UTC 硬线, 不开新臂)"; echo "SKIPPED $tag" >> "$OUT/skipped.txt"; return 99
  fi
  gpu_idle || { log "$tag ABORTED(GPU 一直被占)"; return 98; }
  # timeout 随历史长度放大: h=6 实测 13–15 分钟, h 越长每步的快照越多(近似线性)
  local h=6 a prev=""
  for a in "$@"; do
    if [ "$prev" = "--train-history-len" ]; then h="$a"; fi
    prev="$a"
  done
  local tmo=$(( 900 + 200 * h ))
  log "$tag start  seed=$seed h=$h timeout=${tmo}s args=[$*]"
  local t0=$(date +%s)
  # shellcheck disable=SC2048
  timeout "$tmo" "$PY" main.py "${BASE[@]}" --seed "$seed" $* > "$OUT/$tag.log" 2>&1
  local rc=$?
  # 训练臂末尾一定会走 test(model, ...) 并打印 "MRR (filter_ent)"; 基线的教训是
  # exit=0 也可能是"空成功"(checkpoint 没对上/根本没打印), 所以看指标而不是看 rc。
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

# macro_of <log> —— 只从 final test 块读 Per-GPU macro; 读不到返回空串(不猜)
macro_of() {
  tr '\r' '\n' < "$1" | awk '/Load Model name/{buf=""} {buf=buf"\n"$0} END{print buf}' \
    | grep -a "Per-GPU Hits@1 Macro (unweighted)" | tail -1 | grep -oE '[0-9.]+$' || true
}

PROMO="$OUT/promote.txt"      # P6 待复跑的种子 123 臂

log "==== night_2026_11_18 队列启动 | ARM_TIMEOUT=$ARM_TIMEOUT 硬线=$(date -u -d @$DEADLINE '+%H:%M') UTC ===="

# ── P0 门禁 ────────────────────────────────────────────────────────────────
gpu_idle || { log "FATAL: GPU 不空闲, 整晚作废"; exit 9; }

if [ ! -f "$OUT/backup_models.tar" ]; then
  tar -C ~/xcad/xcad_code/xcad_model -cf "$OUT/backup_models.tar" \
      --ignore-failed-read \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed42" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed123" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed2024" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-nonf-nodf-compat-seed42" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-seed42" \
      "models/xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-nonf-nodf-seed42" \
      > /dev/null 2>&1
  log "P0 旗舰 checkpoint 已备份 -> $OUT/backup_models.tar ($(du -h "$OUT/backup_models.tar" | cut -f1))"
fi

# (a) 新旗标生效性: --train-history-len 3 覆盖 + --stat-suffix 进名(不训练, ~20 秒)
timeout 300 "$PY" main.py "${BASE[@]}" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
    --train-history-len 3 --stat-suffix=-gatecheck --test --seed 42 \
    > "$OUT/p0_flagcheck.log" 2>&1
fc_args=$(tr '\r' '\n' < "$OUT/p0_flagcheck.log" | grep -aoE "'train_history_len': [0-9]+" | head -1)
fc_name=$(tr '\r' '\n' < "$OUT/p0_flagcheck.log" | grep -a "Sanity Check: stat name" | head -1)
log "P0 旗标检查: $fc_args | $fc_name"
case "$fc_name" in
  *his3*gatecheck*) : ;;
  *) log "FATAL: --train-history-len/--stat-suffix 没进 stat 名, 扫描臂会撞名, 整晚作废"; exit 8 ;;
esac

# (b) 回归门禁: 旗舰 joint seed42 test-only, 必须逐位复现 11-17 主表
timeout "$ARM_TIMEOUT" "$PY" main.py --test "${BASE[@]}" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" --seed 42 \
    > "$OUT/p0_regression.log" 2>&1
reg_mrr=$(tr '\r' '\n' < "$OUT/p0_regression.log" | grep -a "MRR (filter_ent)" | tail -1 | grep -oE '[0-9.]+$')
reg_r1=$(tr '\r' '\n' < "$OUT/p0_regression.log" | grep -a "r1_suits" | tail -1 | grep -oE 'H@1=[0-9.]+' | head -1 | cut -d= -f2)
reg_macro=$(macro_of "$OUT/p0_regression.log")
log "P0 回归: MRR=$reg_mrr r1_H@1=$reg_r1 macro=$reg_macro (期望 0.065962 / 0.884843 / 0.5085)"
if [ "$reg_mrr" != "0.065962" ] || [ "$reg_r1" != "0.884843" ]; then
  log "FATAL: 回归没逐位复现 —— --stat-suffix 动了数值或 checkpoint 被换, 整晚作废"; exit 7
fi
log "P0 通过, 开跑 P1–P5"

# ── P1 仅骨架(不加任何机制)的历史长度扫描: §7 第 2 项, h=3 最先跑 ──────────
log "---- P1 骨架 h 扫描(候选段逃逸是否 h 依赖) ----"
for h in 3 1 9 12; do
  run_arm "s1_base_h$h" 42 "${V2[@]}" --train-history-len "$h"
done

# ── P2 完整模型的历史长度扫描(图 5 换到修正编号) ──────────────────────────
log "---- P2 joint h 扫描 ----"
for h in 3 1 9 12; do
  run_arm "s2_joint_h$h" 42 "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" --train-history-len "$h"
done

# ── P3 λ/β 敏感性(表 5)—— stat 名不编码 λ/β, 必须带 suffix ────────────────
log "---- P3 compat λ/β 扫描 ----"
run_arm "s3_joint_lam1"  42 "${V2[@]}" "${HET[@]}" --compat --compat-lambda 1.0  --compat-aux-weight 0.5 --stat-suffix=-lam1
run_arm "s3_joint_lam10" 42 "${V2[@]}" "${HET[@]}" --compat --compat-lambda 10.0 --compat-aux-weight 0.5 --stat-suffix=-lam10
run_arm "s3_joint_beta0" 42 "${V2[@]}" "${HET[@]}" --compat --compat-lambda 5.0  --compat-aux-weight 0.0 --stat-suffix=-beta0
run_arm "s3_joint_lamL"  42 "${V2[@]}" "${HET[@]}" --compat                    --compat-aux-weight 0.5 --stat-suffix=-lamL

# ── P4 消融三行的补种(表 4 变真三种子) ────────────────────────────────────
log "---- P4 消融补种 ----"
for s in 123 2024; do
  run_arm "s4_base_s$s"   "$s" "${V2[@]}"
  run_arm "s4_compat_s$s" "$s" "${V2[@]}" "${COMPAT[@]}"
  run_arm "s4_het_s$s"    "$s" "${V2[@]}" "${HET[@]}"
done

# ── P5 maas 旗舰补第三种子(0.7364 那条头条) ───────────────────────────────
log "---- P5 maas prior-ts 补种 ----"
for s in 44 45; do
  run_arm "s5_maas_priorts_s$s" "$s" -d maas --probe-rel 0 --base-prior --head-prior --prior-ts
done

# ── P6 自动加种: macro5 超过 旗舰+噪声(0.5285)的臂用 seed 123 复跑 ─────────
log "---- P6 晋级检查(macro5 > 0.5285 才补 seed 123) ----"
: > "$PROMO"
while IFS='|' read -r tag seed args; do
  [ -f "$OUT/$tag.log" ] || continue
  m=$(macro_of "$OUT/$tag.log")
  [ -z "${m:-}" ] && continue
  if awk -v a="$m" 'BEGIN{exit !(a>0.5285)}'; then
    echo "${tag}_s123|123|$args" >> "$PROMO"
    log "P6 晋级 $tag macro=$m -> 用 seed123 复跑"
  fi
done < "$OUT/manifest.txt"
while IFS='|' read -r tag seed args; do
  # shellcheck disable=SC2086
  run_arm "$tag" "$seed" $args
done < "$PROMO"
[ -s "$PROMO" ] || log "P6 无臂晋级"

# ── P7 汇总 ────────────────────────────────────────────────────────────────
log "---- P7 汇总 ----"
"$PY" "$MK/summarize_night_logs.py" "$OUT" 2>&1 | tee -a "$QUEUE"
log "==== 队列结束 $(ts) | 读数在 $OUT/night_summary.md ===="
log "问题臂: $(wc -l < "$OUT/problems.txt" 2>/dev/null || echo 0) 个 | 跳过臂: $(wc -l < "$OUT/skipped.txt" 2>/dev/null || echo 0) 个"
