#!/bin/bash
# cw5_2026_11_18.sh — 类别加权轴的补种子队列（判定"主表 ours 行是否换成 inv 档"）
#
# 要回答的问题（唯一）：三种子的 inv 档 macro5 读数是 0.6023/0.5156/0.5495，
#   样本标准差 0.0437 = v2020 实测 run-to-run 噪声 0.0172 的 5 倍 ⇒ "加权把稀有类救活了"
#   方向可信，但量级尚不可判。补两个种子（连同 none 档同种子）把它变成可判的数：
#   到底是 5 种子均值稳、还是双态本身是配置的属性。
#
# 4 臂 = {none, inv} × {seed 7, seed 2025}，串行，单价实测 856–863 s ⇒ 约 57 分钟。
# 为什么也跑 none：换表声明是【同种子配对】改进（Δ5），只有 inv 的新种子撑不起；
#   头图要报稳定性，两边的标准差必须来自同一个 n。
# 为什么是 7 和 2025：在看到任何读数之前写死（预注册），事后不改。
#   v2020 此前只用过 42/123/2024，maas 用过 42–45；开跑前实测 models/ 里没有 seed7/seed2025。
#
# 门禁：
#   G-CAL（已过，0 GPU）：cw_log_parse.py 新增 filter_ent 两字段后，逐 seed 复现主表 §3 三行原始值。
#   G0    0-GPU：expected_weights.json 存在且 inv 权重向量可读（臂日志必须逐位复现，由 summarizer 校）。
#   G-COLLIDE：seed 7/2025 的 model/stat 目录必须不存在（checkpoint 名不编码 --compat-class-weight，
#         见 §30 撞名坑；none 臂因此必须带 --stat-suffix=-cwctrl，inv 臂带 -cw-inv）。
#   G4    每臂开跑前确认 GPU 空闲（3090 串行，单卡）。
#   快速失败：任一臂 rc!=0 或没有 final-test 块 ⇒ 停队列，不烧后面的卡。
#
# 判线（W1a/W1b/W1c、W2、W3）写在 §13.3，早于任何新读数；本脚本只跑臂，判定在
#   summarize_cw5_2026_11_18.py 里机械套用。
set -u

cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
MK=~/xcad/xcad_code/scripts
OUT=~/xcad/outputs/reports/class_weight_ext_2026_11_18
OLD=~/xcad/outputs/reports/class_weight_2026_11_18
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
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)

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
  timeout 2400 "$PY" main.py "${BASE[@]}" --seed "$seed" $* > "$OUT/$tag.log" 2>&1
  local rc=$?
  local has=$(tr '\r' '\n' < "$OUT/$tag.log" | grep -ac "Hits (filter_ent) @ 1:")
  local mac=$(macro_of "$OUT/$tag.log")
  log "$tag done   rc=$rc 用时=$(( $(date +%s) - t0 ))s final-test块=$has macro5=${mac:-无}"
  echo "$tag|$seed|$mac" >> "$OUT/readings.txt"
  echo "$tag|$seed|$*" >> "$OUT/manifest.txt"
  if [ "$rc" -ne 0 ] || [ "$has" -lt 1 ]; then
    log "$tag FAILED —— 停队列（不烧后面的卡）: $(tr '\r' '\n' < "$OUT/$tag.log" | grep -aE 'Traceback|Error|not exist' | head -2)"
    echo "FAILED $tag rc=$rc" >> "$OUT/problems.txt"
    exit 6
  fi
  return 0
}

: > "$OUT/problems.txt"; : > "$OUT/readings.txt"; : > "$OUT/manifest.txt"
log "==== 补种子队列启动（{none,inv} × {7,2025}，4 臂串行，预计 57 分钟）===="

# ── G0 期望权重文件在场 ────────────────────────────────────────────────────
if [ ! -f "$OLD/expected_weights.json" ]; then
  log "FATAL: $OLD/expected_weights.json 缺失，inv 权重向量无法校验，不开 GPU"
  exit 7
fi
log "G0 PASS  expected_weights.json 在场"

# ── G-COLLIDE 新种子不得与既有 checkpoint/stat 撞名 ────────────────────────
for S in 7 2025; do
  for SUF in "" "-cw-inv" "-cwctrl"; do
    hit=$(find ~/xcad/xcad_code/xcad_model/models -maxdepth 1 -name "*seed${S}${SUF}" 2>/dev/null | head -1)
    if [ -n "$hit" ]; then
      log "FATAL: 撞名 —— 已存在 $hit（seed=$S suffix=${SUF:-<旗舰>}），不开 GPU"
      exit 8
    fi
  done
done
log "G-COLLIDE PASS  seed 7/2025 的旗舰与两种后缀目录都不存在"

# ── 4 臂：配对相邻，中途死掉也至少留下完整一对 ─────────────────────────────
for S in 7 2025; do
  run_arm "none_s${S}" "$S" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
      --compat-class-weight none --stat-suffix=-cwctrl
  run_arm "inv_s${S}"  "$S" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
      --compat-class-weight inv  --stat-suffix=-cw-inv
done

log "==== 4 臂跑完，出 5 种子判线汇总 ===="
"$PY" "$MK/summarize_cw5_2026_11_18.py" 2>&1 | tee -a "$OUT/queue.out"
log "==== 补种子队列结束 ===="
