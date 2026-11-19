#!/bin/bash
# cw_2026_11_18.sh — P100/V100M32 类别加权臂（2026-11-18 用户放行："跑 P100 类加权臂"）
#
# 要回答的问题（唯一）：§38 实测随机森林只吃算法 13 维静态画像，在同一测试边集上
#   P100 拿到 0.1214、V100M32 拿到 0.2308，而完整模型三种子这两类**全为 0.0000**，
#   两类贡献了剩余 macro 差距的 94%。⇒ "信号在画像里、模型没用上"。
#   本臂只动一件东西：compat 辅助 CE 的类别权重（sqrt / inv 逆频率，cap=5，按计数归一化
#   保持 β=0.5 的有效 loss 尺度不变）。其余配置与旗舰逐字相同（-d xcad_v2、h=6、dp0.4、
#   n-epochs 30/eval 5、λ=5、β=0.5、het2）。
#   为什么加在辅助 CE 而不是主 CE：§31/§37 实测完整模型的 r1 读数 = compat 头直读
#   （无约束 H@1 = 段内 = compat 0.884843，agree=0.999747）⇒ 编码器在 r1 无可报独立贡献，
#   能改变稀有类排名的训练信号只有这条辅助损失。
#
# 6 臂 = {sqrt, inv} × {seed 42, 123, 2024}；单价实测 856–863 s ⇒ 串行约 87 分钟。
# 控制组不用重跑：class-weight=none 就是旗舰本身（0.5085/0.5069/0.5069）。
# checkpoint/stat 名不编码 --compat-class-weight ⇒ 每臂必须带 --stat-suffix，否则覆盖旗舰。
#
# 门禁（不过就不开卡 / 不开正式臂）：
#   G0 0-GPU 门禁 gate_cw：控制组 macro5 逐位复现 + 稀有类三种子恒 0 + 逐类 n 对上主表 +
#      从 train.txt 独立重算期望权重向量（存 expected_weights.json，臂日志必须复现）。
#   G1 旗舰 checkpoint 先打包备份（撞名会覆盖，见 §30 新坑）。
#   G2 test-only 复现：用旗舰 seed42 checkpoint + 显式 --compat-class-weight none 跑 --test，
#      macro5 必须仍 = 0.5085 ⇒ 新 flag 默认路径不改评测、不改 model_name。
#   G3 冒烟：sqrt、--n-epochs 2、--stat-suffix=-cwsmoke，必须 exit 0 且有 final-test 块
#      （已知坑：n-epochs 1 必 FileNotFoundError，所以取 2）。
#   G4 每臂开跑前确认 GPU 空闲（3090 串行，单卡）。
#
# 判线（**开卡前写死，不看完数字再改**）：
#   C    控制组 macro5 三种子均值 = (0.5085+0.5069+0.5069)/3 = 0.507433。
#   L1   不回退线：某档三种子 macro5 均值 >= C - 0.02 = 0.487433。
#        （v2020 run-to-run 噪声 = 0.0172，判线阈值取 0.02，见 §30；低于 ⇒ 加权把整体排序做坏）
#   L2   稀有类起效线：P100 或 V100M32 的 Hits@1 在该档 >=2 个种子上 >= 0.05。
#        （控制组这两类严格为 0.0000 ⇒ 任何 >0 都是新行为；0.05 = V100M32 的 10 条边 /
#         P100 的 28 条边，排除"偶然命中一两条"的抖动）
#   L3   有效线（能写进论文的改进）：某档三种子 macro5 均值 >= C + 0.02 = 0.527433。
#   L4   天花板参照（不是判线，只标定成色）：RF 同测试边集 macro 0.582、P100 0.1214、
#        V100M32 0.2308；某档 >= 0.552 ⇒ 已逼近画像可分度。
#   STOP 停止线：两档 L2 全不过 ⇒ 结论钉死为"类别加权救不了 P100/V100M32"，
#        本轴不再开臂；要再动损失函数形式（focal/重标标签）需重新放行。
#   必须一起报的代价，不许只报有利面：
#     (i) T4/MISC 逐类是否掉（稀有类加权必然挤占高频类）；
#     (ii) agree（模型段内 top-1 与 compat argmax 同 GPU 的边占比）是否从 0.9997 移动 ——
#          移动即"编码器第一次偏离 compat 头"，无论方向都是有信息量的机制读数；
#     (iii) 稀有类起来但 macro 均值落在 L1~L3 之间 ⇒ 只写"trade"，不写"改进"。
set -u

cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
MK=~/xcad/xcad_code/scripts
OUT=~/xcad/outputs/reports/class_weight_2026_11_18
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
FLAGSHIP='xcad_v2-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed'

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
  local cw=$(grep -ac '^\[class-weight\]' "$OUT/$tag.log")
  log "$tag done   rc=$rc 用时=$(( $(date +%s) - t0 ))s final-test块=$has macro5=${mac:-无} class-weight行=$cw"
  if [ "$has" -lt 1 ] || [ "$cw" -lt 2 ]; then
    log "$tag NO_METRICS !! $(tr '\r' '\n' < "$OUT/$tag.log" | grep -aE 'Traceback|Error|not exist' | head -2)"
    echo "NO_METRICS $tag rc=$rc" >> "$OUT/problems.txt"
  else
    echo "$tag|$seed|$mac" >> "$OUT/readings.txt"
  fi
  echo "$tag|$seed|$*" >> "$OUT/manifest.txt"
  return $rc
}

: > "$OUT/problems.txt"; : > "$OUT/readings.txt"; : > "$OUT/manifest.txt"
log "==== 类别加权队列启动（{sqrt,inv} × 三种子，6 臂串行，预计 87 分钟）===="

# ── G0 0-GPU 门禁 ──────────────────────────────────────────────────────────
"$PY" "$MK/gate_cw_2026_11_18.py" 2>&1 | tee -a "$OUT/queue.out"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log "FATAL: G0 没过 —— 控制组没复现 / 稀有类非恒 0 / 权重口径对不上，不开 GPU"
  exit 7
fi

# ── G1 旗舰 checkpoint 备份 ────────────────────────────────────────────────
if [ ! -f "$OUT/backup_models.tar" ]; then
  gpu_idle || { log "FATAL: GPU 不空闲，本轮作废"; exit 9; }
  (cd ~/xcad/xcad_code/xcad_model && tar -cf "$OUT/backup_models.tar" \
      --ignore-failed-read \
      "models/${FLAGSHIP}42" "models/${FLAGSHIP}123" "models/${FLAGSHIP}2024") \
      > /dev/null 2>&1
  log "G1 旗舰 checkpoint+stat 已备份 -> $OUT/backup_models.tar ($(du -h "$OUT/backup_models.tar" | cut -f1))"
fi

# ── G2 test-only：新 flag 的默认路径必须逐位不动评测 ───────────────────────
gpu_idle || { log "FATAL: GPU 不空闲，本轮作废"; exit 9; }
log "G2 start  test-only 旗舰 seed42 + --compat-class-weight none"
timeout 2400 "$PY" main.py "${BASE[@]}" --seed 42 "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
    --compat-class-weight none --test > "$OUT/g2_testonly_none.log" 2>&1
rc=$?
mac=$(macro_of "$OUT/g2_testonly_none.log")
log "G2 done    rc=$rc macro5=${mac:-无}（期望 0.5085 逐位）"
if [ "$rc" -ne 0 ] || [ "${mac:-x}" != "0.5085" ]; then
  log "FATAL: G2 没过 —— 默认路径改动了评测或 model_name，不开正式臂"
  echo "G2_FAIL rc=$rc macro=$mac" >> "$OUT/problems.txt"
  exit 8
fi

# ── G3 冒烟：新代码路径（独立 suffix，绝不碰正式臂/旗舰 checkpoint）────────
log "G3 start  sqrt 冒烟 n-epochs=2 suffix=-cwsmoke"
timeout 1200 "$PY" main.py "${BASE[@]}" --seed 42 "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
    --n-epochs 2 --evaluate-every 1 --compat-class-weight sqrt \
    --stat-suffix=-cwsmoke > "$OUT/g3_smoke_sqrt.log" 2>&1
rc=$?
has=$(tr '\r' '\n' < "$OUT/g3_smoke_sqrt.log" | grep -ac "Hits (filter_ent) @ 1:")
mac=$(macro_of "$OUT/g3_smoke_sqrt.log")
log "G3 done    rc=$rc final-test块=$has macro5=${mac:-无}"
[ "$rc" -eq 0 ] && [ "$has" -ge 1 ] || { log "FATAL: G3 冒烟没过，不开正式臂"; exit 9; }
rm -rf ~/xcad/xcad_code/xcad_model/models/*-cwsmoke ~/xcad/xcad_code/xcad_model/stats/*-cwsmoke* \
       ~/xcad/xcad_code/xcad_model/models/xcad_v2*"-cwsmoke"* 2>/dev/null
log "G3 冒烟 checkpoint 已清理"

# ── 正式 6 臂 ──────────────────────────────────────────────────────────────
for MODE in sqrt inv; do
  for S in 42 123 2024; do
    run_arm "${MODE}_s${S}" "$S" "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
        --compat-class-weight "$MODE" --stat-suffix=-cw-"$MODE"
  done
done

log "==== 6 臂跑完，出判线汇总 ===="
"$PY" "$MK/summarize_cw_2026_11_18.py" 2>&1 | tee -a "$OUT/queue.out"
log "==== 类别加权队列结束 ===="
