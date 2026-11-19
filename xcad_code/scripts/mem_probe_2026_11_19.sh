#!/bin/bash
# @File    : mem_probe_2026_11_19.sh
# B4 缺口补测（0 新臂）：ours 的【评测/推理侧】显存峰值 + 非零利用率窗口 + 逐边吞吐。
# 用 test-only 复用旗舰 checkpoint（30 秒级），不训练 ⇒ 不产生新臂、不覆盖任何已有读数。
# 口径边界（必须一起报）：这条只回答"评测一遍 90,785 条 test 边要多少显存/多久"，
#   不含梯度与优化器状态 ⇒ 不能当"训练显存峰值"引用；训练侧峰值需要 1 臂 ×14.4 分钟，未批。
# 配置数组逐字抄自 night_2026_11_19.sh（P0 回归用的就是这一条命令，可交叉核对）。
set -uo pipefail

PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/outputs/reports/night_2026_11_19
CSV=$OUT/mem_probe_samples.csv
LOG=$OUT/mem_probe_testonly.log

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

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used:-99999}" -ge 2000 ]; then echo "GPU 被占 ${used} MiB, 不插队"; exit 8; fi

cd ~/xcad/xcad_code/xcad_model/src || exit 1
rm -f "$CSV" "$LOG"

# 轮询 200ms：memory.used / utilization.gpu（expandable_segments 会压低"已分配"读数，
# 所以同时记 query-gpu 与进程占位两个口径不现实，取 nvidia-smi 的 device memory.used，
# 这是对外可引用的那张卡的占用值）
(
  while :; do
    printf '%s,%s\n' "$(date +%s.%N)" \
      "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ';')" \
      >> "$CSV"
    sleep 0.2
  done
) &
POLL=$!
trap 'kill $POLL 2>/dev/null' EXIT

T0=$(date +%s)
timeout 600 "$PY" main.py --test "${BASE[@]}" --seed 42 "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" \
    > "$LOG" 2>&1
RC=$?
T1=$(date +%s)
kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
trap - EXIT

echo "rc=$RC  墙钟=$((T1-T0))s"
python3 - "$CSV" <<'PY'
import sys
rows = []
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if "," not in line:
        continue
    ts, payload = line.split(",", 1)          # 注意：payload 内部本来就是逗号分隔
    if not payload:
        continue
    for seg in payload.rstrip(";").split(";"):
        if not seg or "," not in seg:
            continue
        m, u = seg.split(",")[:2]
        try:
            rows.append((float(ts), int(m), int(u)))
        except ValueError:
            print(f"跳过异常样本: {seg!r}")
if not rows:
    print("无样本"); sys.exit(1)
t0 = rows[0][0]
mem = [r[1] for r in rows]; util = [r[2] for r in rows]
busy = [r[0] for r in rows if r[2] > 0]
print(f"样本数={len(rows)} 覆盖={rows[-1][0]-t0:.1f}s "
      f"device 显存峰值={max(mem)} MiB 均值={sum(mem)/len(mem):.0f} MiB "
      f"利用率峰值={max(util)}%")
if busy:
    print(f"利用率>0 采样数={len(busy)}  非零窗口={max(busy)-min(busy):.1f}s")
else:
    print("利用率全程 0（采样没盖住计算段 ⇒ 吞吐不可推）")
print("显存时间线（每 10 个样本一点）：")
for ts, m, u in rows[::10][:45]:
    print(f"  {ts-t0:7.1f}s  mem={m:6d} MiB  util={u:3d}%")
PY
echo "--- 读数（应与 P0 逐位相同：0.065962 / 0.884843 / 0.5085）---"
tr '\r' '\n' < "$LOG" | grep -a "MRR (filter_ent)\|r1_suits  \|Per-GPU Hits@1 Macro" | tail -3
echo "--- 阶段标记（判加载 vs 计算的分界）---"
tr '\r' '\n' < "$LOG" | grep -an "id-space\|Sanity Check\|Load Model name\|Evaluat\|MRR (filter_ent)" | head -10
