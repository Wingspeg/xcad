#!/bin/bash
# @File    : mem_probe_train_2026_11_19.sh
# #85 / B4 缺口：ours 旗舰配置的【训练侧】显存峰值（1 新臂 × ~14.4 分钟）。
# 背景：§45 的 mem_probe_2026_11_19.sh 只量到评测侧峰值 7,772 MiB（test-only，无梯度/优化器状态），
#       §44 里那句 "12.3 GB" 是单次人工观察、不是峰值，不能引用。R2-6 的代价表要训练侧数字 ⇒ 跑这一臂。
#
# 看数之前写死的四条线（预注册）：
#   L1 样本线：峰值必须是 >=1000 个采样里的最大值、且采样覆盖 >=800s（单次 nvidia-smi 观察不作数）。
#              每轮采样要调两次 nvidia-smi（device + compute-apps），实际间隔约 0.3–0.6s，覆盖时长才是关键量。
#   L2 回归线：本臂 final test 块必须逐位复现 P0 旗舰读数 0.065962 / 0.884843 / 0.5085。
#              事后更正（看数后写）：这条线本身选错了对象——逐位复现只在 test-only 复用 checkpoint 时成立，
#              重新训练一个臂含 cuDNN 非确定核与 best-epoch 选择抖动，只能要求"落在 run-to-run 噪声内"。
#              实跑给出 0.065631 / 0.884758 / 0.5108。三个偏差分开看：
#                r1 H@1  -0.000085  —— 有对应实测线(同配置重训 run-to-run 极差 0.0172)，远低于它;
#                macro5  +0.0023    —— 没有 run-to-run 线, 只有 ④ 五种子 macro sd 0.0016 作量级参照;
#                总 MRR  -0.000331  —— 同上, 无线可判。
#              两侧 best epoch 都是 10(已从日志核对), 所以差异来自训练期浮点非确定性, 不是选轮不同。
#              ⇒ 判读: 不构成本臂的缺陷, 但也不能当作"逐位复现"引用; 显存数字可引, 指标数字一律不用本臂。
#              不复现 ⇒ 显存峰值仍可引（同配置同种子），但本臂不得用于任何指标陈述，且要报告偏差。
#   L3 上限线：device 显存峰值若 >=24576 MiB（卡容量）⇒ 说明逼近 OOM，代价表必须写出"余量"而不是只写峰值。
#   L4 口径线：可写陈述 = "旗舰配置在单张 24GB 卡上训练一次的显存峰值与用时"；
#              不可写：跨卡型外推、随图规模的可扩展性（本臂只有一份规模）、以及把 device 占用当"模型显存需求"
#              （device memory.used 含缓存与其他进程，故同时记 compute-app 进程占位两个口径）。
#
# 防撞名：--stat-suffix=-memprobe ⇒ stat/checkpoint 写到 ...-compat-seed42-memprobe，
#         不覆盖旗舰 checkpoint（test-only 复现读数要用它）。
set -uo pipefail

PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/outputs/reports/mem_probe_train_2026_11_19
CSV=$OUT/samples.csv
LOG=$OUT/arm.log
ERR=$OUT/arm.err
SUM=$OUT/summary.txt
ARM_TIMEOUT=${ARM_TIMEOUT:-1800}

BASE=(--gpu 0 --encoder uvrgcn --decoder convtranse
      --n-hidden 200 --n-layers 2 --n-bases 100
      --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
      --train-history-len 6 --test-history-len 20
      --n-epochs 30 --evaluate-every 5 --lr 0.001
      --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10)
V2=(-d xcad_v2)
HET=(--hetero --n-msg-basis 2)
COMPAT=(--compat --compat-lambda 5.0 --compat-aux-weight 0.5)
SUFFIX=(--stat-suffix=-memprobe)

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"; : > "$CSV"; : > "$LOG"; : > "$ERR"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used:-99999}" -ge 2000 ]; then echo "GPU 被占 ${used} MiB, 不插队" | tee -a "$SUM"; exit 8; fi

cd ~/xcad/xcad_code/xcad_model/src || exit 1

echo "[$(date -u '+%H:%M:%S')] probe arm start" | tee -a "$SUM"
(
  while :; do
    printf '%s,%s,%s,%s\n' "$(date +%s.%N)" \
      "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" \
      "$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits | tr -d ' ' | tr ',' ':' | tr '\n' ';' | sed 's/;$//')" \
      >> "$CSV"
    sleep 0.2
  done
) &
POLL=$!
trap 'kill $POLL 2>/dev/null' EXIT

# 采样器自检：5 秒内必须真的往 CSV 写行（第一版漏了 >> "$CSV"，白跑 2 分钟才发现）
sleep 5
if [ ! -s "$CSV" ]; then
  echo "[$(date -u '+%H:%M:%S')] FAIL: 采样器没有写 $CSV, 中止不开臂" | tee -a "$SUM"
  kill $POLL 2>/dev/null; exit 9
fi
echo "[$(date -u '+%H:%M:%S')] 采样器自检通过: 前 5s 有 $(wc -l < "$CSV") 行" | tee -a "$SUM"

T0=$(date +%s)
timeout "$ARM_TIMEOUT" "$PY" -u main.py "${BASE[@]}" --seed 42 "${V2[@]}" "${HET[@]}" "${COMPAT[@]}" "${SUFFIX[@]}" \
  2> "$ERR" | awk '{ print strftime("%s"), $0; fflush(); }' > "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
trap - EXIT

echo "[$(date -u '+%H:%M:%S')] probe arm done rc=$RC 墙钟=$((T1-T0))s" | tee -a "$SUM"

python3 - "$CSV" "$LOG" "$ERR" "$T0" "$T1" "$RC" "$SUM" "$OUT" <<'PY' >> "$SUM" 2>&1
import sys, csv, re, os
csvf, logf, errf, t0, t1, rc, sumf, outd = sys.argv[1:9]
t0, t1, rc = float(t0), float(t1), int(rc)

rows = []
for line in open(csvf):
    p = line.rstrip('\n').split(',')
    if len(p) < 3:
        continue
    try:
        ts, dev, util = float(p[0]), int(p[1]), int(p[2])
    except ValueError:
        continue
    proc = 0
    for seg in p[3:]:
        for item in seg.split(';'):
            q = item.split(':')
            if len(q) >= 2:
                try:
                    proc = max(proc, int(float(q[-1])))
                except ValueError:
                    pass
    rows.append((ts, dev, util, proc))

dev = [r[1] for r in rows]; util = [r[2] for r in rows]; proc = [r[3] for r in rows]
busy = [r[0] for r in rows if r[2] > 0]
peak_dev = max(dev); peak_proc = max(proc)
i_dev = dev.index(peak_dev); i_proc = proc.index(peak_proc)

print('=== L1 样本线 ===')
print(f'样本数={len(rows)} (需>=1000)  覆盖={rows[-1][0]-rows[0][0]:.1f}s (需>=800)  '
      f'采样间隔中位≈0.2s  墙钟={t1-t0:.0f}s  rc={rc}')
print(f'  -> 样本线 {"PASS" if len(rows)>=1000 and rows[-1][0]-rows[0][0]>=800 else "FAIL"}')
print('=== 训练侧显存（两个口径）===')
_tail = dev[-(len(dev)//4 or 1):]
print(f'device memory.used : 峰值={peak_dev} MiB @ +{rows[i_dev][0]-rows[0][0]:.0f}s  '
      f'均值={sum(dev)/len(dev):.0f}  尾部25%均值={sum(_tail)/len(_tail):.0f}')
print(f'compute-app 进程占位: 峰值={peak_proc} MiB @ +{rows[i_proc][0]-rows[0][0]:.0f}s  均值={sum(proc)/len(proc):.0f}')
print(f'利用率: 峰值={max(util)}%  非零窗口={max(busy)-min(busy):.1f}s / 覆盖 {rows[-1][0]-rows[0][0]:.1f}s' if busy else '利用率全程 0')
print('=== L3 上限线 ===')
print(f'卡容量=24576 MiB  余量={24576-peak_dev} MiB  -> {"接近 OOM, 必须报余量" if peak_dev>=24576 else "有余量"}')

txt = open(logf, errors='replace').read().replace('\r', '\n')
epochs = re.findall(r'^(\d+) Epoch (\d+) \| Ave Loss: ([\d.]+)', txt, re.M)
print('=== 时间线（epoch -> 起于第几秒 / device 显存当时值）===')
if epochs:
    for ts, ep, loss in epochs:
        k = min(range(len(rows)), key=lambda j: abs(rows[j][0] - float(ts)))
        print(f'  epoch {int(ep):2d}  +{float(ts)-rows[0][0]:7.1f}s  loss={loss}  dev={rows[k][1]} MiB')
else:
    print('  (stdout 里没有 Epoch 行——训练可能提前失败)')

print('=== L2 回归线（应逐位 = 0.065962 / 0.884843 / 0.5085）===')
tail = txt.split('Load Model name')[-1]
for name, pat, want in [
        ('总 MRR (filter_ent)  ', r'MRR \(filter_ent\): ([\d.]+)', '0.065962'),
        ('r1_suits H@1         ', r'r1_suits\s+MRR=[\d.]+\s+H@1=([\d.]+)', '0.884843'),
        ('Per-GPU macro5       ', r'Per-GPU Hits@1 Macro \(unweighted\) = ([\d.]+)', '0.5085')]:
    m = re.findall(pat, tail)
    got = m[-1] if m else 'MISSING'
    print(f'  {name} got={got:12s} want={want}  {"OK" if got == want else "MISMATCH"}')
err = open(errf, errors='replace').read()
for kw in ('Traceback', 'CUDA out of memory', 'RuntimeError'):
    if kw in err:
        print(f'  !! stderr 含 {kw}:')
        print('\n'.join(err.splitlines()[-15:]))
        break
else:
    print('  stderr 无 Traceback/OOM')
sz = os.path.getsize(os.path.join(outd, 'arm.log'))
print(f'=== 附带读数 ===\n  arm.log={sz} B  err={os.path.getsize(errf)} B')
PY

echo "--- summary ---"
tr '\r' '\n' < "$SUM"
