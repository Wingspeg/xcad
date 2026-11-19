#!/usr/bin/env python3
# @File    : analyze_mem_probe_2026_11_19.py
# #85 训练侧显存探针的独立分析件（0 GPU）：只读 samples.csv / arm.log / arm.err / summary.txt。
# 存在的原因：臂启动时服务器上的 mem_probe_train_2026_11_19.sh 里 L2 回归门禁的正则写错了
#   （拿总 H@1 的锚点去要 r1 H@1、把 "Per-GPU Hits@1 Macro (unweighted) =" 写成 "Per-GPU Hits@1 Macro:"）。
#   正在运行的脚本不能就地覆盖（bash 按字节偏移续读，改了会执行到脏代码），故原始件保留、分析重做。
# 判线（与脚本头部同一套，未看数前写死）：
#   L1 峰值须来自 >=1000 个采样且覆盖 >=800s；L2 final test 块逐位 = 0.065962 / 0.884843 / 0.5085；
#   L3 峰值 >=24576 MiB 则必须报余量；L4 口径边界见脚本头部注释。
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/xcad/outputs/reports/mem_probe_train_2026_11_19')
CAP = 24576
FLAG = [('总 MRR (filter_ent)', r'MRR \(filter_ent\): ([\d.]+)', '0.065962'),
        ('r1_suits H@1', r'r1_suits\s+MRR=[\d.]+\s+H@1=([\d.]+)', '0.884843'),
        ('Per-GPU macro5', r'Per-GPU Hits@1 Macro \(unweighted\) = ([\d.]+)', '0.5085')]

rows = []
for line in open(os.path.join(OUT, 'samples.csv'), errors='replace'):
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
if not rows:
    sys.exit('samples.csv 无有效样本')

dev = [r[1] for r in rows]
util = [r[2] for r in rows]
proc = [r[3] for r in rows]
busy = [r[0] for r in rows if r[2] > 0]
span = rows[-1][0] - rows[0][0]
peak_dev, peak_proc = max(dev), max(proc)
i_dev, i_proc = dev.index(peak_dev), proc.index(peak_proc)
q = sorted(dev)
p95 = q[min(len(q) - 1, int(0.95 * len(q)))]

print('=== L1 样本线 ===')
print(f'样本数={len(rows)} (需>=1000)  覆盖={span:.1f}s (需>=800)  实际间隔={span/max(1,len(rows)-1):.3f}s  '
      f'-> {"PASS" if len(rows) >= 1000 and span >= 800 else "FAIL"}')
print('=== 训练侧显存（两个口径，单位 MiB）===')
print(f'device memory.used   : 峰值={peak_dev} @ +{rows[i_dev][0]-rows[0][0]:.0f}s  '
      f'P95={p95}  均值={sum(dev)/len(dev):.0f}  最小={min(dev)}')
print(f'compute-app 本进程占位: 峰值={peak_proc} @ +{rows[i_proc][0]-rows[0][0]:.0f}s  '
      f'均值={sum(proc)/len(proc):.0f}  与 device 口径差={peak_dev-peak_proc}')
if busy:
    print(f'GPU 利用率: 峰值={max(util)}%  非零窗口={max(busy)-min(busy):.1f}s / {span:.1f}s '
          f'= {100*(max(busy)-min(busy))/span:.1f}%')
print('=== L3 上限线 ===')
print(f'卡容量={CAP} MiB  峰值占 {100*peak_dev/CAP:.1f}%  余量={CAP-peak_dev} MiB  '
      f'-> {"逼近 OOM" if peak_dev >= CAP else "有余量"}')

txt = open(os.path.join(OUT, 'arm.log'), errors='replace').read().replace('\r', '\n')
epochs = re.findall(r'^(\d+) Epoch (\d+) \| Ave Loss: ([\d.]+)', txt, re.M)
print('=== 时间线（epoch 起于第几秒 / 当时 device 显存 / 利用率）===')
for ts, ep, loss in epochs:
    k = min(range(len(rows)), key=lambda j: abs(rows[j][0] - float(ts)))
    print(f'  epoch {int(ep):2d}  +{float(ts)-rows[0][0]:7.1f}s  loss={loss:>7s}  '
          f'dev={rows[k][1]:6d}  util={rows[k][2]:3d}%')

print('=== L2 回归线（final test 块，应逐位复现旗舰读数）===')
tail = txt.split('Load Model name')[-1]
ok = True
for name, pat, want in FLAG:
    m = re.findall(pat, tail)
    got = m[-1] if m else 'MISSING'
    good = got == want
    ok &= good
    print(f'  {name:22s} got={got:12s} want={want}  {"OK" if good else "MISMATCH"}')
print(f'  -> L2 {"PASS" if ok else "FAIL（显存峰值仍可引，但本臂不得用于任何指标陈述）"}')

err = open(os.path.join(OUT, 'arm.err'), errors='replace').read()
for kw in ('Traceback', 'CUDA out of memory', 'RuntimeError'):
    if kw in err:
        print(f'  !! stderr 含 {kw}，末 12 行：')
        print('\n'.join(err.splitlines()[-12:]))
        break
else:
    print('  stderr 无 Traceback/OOM')

ckpt = os.path.join(os.path.expanduser('~'), 'xcad/xcad_code/xcad_model/models')
hits = [f for f in os.listdir(ckpt) if f.endswith('-memprobe')] if os.path.isdir(ckpt) else []
print('=== 附带（防撞名与产物体积）===')
for f in hits:
    print(f'  checkpoint {os.path.getsize(os.path.join(ckpt, f))/1e6:.1f} MB  <- {f[-60:]}')
print(f'  samples.csv={os.path.getsize(os.path.join(OUT,"samples.csv"))/1e3:.0f} KB  '
      f'arm.log={os.path.getsize(os.path.join(OUT,"arm.log"))/1e3:.0f} KB')
print(open(os.path.join(OUT, 'summary.txt'), errors='replace').read())
