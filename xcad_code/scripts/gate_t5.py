#!/usr/bin/env python3
"""gate_t5.py — 表 5 补种前的 0-GPU 门禁（跑 GPU 之前必须全 PASS）。

要钉死的两件事：
1. **尺子没变**：三份旗舰（λ=5, β=0.5）日志的 final test macro5 必须逐位复现
   主表文档 `2026-11-17_idspace_main_table.md` §3 的 0.5085 / 0.5069 / 0.5069。
   复现不了 ⇒ 我读的口径不对，补出来的六个臂和旗舰不同源，整轮作废，所以不开 GPU。
2. **旗舰格已经是三种子**：表 5 的 λ=5 与 β=0.5 两格本来就是同一个配置
   （主配置 λ=5/β=0.5），seed {42,123,2024} 都跑过 ⇒ 只需补 {λ1, λ10, β0} × {123,2024} = 6 臂。

只读最后一次 "Load Model name … Using best epoch : N" 之后的打印（口径同 §1 第 4 条），
训练循环里的 eval 块一概不读（那份类频是 12,600 那一套，不是 11,836）。
"""
import os
import re
import sys

# (日志, 期望 macro5, 期望 compat_lambda, 期望 compat_aux_weight)
EXPECT = [
    ("night_2026_11_18/p0_regression.log",     0.5085, "5.0",  "0.5"),   # seed 42
    ("baselines_idspace/v2_joint_s123.log",    0.5069, "5.0",  "0.5"),   # seed 123
    ("baselines_idspace/v2_joint_s2024.log",   0.5069, "5.0",  "0.5"),   # seed 2024
]
ROOT = os.path.expanduser("~/xcad/outputs/reports/")
LAST_EVAL = re.compile(r"Load Model name[^\n]*\n.*?Using best epoch\s*:\s*\d+", re.S)
MACRO = re.compile(r"Per-GPU Hits@1 Macro \(unweighted\)\s*=\s*([0-9.]+)")

rc = 0
for rel, want_macro, want_lam, want_beta in EXPECT:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print("FAIL  %s  日志不存在" % rel)
        rc = 1
        continue
    txt = open(path, errors="ignore").read()
    hits = list(re.finditer(r"Load Model name", txt))
    if not hits:
        print("FAIL  %s  没有 final test 块" % rel)
        rc = 1
        continue
    tail = txt[hits[-1].start():]
    m = MACRO.search(tail)
    if not m:
        print("FAIL  %s  final test 块里没有 macro5 行" % rel)
        rc = 1
        continue
    got = round(float(m.group(1)), 4)
    ns = re.search(r"Namespace\(.*", txt)
    ns = ns.group(0) if ns else ""
    lam = re.search(r"compat_lambda=(\S+?)[,)]", ns)
    beta = re.search(r"compat_aux_weight=(\S+?)[,)]", ns)
    lam = lam.group(1) if lam else "?"
    beta = beta.group(1) if beta else "?"
    ok = (got == want_macro) and (lam == want_lam) and (beta == want_beta)
    print("%s  %s  macro5=%s (期望 %s)  lambda=%s (期望 %s)  beta=%s (期望 %s)"
          % ("PASS" if ok else "FAIL", rel, got, want_macro, lam, want_lam, beta, want_beta))
    if not ok:
        rc = 1

print("GATE " + ("ALL PASS -> 可以开 GPU" if rc == 0 else "FAILED -> 不开 GPU"))
sys.exit(rc)
