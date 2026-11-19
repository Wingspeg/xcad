#!/usr/bin/env python3
"""gate_cw_2026_11_18.py — P100/V100M32 类别加权臂的 0-GPU 门禁 + 预注册数字。

开 GPU 之前必须全 PASS。本脚本同时把"跑出来该等于什么"写死：

G-a 尺子同源 + 问题前提：三份旗舰（λ=5, β=0.5, class-weight=none）日志 final-test 块的
    macro5 必须 = 0.5085 / 0.5069 / 0.5069（主表 §3），且逐类读数里
    **P100 与 V100M32 三种子全为 0.0000** —— 这正是本臂要修的对象；
    若控制组本身不是恒 0，"类加权能不能救稀有类"这个问题的前提就没了。
G-b 类别计数：从 data/xcad_v2 的 train/valid/test.txt 独立重算 r1 正向边的 GPU 类分布，
    test 必须 = 5032/5246/552/811/195（n=11836，与主表逐类一致）；
    CPU 段（索引 0）在三份 split 里都必须为 0 条 ⇒ 加权只作用于 5 个实际存在的类。
G-c 权重可预知：用 train-only 计数按 main.py 口径（present 类逆频率 → 可选开方 →
    cap=5 → 计数归一化使有效 loss 尺度不变）算出 sqrt/inv 两档的期望权重与逐类 loss 占比，
    打印并存成 expected_weights.json。臂日志里 [class-weight] 那几行必须与此逐位相同。

只读最后一次 "Load Model name" 之后的 final-test 块（口径同 §31/§37）。
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw_log_parse import parse  # noqa: E402

ROOT = os.path.expanduser("~/xcad/outputs/reports/")
DATA = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/xcad_v2")
OUT = os.path.join(ROOT, "class_weight_2026_11_18")
GPU_BASE = 124885
CLS = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
CAP = 5.0

CONTROL = [  # (相对日志路径, 期望 macro5, seed)
    ("night_2026_11_18/p0_regression.log", 0.5085, 42),
    ("baselines_idspace/v2_joint_s123.log", 0.5069, 123),
    ("baselines_idspace/v2_joint_s2024.log", 0.5069, 2024),
]
EXPECT_TEST_N = {1: 5032, 2: 5246, 3: 552, 4: 811, 5: 195}

rc = 0
print("==== gate_cw_2026_11_18 (0 GPU) ====")

print("\n── G-a 控制组（旗舰 = class-weight none）──────────────────────────")
ctrl = {}
for rel, want, seed in CONTROL:
    d = parse(os.path.join(ROOT, rel))
    if not d["ok"]:
        print(f"FAIL  {rel}: {d['why']}")
        rc = 1
        continue
    ok_macro = d["macro5"] == want
    rare = (d["per"]["P100"], d["per"]["V100M32"])
    ok_rare = rare == (0.0, 0.0)
    ctrl[seed] = d
    print(f"{'PASS' if (ok_macro and ok_rare) else 'FAIL'}  seed{seed:<5} macro5={d['macro5']} "
          f"(期望 {want})  P100={rare[0]}  V100M32={rare[1]}  "
          f"{'← 稀有类恒 0，本臂要修的对象' if ok_rare else 'FAIL: 控制组稀有类非 0，前提不成立'}")
    print(f"        r1 H@1={d['r1_h1']}  段内 macro5={d['inseg_macro5']}  compat macro5={d['compat_macro5']}  "
          f"agree={d['agree']}  best_epoch={d['best_epoch']}")
    if not (ok_macro and ok_rare):
        rc = 1

print("\n── G-b 类别计数（独立从 txt 重算）─────────────────────────────────")
cnt = {}
for split in ("train", "valid", "test"):
    c = np.zeros(6, dtype=np.int64)
    with open(os.path.join(DATA, split + ".txt")) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 3 or p[1] != "1":
                continue
            t = int(p[2])
            if t >= GPU_BASE:
                i = t - GPU_BASE
                assert 0 <= i <= 5, f"{split}: tail {t} 越出 GPU 段"
                c[i] += 1
    cnt[split] = c
    print(f"  {split:5s} total={int(c.sum()):7d}  " +
          "  ".join(f"{CLS[i]}={int(c[i])}" for i in range(6)))

ok = all(cnt["test"][i] == n for i, n in EXPECT_TEST_N.items()) and int(cnt["test"].sum()) == 11836
print(f"{'PASS' if ok else 'FAIL'}  test 逐类 = 5032/5246/552/811/195 且 total=11836")
rc = rc or (0 if ok else 1)
ok = all(int(cnt[s][0]) == 0 for s in cnt)
print(f"{'PASS' if ok else 'FAIL'}  CPU 段(class 0) 在 train/valid/test 均为 0 条")
rc = rc or (0 if ok else 1)


def weights(mode, counts, cap=CAP):
    """与 main.py 同口径：present 类逆频率 → 可选开方 → cap → 计数归一化使均值=1。"""
    n = counts.sum()
    nz = counts > 0
    k = int(nz.sum())
    idx = np.nonzero(nz)[0]
    raw = n / (k * counts[nz])
    if mode == "sqrt":
        raw = np.sqrt(raw)
    w = np.zeros(6)
    w[idx] = np.minimum(raw, cap)
    w = w / float((counts * w).sum() / n)
    share = counts * w
    share = share / share.sum()
    return raw, w, share, idx, k


print("\n── G-c 期望权重向量（预注册；臂日志的 [class-weight] 必须逐位复现）────")
expected = {}
for mode in ("sqrt", "inv"):
    raw, w, share, idx, k = weights(mode, cnt["train"])
    expected[mode] = dict(w=w.tolist(), share=share.tolist(), raw=raw.tolist(),
                          present=idx.tolist(), k=k, counts=cnt["train"].tolist())
    print(f"  [{mode}] cap={CAP}  K_present={k}  n_train_r1={int(cnt['train'].sum())}")
    print(f"         raw(present) = {np.array2string(raw, precision=4, floatmode='fixed')}")
    print(f"         w(all 6)     = {np.array2string(w, precision=4, floatmode='fixed')}")
    print(f"         loss share   = {np.array2string(share, precision=4, floatmode='fixed')}"
          f"   (均匀档 = {1.0/k:.4f}/类)")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "expected_weights.json"), "w") as f:
    json.dump({"cap": CAP, "gpu_base": GPU_BASE, "classes": CLS,
               "counts": {s: cnt[s].tolist() for s in cnt},
               "expected": expected,
               "control": {str(s): {"macro5": ctrl[s]["macro5"], "per": ctrl[s]["per"],
                                    "r1_h1": ctrl[s]["r1_h1"]} for s in ctrl}},
              f, indent=2, ensure_ascii=False)
print(f"\n  已存 -> {OUT}/expected_weights.json")

print("\nGATE " + ("ALL PASS -> 可以开 GPU" if rc == 0 else "FAILED -> 不开 GPU"))
sys.exit(rc)
