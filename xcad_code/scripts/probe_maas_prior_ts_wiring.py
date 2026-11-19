#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_maas_prior_ts_wiring.py -- 0-GPU 校验 §26 (H) 的【注入接线】。

背景：静态先验表(§23–§25)在训练期是自抄小抄(head 通道可用率 train 1.0000 vs test 0.1525，
16.4% 的训练行没有更早快照却有表，其中 79.8% 的 argmax 就是自己这行的 gold —— 见
probe_maas_prior_train_leak.log)。export_maas_base_prior_ts.py 已经把两张表换成
"第 t 层只统计 ts<t 的边"，并在 exporter 里断言过表本身的形状/零行/逐层一致性。
但 exporter 只证明【表对】，不证明【模型按快照读对了层】。本探针证明后者，五件事：

    T1 静态口径逐位还原：ts=None 时 apply_base_prior 的输出 == 手工 numpy 参考(§23/§24 公式)。
       => 重构(把表选择/alpha 选择塞进函数)没有改动老口径一个 bit。
    T2 时序口径逐位还原：ts∈{1,20,47,48,53,54,65} 时输出 == 用 base/head_prior_ts[ts] 手工算的参考。
       => 按层取表 + 按 head 取行 + 只动卡型段列，三件事都对。
    T3 关键不变量(承 §25.8)：所有 t>=首个 test 快照 的层，时序注入结果与静态注入结果【逐位相同】。
       => 这次改动只换了训练/验证期看到的先验面，test 侧条件一个字没变；
          因此臂间 test 读数差只能来自"模型学到了什么"，不能来自评测口径。
    T4 门控路径：--prior-gate 开时，(实体, ts) 现算的特征确实随 ts 变(无历史行 f1=f2=0)，
       且梯度能流到 gate_w/gate_c(否则门是装饰品)。
    T5 忘记 set_prior_ts 必须当场报错，pop 之后不能重复用同一个 ts。

用法(server)：
    cd ~/xcad/xcad_code && .venv/bin/python scripts/probe_maas_prior_ts_wiring.py
退出码 0 = 五条全过。
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.expanduser("~/xcad/xcad_code/xcad_model")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src import rrgcn  # noqa: E402

D = os.path.join(ROOT, "data", "maas")
FAIL = []


def chk(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


# ── id 空间与表(与 exporter 同一套推导, 不写死数字) ──
n_ents = int(open(os.path.join(D, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(D, "base_of.npy"))
n_bases, pad = int(base_of.max()), int(base_of.max())
tails = set()
test_ts, first_test_ts = set(), None
for ln in open(os.path.join(D, "test.txt")):
    p = ln.split()
    if len(p) >= 4 and int(p[1]) == 0:
        tails.add(int(p[2]))
        test_ts.add(int(p[3]))
prior_lo = min(tails)
prior_n = max(tails) - prior_lo + 1
print(f"[id-space] n_ents={n_ents} prior_lo={prior_lo} prior_n={prior_n} n_bases={n_bases} "
      f"pad={pad} test_ts={sorted(test_ts)[:3]}..{max(test_ts)}")

bp = torch.from_numpy(np.load(os.path.join(D, "base_prior.npy"))).float()
hp = torch.from_numpy(np.load(os.path.join(D, "head_prior.npy"))).float()
bpt = torch.from_numpy(np.load(os.path.join(D, "base_prior_ts.npy"))).float()
hpt = torch.from_numpy(np.load(os.path.join(D, "head_prior_ts.npy"))).float()
T = hpt.shape[0]
assert bp.shape == (n_bases + 1, prior_n) and hp.shape == (n_ents, prior_n)
assert bpt.shape == (T, n_bases + 1, prior_n) and hpt.shape == (T, n_ents, prior_n)

# 真实 head: 从 train/valid 里取 rel0 边的 head, 保证 base_of[head] 有意义
real_hd = []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(D, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            real_hd.append(int(p[0]))
rng = np.random.default_rng(0)
N_ROW = 512   # score 是 [rows, n_ents]: 512×80287×4B≈164MB, 主机内存友好
hd_np = torch.from_numpy(rng.choice(np.array(real_hd, dtype=np.int64), size=N_ROW, replace=False))
# 逆向行(subject = 卡型实体)也要进去: 偏置必须恒 0
hd_np = torch.cat([hd_np, torch.arange(prior_lo, prior_lo + prior_n, dtype=torch.int64)])
rel_np = torch.zeros(hd_np.numel(), dtype=torch.int64)
tk_np = torch.randint(prior_lo, prior_lo + prior_n, (hd_np.numel(),), dtype=torch.int64)
triples = torch.stack([hd_np, rel_np, tk_np], dim=1)
base_of_t = torch.from_numpy(base_of)


def make_stub(gate=False):
    """不跑 RGCN.__init__(要读一堆数据/建参数), 只把 apply_base_prior 用到的字段摆好。"""
    m = rrgcn.RecurrentRGCN.__new__(rrgcn.RecurrentRGCN)
    torch.nn.Module.__init__(m)
    m.device = "cpu"
    m.use_base_prior = True
    m.use_head_prior = True
    m.use_prior_gate = gate
    m.use_prior_ts = True
    m.prior_lo, m.prior_n = prior_lo, prior_n
    m.base_prior, m.head_prior = bp, hp
    m.base_prior_ts, m.head_prior_ts = bpt, hpt
    m.prior_ts_len = T
    m.base_of = base_of_t
    m._ts_norm_b = float(bpt.sum(2).max())
    m._ts_norm_m = float(hpt.sum(2).max())
    m._pending_ts = None
    m.prior_alpha = torch.nn.Parameter(torch.tensor(1.5557))
    m.head_alpha = torch.nn.Parameter(torch.tensor(2.1234))
    if gate:
        m.gate_w = torch.nn.Parameter(torch.tensor([[0.4, -0.3, 0.9], [0.1, 0.7, -0.2]]))
        m.gate_c = torch.nn.Parameter(torch.tensor([0.54133, 0.54133]))
    return m


def ref_static(score, stub):
    sl = slice(prior_lo, prior_lo + prior_n)
    hd = triples[:, 0]
    out = score.clone()
    out[:, sl] = out[:, sl] + stub.prior_alpha * bp[base_of_t[hd]]
    out[:, sl] = out[:, sl] + stub.head_alpha * hp[hd]
    return out


def ref_ts(score, stub, t):
    sl = slice(prior_lo, prior_lo + prior_n)
    hd = triples[:, 0]
    out = score.clone()
    out[:, sl] = out[:, sl] + stub.prior_alpha * bpt[t][base_of_t[hd]]
    out[:, sl] = out[:, sl] + stub.head_alpha * hpt[t][hd]
    return out


# ── T1 静态口径逐位还原 ──
stub = make_stub()
sc = torch.randn(triples.size(0), n_ents)
got = rrgcn.RecurrentRGCN.apply_base_prior(stub, sc.clone(), triples, None)
want = ref_static(sc, stub)
d1 = float((got - want).abs().max())
head_seg_untouched = bool(torch.equal(got[:, :prior_lo], sc[:, :prior_lo]))
tail_cols_beyond = bool(torch.equal(got[:, prior_lo + prior_n:], sc[:, prior_lo + prior_n:]))
chk("T1 静态口径(ts=None)逐位还原 §23/§24", d1 == 0.0 and head_seg_untouched,
    f"maxdiff={d1} head段不动={head_seg_untouched} 卡型段右侧不动={tail_cols_beyond}")
# 逆向行(subject=卡型)偏置恒 0
inv_rows = slice(triples.size(0) - prior_n, triples.size(0))
chk("T1b 逆向行(subject=卡型实体)偏置恒 0",
    float((got[inv_rows, prior_lo:prior_lo + prior_n] - sc[inv_rows, prior_lo:prior_lo + prior_n]).abs().max()) == 0.0)

# ── T2 时序口径逐位还原 ──
probes = [1, 20, 47, 48, 53, 54, T - 1]
worst, ok_all = 0.0, True
for t in probes:
    got_t = rrgcn.RecurrentRGCN.apply_base_prior(stub, sc.clone(), triples, t)
    want_t = ref_ts(sc, stub, t)
    worst = max(worst, float((got_t - want_t).abs().max()))
    ok_all &= bool(torch.equal(got_t, want_t))
chk("T2 时序口径按层取表逐位还原", ok_all, f"ts={probes} maxdiff={worst}")

# ── T3 test 段逐位相同 ⇒ 评测条件不变 ──
same, first_bad = True, None
for t in range(min(test_ts), T):
    a = rrgcn.RecurrentRGCN.apply_base_prior(stub, sc.clone(), triples, t)
    b = rrgcn.RecurrentRGCN.apply_base_prior(stub, sc.clone(), triples, None)
    if not torch.equal(a, b):
        same, first_bad = False, t
        break
cov1 = float((hpt[1].max(1).values > 0).float().mean())
covL = float((hpt[T - 1].max(1).values > 0).float().mean())
chk("T3 所有 test 层注入结果 == 静态注入(逐位)", same,
    f"首个 test ts={min(test_ts)} 层数={T - min(test_ts)} 首个不同层={first_bad}  "
    f"head 覆盖率 ts=1 {cov1:.4f} -> ts={T - 1} {covL:.4f}")
chk("T3b 训练层确实与静态层不同(否则这次改动没咬住)",
    not torch.equal(rrgcn.RecurrentRGCN.apply_base_prior(stub, sc.clone(), triples, 1),
                    rrgcn.RecurrentRGCN.apply_base_prior(stub, sc.clone(), triples, None)),
    "ts=1 注入结果 != 静态注入")

# ── T4 门控路径: 特征按 (实体, ts) 现算 + 梯度可达 ──
# 时序口径下不会再用按实体预存的 prior_gate_feat(stub 里故意不建它 => 一旦回落到静态分支就 AttributeError)
g = make_stub(gate=True)
f_t, a_t = [], {}
for t in (1, 20, T - 1):
    hd = triples[:200, 0]
    _v = hpt[t].sum(1)[hd] / max(g._ts_norm_m, 1e-6)
    f_t.append(float(_v.mean()))
    _feat = torch.stack([bpt[t].sum(1)[base_of_t[hd]] / max(g._ts_norm_b, 1e-6), _v,
                         (_v > 0).float()], 1)
    _al = torch.nn.functional.softplus(_feat @ g.gate_w.t() + g.gate_c)
    a_t[t] = (float(_al[:, 0].mean()), float(_al[:, 1].mean()))
al = {}
# ts=1 整张时序表全 0(没有 ts<1 的边) => 注入恒 0、门梯度也恒 0, 这是正确行为, 所以梯度只在有历史的层上验
for t in (20, T - 1):
    sc2 = sc[:200].clone()
    out2 = rrgcn.RecurrentRGCN.apply_base_prior(g, sc2, triples[:200], t)
    out2.sum().backward()
    al[t] = (float(g.gate_w.grad.abs().sum()), float(g.gate_c.grad.abs().sum()))
    g.gate_w.grad = None
    g.gate_c.grad = None
chk("T4 门控梯度可达 gate_w/gate_c(时序口径)", all(w > 0 and c > 0 for w, c in al.values()),
    "  ".join(f"ts={t}: |grad w|={w:.3f} |grad c|={c:.3f}" for t, (w, c) in al.items()))
# ts=1 表全 0 => 门输入退化; ts=T-1 才有 head 支持度 => 同一批行的 alpha 必须不同
chk("T4b 门控输入特征/输出信任度随 ts 变",
    f_t[0] == 0.0 and f_t[-1] > 0.0 and abs(a_t[1][1] - a_t[T - 1][1]) > 1e-6,
    f"head 支持度均值 ts=1 {f_t[0]:.6f} / ts=20 {f_t[1]:.6f} / ts={T - 1} {f_t[-1]:.6f}  "
    f"alpha_m 均值 ts=1 {a_t[1][1]:.4f} vs ts={T - 1} {a_t[T - 1][1]:.4f}")
sc1 = sc[:200].clone()
out1 = rrgcn.RecurrentRGCN.apply_base_prior(g, sc1, triples[:200], 1)
chk("T4c ts=1(没有更早历史)注入恒 0 => 首个快照逐位等于不开先验的原模型",
    torch.equal(out1, sc[:200]))

# ── T5 忘设 / 重复取 必须报错 ──
def expect_assert(fn, name):
    try:
        fn()
    except AssertionError as e:
        print(f"[PASS] {name}  ({str(e).splitlines()[0][:60]})")
        return
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"[FAIL] {name}  抛的不是 AssertionError: {type(e).__name__}: {e}")
        return
    FAIL.append(name)
    print(f"[FAIL] {name}  没有报错")


s5 = make_stub()
expect_assert(lambda: rrgcn.RecurrentRGCN._pop_prior_ts(s5),
              "T5 未 set_prior_ts 直接调用 => 报错")
s5.set_prior_ts(7)
v1 = rrgcn.RecurrentRGCN._pop_prior_ts(s5)
expect_assert(lambda: rrgcn.RecurrentRGCN._pop_prior_ts(s5), "T5b 同一个 ts pop 两次 => 报错")
chk("T5c set/pop 传值正确", v1 == 7, f"pop 得到 {v1}")
expect_assert(lambda: (s5.set_prior_ts(T), rrgcn.RecurrentRGCN._pop_prior_ts(s5)),
              "T5d ts 越出表长 => 报错")
# 静态口径(use_prior_ts=False)下 set_prior_ts / _pop_prior_ts 都必须无害
s6 = make_stub()
s6.use_prior_ts = False
s6.set_prior_ts(3)
chk("T5e 静态口径下 set/pop 是 no-op(老臂一个字节都不用改)",
    rrgcn.RecurrentRGCN._pop_prior_ts(s6) is None and s6._pending_ts is None)

print("\n" + ("[ALL PASS] 5 条全过, 可以开臂" if not FAIL else f"[FAILED] {FAIL}"))
sys.exit(1 if FAIL else 0)
