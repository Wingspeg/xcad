#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_xcad_head_prior.py -- (D)：把 §26 的时序口径输出层先验搬回主数据集 xcad_v2。

对应 maas 那两张表的哪一半
--------------------------------
maas 有两条通道（base 粗粒度 + head 自己），v2020 只有 head 这一条 —— 数据里没有
`base_id` 层。所以本脚本只产：

    head_prior_ts[t, v, k] = log1p( #{train+valid 里 rel==1 且 ts < t
                                      且 head==v 且 tail==gpu_k 的边} )
    head_prior[v, k]       = 同一张表但不做【严格更早】限制（= §23/§24 的静态口径）

下标 t 直接用 .txt 第 4 列的原始 τ（v2020: train 1..49 / valid 50..55 / test 56..61），
与 maas 的 `head_prior_ts.npy` 同约定 ⇒ `main.py::snap_ts()` / `set_prior_ts()` 一行不用改。
打分侧也不用改：`--head-prior` 默认读 `../data/<dataset>/head_prior.npy`、`--prior-ts` 读
`head_prior_ts.npy`，偏置列段由 `--probe-rel 1` 自动推出 [124886,124890)。

为什么正式臂用【时序口径】而不是静态口径（§28.2-③ 的 fit 侧实测）
--------------------------------------------------------------
静态表对全部 97,246 条训练行的 argmax 100.00% 命中本行自己的 gold（表就是由这些行造的，
且 v2020 每个 head 的历史近乎只对应一个卡型），而时序表只在 8.35% 的训练行上非空。
maas 的 §25 探针量到的是"16.4% 训练行无更早快照却有表、其中 79.8% 命中自己"——
**v2020 的这个捷径更甜也更常开**，用静态口径会让 `head_alpha` 学在一条测试期 91.4%
为 0 的通道上（正是 §25 在 maas 上的失效形状）。

test 侧两个口径应当【逐位相同】（§28 实测同点率 1.000000）；这条写成断言，
它保证 (D) 的任何 test 读数差异只可能来自"模型学到了什么"，不可能来自"评测看到了什么"。

泄漏纪律：test.txt 的边不进任何表；只用 train.txt + valid.txt。

Usage (server):
    ~/xcad/xcad_code/.venv/bin/python xcad_code/src/data/export_xcad_head_prior.py
"""
import json
import os

import numpy as np

D = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/xcad_v2")
OUT_TS = os.path.join(D, "head_prior_ts.npy")
OUT_STATIC = os.path.join(D, "head_prior.npy")
OUT_JSON = os.path.join(D, "head_prior_report.json")

REL = 1                                    # r1_suits —— 与 §2.4/§28 的评测关系一致
REF_LOOKUP = (0.478033, 0.999022, 0.428743)   # §2.4/§28.1 的查表 H@1: 全体/复发(1022/1023)/全新
REF_AVAIL = (0.0835, 0.0864)                   # §28 探针: 拟合期/测试期"有严格更早历史"的行比例
n_ents = int(open(os.path.join(D, "stat.txt")).read().split()[0])

# ── rel=1 的 tail 段必须是末尾连续段（与 main.py 里 --probe-rel 那三条断言同判据） ──
E = {}
for s in ("train", "valid", "test"):
    a = np.loadtxt(os.path.join(D, f"{s}.txt"), dtype=np.int64)
    r = a[a[:, 1] == REL]
    E[s] = np.stack([r[:, 0], r[:, 2], r[:, 3]], axis=1)      # head, tail, ts
tails = sorted(set(np.concatenate([E[s][:, 1] for s in E]).tolist()))
tail_lo, tail_hi = tails[0], tails[-1]
assert tail_hi == n_ents - 1 and len(tails) == tail_hi - tail_lo + 1, (
    f"rel{REL} 的 tail 段 [{tail_lo},{tail_hi}] 不是末尾连续段（{len(tails)} 个 tail, "
    f"num_nodes={n_ents}）=> 偏置列无从对齐")
n_types = len(tails)
col = np.zeros(n_ents, dtype=np.int64)     # tail 实体 id -> 先验表列（只填 tail 段那几个）
col[tails] = np.arange(n_types)
print(f"[id-space] n_ents={n_ents} rel={REL} tail 段=[{tail_lo},{tail_hi}] n_types={n_types} "
      f"→ prior_lo={tail_lo} prior_n={n_types}")

for s in E:
    assert E[s][:, 0].max() < tail_lo, f"{s}: rel{REL} 的 head 落进了 tail 段"
    assert E[s][:, 2].min() >= 1, f"{s}: 出现 ts=0，口径要重推"
    assert np.isin(E[s][:, 1], tails).all(), f"{s}: rel{REL} 出现 tail 段之外的实体"
T = int(max(E[s][:, 2].max() for s in E)) + 1                 # 下标 = 原始 τ；第 0 层恒全 0
print("[splits] " + "  ".join(
    f"{s}: n={len(E[s])} ts=[{E[s][:, 2].min()},{E[s][:, 2].max()}] 唯一={len(np.unique(E[s][:, 2]))}"
    for s in E))
assert (E["train"][:, 2].max() < E["valid"][:, 2].min()
        and E["valid"][:, 2].max() < E["test"][:, 2].min()), "三段 ts 有交叠"
t_test_lo = int(E["test"][:, 2].min())
HIST = np.concatenate([E["train"], E["valid"]])               # 只有这两段进表
print(f"[edges] history(rel{REL})={len(HIST)}  test(rel{REL})={len(E['test'])}  T={T}")


# ── 建表：按 τ 计数 -> 沿 τ 前缀和 -> (可选)右移一位(=严格更早) -> log1p 走 float64 ──
def build(strict_earlier):
    day = np.zeros((T, n_ents, n_types), dtype=np.float64)
    np.add.at(day, (HIST[:, 2], HIST[:, 0], col[HIST[:, 1]]), 1.0)
    np.cumsum(day, axis=0, out=day)                            # day[t] = ts<=t
    if strict_earlier:
        shifted = np.zeros_like(day)
        shifted[1:] = day[:-1]                                 # shifted[t] = ts<t
        day = shifted
    assert not day[0].any(), "ts=0 层不该有任何计数"
    res = np.empty(day.shape, dtype=np.float32)
    for t in range(T):                                         # 与 §23/§26 同一算法：float64 log1p
        res[t] = np.log1p(day[t])
    assert (np.diff(day, axis=0) >= 0).all(), "计数沿 ts 出现倒退"
    return res, day


head_prior_ts, cnt_ts = build(True)
head_prior_static, cnt_st = build(False)
assert not head_prior_ts[:, tail_lo:].any() and not head_prior_static[:, tail_lo:].any(), \
    "GPU 实体不该有自己的历史出边计数"

tot = cnt_st[T - 1]                                            # 全历史每 head 每卡型的计数
glob = tot.sum(axis=0).astype(np.int64)

# 众数：主键=计数降序，并列时按全局卡型频次降序，再并列按类号升序 —— 与 §28 探针 mode_of()
# 的 np.lexsort((-glob, -d)) 逐位同规则（写成复合分数是为了向量化；量级差保证不会串键）。
rank = np.empty(n_types, dtype=np.int64)
rank[np.argsort(-glob, kind='stable')] = np.arange(n_types)
score = (tot * 1e8
         + (n_types - rank)[None, :].astype(np.float64) * 1e4
         + (n_types - np.arange(n_types))[None, :].astype(np.float64))
mode = score.argmax(axis=1)
glob_mode = int(np.lexsort((np.arange(n_types), -glob))[0])
print(f"[static] 全局类计数 {dict(zip(tails, glob.tolist()))} -> 众数列={glob_mode}")

# ── C1: test 侧两口径逐位相同（§28 实测同点率 1.000000） ──
same_test = all(np.array_equal(head_prior_ts[t], head_prior_static[t])
                for t in range(t_test_lo, T))
assert same_test, (f"test 段(t>={t_test_lo}) 时序表与静态表不一致 => 本轮只该改训练/验证期的"
                   "读表口径，test 侧必须逐位不变")
# 反向自检：train/valid 段【必须】不同，否则时序口径等于没改
for t_chk, nm in ((int(E["train"][:, 2].max()), "train 末快照"),
                  (int(E["valid"][:, 2].min()), "valid 首快照")):
    assert not np.array_equal(head_prior_ts[t_chk], head_prior_static[t_chk]), \
        f"{nm} 两口径相同 => 时序口径没有生效"

# ── C2: 自泄漏按构造为 0；可用率与捷径率必须复现 §28 探针 ──
first_ts = np.full(n_ents, 10 ** 9, dtype=np.int64)
np.minimum.at(first_ts, (HIST[:, 0],), HIST[:, 2])
rep_split = {}
print("\n[availability] 两口径对照（§28 探针：fit 时序 0.0835 / test 0.0864 / 静态捷径率 1.0000）")
print("   集合     n        静态可用率  时序可用率  静态命中gold  时序命中gold(仅可用行)  自泄漏行")
for nm, arr in (("train", E["train"]), ("valid", E["valid"]), ("test", E["test"]),
                ("fit(=train+valid)", HIST)):
    hd, td, ts = arr[:, 0], arr[:, 1], arr[:, 2]
    c = col[td]
    # 静态口径：表含本行自己 => train/valid 行可用率恒 1.0000（那正是捷径的成因）
    av_st = tot[hd].sum(axis=1) > 0
    ok_ts = ts > first_ts[hd]                                  # 严格更早有没有边（= 时序可用）
    # 自泄漏 = 直接读【时序表本体】，非空却无更早历史；按构造必须为 0
    av_ts_row = head_prior_ts[ts, hd].max(axis=1) > 0
    leak = int((av_ts_row & ~ok_ts).sum())
    assert int((av_ts_row != ok_ts).sum()) == 0, \
        f"{nm}: 时序表非空与'有严格更早历史'不等价 {int((av_ts_row != ok_ts).sum())} 行"
    hit_ts = float((mode[hd][ok_ts] == c[ok_ts]).mean()) if ok_ts.any() else float("nan")
    hit_st = float((mode[hd] == c).mean())                     # 静态口径：表含本行自己
    print(f"   {nm:8s} {len(hd):8d}   {av_st.mean():.4f}      {ok_ts.mean():.4f}"
          f"      {hit_st:.4f}        {hit_ts:.4f}              {leak}")
    rep_split[nm] = {"n": int(len(hd)), "static_avail": round(float(av_st.mean()), 4),
                    "ts_avail": round(float(ok_ts.mean()), 4),
                    "static_hit_rate": round(hit_st, 4),
                    "ts_hit_rate_on_available": round(hit_ts, 4),
                    "self_leak_rows": leak}
    assert leak == 0, f"{nm}: {leak} 行在无更早快照时表非空 => 自泄漏没修掉"
fit_av, te_av = rep_split["fit(=train+valid)"]["ts_avail"], rep_split["test"]["ts_avail"]
fit_st = rep_split["fit(=train+valid)"]["static_avail"]
te_st = rep_split["test"]["static_avail"]
gap_ts = fit_av - te_av
print(f"   >>> 可用率 fit - test：静态 {fit_st:.4f}-{te_st:.4f}=+{fit_st - te_st:.4f}"
      f"（§25 maas 是 +0.8475）   时序 {gap_ts:+.4f}（§28 探针 −0.0029）")
assert abs(gap_ts - (REF_AVAIL[0] - REF_AVAIL[1])) <= 0.002, \
    f"时序可用率差 {gap_ts:+.4f} 与 §28 探针不符"
assert abs(fit_av - REF_AVAIL[0]) <= 0.002 and abs(te_av - REF_AVAIL[1]) <= 0.002, \
    f"可用率 fit={fit_av} test={te_av} 与 §28 的 {REF_AVAIL} 不符"
assert rep_split["fit(=train+valid)"]["static_hit_rate"] >= 0.999, \
    "静态口径下训练行的 argmax 应当几乎全部命中本行 gold（那正是捷径的成因），对不上说明表建错了"
assert abs(fit_st - 1.0) < 1e-9, "静态表对 train+valid 行必须全部可用（它由这些行造出来）"

# ── C3: 查表读数必须复现 §2.4/§28.1 的 0.4780/0.9990/0.4287 ──
te = E["test"]
cls_te = col[te[:, 1]]
pred_te = mode[te[:, 0]]
is_rec = tot[te[:, 0]].sum(axis=1) > 0
read = (float((pred_te == cls_te).mean()),
        float((pred_te[is_rec] == cls_te[is_rec]).mean()),
        float((pred_te[~is_rec] == cls_te[~is_rec]).mean()))
print(f"\n[lookup] 查表 H@1 实测={read[0]:.6f}/{read[1]:.6f}/{read[2]:.6f}  "
      f"ref(§2.4)={REF_LOOKUP[0]:.6f}/{REF_LOOKUP[1]:.6f}/{REF_LOOKUP[2]:.6f}  "
      f"复发 {int(is_rec.sum())}/{len(te)} = {is_rec.mean()*100:.2f}%")
for got, ref, nm in zip(read, REF_LOOKUP, ("全体", "复发", "全新")):
    assert abs(got - ref) <= 1.5e-4, f"查表 {nm} 读数 {got:.6f} 与 §2.4 的 {ref:.6f} 不符"
# 全新层的查表读数按构造必须等于"恒答全局众数"：① 这些行的预测全是 glob_mode；
# ② 于是它们的准确率 == 常数先验准确率（§2.4 里"记忆查表在全新层 = 类别先验"那格）
assert (mode[te[~is_rec, 0]] == glob_mode).all(), "无历史 head 的查表预测应当全部退回全局众数"
assert read[2] == float((np.full(int((~is_rec).sum()), glob_mode) == cls_te[~is_rec]).mean())

# ── 静态 cube 只用于对照；rrgcn 的 --head-prior 读的是 2-D 表（= 最后一个 τ 层，全历史）──
st_cube = head_prior_static
assert (st_cube[T - 1] == st_cube[t_test_lo]).all(), "静态 cube 在 test 段应当已经饱和"
head_prior_static = st_cube[T - 1]                 # (n_ents, n_types)

# ── C4: 形状/边界必须能被 rrgcn 的三条断言接住 ──
assert head_prior_static.shape == (n_ents, n_types)
assert head_prior_ts.shape == (T, n_ents, n_types)
assert float(np.abs(head_prior_static[tail_lo:tail_lo + n_types]).max()) < 1e-6
assert float(np.abs(head_prior_ts[:, tail_lo:tail_lo + n_types]).max()) < 1e-6
assert np.array_equal(head_prior_ts[t_test_lo], head_prior_static), \
    "首个 test 快照的时序表必须等于静态表"

rep = {
    "convention": "head_prior_ts[t,v,k] = log1p(#{train+valid rel==1 edges with ts<t, head==v, tail==gpu_k})",
    "dataset": "xcad_v2", "rel": REL, "T": T, "n_ents": n_ents,
    "prior_lo": tail_lo, "prior_n": n_types, "tails": tails,
    "ts_axis": "dim0 == 第 4 列原始 τ (train 1..49 / valid 50..55 / test 56..61)；第 0 层恒全 0",
    "history_edges_in_table": int(len(HIST)),
    "test_first_ts": t_test_lo,
    "test_side_static_vs_temporal_identical": bool(same_test),
    "avail_static_fit_minus_test": round(fit_st - te_st, 4),
    "avail_temporal_fit_minus_test": round(gap_ts, 4),
    "maas_reference_avail_gap_static": 0.8475,
    "lookup_h1": {"all": round(read[0], 6), "recurrent": round(read[1], 6),
                  "novel": round(read[2], 6), "reference_2_4": list(REF_LOOKUP)},
    "recurrent_share": round(float(is_rec.mean()), 4),
    "splits": rep_split,
    "log1p_max_static": round(float(head_prior_static.max()), 4),
    "log1p_max_ts": round(float(head_prior_ts.max()), 4),
    "heads_with_history": int((tot[:tail_lo].sum(axis=1) > 0).sum()),
    "self_leak_rows_all_splits": 0,
}
np.save(OUT_TS, head_prior_ts)
np.save(OUT_STATIC, head_prior_static)
with open(OUT_JSON, "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(f"\n[write] {OUT_TS} shape={head_prior_ts.shape} "
      f"({head_prior_ts.nbytes / 1e6:.1f} MB float32)  log1p<= {rep['log1p_max_ts']}")
print(f"[write] {OUT_STATIC} shape={head_prior_static.shape} "
      f"({head_prior_static.nbytes / 1e6:.1f} MB)  log1p<= {rep['log1p_max_static']}")
print(json.dumps(rep, ensure_ascii=False, indent=2))
print("\nALL CHECKS OK")
