#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_maas_base_prior.py -- 输出层先验臂（--base-prior / --head-prior）的 0-GPU 机制检查。

各臂（凡是 maas_arms/<tag>.pt 存在的）+ 闭式通道放在同一批 121,371 行、同一把过滤式尺子上
逐边对照。E1 相对 §23 多了一条 head 通道，所以多了一张"加法合并"闭式表(alpha=1+1 的靶子)：

Q1 这一臂是不是"只是把先验搬过来"？
    模型 top-1 与闭式 base_hist top-1 的一致率(Jaccard) 分层给出；再加逐边得失表
    (模型对/先验错 vs 先验对/模型错)。若一致率≈1，学到的部分没有增量。
Q2 学到的 encoder 还有没有净贡献？
    在【先验答错】的边上模型的 H@1（base / baseemb / basefeat / baseprior 四臂对照）。
    这列不掉到 0 就说明 encoder 在先验之外还找到东西；关键是它在 baseprior 臂里比 base 臂高多少。
Q3 与"非学习合并天花板"还差多少？
    复发层同时给 entity_memory 闭式(0.7515)；再给 hit-level oracle（两条通道任一答对即算对），
    这是任何"只在两条通道之间选择"的模型都超不过的上界 —— 要证明"学出来的融合"有价值，
    必须越过它。
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

D = os.path.expanduser("~/xcad/xcad_code/xcad_model")
ARMS = os.path.join(D, "src/maas_arms")
DATA = os.path.join(D, "data/maas")
sys.path.insert(0, os.path.join(D, "src"))
import generic_strata as prod                                   # noqa: E402

TAGS_ALL = ("base", "baseemb", "basefeat", "baseprior", "headprior", "priorgate",
            "priorts", "priortsgate", "priorts_s43")
PRIOR_ARMS = ("baseprior", "headprior", "priorgate", "priorts", "priortsgate")
# checkpoint 文件名里的旗标串是【固定顺序】拼出来的(baseprior→headprior→priorgate→priorts),
# 与 dump 的 tag 不同名, 所以 Q4 取标量要按这张表找文件。
TAG2CK = {"baseprior": "-baseprior-seed42", "headprior": "-headprior-seed42",
          "priorgate": "-priorgate-seed42", "priorts": "-headprior-priorts-seed42",
          "priortsgate": "-priorgate-priorts-seed42", "priorts_s43": "-headprior-priorts-seed43"}
TAGS = tuple(t for t in TAGS_ALL if os.path.exists(os.path.join(ARMS, f"{t}.pt")))
n_ents = int(open(os.path.join(DATA, "stat.txt")).read().split()[0])
base_of = np.load(os.path.join(DATA, "base_of.npy"))
pad = int(base_of.max())

tails = set()
for fn in ("train.txt", "valid.txt", "test.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            tails.add(int(p[2]))
gpu_base = min(tails)
n_types = len(tails)
assert max(tails) == n_ents - 1 and n_types == n_ents - gpu_base

hrows, rows = [], []
for fn in ("train.txt", "valid.txt"):
    for ln in open(os.path.join(DATA, fn)):
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            hrows.append((int(p[0]), int(p[2]) - gpu_base))
for ln in open(os.path.join(DATA, "test.txt")):
    p = ln.split()
    if len(p) >= 4 and int(p[1]) == 0:
        rows.append((int(p[0]), int(p[2]) - gpu_base, int(p[3])))
T = pd.DataFrame(rows, columns=["hd", "tc", "tau"])
H = pd.DataFrame(hrows, columns=["hd", "tc"])
hist_pairs = set(map(tuple, H.values))
hist_heads = set(H.hd)

ranks = {}
for tag in TAGS:
    d = torch.load(os.path.join(ARMS, f"{tag}.pt"), map_location="cpu")
    assert np.array_equal(d["heads"].numpy(), T.hd.values), f"{tag} 行序不一致"
    assert np.array_equal(d["tails"].numpy() - gpu_base, T.tc.values), f"{tag} gold 不一致"
    ranks[tag] = d["filt_rank"].numpy().astype(np.float64)
n = len(T)

per = {}
for h, tau, t in zip(T.hd, T.tau, T.tc):
    per.setdefault((int(h), int(tau)), set()).add(int(t))
gold = T.tc.to_numpy()
filt = [{x for x in per[(int(h), int(tau))] if x != int(g)} for h, tau, g in
        zip(T.hd, T.tau, gold)]


def chan_ranks(row_tables):
    """任意"每行一张 {卡型: 分}"表 -> 同尺排名(与生产实现同一个函数)。"""
    uniq, index_of, rk_id = [], {}, np.empty(n, dtype=np.int64)
    for i, t in enumerate(row_tables):
        k = frozenset(t.items())
        j = index_of.get(k)
        if j is None:
            j = index_of[k] = len(uniq)
            uniq.append(t)
        rk_id[i] = j
    return prod._ranks_from_cols(rk_id, uniq, np.asarray(gold), filt, n_types,
                                 np.random.default_rng(7), (0, n_types - 1)).astype(np.float64)


# 闭式通道：base 历史直方图 / head 自己的历史 / 全局最热卡型
Hb = H.assign(b=base_of[H.hd.values])
tab_b = {int(b): {int(c): int(v) for c, v in sub.groupby("tc")["n"].sum().items()}
         for b, sub in Hb.groupby(["b", "tc"]).size().rename("n").reset_index().groupby("b")}
tab_h = {int(h): {int(c): int(v) for c, v in sub.groupby("tc")["n"].sum().items()}
         for h, sub in H.groupby(["hd", "tc"]).size().rename("n").reset_index().groupby("hd")}
tc_all = H.groupby("tc").size()
tab_t = {int(c): int(v) for c, v in tc_all.items()}

bh_rank = chan_ranks([tab_b.get(int(b), {}) for b in base_of[T.hd.values]])
em_rank = chan_ranks([tab_h.get(int(h), {}) for h in T.hd.values])
tp_rank = chan_ranks([tab_t] * n)

# 闭式"加法合并"通道 = 两表 log1p 逐候选相加 —— 它就是 alpha_b=alpha_m=1、encoder 输出
# 为常数时 headprior 臂应当复现的那张表。臂与它逐边对照 => 学到的两个标量偏离 1+1 有没有用。
_bl = {b: {c: float(np.log1p(v)) for c, v in d.items()} for b, d in tab_b.items()}
_hl = {h: {c: float(np.log1p(v)) for c, v in d.items()} for h, d in tab_h.items()}


def _add_tbl(h, b):
    d = dict(_bl.get(b, {}))
    for c, v in _hl.get(h, {}).items():
        d[c] = d.get(c, 0.0) + v
    return d


add_rank = chan_ranks([_add_tbl(int(h), int(b))
                       for h, b in zip(T.hd.values, base_of[T.hd.values])])

new_head = ~T.hd.isin(hist_heads).values
recur = np.array([(int(h), int(t)) in hist_pairs for h, t in zip(T.hd.values, T.tc.values)])
COH = {"全部 test 边": np.ones(n, bool), "全新 head": new_head, "复发边(硬线层)": recur}


def m(r, mask):
    x = r[mask]
    return (f"H@1={np.mean(x <= 1):.4f} H@3={np.mean(x <= 3):.4f} H@10={np.mean(x <= 10):.4f} "
            f"中位={np.median(x):.1f} p90={np.quantile(x, .9):.1f} n={int(mask.sum())}")


print(f"[setup] n={n} n_ents={n_ents} gpu_base={gpu_base} n_types={n_types} n_bases={pad} "
      f"全新 head={new_head.mean():.4f} 复发边={recur.mean():.4f}")
_ck = torch.load(os.path.join(ARMS, "baseprior.pt"), map_location="cpu")
print(f"[dump] candidate_universe={_ck.get('candidate_universe')} num_nodes={int(_ck['num_nodes'])} "
      f"dump 自带 entity_memory/tail_prior 秩(下面重算一遍对照)")

print("\n=== Q0 管道自检：重算的闭式 vs dump 里生产实现算的（H@1，全体/复发）===")
for nm, mine, theirs in (("entity_memory", em_rank, np.asarray(_ck["baseline"]["entity_memory"], dtype=np.float64)),
                         ("tail_prior", tp_rank, np.asarray(_ck["baseline"]["tail_prior"], dtype=np.float64))):
    for c, mask in COH.items():
        if c == "全新 head":
            continue
        print(f"  {nm:14s} {c:14s} 重算 H@1={np.mean(mine[mask] <= 1):.4f}  "
              f"生产 H@1={np.mean(theirs[mask] <= 1):.4f}  秩不同行数={int((mine[mask] != theirs[mask]).sum())}")

print("\n=== Q1 同尺四臂 + 三通道（H@1/H@3/H@10）===")
for c, mask in COH.items():
    print(f"  {c:16s} n={int(mask.sum()):7d}")
    for tag in TAGS:
        print(f"      {tag:10s} {m(ranks[tag], mask)}")
    for nm, r in (("base_hist", bh_rank), ("entity_mem", em_rank),
                  ("add_hist", add_rank), ("tail_prior", tp_rank)):
        print(f"      {nm:10s} {m(r, mask)}")

print("\n=== Q1b 先验臂是不是只把先验搬过来：top-1 同率 + 逐边得失（每条先验臂一行）===")
for arm in [a for a in PRIOR_ARMS if a in TAGS]:
    print(f"  --- {arm} ---")
    for c, mask in COH.items():
        mh = ranks[arm][mask] <= 1
        ph = bh_rank[mask] <= 1
        eh = em_rank[mask] <= 1
        print(f"  {c:14s} n={int(mask.sum()):7d}  ∩base_hist="
              f"{np.mean(np.logical_and(mh, ph) | np.logical_and(~mh, ~ph)):.4f}(top-1 同率)  "
              f"∩entity_mem={np.mean(np.logical_and(mh, eh) | np.logical_and(~mh, ~eh)):.4f}  "
              f"模型对&先验错={int(np.mean(mh & ~ph) * mask.sum()):6d}  先验对&模型错="
              f"{int(np.mean(ph & ~mh) * mask.sum()):6d}")
        print(f"                   模型对&记忆错={int(np.mean(mh & ~eh) * mask.sum()):6d}  "
              f"记忆对&模型错={int(np.mean(eh & ~mh) * mask.sum()):6d}  "
              f"两者都错(模型单独答对的空间)= 全体 miss {int(np.mean(~mh & ~eh) * mask.sum()):6d}")

print("\n=== Q2 encoder 的净贡献：在【先验答错】的边上各臂 H@1 ===")
for c, mask in COH.items():
    for nm, pr in (("base_hist", bh_rank), ("entity_memory", em_rank)):
        sel = mask & (pr > 1)
        line = "  ".join(f"{tag}={np.mean(ranks[tag][sel] <= 1):.4f}" for tag in TAGS)
        print(f"  {c:14s} {nm:13s}错 n={int(sel.sum()):7d}  {line}  "
              f"| 该通道单独 H@1={np.mean(pr[sel] <= 1):.4f}")

print("\n=== Q3 与『非学习合并』的差距：hit-level oracle（任一通道 top-1 命中即算命中）===")
for arm in [a for a in PRIOR_ARMS if a in TAGS]:
    print(f"  --- {arm} ---")
    for c, mask in COH.items():
        or_b = (bh_rank <= 1) | (em_rank <= 1)
        or_all = or_b | (tp_rank <= 1)
        print(f"  {c:14s} {arm}={np.mean(ranks[arm][mask] <= 1):.4f}  "
              f"oracle(base∪mem)={np.mean(or_b[mask]):.4f}  oracle(+tail)={np.mean(or_all[mask]):.4f}  "
              f"| 单通道 base={np.mean(bh_rank[mask] <= 1):.4f} mem={np.mean(em_rank[mask] <= 1):.4f} "
              f"add={np.mean(add_rank[mask] <= 1):.4f}")

print("\n=== Q5 encoder 的独立空间：【两条闭式通道都答错】的边上各臂 H@1（挤压的直接读数）===")
for c, mask in COH.items():
    for nm, sel0 in (("base&mem 都错", (bh_rank > 1) & (em_rank > 1)),
                     ("base&mem&add 都错", (bh_rank > 1) & (em_rank > 1) & (add_rank > 1))):
        sel = mask & sel0
        if sel.sum() < 50:
            print(f"  {c:14s} {nm:18s} n={int(sel.sum()):5d}  (太少，跳过)")
            continue
        line = "  ".join(f"{tag}={np.mean(ranks[tag][sel] <= 1):.4f}" for tag in TAGS)
        print(f"  {c:14s} {nm:18s} n={int(sel.sum()):6d}  {line}")

print("\n=== Q4 学到的信任度 ===")
for arm in [a for a in PRIOR_ARMS if a in TAGS]:
    for f in sorted(os.listdir(os.path.join(D, "models"))):
        if TAG2CK.get(arm, f"-{arm}-seed42") in f:
            ck = torch.load(os.path.join(D, "models", f), map_location="cpu")
            sd = ck["state_dict"]
            al = "  ".join(f"{k}={float(sd[k]):.4f}" for k in ("prior_alpha", "head_alpha") if k in sd)
            print(f"  {arm:9s} {f}\n            best epoch={ck.get('epoch')}  {al}  "
                  f"base_prior max={float(sd['base_prior'].max()):.4f}"
                  + (f"  head_prior max={float(sd['head_prior'].max()):.4f}" if "head_prior" in sd else ""))

print("\n=== Q6 (§25 自证伪线) 门控有没有真的按行分离信任度 ===")
# 静态口径的门控输入按实体存, 时序口径按 (实体, ts) 现算 —— 但 test 行的 ts 全 >= 54,
# 而那些层的表与静态表【逐位相同】(exporter + probe_maas_prior_ts_wiring T3), 所以同一套
# 静态特征就是这两条臂在 test 上真正用到的特征 => 下面这段对 priorgate / priortsgate 都成立。
_gf_path = os.path.join(D, "models")
for f in sorted(os.listdir(_gf_path)):
    if "-priorgate" not in f or "-seed42" not in f:
        continue
    sd = torch.load(os.path.join(_gf_path, f), map_location="cpu")["state_dict"]
    if "gate_w" not in sd:
        print("  checkpoint 里没有 gate_w —— 这一臂没开门控？"); break
    gw, gc = sd["gate_w"].float(), sd["gate_c"].float()
    _bp = torch.from_numpy(np.load(os.path.join(DATA, "base_prior.npy"))).float()
    _hp = torch.from_numpy(np.load(os.path.join(DATA, "head_prior.npy"))).float()
    _sb, _sh = _bp.sum(1), _hp.sum(1)
    _sb = _sb / max(float(_sb.max()), 1e-6)
    _sh = _sh / max(float(_sh.max()), 1e-6)
    gf = torch.stack([_sb[torch.from_numpy(base_of)], _sh, (_sh > 0).float()], 1)
    al = torch.nn.functional.softplus(gf @ gw.t() + gc)              # [n_ents, 2] 逐【实体】
    # 注意：alpha 只随实体变 => 必须先在实体域算，再按行取；不要把逐行 mask 直接切实体数组。
    ab_e, am_e = al[:, 0].numpy(), al[:, 1].numpy()                  # [n_ents]
    has_h_ent = (_sh > 0).numpy()                                   # [n_ents] 该实体有 head 历史
    is_head_ent = np.arange(n_ents) < gpu_base
    g1, g0 = has_h_ent & is_head_ent, (~has_h_ent) & is_head_ent
    ent = T.hd.values                                               # [n] 逐行 head 实体
    has_h = has_h_ent[ent]                                          # [n] 逐行
    print(f"  gate_w={np.round(gw.numpy(), 4).tolist()}  gate_c={np.round(gc.numpy(), 4).tolist()}")
    print(f"  实体域：有 head 历史 {int(g1.sum()):6d}/{int(is_head_ent.sum())} 个 head, "
          f"alpha_b={ab_e[g1].mean():.4f}  alpha_m={am_e[g1].mean():.4f}; "
          f"无历史 {int(g0.sum()):6d} 个, alpha_b={ab_e[g0].mean():.4f}  alpha_m={am_e[g0].mean():.4f}")
    for c, mask in COH.items():
        g = has_h[mask]
        _ab, _am = ab_e[ent[mask]], am_e[ent[mask]]
        print(f"      {c:14s} 有 head 历史 {int(g.sum()):6d} 行 / 无 {int((~g).sum()):6d} 行  "
              f"alpha_b {(_ab[g].mean() if g.any() else float('nan')):.4f} vs {(_ab[~g].mean() if (~g).any() else float('nan')):.4f}   "
              f"alpha_m {(_am[g].mean() if g.any() else float('nan')):.4f} vs {(_am[~g].mean() if (~g).any() else float('nan')):.4f}")
    d_ab = ab_e[ent][~has_h].mean() - ab_e[ent][has_h].mean()
    print(f"  >>> 自证伪线：两组 alpha_b 均值差(逐行加权) = {d_ab:+.4f}  "
          f"(|差| < 0.10 => 门没按行分开 = §25 判失败)  参照: §24 全局标量 alpha_b=1.4006 / §23 1.5557")
