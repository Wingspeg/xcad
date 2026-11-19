# -*- coding: utf-8 -*-
"""
generic_strata.py -- R1-1 的分层记账，去掉 xCAD 专属部分后的通用版。

为什么需要它
------------
main.py 里现有的探针(r1_strata / seg_probe / compat)全部假设 rel id 1 是
r1_suits、tail 段是 6 个 GPU 型号。第二份数据(MIT Supercloud)只有 1 个关系
placement、tail 是节点身份、卡型方差为 0，那些假设不成立，硬跑会打印一堆没有
意义的"per-GPU macro"。这里只保留真正决定 R1-1 结论的东西：

    test 边按 (a) head 是否在 history 里复发、(b) 这条 (head,tail) 边本身是否在
    history 里复发 拆开，每层给 MRR/H@1/H@3/H@10，
    并在【同一批行、同一套 filtered 口径】上重算两个闭式通道：
        entity_memory : score = history 里该 (head,tail) 出现次数
        tail_prior    : score = history 里该 tail 出现次数(与 head 无关，即类别先验)
    这样"模型 H@1 是不是只是把先验/查表复现了一遍"不需要跨脚本对齐口径。

排名口径与 src/rrgcn.py::predict 一致：
    rank = 1 + #{未被过滤的候选里分数严格大于真值的个数}
闭式基线的分数有大量并列(多数候选为 0)，并列必须【随机】打破而不是免费给 rank=1，
所以加 eps*U(0,1) 抖动(固定 seed，可复现)。

实现上不能物化 [n_test, n_ent] 的分数矩阵：maas 是 121,371 行 × 80,287 实体，
两个 float32 矩阵 + 一个 float64 副本 = 117 GB，第一版就是在 3090 那台 125 GB 的
机器上被 test 阶段的这一步 OOM killer 打死的(训练本身没问题)。闭式通道因此走
_ranks_from_cols(只遍历每行的正分候选)，dense 版留作参照实现，两者的等价性由
check_sparse_vs_dense() 断言。

纯记账，不 consume torch 的随机流，不影响任何在跑的训练。
"""

import numpy as np
import torch


def _arr(snap):
    """快照 -> int64 [M,4]。utils.split_by_time 会把 tau 剥掉，所以 3 列也要能吃。"""
    a = snap.cpu().numpy() if isinstance(snap, torch.Tensor) else np.asarray(snap)
    if not a.size:
        return np.zeros((0, 4), dtype=np.int64)
    a = a.reshape(-1, a.shape[-1]).astype(np.int64)
    if a.shape[1] == 3:
        a = np.column_stack([a, np.zeros(len(a), dtype=np.int64)])
    return a


def history_stats(snaps, rel):
    """history 快照 -> (heads_any, heads_rel, head_rel_days, pairs_rel, pairs_cnt).

    只用 history 侧 id，不碰 test 真值，所以分层器是无标签的。
    head_rel_days = 该 head 以 rel 出现过的【快照个数】；
    pairs_cnt     = (head, tail) 在 history 里的总边数(entity memory 的分数)。
    """
    heads_any, heads_rel, pairs, pairs_cnt, tail_cnt = set(), set(), set(), {}, {}
    head_rel_days = {}
    for snap in snaps:
        a = _arr(snap)
        if not len(a):
            continue
        heads_any.update(a[:, 0].tolist())
        f = a[a[:, 1] == rel]
        if not len(f):
            continue
        heads_rel.update(f[:, 0].tolist())
        for h in np.unique(f[:, 0]):
            head_rel_days[int(h)] = head_rel_days.get(int(h), 0) + 1
        for h, t in zip(f[:, 0].tolist(), f[:, 2].tolist()):
            pairs.add((int(h), int(t)))
            pairs_cnt[(int(h), int(t))] = pairs_cnt.get((int(h), int(t)), 0) + 1
            tail_cnt[int(t)] = tail_cnt.get(int(t), 0) + 1
    return heads_any, heads_rel, head_rel_days, pairs, pairs_cnt, tail_cnt


def _metrics(r):
    r = np.asarray(r, dtype=np.float64)
    return {"n": int(len(r)), "mrr": round(float((1.0 / r).mean()), 6),
            "h1": round(float((r <= 1).mean()), 6),
            "h3": round(float((r <= 3).mean()), 6),
            "h10": round(float((r <= 10).mean()), 6)}


def _rank_from_scores(scores, gold, filt):
    """参照实现(dense)：与 rrgcn.predict 同口径的 filtered rank(并列用固定 seed 的抖动随机打破)。

    只留给 _selftest 当等价性参照 —— 它会物化 float64 的 [n_rows, n_ent]，
    maas(121,371 行 × 80,287 实体)下要 78 GB，正是 3090 那台机器 OOM 的地方。
    生产路径走 _ranks_from_cols(稀疏，同口径)。
    """
    sc = scores.astype(np.float64) + \
        np.random.default_rng(1).random(scores.shape) * 1e-6
    out = np.empty(len(gold), dtype=np.int64)
    for i in range(len(gold)):
        s = sc[i].copy()
        g = int(gold[i])
        if filt[i]:
            s[list(filt[i])] = -np.inf
            s[g] = sc[i][g]
        out[i] = 1 + int((s > s[g]).sum())
    return out


def _ranks_from_cols(row_key, col_sets, gold, filt, n_ent, rng, universe=None):
    """_rank_from_scores 的稀疏等价版：只遍历每行的【正分候选】，不建 [n_rows, n_ent]。

    col_sets[k] = {entity: count>0}；row_key[i] 指向第 i 行用哪个分数表
    (entity_memory 按 head 取，tail_prior 全行共用同一张表)。
    universe=(lo,hi)：候选集换成这个闭区间(诊断口径，必须和 main.py --ent-segment 给
    模型的段一致)；区间外的候选与区间外的过滤项都不计，n_ent 被区间长度顶掉。

    等价性(和 dense 逐行同口径，rank = 1 + #{未被过滤且严格大于真值})：
      * 正分候选 C、被过滤集 F(dense 里写 -inf)、真值 g 三者分开看；
      * 未打分列的分数恒为 0 + eps*u，u~U(0,1)，于是
        (a) 真值正分(count>=1)时：0+eps < count <= 真值分，未打分列一个都不算大于
            真值 —— dense 里那 n_ent 列的比较整体退化为 0；
        (b) 真值 0 分时：未打分列 j 排到真值前 <=> u_j > u_g，|B| 个独立同分布的
            比较合起来就是 Binomial(|B|, 1 - u_g)，u_g 取 dense 里真值那一列的同一次
            抖动。所以这里抽【计数】而不是抽【n_ent 个抖动】，分布完全一致。
      * 正分候选之间的并列(count 相等)仍按 dense 一样用 eps 抖动打破。
    """
    out = np.empty(len(gold), dtype=np.int64)
    if universe is None:
        in_uni = lambda c: True
        n_uni = n_ent
        fast_n_f = True
    else:
        glo, ghi = int(universe[0]), int(universe[1])
        in_uni = lambda c: glo <= c <= ghi
        n_uni = ghi - glo + 1
        fast_n_f = False
        assert int(gold.min()) >= glo and int(gold.max()) <= ghi, (
            f"闭式基线候选段 [{glo}, {ghi}] 容不下真值: gold∈[{int(gold.min())}, "
            f"{int(gold.max())}] —— --ent-segment 与数据里的 tail 段不符")
    for i in range(len(gold)):
        g = int(gold[i])
        table = col_sets[row_key[i]]
        gs = table.get(g, 0)
        f = filt[i]
        cols = [c for c in table if c != g and c not in f and in_uni(c)]
        n_f = (len(f) - (1 if g in f else 0)) if fast_n_f \
            else len([x for x in f if x != g and in_uni(x)])
        u_g = rng.random() * 1e-6
        if gs > 0:
            sc_g = gs + u_g
            higher = 0
            if cols:
                vals = np.fromiter((table[c] for c in cols), dtype=np.float64,
                                   count=len(cols))
                higher = int(np.sum(vals + rng.random(len(cols)) * 1e-6 > sc_g))
            rest = 0                      # 见上 (a)
        else:
            higher = len(cols)            # 见上 (b)：正分候选全部严格大于 eps
            n_rest = n_uni - 1 - n_f - len(cols)
            rest = int(rng.binomial(n_rest, 1.0 - u_g * 1e6)) if n_rest > 0 else 0
        out[i] = 1 + higher + rest
    return out


def baseline_ranks(rel, test_snaps, hist_snaps, n_ent, universe=None):
    """在【模型评测的同一批 test 行】上算 entity_memory / tail_prior 的 rank。

    返回 (rows, base)；rows = [(s, r, o, tau), ...]，顺序与 test_snaps 里该关系
    的正向行一致 —— 也就是 main.py 收集模型 rank 的顺序，两边可以逐行对齐。
    filtered 口径与 utils.load_all_answers_for_time_filter(data.test) 一致：
    过滤集 = 同一 test 快照内、同 (head, rel) 的其他答案。
    """
    _, _, _, pairs, pairs_cnt, tail_cnt = history_stats(hist_snaps, rel)
    rows, gold, filt = [], [], []
    for snap in test_snaps:
        a = _arr(snap)
        f = a[a[:, 1] == rel]
        per = {}
        for h, o in zip(f[:, 0].tolist(), f[:, 2].tolist()):
            per.setdefault(int(h), set()).add(int(o))
        for h, r_, o, tau in f.tolist():
            h, o = int(h), int(o)
            rows.append((h, int(r_), o, int(tau)))
            gold.append(o)
            filt.append({x for x in per[h] if x != o})
    # 分数必须给【所有】候选打分，只填真值列会让基线退化成"命中即 rank1"，
    # 那样它就不是竞争排名而是查表命中率了 —— 未打分候选由 _ranks_from_cols 按
    # dense 同一口径计成 0+eps 那一档，不是一律排除。
    per_head = {}
    for (h, o), c in pairs_cnt.items():
        per_head.setdefault(h, {})[o] = c
    gold = np.array(gold, dtype=np.int64)
    n_row = len(rows)
    # entity_memory 的分数表按 head 取；tail_prior 全行共用同一张(与 head 无关)。
    mem_tables = [per_head.get(h, {}) for h, _, _, _ in rows]
    row_key = np.arange(n_row, dtype=np.int64)
    rng = np.random.default_rng(1)
    r_mem = _ranks_from_cols(row_key, mem_tables, gold, filt, n_ent, rng, universe)
    r_pop = _ranks_from_cols(np.zeros(n_row, dtype=np.int64),
                             [tail_cnt] * n_row if n_row else [], gold, filt, n_ent, rng,
                             universe)
    return rows, {
        "entity_memory": r_mem,
        "tail_prior": r_pop,
        "lookup_hit": np.array([(h, o) in pairs for h, _, o, _ in rows]),
        "rows": rows,
    }


def generic_report(heads, tails, ranks, rel, history_list, window_snaps,
                   test_snaps, num_nodes, mode="test", dump_path=None,
                   candidate_universe=None):
    """打印通用分层表；heads/tails/ranks 是模型在该关系正向边上的 1-D 张量。

    candidate_universe=(lo,hi) 必须与 main.py --ent-segment 传给模型排名的同一段：
    模型和闭式基线用的是同一把尺子，这张表才可以横向比。None = 全实体(论文主表口径)。
    """
    h = heads.numpy()
    n_total = len(h)
    if n_total == 0:
        print(f"  [rel{rel} 分层] no data")
        return {}
    rows, base = baseline_ranks(rel, test_snaps, history_list, num_nodes,
                                universe=candidate_universe)
    if len(rows) != n_total:
        print(f"  !! [rel{rel} 分层] 基线行数 {len(rows)} != 模型行数 {n_total}，"
              f"闭式基线不可比，只报模型")
        base = None
    else:
        for i, ((rh, _, ro, _), (mh, mt)) in enumerate(zip(rows, zip(h, tails.numpy()))):
            assert (rh, ro) == (int(mh), int(mt)), (i, rh, ro, mh, mt)

    heads_any, heads_rel, head_days, pairs, _, _ = history_stats(history_list, rel)
    win_any, win_rel, _, win_pairs, _, _ = history_stats(window_snaps, rel)
    t = tails.numpy()
    rec_edge = np.array([(int(a), int(b)) in pairs for a, b in zip(h, t)])
    win_edge = np.array([(int(a), int(b)) in win_pairs for a, b in zip(h, t)])
    strata = [
        ("全部 test 边", np.ones(n_total, bool)),
        (f"head 在 history 以 rel{rel} 复发", np.isin(h, list(heads_rel))),
        ("head 在 history 从未出现", ~np.isin(h, list(heads_any))),
        ("head 复发 >=2 快照", np.isin(h, [k for k, v in head_days.items() if v >= 2])),
        ("边 (head,tail) 在 history 里", rec_edge),
        ("边 (head,tail) 不在 history 里", ~rec_edge),
        ("边在编码器窗口(最近快照)里", win_edge),
        ("边不在编码器窗口里", ~win_edge),
    ]
    print()
    print("=" * 28 + f" rel{rel} 分层 [{mode}] " + "=" * 28)
    print(f"  边数={n_total}  实体={num_nodes}  history 快照={len(history_list)}  "
          f"编码器窗口={len(window_snaps)}")
    print(f"  候选集口径={'全实体(主表口径)' if candidate_universe is None else '类型段 %s' % (candidate_universe,)}"
          f" —— 模型与两条闭式通道同一段")
    res = {}
    for name, mask in strata:
        if mask.sum() == 0:
            print(f"    {name:34s} n=0")
            continue
        r = ranks.numpy()[mask]
        m = _metrics(r)
        line = (f"    {name:34s} n={m['n']:6d} ({m['n'] / n_total * 100:5.1f}%)  "
                f"MRR={m['mrr']:.6f}  H@1={m['h1']:.6f}  H@3={m['h3']:.6f}  "
                f"H@10={m['h10']:.6f}")
        if base is not None:
            for k in ("entity_memory", "tail_prior"):
                b = _metrics(base[k][mask])
                line += (f"\n    {'':34s}   {k}: MRR={b['mrr']:.6f} "
                         f"H@1={b['h1']:.6f} H@3={b['h3']:.6f} H@10={b['h10']:.6f}")
            line += (f"\n    {'':34s}   查表命中(gold 边在 history 出现过)="
                     f"{float(base['lookup_hit'][mask].mean()):.6f}")
        print(line)
        res[name] = m
    if dump_path and mode == "test":
        torch.save({"rel": int(rel), "heads": heads.cpu(), "tails": tails.cpu(),
                    "filt_rank": ranks.cpu(), "edge_in_history": rec_edge,
                    "num_nodes": int(num_nodes),
                    "candidate_universe": (tuple(int(x) for x in candidate_universe)
                                           if candidate_universe is not None else None),
                    "baseline": {k: v for k, v in (base or {}).items()}},
                   dump_path)
        print(f"\n  [dump] {dump_path}  n={n_total}")
    print("=" * 76)
    return res


def check_sparse_vs_dense(n_ent=60, n_row=4000, seed=0):
    """_ranks_from_cols 与 dense 参照实现的等价性检查(小实体空间，两者都跑得动)。

    为什么只能统计等价、不能逐位相同：dense 抽的是整块 [n_row, n_ent] 的抖动，
    真值那一列拿到的是流里第 (i, g) 个数；稀疏版每行只抽 1+|候选| 个。并列由抖动
    打破，所以【有并列的行】两边各是一个合法随机解，【无并列的行】必须逐位相同 ——
    这条是硬断言。无并列行覆盖不到的"0 分真值 vs 未打分候选"那一档，用均值/分布对齐。
    """
    rng = np.random.default_rng(seed)
    rows_per_table = 40                        # 一张分数表服务 40 行(模拟同一 head)
    tables, row_key, gold, filt = [], [], [], []
    for k in range(n_row // rows_per_table):
        n_cand = int(rng.integers(1, 12))
        cols = rng.choice(n_ent, size=n_cand, replace=False)
        tables.append({int(c): int(rng.integers(1, 6)) for c in cols})
        row_key.extend([k] * rows_per_table)
    row_key = np.array(row_key, dtype=np.int64)
    for i in range(n_row):
        t = tables[row_key[i]]
        keys = list(t)
        if i % 3:                              # 2/3 的行真值有分，1/3 落在未打分档
            gold.append(keys[int(rng.integers(len(keys)))])
        else:
            gold.append(int(next(c for c in range(n_ent) if c not in t)))
        g = gold[-1]
        others = [c for c in keys if c != g]
        n_f = min(int(rng.integers(0, 3)), len(others))
        filt.append(set(int(c) for c in rng.choice(others, size=n_f, replace=False))
                    if others else set())
    gold = np.array(gold, dtype=np.int64)

    dense_mat = np.zeros((n_row, n_ent), dtype=np.float32)
    for i in range(n_row):
        for c, v in tables[row_key[i]].items():
            dense_mat[i, c] = v
    ref = _rank_from_scores(dense_mat, gold, filt)
    fast = _ranks_from_cols(row_key, tables, gold, filt,
                            n_ent, np.random.default_rng(1))
    # universe 取全区间时必须与不传 universe 逐位相同 —— 只自校验管道，不引入新口径。
    fast_uni = _ranks_from_cols(row_key, tables, gold, filt, n_ent,
                                np.random.default_rng(1), universe=(0, n_ent - 1))
    assert np.array_equal(fast, fast_uni), "universe=(0,n_ent-1) 与全实体口径不一致"

    # 逐位相同的充要条件：真值正分，且没有"与真值同分且未被过滤"的候选(那才是抖动
    # 能翻盘的地方)。真值 0 分的行天然带随机项，只进统计对齐。
    deterministic = np.ones(n_row, dtype=bool)
    for i in range(n_row):
        t = tables[row_key[i]]
        g = int(gold[i])
        gs = t.get(g, 0)
        others = [v for c, v in t.items() if c != g and c not in filt[i]]
        deterministic[i] = gs > 0 and gs not in others
    n_det = int(deterministic.sum())
    exact = int(np.sum(ref[deterministic] == fast[deterministic]))
    assert exact == n_det, f"deterministic rows disagree: {n_det - exact}/{n_det}"
    n_zero_gold = int(np.sum([gold[i] not in tables[row_key[i]] for i in range(n_row)]))
    assert n_zero_gold > 0, "check is vacuous: no zero-score gold rows were generated"

    # 全体：并列行与 0 分真值行各是一个合法随机解，比均值与 H@k。
    mr, mf = float(ref.mean()), float(fast.mean())
    h1r, h1f = float((ref <= 1).mean()), float((fast <= 1).mean())
    h10r, h10f = float((ref <= 10).mean()), float((fast <= 10).mean())
    assert abs(mr - mf) / mr < 0.02, (mr, mf)
    assert abs(h1r - h1f) < 0.01 and abs(h10r - h10f) < 0.01, (h1r, h1f, h10r, h10f)
    print(f"[sparse==dense] 确定性行 {exact}/{n_det} 逐位相同  "
          f"(真值 0 分行 {n_zero_gold} 走二项抽样)  "
          f"mean rank dense={mr:.3f} sparse={mf:.3f}  "
          f"H@1 {h1r:.4f}/{h1f:.4f}  H@10 {h10r:.4f}/{h10f:.4f}")
    return {"exact": exact, "n_det": n_det, "n_zero_gold": n_zero_gold,
            "mean_dense": mr, "mean_sparse": mf,
            "h1": (h1r, h1f), "h10": (h10r, h10f)}


def _selftest():
    """合成数据：2 个 history 快照 + 1 个 test 快照，校验 rank 口径与分层计数。"""
    check_sparse_vs_dense()
    hist = [np.array([[0, 0, 5, 1], [0, 0, 6, 1]]), np.array([[0, 0, 5, 2], [1, 0, 7, 2]])]
    test = [np.array([[0, 0, 5, 3], [0, 0, 6, 3], [1, 0, 8, 3], [2, 0, 5, 3]])]
    heads = torch.tensor([0, 0, 1, 2])
    tails = torch.tensor([5, 6, 8, 5])
    ranks = torch.tensor([1, 2, 1, 3])
    res = generic_report(heads, tails, ranks, 0, hist, hist, test, 10, "selftest")
    assert res["全部 test 边"]["n"] == 4
    assert res["边 (head,tail) 在 history 里"]["n"] == 2      # (0,5) (0,6)；(1,8) (2,5) 不在
    assert res["head 在 history 以 rel0 复发"]["n"] == 3       # head 0,0,1
    assert res["head 在 history 从未出现"]["n"] == 1          # head 2
    print("\n[selftest] OK")


if __name__ == "__main__":
    _selftest()
