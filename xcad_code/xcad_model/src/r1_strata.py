# -*- coding: utf-8 -*-
"""
r1_strata.py — R1-1「归因不唯一」的决定性分层: 把 test 的 r1_suits 边按
**head 算法是否在历史里复发** 拆开, 分别算排序指标与 per-GPU macro。

为什么要这个:
    RE-GCN 的增益按论文叙事来自"实体随时间复发 → dynamic_emb 被训练 → 排序变好"。
    但 xCAD 的 r1 test 边里绝大多数 head 在训练侧从未出现。若增益全部落在
    "新 head" 这一层, 说明模型靠的是与时间无关的属性先验(compat 头读静态特征),
    而不是关系演化; 若集中在"复发 head" 层, 命题 P 被推翻, 演化叙事成立。

分层器必须是**无标签**的: 只用 history(train/valid 快照)里出现过的 id 判定,
不碰 test 真值。这里同时给两套判定口径:
    FULL : history_list 全量 —— 决定 dynamic_emb 是否吃过梯度
    WIN  : 编码器在 test 时真正聚合的最后 test_history_len 个快照
    r1   : 只算 head 以 r1_suits 出现过的
    any  : head 以任意关系出现过的(节点 embedding 跨关系共享)

基准线: prior = 用 history 里 r1 tail 的众数类去猜, 每个分层给一次,
        用来判断"模型 H@1 是否只是复现了类别先验"。

纯记账代码, 不消耗随机数、不做反向传播 —— 挂在 main.py::test() 末尾
不会影响任何在跑的训练。

自测: python r1_strata.py  (合成数据, 校验 n 拆分/macro 口径/prior)
"""

import numpy as np
import torch

GPU_CLASS_NAMES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
N_GPU_CLASSES = 5          # GPU 段里 CPU 从不是 r1 真值, 只有 5 类参与 macro


def head_sets(snaps):
    """从快照列表(每个是 [M,4] ndarray/tensor)提取 head 集合。

    返回 (r1_heads, any_heads):
        r1_heads : 以关系 id==1 (r1_suits) 出现过的 head
        any_heads: 以任意关系出现过的 head
    """
    r1, any_ = set(), set()
    for snap in snaps:
        arr = np.asarray(snap.cpu().numpy() if isinstance(snap, torch.Tensor) else snap)
        if arr.size == 0:
            continue
        any_.update(arr[:, 0].tolist())
        r1.update(arr[arr[:, 1] == 1, 0].tolist())
    return r1, any_


def prior_tail(snaps):
    """history 里 r1 tail 的众数类(无标签先验); 没有 r1 边时返回 None。"""
    cnt = {}
    for snap in snaps:
        arr = np.asarray(snap.cpu().numpy() if isinstance(snap, torch.Tensor) else snap)
        if arr.size == 0:
            continue
        for t in arr[arr[:, 1] == 1, 2].tolist():
            cnt[t] = cnt.get(t, 0) + 1
    if not cnt:
        return None
    return max(cnt.items(), key=lambda kv: kv[1])[0]


def _metrics(ranks, tails, gpu_base):
    """一组 r1 边的排序指标 + 两种 macro 口径。"""
    r = ranks.float()
    out = {
        "n": int(r.numel()),
        "mrr": (1.0 / r).mean().item(),
        "h1": (r <= 1).float().mean().item(),
        "h3": (r <= 3).float().mean().item(),
        "h10": (r <= 10).float().mean().item(),
        "per_class": {},
    }
    hits, occ = [], []
    for k in range(1, N_GPU_CLASSES + 1):
        gid = gpu_base + k
        m = (tails == gid)
        n = int(m.sum())
        if n == 0:
            hits.append(0.0)
            out["per_class"][GPU_CLASS_NAMES[k]] = (0.0, 0)
        else:
            h = (ranks[m] == 1).float().mean().item()
            hits.append(h)
            occ.append(h)
            out["per_class"][GPU_CLASS_NAMES[k]] = (h, n)
    out["macro5"] = sum(hits) / N_GPU_CLASSES      # 缺失类记 0 —— 与论文表格同口径
    out["macro_occ"] = (sum(occ) / len(occ)) if occ else float("nan")
    out["n_occ"] = len(occ)
    return out


def stratified_report(heads, tails, ranks, history_list, window_snaps,
                      gpu_base, mode="test"):
    """打印 r1 分层对照表, 并返回 dict(供上层留存)。

    参数都是 1-D 同长张量: heads/tails = r1 边的 head/tail entity id,
    ranks = filter 后真值排名(1-based)。
    """
    heads_np = heads.numpy()
    n_total = int(heads.numel())
    if n_total == 0:
        print("  [r1 分层] no data")
        return {}

    p = prior_tail(history_list)
    full_r1, full_any = head_sets(history_list)
    win_r1, win_any = head_sets(window_snaps)

    res = {"total": _metrics(ranks, tails, gpu_base)}
    print()
    print("=" * 30 + f" r1 分层 [{mode}] " + "=" * 30)
    print(f"  r1 边数={n_total}  gpu_base={gpu_base}  先验类={p}"
          f"  (history r1 tail 众数)")
    _line("全部 test 边", res["total"], n_total, tails, p)

    for label, s in (("FULL & r1", full_r1), ("WIN & r1", win_r1),
                     ("FULL & any", full_any), ("WIN & any", win_any)):
        is_rec = np.isin(heads_np, list(s))
        print(f"\n  ── 分层器 {label}: 复发 {(is_rec.sum()):d}/{n_total} "
              f"= {is_rec.sum()/n_total*100:.2f}%")
        res[label] = {}
        for sub, mask in (("复发 head", is_rec), ("全新 head", ~is_rec)):
            if mask.sum() == 0:
                print(f"    {sub:9s} n=0")
                continue
            idx = torch.from_numpy(mask)
            m = _metrics(ranks[idx], tails[idx], gpu_base)
            res[label][sub] = m
            _line(sub, m, n_total, tails[idx], p)
    print("=" * 76)
    return res


def _line(name, m, n_total, tails, p):
    share = m["n"] / n_total * 100 if n_total else 0.0
    pr = float("nan")
    if p is not None:
        pr = (tails == p).float().mean().item()
    print(f"    {name:9s} n={m['n']:6d} ({share:5.1f}%)  MRR={m['mrr']:.6f}  "
          f"H@1={m['h1']:.6f}  H@3={m['h3']:.6f}  H@10={m['h10']:.6f}")
    print(f"    {'':9s} macro5={m['macro5']:.4f}  macro_occ={m['macro_occ']:.4f}"
          f"(c={m['n_occ']})  prior-H@1={pr:.4f}   " +
          "  ".join(f"{k}={v[0]:.3f}/{v[1]}" for k, v in m["per_class"].items()))


def _selftest():
    """合成数据校验记账逻辑: 复发/全新拆分无重叠、macro 口径、prior 计算。"""
    rng = np.random.RandomState(0)
    # 两个历史快照: tau1 有算法 10,11 的 r1 边; tau2 有算法 12 的 r0 边
    hist = [
        np.array([[10, 1, 124886, 1], [11, 1, 124887, 1], [10, 0, 5, 1]]),
        np.array([[12, 0, 6, 2], [10, 1, 124886, 2]]),
    ]
    heads = torch.tensor([10, 11, 12, 99])          # 12 只在 r0 出现, 99 全新
    tails = torch.tensor([124886, 124887, 124886, 124890])
    ranks = torch.tensor([1, 2, 5, 1])
    r = stratified_report(heads, tails, ranks, hist, hist, gpu_base=124885, mode="selftest")
    t = r["total"]
    assert t["n"] == 4
    assert abs(t["mrr"] - (1 + 0.5 + 0.2 + 1) / 4) < 1e-6
    rec = r["FULL & r1"]["复发 head"]
    nov = r["FULL & r1"]["全新 head"]
    assert rec["n"] == 2 and nov["n"] == 3 - 1, (rec["n"], nov["n"])
    assert rec["n"] + nov["n"] == t["n"]
    # prior 类 = 124886 (出现 2 次)
    assert prior_tail(hist) == 124886
    # r1 分层: 复发={10,11}; 全新={12,99}
    assert abs(rec["h1"] - 0.5) < 1e-6
    # any 分层: 复发={10,11,12}; 全新={99}
    assert r["FULL & any"]["复发 head"]["n"] == 3
    assert r["FULL & any"]["全新 head"]["n"] == 1
    # macro5 缺失类记 0: 复发组 T4 全中(rank1)、MISC 全失(rank2) -> (1.0+0)/5
    assert abs(rec["macro5"] - (1.0 + 0.0 + 0.0 + 0.0 + 0.0) / 5) < 1e-6, rec["macro5"]
    assert abs(rec["macro_occ"] - 0.5) < 1e-6, rec["macro_occ"]
    print("\n[selftest] OK")


if __name__ == "__main__":
    _selftest()
