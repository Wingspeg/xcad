# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle

import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.r1_strata import stratified_report, head_sets
from src.generic_strata import generic_report
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

MAX_TRIPLES_PER_STEP = 2048  # 每时间步最多采样多少条正例三元组,避免 [B, 124891] score 矩阵 OOM
# (仅对 loss 目标边采样,history 图保持完整,与官方"全实体 CE"范式兼容)


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, static_graph, mode,
         snap_ts=None):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :param snap_ts: 逐快照的 ts(数据第 4 列), 与 test_list 一一对应。
                    只在 --prior-ts(时序口径输出层先验)时必须给: 模型要知道每个快照读哪一层先验表。
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    if getattr(model, "use_prior_ts", False):
        assert snap_ts is not None and len(snap_ts) == len(test_list), (
            f"--prior-ts: test() 缺逐快照 ts (给了 {None if snap_ts is None else len(snap_ts)}, "
            f"需要 {len(test_list)}) —— 少了就会静默沿用上一个快照的先验层")
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    # 按关系类型分组收集 filter rank
    r3_ranks_filter = []       # r3_drives: id=3(正) 和 id=7(逆=3+4)
    relX_ranks = {0: [], 1: [], 2: []}  # placement, r1_suits, r2_requires 正向
    r1_gpu_tails = []   # r1 tail entity ID (1-5)
    r1_gpu_ranks = []   # 1-based filter rank
    r1_gpu_heads = []   # r1 head(算法)entity ID —— 供 head 复现性分层用

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    # 分层器要在 input_list 被 pop/append 之前定住(编码器 test 时真正看到的窗口)
    window_snaps = list(input_list)
    # ── R1-1 的探针/分层/落盘 key 在哪个【正向】关系上 ──
    # xCAD: PR=1 (r1_suits, tail 段 = 6 个 GPU 型号)。第二份数据集(MIT)只有
    # PR=0 (placement, tail = 节点身份, 卡型方差 0): GPU 类别专属报表整段跳过,
    # 改走 src/generic_strata —— 同一套 filtered rank 口径, 只是没有 6 类宏平均。
    PR = int(getattr(args, "probe_rel", 1))

    # ── 探针(R1-1 同口径对照用, 纯记录, 不影响任何数值) ──
    #   seg_probe     : r1 正向边的 top-1 落在哪 + gold 在 GPU 段内的排名(已过滤)
    #   compat_probe  : compat 头自己的 6 类 argmax —— 这就是模型内部的"静态通道"直读
    _GPU_REL = (PR == 1)      # rel id 1 == r1_suits: tail 是 6 个 GPU 型号段
    _gb_probe = getattr(model, "gpu_base", 0)
    _dump_scores = bool(getattr(args, "dump_seg_scores", None))
    model.seg_probe = ({"gpu_base": _gb_probe, "n_cls": 6,
                        "seg_rank": [], "top1": [], "gold": []} if _GPU_REL else None)
    if _GPU_REL and _dump_scores:
        # B7/hybrid 的图通道: 过滤前 GPU 段分数。只有显式给 --dump-seg-scores 才建这三个桶,
        # 不给 ⇒ predict() 里的导出分支整块跳过 ⇒ 与论文口径逐位相同(由 G-SCORE 验证)。
        model.seg_probe.update({"raw_seg": [], "raw_head": [], "raw_gold": []})
        print(f"[dump-seg-scores] 开启过滤前 GPU 段分数导出  mode={mode}  "
              f"(只读探针, 不参与任何指标计算)")
    if hasattr(model, "decoder_ob"):
        model.decoder_ob.compat_probe = ({"pred": [], "gold": [], "head": [], "logit": []}
                                         if _GPU_REL else None)

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        if use_cuda:
            test_triples_input = test_triples_input.to(args.gpu)
        # 时序口径先验: 这个快照读第 ts 层表(= 严格早于该天的历史)。snap_ts=None 时是 no-op。
        if snap_ts is not None:
            model.set_prior_ts(snap_ts[time_idx])
        test_triples, rank_raw, rank_filter, rank_raw_r, rank_filter_r = model.predict(
            history_glist, num_rels, static_graph, test_triples_input, use_cuda,
            all_ans_ent=all_ans_list[time_idx],
            all_ans_rel=all_ans_r_list[time_idx],
            eval_bs=1000,
        )

        # ----- 按 relation 分组收集 rank -----
        rel_ids = test_triples[:, 1].cpu()         # [2*N_snap], relation id 含逆
        # r3_drives: 正向=3, 逆向=7(正向+4)
        r3_mask = (rel_ids == 3) | (rel_ids == 7)
        r3_ranks_filter.append(rank_filter[r3_mask].cpu())
        for rel in [0, 1, 2]:
            relX_ranks[rel].append(rank_filter[rel_ids == rel].cpu())

        r1_mask = (test_triples[:, 1] == PR)
        if r1_mask.sum() > 0:
            r1_gpu_tails.append(test_triples[r1_mask, 2].cpu())
            r1_gpu_ranks.append(rank_filter[r1_mask].cpu())
            r1_gpu_heads.append(test_triples[r1_mask, 0].cpu())

        mrr_snap = torch.mean(1.0 / rank_raw.float()).item()
        mrr_filter_snap = torch.mean(1.0 / rank_filter.float()).item()
        mrr_snap_r = torch.mean(1.0 / rank_raw_r.float()).item()
        mrr_filter_snap_r = torch.mean(1.0 / rank_filter_r.float()).item()

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
    
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    # ----- 按关系类型分组打印 MRR -----
    def stat_ranks_by_rel(ranks_list, name):
        if len(ranks_list) == 0:
            return None
        all_ranks = torch.cat(ranks_list).float()
        mrr  = torch.mean(1.0 / all_ranks).item()
        h1   = torch.mean((all_ranks <= 1).float()).item()
        h3   = torch.mean((all_ranks <= 3).float()).item()
        h10  = torch.mean((all_ranks <= 10).float()).item()
        return mrr, h1, h3, h10

    print()
    print("=" * 30 + " Per-Relation filter MRR " + "=" * 30)
    if _GPU_REL:
        r3_stat = stat_ranks_by_rel(r3_ranks_filter, "r3_drives")
        if r3_stat:
            mrr, h1, h3, h10 = r3_stat
            print(f"  r3_drives (attr, id=3+7):  MRR={mrr:.6f}  H@1={h1:.6f}  H@3={h3:.6f}  H@10={h10:.6f}")
        else:
            print("  r3_drives: no data")
        rel_names = {0: "placement", 1: "r1_suits", 2: "r2_requires"}
        for rel in [0, 1, 2]:
            st = stat_ranks_by_rel(relX_ranks[rel], rel_names[rel])
            if st:
                mrr, h1, h3, h10 = st
                print(f"  {rel_names[rel]:25s} MRR={mrr:.6f}  H@1={h1:.6f}  H@3={h3:.6f}  H@10={h10:.6f}")
            else:
                print(f"  {rel_names[rel]}: no data")
    else:
        # 非 xCAD 数据集(MIT): relX_ranks 只有 0..2 三个桶, 按实际关系数打印正向桶
        for rel in range(min(num_rels, 3)):
            st = stat_ranks_by_rel(relX_ranks[rel], f"rel{rel}")
            if st:
                mrr, h1, h3, h10 = st
                print(f"  rel{rel} (fwd) {mrr:.6f} | H@1 {h1:.6f} H@3 {h3:.6f} "
                      f"H@10 {h10:.6f}")
    print("=" * 70)

    if not _GPU_REL and r1_gpu_tails:
        generic_report(torch.cat(r1_gpu_heads), torch.cat(r1_gpu_tails),
                       torch.cat(r1_gpu_ranks), PR, history_list, window_snaps,
                       test_list, num_nodes, mode,
                       getattr(args, "dump_preds", None) if mode == "test" else None,
                       # 闭式基线必须和模型用同一个候选段，否则两张表不是同一把尺子
                       candidate_universe=getattr(model, "ent_segment", None))

    if _GPU_REL and r1_gpu_tails:
        tails_cat = torch.cat(r1_gpu_tails)
        ranks_cat = torch.cat(r1_gpu_ranks)
        _gb = getattr(model, 'gpu_base', 0)
        # GPU_TYPES 顺序固定 [CPU, T4, MISC, P100, V100, V100M32]; CPU(_gb+0) 从不是 r1 真值
        GPU_LABELS = {i: n for i, n in
                      enumerate(["CPU", "T4", "MISC", "P100", "V100", "V100M32"], start=_gb)}
        h1_list, n_list = [], []
        for k in range(1, 6):
            gpu_id = _gb + k
            mask = (tails_cat == gpu_id)
            n = mask.sum().item()
            h1 = (ranks_cat[mask] == 1).float().mean().item() if n > 0 else 0.0
            h1_list.append(h1)
            n_list.append(n)
            print(f"    {GPU_LABELS[gpu_id]:>8}: Hits@1={h1:.4f}  (n={n})")
        macro = sum(h1_list) / 5
        print(f"  Per-GPU Hits@1 Macro (unweighted) = {macro:.4f}")
        if sum(n_list) != len(tails_cat):
            print(f"  !! [id-space] GPU 段覆盖不全: 落入 5 类 {sum(n_list)}/{len(tails_cat)}"
                  f" -> gpu_base={_gb} 判定可能有误, 本行 macro 不可用")

        # ----- R1-1 决定性实验: 按 head 是否在历史里复发分层 -----
        # 分层器只用 history 侧 id, 不使用 test 真值, 因此是无标签的。
        stratified_report(torch.cat(r1_gpu_heads), tails_cat, ranks_cat,
                          history_list, window_snaps, _gb, mode)
    else:
        print("  [r1_suits per-GPU] no data")

    # ────── R1-1 同口径对照: GPU 段内排名 + compat 头直读 ──────
    # 全实体 H@1 惩罚"top-1 落到非 GPU 实体", 而 5 类分类不会; 这里把两种口径都测出来,
    # 让 Ours 与 sklearn 的数字可以逐位对齐(R1-1 §4.5)。
    _gb2 = getattr(model, "gpu_base", 0)
    _L = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
    print(f"\n{'='*28} 同口径对照 [{mode}] {'='*28}")
    _sp = getattr(model, "seg_probe", None)
    _seg_tensors = None
    _raw_tensors = None
    if _sp and _sp["seg_rank"]:
        sr, top1, gold = (torch.cat(_sp[k]) for k in ("seg_rank", "top1", "gold"))
        _seg_tensors = (sr, top1, gold)
        in_seg = ((top1 >= _gb2) & (top1 < _gb2 + 6)).float().mean().item()
        print(f"  r1 正向边 n={len(sr)}  无约束 top-1 落在 GPU 段的比例 = {in_seg:.6f}")
        print(f"  GPU 段内(filtered) H@1 = {(sr == 0).float().mean().item():.6f}"
              f"   —— 与 5 类分类 accuracy 同口径")
        _hits = []
        for k in range(1, 6):
            m = (gold == _gb2 + k)
            h = (sr[m] == 0).float().mean().item() if m.any() else 0.0
            _hits.append(h)
            print(f"    {_L[k]:>8}: 段内 H@1={h:.4f} (n={int(m.sum())})")
        print(f"    段内 per-GPU macro5 = {sum(_hits)/5:.4f}")

        # ── B7/hybrid: 过滤前分数的同口径读数(G-SCORE 门禁读这一段) ──
        # 期望: 与上面的 filtered 读数逐位相同(11,830/11,836 条 (head,ts) 键只有 1 个候选类型,
        #       过滤对这些行是恒等操作)。不一致的边数必须打印出来, 不能默默吞掉。
        if _sp.get("raw_seg"):
            rs = torch.cat(_sp["raw_seg"])
            rh, rg = (torch.cat(_sp[k]) for k in ("raw_head", "raw_gold"))
            assert rs.size(0) == len(sr) == len(rh), (
                f"过滤前导出行数 {rs.size(0)} != 探针行数 {len(sr)} —— 导出条件与探针条件不一致")
            assert bool((rg == gold).all()), "过滤前导出的 gold 序列与探针 gold 序列不同 ⇒ 行序错配"
            rc = rs.argmax(1)                                   # 段内 0..5 (含 CPU 列)
            rc_ok = (rc == (gold - _gb2)).float().mean().item()
            _rh2 = []
            for k in range(1, 6):
                m = (gold == _gb2 + k)
                _rh2.append((rc[m] == k).float().mean().item() if m.any() else 0.0)
                print(f"    {_L[k]:>8}: 段内(过滤前) H@1={_rh2[-1]:.4f} (n={int(m.sum())})")
            print(f"  [G-SCORE] GPU 段内(过滤前) top-1 acc = {rc_ok:.6f}  "
                  f"macro5 = {sum(_rh2)/5:.6f}   对照 filtered 读数 "
                  f"{(sr == 0).float().mean().item():.6f} / {sum(_hits)/5:.6f}")
            _post_cls = top1 - _gb2
            _dis = (rc != _post_cls)
            print(f"  [G-SCORE] 过滤前 vs 过滤后 段内 argmax 不一致边数 = "
                  f"{int(_dis.sum())}/{len(sr)}  (过滤后无约束 top-1 落在段外的边数 = "
                  f"{int(((_post_cls < 0) | (_post_cls >= 6)).sum())})")
            _raw_tensors = (rs, rh, rg, rc)
        model.seg_probe = None

    _cp = getattr(getattr(model, "decoder_ob", None), "compat_probe", None)
    _compat_tensors = None
    if _cp and _cp["pred"]:
        pred, gld, hds = (torch.cat(_cp[k]) for k in ("pred", "gold", "head"))
        _compat_tensors = (pred, gld, hds)
        print(f"\n  compat 头直读(6 类 argmax, 与 sklearn 同口径): n={len(pred)} "
              f"acc={(pred == gld).float().mean().item():.6f} "
              f"预测成 CPU 的比例={(pred == 0).float().mean().item():.6f}")
        _hits = []
        for k in range(1, 6):
            m = (gld == k)
            h = (pred[m] == k).float().mean().item() if m.any() else 0.0
            _hits.append(h)
            print(f"    {_L[k]:>8}: H@1={h:.4f} (n={int(m.sum())})")
        print(f"    compat 头 per-GPU macro5 = {sum(_hits)/5:.4f}")
        _full_r1, _full_any = head_sets(history_list)
        _win_r1, _ = head_sets(window_snaps)
        _hn = hds.numpy()
        for _tag, _s in (("FULL&r1", _full_r1), ("WIN&r1", _win_r1), ("FULL&any", _full_any)):
            _rec = np.isin(_hn, list(_s))
            for _sub, _m in (("复发", _rec), ("全新", ~_rec)):
                if _m.sum() == 0:
                    print(f"    [{_tag} {_sub}] n=0")
                    continue
                _mm = torch.from_numpy(_m)
                print(f"    [{_tag} {_sub}] n={int(_m.sum()):6d}  "
                      f"compat头 acc={(pred[_mm] == gld[_mm]).float().mean().item():.6f}")
        model.decoder_ob.compat_probe = None

        # ── compat argmax 与 模型段内 top-1 的一致性 ──
        # 若两者在几乎所有 r1 边上落到同一个 GPU 实体, 则 r1 的排序完全由 6 类属性头决定,
        # 图编码器/时间演化在 r1 上没有留下任何决策权(R1-1 最硬的因果隔离证据)。
        if _seg_tensors is not None:
            sr, top1, gold = _seg_tensors
            if len(sr) == len(pred):
                _align = (gold == gld + _gb2).float().mean().item()
                _model_cls = top1 - _gb2                      # 无约束 top-1 的 GPU 类(段外则为负/>=6)
                _same = (_model_cls == pred)
                _in_seg = (_model_cls >= 0) & (_model_cls < 6)
                print(f"\n  [一致性自检] 两条探针的 gold 序列对齐率 = {_align:.6f}"
                      f"   (<0.999 则下面的同点率不可用)")
                if _align > 0.999:
                    print(f"  compat argmax 与 模型段内 top-1 是同一个 GPU 的边 = "
                          f"{_same.float().mean().item():.6f} ({int(_same.sum())}/{len(pred)})")
                    print(f"    其中无约束 top-1 落在段内的 {int(_in_seg.sum())} 条上: "
                          f"{_same[_in_seg].float().mean().item():.6f}")
                    _mc = (sr == 0)          # 模型答对(段内第一)
                    _cc = (pred == gld)      # compat 头答对
                    _agree = (_mc == _cc).float().mean().item()
                    print(f"    两者\"对不对\"一致率 = {_agree:.6f}   "
                          f"都对={int((_mc & _cc).sum())}  只 compat 对={int((~_mc & _cc).sum())}  "
                          f"只模型对={int((_mc & ~_cc).sum())}  都错={int((~_mc & ~_cc).sum())}")
                    for _tag, _s in (("FULL&r1", _full_r1), ("WIN&r1", _win_r1)):
                        _rec = torch.from_numpy(np.isin(_hn, list(_s)))
                        for _sub, _m in (("复发", _rec), ("全新", ~_rec)):
                            if _m.sum() == 0:
                                continue
                            print(f"    [{_tag} {_sub}] n={int(_m.sum()):6d}  同点率="
                                  f"{_same[_m].float().mean().item():.6f}  对错一致率="
                                  f"{(_mc[_m] == _cc[_m]).float().mean().item():.6f}")
            else:
                print(f"\n  [一致性自检] 两条探针长度不同 ({len(sr)} vs {len(pred)}), 跳过同点率")
    # ────── R1-1 预测类别混淆矩阵 ──────
    # 用途: 分层表里"复发层 V100 150 条全错"这种读数, 需要知道模型把它们答成了什么,
    #       才能区分"频率先验失败"(整类答成 MISC/T4)与"记忆失败"(答成别的少数类)。
    def _confusion(_row, _col, _title):
        print(f"\n  {_title}   (行=真值类, 列=预测类, 数字=边数)")
        print("          " + "".join(f"{x:>8}" for x in _L))
        for k in range(6):
            _m = (_row == k)
            if int(_m.sum()) == 0:
                continue
            _cnt = torch.bincount(_col[_m].clamp(0, 5), minlength=6).tolist()
            print(f"    {_L[k]:>6}" + "".join(f"{c:>8d}" for c in _cnt) + f"   | n={int(_m.sum())}")

    if _compat_tensors is not None:
        _p2, _g2, _h2 = _compat_tensors
        _confusion(_g2, _p2, "compat 属性头直读")
        _full_r1b, _full_anyb = head_sets(history_list)
        _win_r1b, _ = head_sets(window_snaps)
        _hn2 = _h2.numpy()
        for _tag, _s in (("FULL&r1", _full_r1b), ("WIN&r1", _win_r1b)):
            _rec = torch.from_numpy(np.isin(_hn2, list(_s)))
            for _sub, _m in (("复发", _rec), ("全新", ~_rec)):
                if int(_m.sum()) >= 50:
                    _confusion(_g2[_m], _p2[_m], f"compat 属性头直读 [{_tag} {_sub}]")
    if _seg_tensors is not None:
        _sr3, _t13, _g3 = _seg_tensors
        _cls3 = _t13 - _gb2
        _in3 = (_cls3 >= 0) & (_cls3 < 6)
        print(f"\n  模型无约束 top-1 的 GPU 类分布(段外计 {int((~_in3).sum())} 条):")
        _cnt3 = torch.bincount(_cls3[_in3].clamp(0, 5), minlength=6).tolist()
        print("          " + "".join(f"{x:>8}" for x in _L))
        print(f"    {'预测':>6}" + "".join(f"{c:>8d}" for c in _cnt3))
        _confusion(_g3 - _gb2, _cls3.clamp(0, 5), "模型无约束 top-1 当作 6 类预测")
    # ────── R1-1 反驳用逐边落盘: 冲突分层 / 构造性 cohort 都靠它, 不再需要重跑模型 ──────
    # 只在最终 test 那一次写文件; 内容全是 id/类别/排名, 不含任何答案之外的信息。
    if mode == "test" and _GPU_REL and getattr(args, "dump_preds", None) and len(r1_gpu_heads) > 0:
        _row = {
            "heads":        torch.cat(r1_gpu_heads).cpu(),
            "tails":        tails_cat.cpu(),
            "filt_rank":    ranks_cat.cpu(),
            "gpu_base":     int(_gb2),
        }
        if _seg_tensors is not None:
            _sr, _t1, _gd = _seg_tensors
            _row.update({"seg_rank": _sr.cpu(), "model_top1": _t1.cpu(), "seg_gold": _gd.cpu()})
        if _compat_tensors is not None:
            _pd, _gl, _hd = _compat_tensors
            _row.update({"compat_pred": _pd.cpu(), "compat_gold": _gl.cpu(),
                         "compat_head": _hd.cpu()})
        if _raw_tensors is not None:
            _row.update({"raw_seg": _raw_tensors[0].cpu(), "raw_argmax": _raw_tensors[3].cpu()})
        torch.save(_row, args.dump_preds)
        print(f"\n  [dump-preds] {args.dump_preds}  n={len(_row['heads'])}  "
              f"keys={sorted(_row.keys())}")

    # ── B7/hybrid: 图通道(过滤前 GPU 段分数)落盘 ──
    # 独立于 --dump-preds: 混合器要在【非测试】边(valid)上拟合, 那一次调用 mode="train"
    # 走不到上面的分支, 所以这里按 mode 分成 .test.pt / .valid.pt 两个文件。
    _ds = getattr(args, "dump_seg_scores", None)
    if _ds and _GPU_REL:
        if _raw_tensors is not None:
            _rs, _rh, _rg, _rc = _raw_tensors
            _h_cat = torch.cat(r1_gpu_heads).cpu()
            assert len(_h_cat) == len(_rh) and bool((_h_cat == _rh).all()), (
                f"过滤前导出的 head 序列与主报表的 r1 head 序列不一致 "
                f"({len(_rh)} vs {len(_h_cat)}) ⇒ 行序不可信, 禁止用于 hybrid")
            _p = f"{_ds}.{'test' if mode == 'test' else 'valid'}.pt"
            torch.save({"raw_seg": _rs, "heads": _rh, "gold": _rg, "seg_argmax": _rc,
                        "gpu_base": int(_gb2), "mode": mode, "n_rows": int(_rs.size(0))}, _p)
            print(f"\n  [dump-seg-scores] {_p}  n={int(_rs.size(0))}  "
                  f"scores={tuple(_rs.shape)}  mode={mode}")
        else:
            print("\n  !! [dump-seg-scores] 旗标已开但没拿到分数 ⇒ 本次未落盘; "
                  "不要把同前缀的旧文件当本轮产物")

    print("=" * 76)

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    set_seed(args.seed)
    print(f"[seed] {args.seed}")

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    # ── (H) 时序口径先验: 每个快照对应的 ts(数据文件第 4 列) ──
    # split_by_time 只把 ts 用来分组、存下来时丢掉了第 4 列(rgcn/utils.py: snapshot.append(train[:3])),
    # 所以模型侧拿不到"这批三元组属于哪一天"。这里按【和 split_by_time 完全相同的连续段口径】把
    # 每个快照的 ts 复原出来, 长度必须和快照数一致、且严格递增 —— 不对就说明数据没按时间排序,
    # 时序表会静默读错层, 所以当场断言而不是容错。
    def snap_ts(split):
        out, last = [], None
        for row in split:
            t = int(row[3])
            if t != last:
                out.append(t)
                last = t
        return out

    train_ts = valid_ts = test_ts = None
    if args.prior_ts:
        train_ts, valid_ts, test_ts = snap_ts(data.train), snap_ts(data.valid), snap_ts(data.test)
        for _name, _ts, _snaps in (("train", train_ts, train_list), ("valid", valid_ts, valid_list),
                                   ("test", test_ts, test_list)):
            assert len(_ts) == len(_snaps), f"{_name}: ts 段数 {len(_ts)} != 快照数 {len(_snaps)}"
            assert all(a < b for a, b in zip(_ts, _ts[1:])), f"{_name}: ts 不严格递增 {_ts}"
        print(f"[prior-ts] 快照 ts: train {train_ts[0]}..{train_ts[-1]}({len(train_ts)})  "
              f"valid {valid_ts[0]}..{valid_ts[-1]}({len(valid_ts)})  "
              f"test {test_ts[0]}..{test_ts[-1]}({len(test_ts)})")

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # ── GPU 段起点自动识别 ──
    # 旧 data/xcad     : 段内【局部】编号, GPU 落在 id 0..5(四套编号互相撞号)
    # 新 data/xcad_v2  : 全局四段编号, GPU 落在 num_nodes-6 .. num_nodes-1
    # 按数据实际取值判定, 使同一份代码能在两套编号上跑对照实验。
    _r1_tails = [t[2] for _split in (data.train, data.valid, data.test) for t in _split if t[1] == 1]
    _r1_tail_max = max(_r1_tails) if _r1_tails else None
    gpu_base = num_nodes - 6 if (_r1_tail_max is not None and _r1_tail_max >= num_nodes - 6) else 0
    print(f"[id-space] num_nodes={num_nodes} r1_tail_max={_r1_tail_max} -> gpu_base={gpu_base} "
          f"({'GLOBAL 四段编号' if gpu_base else 'LEGACY 局部编号'})")

    # ── compat 辅助 CE 的类别加权(2026-11-18 放行) ──
    # 计数只用【train】的 r1 正向边, 与 rrgcn.get_loss 里 gpu_labels = tail - gpu_base 同口径;
    # valid/test 的标签一律不参与, 避免信息泄漏。
    compat_class_weights = None
    if args.compat_class_weight != "none":
        assert args.compat, "--compat-class-weight 需要配合 --compat 使用(compat 头关闭时它无对象可加权)"
        assert args.compat_aux_weight > 0, "--compat-class-weight 需要 --compat-aux-weight > 0"
        _cw_lab = [int(t[2]) - gpu_base for t in data.train if t[1] == 1]
        assert _cw_lab, "--compat-class-weight: train 里没有 r1 正向边"
        assert min(_cw_lab) >= 0 and max(_cw_lab) <= 5, (
            f"--compat-class-weight: train r1 标签换算越界 [{min(_cw_lab)}, {max(_cw_lab)}] (gpu_base={gpu_base})")
        _cw_cnt = np.bincount(_cw_lab, minlength=6).astype(np.float64)
        _cw_n = _cw_cnt.sum()
        _cw_k = int((_cw_cnt > 0).sum())
        _w = np.zeros(6, dtype=np.float64)
        _nz = _cw_cnt > 0
        _cw_cls = np.nonzero(_nz)[0]          # 出现过的类别索引(全 6 类里的子集)
        _raw = _cw_n / (_cw_k * _cw_cnt[_nz])               # 逆频率: 每类总权重相等
        if args.compat_class_weight == "sqrt":
            _raw = np.sqrt(_raw)
        _cap = float(args.compat_class_weight_cap)
        assert _cap >= 1.0, f"--compat-class-weight-cap={_cap} < 1, 上限低于均匀权重没有意义"
        _w[_nz] = np.minimum(_raw, _cap)
        _scale = float((_cw_cnt * _w).sum() / _cw_n)         # 计数加权均值
        assert _scale > 0, "类别权重全为 0"
        _w = _w / _scale                                     # 归一化: 保持辅助 loss 的有效尺度
        compat_class_weights = torch.as_tensor(_w, dtype=torch.float32)
        print(f"[class-weight] mode={args.compat_class_weight} cap={_cap}  "
              f"train r1 counts={_cw_cnt.astype(int).tolist()}  "
              f"present classes {_cw_cls.astype(int).tolist()} (K_present={_cw_k}, n={int(_cw_n)})")
        print(f"[class-weight] raw(present only)={np.array2string(_raw, precision=4, floatmode='fixed')}  "
              f"capped&renorm w(all 6)={np.array2string(_w, precision=4, floatmode='fixed')}")
        # 每类对辅助 loss 的贡献占比 = c_i*w_i / sum(c*w) —— 这才是"极少数类会不会喧宾夺主"的实际量
        _share = _cw_cnt * _w
        _share = _share / _share.sum()
        print(f"[class-weight] loss share={np.array2string(_share, precision=4, floatmode='fixed')}  "
              f"(uniform share would be {1.0/_cw_k:.4f} per present class, "
              f"raw-frequency share {float(_cw_cnt[_nz].min()/_cw_n):.4f}~{float(_cw_cnt[_nz].max()/_cw_n):.4f})")

    # ── §23/E1 输出层先验: 偏置只能加在【评测关系的 tail 段】列上, 段边界按数据实际取值判 ──
    prior_lo = prior_n = 0
    if args.base_prior or args.head_prior:
        _pt = [t[2] for _split in (data.train, data.valid, data.test)
               for t in _split if t[1] == args.probe_rel]
        assert _pt, f"--base-prior/--head-prior: rel{args.probe_rel} 在数据里没有边"
        _plo, _phi = min(_pt), max(_pt)
        assert _phi == num_nodes - 1 and len(set(_pt)) == _phi - _plo + 1, (
            f"--base-prior/--head-prior: rel{args.probe_rel} 的 tail 段 [{_plo},{_phi}] 不是末尾连续段 "
            f"(不同 tail 实体 {len(set(_pt))} 个, num_nodes={num_nodes}) => 偏置列无从对齐")
        prior_lo, prior_n = _plo, _phi - _plo + 1
        print(f"[prior] 偏置列段 = [{prior_lo}, {prior_lo + prior_n}) "
              f"= rel{args.probe_rel} 的 {prior_n} 个 tail 实体 (num_nodes={num_nodes}) "
              f"base={args.base_prior} head={args.head_prior})")

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    # 实验开关标识
    het_flag    = f"-het{args.n_msg_basis}" if args.hetero else ""
    nf_flag     = "-nf" if args.node_feat else "-nonf"
    fusion_flag = f"-f{args.feat_fusion[:4]}" if args.node_feat else ""
    scope_flag  = f"-s{args.feat_scope[:4]}" if args.node_feat else ""
    alpha_flag  = f"-afix{args.feat_alpha_fixed}" if args.node_feat and args.feat_alpha_fixed is not None else ("-alearn" if args.node_feat else "")
    rd_flag     = "-rd" if args.rel_decay else ""
    dfeat_flag  = "-df" if args.decoder_feat else "-nodf"
    compat_flag = "-compat" if args.compat else ""
    staticrel_flag = "-staticrel" if args.no_rel_evolution else ""
    baseemb_flag  = "-baseemb" if args.base_emb else ""
    basefeat_flag = "-basefeat" if args.base_feat else ""
    baseprior_flag = "-baseprior" if args.base_prior else ""
    headprior_flag = "-headprior" if args.head_prior else ""
    priorgate_flag = "-priorgate" if args.prior_gate else ""
    priorts_flag = "-priorts" if args.prior_ts else ""

    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}-seed{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu,
                het_flag, nf_flag, fusion_flag, scope_flag, alpha_flag, rd_flag, dfeat_flag, compat_flag, staticrel_flag, baseemb_flag, basefeat_flag,
                baseprior_flag, headprior_flag, priorgate_flag, priorts_flag, args.seed)
    model_name = model_name + args.stat_suffix
    # models/ 是运行时目录, git 不保存空目录 => 干净克隆里它不存在。不建的话, 训练会在
    # 第一次保存最优 checkpoint 时才抛 FileNotFoundError, 白跑一轮。
    os.makedirs('../models', exist_ok=True)
    model_state_file = '../models/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis,
                        use_hetero=args.hetero,
                        n_msg_basis=args.n_msg_basis,
                        use_node_feat=args.node_feat,
                        feat_dir=args.feat_dir,
                        feat_fusion=args.feat_fusion,
                        feat_scope=args.feat_scope,
                        feat_alpha_fixed=args.feat_alpha_fixed,
                        use_decoder_feat=args.decoder_feat,
                        decoder_feat_lambda=args.decoder_feat_lambda,
                        use_rel_decay=args.rel_decay,
                        use_rel_evolution=not args.no_rel_evolution,
                        use_compat=args.compat,
                        compat_lambda=args.compat_lambda,
                        compat_aux_weight=args.compat_aux_weight,
                        compat_class_weights=compat_class_weights,
                        gpu_base=gpu_base,
                        use_base_emb=args.base_emb,
                        base_map_path=(args.base_map if args.base_map else
                                       "../data/" + args.dataset + "/base_of.npy"),
                        use_base_feat=args.base_feat,
                        base_feat_path=(args.base_feat_path if args.base_feat_path else
                                        "../data/" + args.dataset + "/base_feat.npy"),
                        base_alpha_init=args.base_alpha_init,
                        use_base_prior=args.base_prior,
                        base_prior_path=(args.base_prior_path if args.base_prior_path else
                                         "../data/" + args.dataset + "/base_prior.npy"),
                        base_prior_alpha_init=args.base_prior_alpha_init,
                        use_head_prior=args.head_prior,
                        head_prior_path=(args.head_prior_path if args.head_prior_path else
                                         "../data/" + args.dataset + "/head_prior.npy"),
                        head_prior_alpha_init=args.head_prior_alpha_init,
                        use_prior_gate=args.prior_gate,
                        use_prior_ts=args.prior_ts,
                        base_prior_ts_path=(args.base_prior_ts_path if args.base_prior_ts_path else
                                            "../data/" + args.dataset + "/base_prior_ts.npy"),
                        head_prior_ts_path=(args.head_prior_ts_path if args.head_prior_ts_path else
                                            "../data/" + args.dataset + "/head_prior_ts.npy"),
                        prior_lo=prior_lo, prior_n=prior_n)

    # 候选集口径(诊断用)：test 阶段实体排名在哪个类型段里做。None = 全实体(论文主表口径)。
    model.ent_segment = tuple(args.ent_segment) if getattr(args, "ent_segment", None) else None
    if model.ent_segment is not None:
        _lo, _hi = model.ent_segment
        print(f"[cand-set] 实体排名 = 正向 tail 段 [{_lo}, {_hi}] / 逆向 head 段 [0, {_lo}) "
              f"—— 非主表口径，闭式基线会走同一段(见 generic_strata 的 universe)；"
              f"总表 filter_ent 混合正逆两段不可读，只读 rel{args.probe_rel} 分层表")

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list, 
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph, 
                                                            "test", snap_ts=test_ts)
        if getattr(args, "dump_seg_scores", None):
            # B7/hybrid 的混合器必须拟合在模型【没见过标签】的非测试边上。这里复用同一进程、
            # 同一 checkpoint, 只做一次 valid 打分: history = train_list(不含 valid),
            # 与训练期选型那一次调用完全同一口径 ⇒ 图通道与画像通道在 valid 上的信息视界对称。
            # 报表只用于自检(G-ALIGN), 不参与任何已发表数字, 也不覆盖 checkpoint。
            print("\n" + "=" * 30 + " B7: valid 侧图通道打分(mixer 拟合集) " + "=" * 30)
            test(model,
                 train_list,
                 valid_list,
                 num_rels,
                 num_nodes,
                 use_cuda,
                 all_ans_list_valid,
                 all_ans_list_r_valid,
                 model_state_file,
                 static_graph,
                 mode="train", snap_ts=valid_ts)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
        # 以前这里会直接落到 return 上抛 UnboundLocalError(2026-11-12 探针校验时暴露),
        # 明确退出, 让批量 --test 复跑脚本能报出是哪一臂的 checkpoint 没对上。
        sys.exit(1)
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                # 训练时 snapshot 边采样:只对 loss 目标 triples (output[0]) 采样,
                # history_glist 保持完整。get_loss 内部仍按 124891 实体算全实体 CE,
                # 只是 B 从 ~13万 降到 2048,避免 [B, num_ents] score 矩阵 OOM。
                cur_triples = output[0]
                if cur_triples.size(0) > MAX_TRIPLES_PER_STEP:
                    perm = torch.randperm(cur_triples.size(0), device=cur_triples.device)[:MAX_TRIPLES_PER_STEP]
                    cur_triples = cur_triples[perm]

                # 时序口径先验: 训练同样只能看"严格早于本快照"的历史(train_ts 与 train_list 同序)。
                # train_ts=None(不开 --prior-ts)时 set_prior_ts 是 no-op, 行为逐字等于 §23–§25。
                if train_ts is not None:
                    model.set_prior_ts(train_ts[train_sample_num])
                loss_e, loss_r, loss_static = model.get_loss(history_glist, cur_triples, static_graph, use_cuda)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                # === 撤 AMP:还原成普通 backward + clip + step + zero_grad ===
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    use_cuda, 
                                                                    all_ans_list_valid, 
                                                                    all_ans_list_r_valid, 
                                                                    model_state_file, 
                                                                    static_graph, 
                                                                    mode="train", snap_ts=valid_ts)
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            static_graph, 
                                                            mode="test", snap_ts=test_ts)
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 weight decay for Adam")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")
    parser.add_argument("--hetero", action="store_true", default=False,
                        help="开启异质 RGCN: per-relation-type basis 分解变换")
    parser.add_argument("--n-msg-basis", type=int, default=2,
                        help="异质 RGCN 的 message basis 数量 (默认2)")
    # === node-feat: 类型感知异质表征 ===
    parser.add_argument("--node-feat", action="store_true", default=False,
                        help="开启类型感知异质表征: 加载 node_feat_algo/data/compute.npy, 用投影层替代/融合 dynamic_emb")
    parser.add_argument("--feat-dir", type=str, default=None,
                        help="node_feat_algo.npy / node_feat_data.npy / node_feat_compute.npy 所在目录 "
                             "(相对 cwd); 不给时取 ../data/<dataset>, 与 -d 指向同一份编号")
    parser.add_argument("--feat-fusion", type=str, default="residual", choices=["residual", "replace"],
                        help="特征注入方式: 'residual' = dynamic_emb + alpha*feat_proj (默认); 'replace' = 直接用 feat_proj")
    parser.add_argument("--feat-scope", type=str, default="all", choices=["all", "algo"],
                        help="特征注入范围: 'all' = 三类节点全注入(默认); 'algo' = 只融合 Algorithm 段特征")
    parser.add_argument("--feat-alpha-fixed", type=float, default=None,
                        help="固定 feat_alpha 为该值且不训练; 不给则可学习(默认)")
    parser.add_argument("--decoder-feat", action="store_true", default=False,
                        help="decoder 端注入节点特征偏置(ConvTransE score + lambda*feat_bias)")
    parser.add_argument("--decoder-feat-lambda", type=float, default=None,
                        help="decoder-feat 的 lambda 权重: None=可学习(默认), 数值=固定")
    parser.add_argument("--rel-decay", action="store_true", default=False,
                        help="开启关系感知时间衰减: 历史越老的快照,按每种关系各自的 decay rate 衰减消息")
    # === R1-1 具名消融①: 去掉 GRU 关系演化 ===
    parser.add_argument("--no-rel-evolution", action="store_true", default=False,
                        help="消融: 跳过 relation GRUCell, 每个快照都用静态 emb_rel 做消息传递(默认关闭=原行为)")
    # === compat: 算法画像驱动的 GPU 兼容性头 ===
    parser.add_argument("--compat", action="store_true", default=False,
                        help="开启兼容性头: 算法画像(r1_suits 边 头算法 GPU 类型)以可微方式接进 r1 打分")
    parser.add_argument("--compat-lambda", type=float, default=None,
                        help="compat 权重: None=可学习(默认), 数值=固定该值")
    parser.add_argument("--compat-aux-weight", type=float, default=0.5,
                        help="compat 辅助分类 loss 权重 (默认0.5)")
    parser.add_argument("--compat-class-weight", type=str, default="none",
                        choices=["none", "sqrt", "inv"],
                        help="compat 辅助 CE 的类别加权: none=原行为(逐位不变); "
                             "sqrt=逆频率开方; inv=逆频率. 计数只取【train】的 r1 正向边, "
                             "加权后按计数重新归一化使有效 loss 尺度不变(不改变 beta 的含义)")
    parser.add_argument("--compat-class-weight-cap", type=float, default=5.0,
                        help="类别权重的上限, 防止极少数类主导整个辅助 loss (默认5.0)")
    parser.add_argument("--dump-preds", type=str, default=None, metavar="PATH",
                        help="最终 test 时把 r1 每条边的 head/tail/段内排名/compat argmax 落盘到 PATH(供 r1_conflict.py 离线分层)")
    parser.add_argument("--dump-seg-scores", type=str, default=None, metavar="PREFIX",
                        help="B7/hybrid 用: 落盘 r1 正向边【过滤前】的 GPU 段 6 列分数到 PREFIX.test.pt / PREFIX.valid.pt"
                             "(后者是同进程追加的一次 valid 打分, history 只到 train 末)。"
                             "纯只读探针, 不开本旗标时 predict()/test() 的数值路径与论文口径逐位相同")
    # === 第二份数据集(MIT Supercloud)评测口径开关 ===
    parser.add_argument("--probe-rel", type=int, default=1, metavar="REL_ID",
                        help="R1-1 的探针/分层/落盘 key 在哪个【正向】关系上。"
                             "xCAD=1(r1_suits, tail 是 6 个 GPU 型号段, 默认=原行为不改); "
                             "只有 1 个关系的数据集(MIT: rel0=placement, tail 是节点身份)用 0, "
                             "此时 6 类 GPU 报表整段跳过, 改走 generic_strata(同 filtered 口径)。")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility")
    # === 候选集口径诊断开关(不改模型、不改训练，只改 test 阶段实体排名的范围) ===
    parser.add_argument("--ent-segment", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                        help="实体排名只在闭区间 [LO,HI] 这段(tail 类型段)里做，逆向行用 [0,LO) 的 head 段；"
                             "默认 None = 现行论文的全实体口径。用途: 把'模型不及闭式基线'拆成"
                             "'候选空间太宽'与'真没学到判别'两部分(第二数据集诊断, screening §18.4-B)。")

    # === base 共享表征开关(screening §20 的修法; 只改 head 表征的构成, 输出层/训练目标/超参都不动) ===
    parser.add_argument("--base-emb", action='store_true', default=False,
                        help="head 表征 = dynamic_emb[id] + base_emb[base_of[id]]。"
                             "用于 head id 大多从未在历史出现的数据集(maas 84.8%% 的测试边): "
                             "没学过的 id 向量至少继承父节点(base 模型)的可训练表征。"
                             "tail 段与无父实体走 pad 索引并被 mask 置 0 => 可打分实体表征逐位不变。")
    parser.add_argument("--base-map", type=str, default=None, metavar="PATH",
                        help="base_of.npy (int64 [num_ents], pad=n_bases) 路径; "
                             "默认 ../data/<dataset>/base_of.npy, 由 src/data/export_maas_base_map.py 生成")
    # === §21.3 Q1 的下一步: 把 base 的历史卡型直方图【注入】而不是让 SGD 自己发现 ===
    parser.add_argument("--base-feat", action='store_true', default=False,
                        help="head 表征 = dynamic_emb[id] + mask * alpha * Linear(base 的历史卡型分布特征)。"
                             "特征表由 src/data/export_maas_base_feat.py 从 train+valid 边生成, "
                             "内容就是 §20 那条 H@1 0.2980 的闭式通道, 模型只学投影与信任度。")
    parser.add_argument("--base-feat-path", type=str, default=None, metavar="PATH",
                        help="base_feat.npy ([n_bases+1, 31], 已 z-score, pad 行全 0); "
                             "默认 ../data/<dataset>/base_feat.npy")
    parser.add_argument("--base-alpha-init", type=float, default=1.0,
                        help="注入项的可学习标量初值(默认 1.0; 特征已 z-score, 不再需要手工定尺度)")

    # === §23: 同一张表换到【输出层】注入(screening §22 Q1/Q2: 表征层注入只吃到闭式的 1/3) ===
    parser.add_argument("--base-prior", action='store_true', default=False,
                        help="候选 logit += alpha * log1p(base 的历史卡型计数)。"
                             "只动卡型段的列, head 段一位不改; 逆向三元组的 subject 落在 pad 行 => 偏置恒 0。"
                             "与 --base-emb/--base-feat 正交(那两个改 head 表征, 这个改输出层)。")
    parser.add_argument("--base-prior-path", type=str, default=None, metavar="PATH",
                        help="base_prior.npy ([n_bases+1, n_types], log1p 计数, pad 行全 0); "
                             "默认 ../data/<dataset>/base_prior.npy, 由 src/data/export_maas_base_prior.py 生成")
    parser.add_argument("--base-prior-alpha-init", type=float, default=1.0, metavar="A",
                        help="输出层先验的可学习标量初值(默认 1.0; alpha=0 退回原模型, 大则接近闭式先验)")

    # === E1: 第二条输出层先验通道 = head 实体自己的历史卡型计数(screening §23.6) ===
    # §23 的 baseprior 臂在复发层 0.4450 而查表通道同层 0.7504 —— 缺的就是"这个 model
    # 上次落在哪张卡"。两张表各配一个标量, 信任度由 CE loss 自己定, 目标是胜过 Q3 的
    # 手工合并 oracle(全体 0.3947 / 复发层 0.7986)。
    parser.add_argument("--head-prior", action='store_true', default=False,
                        help="候选 logit += alpha_m * log1p(head 自己的历史卡型计数)。"
                             "按实体 id 直接索引, 无历史边的 head 整行 0 => 等价于不注入; "
                             "卡型实体自己的行也全 0 => 逆向三元组不受影响。"
                             "可与 --base-prior 同时开(两项相加, 各自一个标量)。")
    parser.add_argument("--head-prior-path", type=str, default=None, metavar="PATH",
                        help="head_prior.npy ([n_ents, n_types], log1p 计数, 无历史行全 0); "
                             "默认 ../data/<dataset>/head_prior.npy, "
                             "由 src/data/export_maas_base_prior.py 与 base_prior.npy 一同生成")
    parser.add_argument("--head-prior-alpha-init", type=float, default=1.0, metavar="A",
                        help="head 通道先验的可学习标量初值(默认 1.0; alpha=0 退回单通道 §23)")

    # === §25 (F): 信任度按行分配(screening §24.6b: 全局标量把两条通道的信用互相拖走) ===
    parser.add_argument("--prior-gate", action='store_true', default=False,
                        help="把 --base-prior/--head-prior 的两个全局标量 alpha 换成按行的小门 "
                             "alpha_c(v)=softplus(w_c·f(v)+b_c), f = [base 通道支持度, head 通道支持度, "
                             "head 有无历史], 全部由两张偏置表自己导出(不引入新数据)。"
                             "w 初值 0 => 起点 alpha≡1.0, 与不开门的臂初值逐字相同。")

    # === §26 (H): 先验表的时序口径(screening §25.6 探针: 静态表在训练期是自抄小抄) ===
    # 静态表 = train+valid 全历史, 于是 train 每一行的 head 通道都非空、16.4% 的训练行在该 head
    # 的任何更早快照上根本没有边而表里的 argmax 恰是它自己的 gold(79.8%) => alpha_m 学在"常开且
    # 可信"的假象上(train 可用率 1.0000 vs test 0.1525)。时序表第 t 层只统计 ts < t 的边,
    # exporter 已断言所有 t>=首个 test 快照 的层与静态表逐位相同 => test 侧条件不变,
    # 变的只有训练/验证期读到哪一层。所以每次 get_loss/predict 前必须 set_prior_ts(τ)。
    parser.add_argument("--prior-ts", action='store_true', default=False,
                        help="把 --base-prior/--head-prior 读的表换成【按快照时序】的表: "
                             "第 t 层只含 ts<t 的历史边。需要 base_prior_ts.npy / head_prior_ts.npy "
                             "(src/data/export_maas_base_prior_ts.py 生成)。关掉时逐字等于 §23–§25。")
    parser.add_argument("--base-prior-ts-path", type=str, default=None, metavar="PATH",
                        help="base_prior_ts.npy ([T, n_bases+1, n_types]); "
                             "默认 ../data/<dataset>/base_prior_ts.npy")
    parser.add_argument("--head-prior-ts-path", type=str, default=None, metavar="PATH",
                        help="head_prior_ts.npy ([T, n_ents, n_types]); "
                             "默认 ../data/<dataset>/head_prior_ts.npy")

    # === 超参扫描的 checkpoint 防撞名保护(不改任何数值行为) ===
    # model_name 只编码 flag 组合 + seed, 不编码 --compat-lambda / --compat-aux-weight /
    # --n-epochs, 于是 λ∈{1,10}、β=0 这类扫描臂会逐字覆盖旗舰 checkpoint(已出过事故)。
    # 默认空串 => 不开此 flag 时 stat 名与历史逐位相同, 老 checkpoint 照常加载。
    parser.add_argument("--stat-suffix", type=str, default="", metavar="S",
                        help="追加到 stat/checkpoint 名末尾的一段字符串, 如 --stat-suffix=-lam1"
                             "(argparse 要求等号形式, 空格形式会把 - 开头的值当成选项)。"
                             "用于扫描臂与旗舰臂同名冲突; 默认空 = 名称不变。")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    # --feat-dir 不给时跟着 -d 走: 特征矩阵和边表必须来自同一个编号空间,
    # 否则换 -d 的臂会静默读到另一份目录里的画像 (发布时的默认值写死了 ../data/xcad)。
    if args.feat_dir is None:
        args.feat_dir = "../data/" + args.dataset
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()



