#!/usr/bin/env python3
"""
diagnose_r1.py: 分析 checkpoint 对 r1_suits (relation=1) 的预测行为。

用法: python diagnose_r1.py --checkpoint ../models/<checkpoint_name> [--seed 42]

输出: 混淆矩阵, Per-GPU Hits@1, 排名分布, 算法特征差异, 预测分布/熵, score spot-check
"""
import argparse
import numpy as np
import torch
import sys, os
from collections import Counter
from math import log2
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rrgcn import RecurrentRGCN
import rgcn.utils as utils
from rgcn.utils import sort_and_rank, filter_score, build_sub_graph

# GPU type 映射: 从 export_to_regcn_format.py 的 gpu_type_pool_ids 得知,
# 顺序 = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"].
# 反映在 graph 中 entity ID = local idx (0-based).
# r1_suits tail 实体 ID 范围: 1-5 (local idx, CPU 无 r1 边)
GPU_TYPE_LABELS = {0: "CPU", 1: "T4", 2: "MISC", 3: "P100", 4: "V100", 5: "V100M32"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint 路径 (baseline 或 compat)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="评测时的历史窗口长度 (同 main.py 默认)")
    parser.add_argument("--train-history-len", type=int, default=6,
                        help="训练时的历史窗口长度, 需与 checkpoint 一致 (默认 6)")
    parser.add_argument("--compat", action="store_true", default=False,
                        help="开启 compat 兼容性头 (需与 checkpoint 一致)")
    parser.add_argument("--compat-lambda", type=float, default=None,
                        help="compat 偏置权重 (None=可学习, 数值=固定)")
    parser.add_argument("--compat-aux-weight", type=float, default=0.5,
                        help="compat 辅助分类 loss 权重")
    parser.add_argument("--feat-dir", type=str, default="../data/xcad",
                        help="节点特征矩阵所在目录")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[device] {device}")
    print(f"[checkpoint] {args.checkpoint}")
    print(f"[seed] {args.seed}")
    print(f"[compat] use_compat={args.compat}, compat_lambda={args.compat_lambda}, "
          f"compat_aux_weight={args.compat_aux_weight}")
    print(f"[compat] feat_dir={args.feat_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 常量 ──
    GPU_START = 1  # GPU type entity ID start (r1_suits tail is entities 1-5)

    # ── 加载数据 ──
    data = utils.load_data("xcad")
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # ── 加载 entity2id.txt 确认映射 ──
    e2i_path = os.path.join(os.path.dirname(__file__), "..", "data", "xcad", "entity2id.txt")
    entity_id2name = {}
    try:
        with open(e2i_path) as f:
            for line in f:
                name, eid = line.strip().split("\t")
                entity_id2name[int(eid)] = name
    except:
        print("[warn] could not load entity2id.txt")

    print(f"[data] num_nodes={num_nodes}, num_rels={num_rels}")
    print(f"[data] train={len(train_list)} valid={len(valid_list)} test={len(test_list)} snapshots")

    # ── 构造模型 (同 baseline: --n-hidden 200 --n-layers 2 --n-bases 100 --dropout 0.4) ──
    model = RecurrentRGCN(
        "convtranse",              # decoder_name
        "uvrgcn",                  # encoder_name
        num_nodes,                 # num_ents
        num_rels,                  # num_rels
        0,                         # num_static_rels
        0,                         # num_words
        200,                       # h_dim
        "sub",                     # opn
        sequence_len=args.train_history_len,
        num_bases=100,
        num_basis=100,
        num_hidden_layers=2,
        dropout=0.4,
        self_loop=True,
        input_dropout=0.4,
        hidden_dropout=0.4,
        feat_dropout=0.4,
        use_compat=args.compat,
        compat_lambda=args.compat_lambda,
        compat_aux_weight=args.compat_aux_weight,
        feat_dir=args.feat_dir,
    )

    # ── 加载 checkpoint ──
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"[model] Loaded epoch={checkpoint.get('epoch', '?')}")

    # ── all_ans for filter ──
    all_ans_list = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    use_cuda = torch.cuda.is_available()

    # ── 评测循环 (同 main.py test()) ──
    history_list = train_list + valid_list
    # 只取最后 test_history_len 个快照作为历史输入
    current_history = [snap for snap in history_list[-args.test_history_len:]]

    records = []  # 每行: [head_algo_id, real_gpu_entity_id, pred_top1_entity_id, filter_rank]

    for time_idx, test_snap in enumerate(test_list):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                         for g in current_history]

        with torch.no_grad():
            # model forward → 取最后一个时间步的节点表征
            evolve_embs, _, r_emb, _, _ = model.forward(history_glist, None, use_cuda)
            embedding = evolve_embs[-1]  # [num_ents, h_dim]

            # 准备测试三元组 (正向 + 逆向, 同 predict())
            test_triples = torch.LongTensor(test_snap).to(device)
            inverse_triples = test_triples[:, [2, 1, 0]].clone()
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
            all_triples = torch.cat([test_triples, inverse_triples])  # [2*N_snap, 3]

            # batch decode + per-batch filter + r1 processing (不物化全量 score 矩阵)
            eval_bs = 500
            N_trip = all_triples.size(0)
            for s in range(0, N_trip, eval_bs):
                e = min(s + eval_bs, N_trip)
                batch_trip = all_triples[s:e]
                batch_score = model.decoder_ob.forward(embedding, r_emb, batch_trip, mode="test")
                # filter 本批次
                filter_score(batch_trip, batch_score, all_ans_list[time_idx])
                # 筛选 r1 正向边
                r1_mask = (batch_trip[:, 1] == 1)
                if r1_mask.sum() > 0:
                    r1_trip_b = batch_trip[r1_mask]
                    r1_sco_b = batch_score[r1_mask]
                    pred_top1_b = r1_sco_b.argmax(dim=1)
                    for j in range(len(r1_trip_b)):
                        head = r1_trip_b[j, 0].item()
                        real_tail = r1_trip_b[j, 2].item()
                        pred_tail = pred_top1_b[j].item()
                        # filter rank (1-based)
                        raw_rank = sort_and_rank(r1_sco_b[j:j+1], r1_trip_b[j:j+1, 2]).item()
                        rank = raw_rank + 1
                        records.append([head, real_tail, pred_tail, rank])
                del batch_score

        # slide history
        current_history.pop(0)
        current_history.append(test_snap)

    # ──────────────────────────────────────────
    # 分析
    # ──────────────────────────────────────────
    records = np.array(records)
    num_r1 = len(records)
    print(f"\n{'='*60}")
    print(f"  r1_suits 测试边总数: {num_r1}")
    print(f"  其中 tail=1(T4)={(records[:,1]==1).sum()}, tail=2(MISC)={(records[:,1]==2).sum()}, "
          f"tail=3(P100)={(records[:,1]==3).sum()}, tail=4(V100)={(records[:,1]==4).sum()}, "
          f"tail=5(V100M32)={(records[:,1]==5).sum()}")
    print(f"{'='*60}")

    heads      = records[:, 0].astype(int)      # head 算法 id
    real_gpu   = records[:, 1].astype(int)      # 真实 GPU 实体 id
    pred_ent   = records[:, 2].astype(int)      # 预测 top1 实体 id
    rank       = records[:, 3].astype(int)      # filter rank (1-based)

    # ── 0. 确认 entity 2 的 label ──
    print(f"\n{'─'*50}")
    print("0. entity 2 (模型始终预测的实体) 是什么?")
    print(f"{'─'*50}")
    ent2_name = entity_id2name.get(2, "?(load failed)")
    print(f"  全局实体 ID 2: entity2id.txt name = {ent2_name}")
    print(f"  GPU 类型映射 (local idx)→(GPU type):")
    for eid in sorted(GPU_TYPE_LABELS):
        n = entity_id2name.get(124885 + eid, "")
        print(f"    entity {eid} (global {124885+eid}) = {GPU_TYPE_LABELS[eid]:>8}  (name in e2i: {n})")
    print(f"\n  注意: test.txt 中 r1_suits 的 tail 实体 ID = local idx (0-5),")
    print(f"        不是全局 GPU entity ID (124885-124890).")
    print(f"        模型预测的 entity 2 = local idx 2 = {GPU_TYPE_LABELS[2]}")

    # ── 0b. top1 预测的分布和熵 ──
    top1_counts = Counter(pred_ent.tolist())
    top5_ents = sum(v for k, v in top1_counts.items() if k in (1,2,3,4,5))
    other_ents = num_r1 - top5_ents
    print(f"\n{'─'*50}")
    print("0b. Top1 预测分布 (所有实体):")
    print(f"{'─'*50}")
    total_f = float(num_r1)
    H = 0.0
    for ent_id in sorted(top1_counts.keys()):
        cnt = top1_counts[ent_id]
        p   = cnt / total_f
        if ent_id in GPU_TYPE_LABELS:
            label = f"GPU_{ent_id}({GPU_TYPE_LABELS[ent_id]})"
        else:
            label = f"非GPU实体{ent_id}"
        H += -p * log2(p) if p > 0 else 0
        print(f"  {label:<32} {cnt:>6}  ({p*100:5.2f}%)")
    if other_ents > 0:
        print(f"  {'其他实体汇总':<32} {other_ents:>6}  ({(other_ents/total_f)*100:5.2f}%)")

    num_classes = len(top1_counts)
    H_max = log2(num_classes) if num_classes > 1 else 1.0
    H_norm = H / H_max if H_max > 0 else 0.0
    print(f"\n  熵分析:")
    print(f"  H={H:.4f}  (max possible={H_max:.2f}, normalized={H_norm:.4f})")

    # GPU 实体 ID 范围
    gpu_offset = GPU_START
    real_gpu_local = real_gpu - gpu_offset      # 0-based local GPU id
    pred_local = pred_ent - gpu_offset           # FIX: 必须减 offset!
    pred_on_gpu = ((pred_ent >= 1) & (pred_ent <= 5)) | ((pred_ent >= 124885) & (pred_ent <= 124890))

    unique_real_gpus = sorted(set(real_gpu_local))
    n_gpu = len(unique_real_gpus)
    print(f"\n真实 GPU types: {unique_real_gpus}  (0-based local idx, 原始实体 ID {GPU_START}+local)")
    print(f"  对应 GPU 类型: {', '.join([f'idx{g}={GPU_TYPE_LABELS.get(g,chr(63))}' for g in unique_real_gpus])}")
    print(f"共 {n_gpu} 种 GPU")

    # ── 1. 预测 top1 落在 GPU 内 vs 外的比例 ──
    on_gpu_pct = pred_on_gpu.mean() * 100
    off_gpu_pct = 100 - on_gpu_pct
    print(f"\n{'─'*50}")
    print("1. 预测 top1 落点")
    print(f"{'─'*50}")
    print(f"  落在 GPU 节点: {on_gpu_pct:.2f}% ({pred_on_gpu.sum()}/{num_r1})")
    print(f"  落在其他实体: {off_gpu_pct:.2f}% ({(~pred_on_gpu).sum()}/{num_r1})")

    # ── 2. 混淆矩阵 ──
    valid = pred_on_gpu
    v_real = real_gpu_local[valid]
    v_pred = pred_local[valid]

    print(f"\n{'─'*50}")
    print("2. 混淆矩阵 (真实 GPU × 预测 top1 GPU)")
    print(f"   注: 只统计预测 top1 落在 GPU 内的样本. top1 落在 GPU 外: {(~pred_on_gpu).sum()} 条")
    print(f"   行列标签: 0=CPU, 1=T4, 2=MISC, 3=P100, 4=V100, 5=V100M32")
    print(f"{'─'*50}")

    gpu_labels_list = [GPU_TYPE_LABELS.get(g, f"g{g}") for g in unique_real_gpus]
    cm = np.zeros((n_gpu, n_gpu), dtype=int)
    for r, p in zip(v_real, v_pred):
        if p in unique_real_gpus:
            ri = unique_real_gpus.index(r)
            pi = unique_real_gpus.index(p)
            cm[ri, pi] += 1

    # 打印矩阵
    header_2d = "真实\\预测  " + "  ".join([f"{lbl:>7}" for lbl in gpu_labels_list])
    print("   " + " " * 12 + "  ".join([f"{lbl:>7}" for lbl in gpu_labels_list]))
    for i, g in enumerate(unique_real_gpus):
        lbl = GPU_TYPE_LABELS.get(g, f"g{g}")
        row = f"   {lbl:<10}" + "  ".join([f"{cm[i,j]:>7}" for j in range(n_gpu)])
        print(row)

    # ── 3. Per-GPU Hits@1 ──
    print(f"\n{'─'*50}")
    print("3. Per-GPU Hits@1")
    print(f"{'─'*50}")
    for g in unique_real_gpus:
        mask = (real_gpu_local == g)
        n_total = mask.sum()
        if n_total > 0:
            h1 = (rank[mask] == 1).mean() * 100
            lbl = GPU_TYPE_LABELS.get(g, f"GPU{g}")
            print(f"   {lbl:>8}: Hits@1={h1:6.2f}%  (n={n_total:>6})")

    # ── 4. 排名分布 ──
    print(f"\n{'─'*50}")
    print("4. 正确 dst 排名分布 (filter rank)")
    print(f"{'─'*50}")
    for r_val in range(1, 6):
        pct = (rank == r_val).mean() * 100
        print(f"   排名 = {r_val:>2}: {pct:6.2f}%  (n={(rank==r_val).sum():>6})")
    pct_gt5 = (rank > 5).mean() * 100
    print(f"   排名 > 5: {pct_gt5:6.2f}%  (n={(rank>5).sum():>6})")

    # ── 5. 算法特征差异 ──
    print(f"\n{'─'*50}")
    print("5. 算法特征差异 (预测对 vs 预测错)")
    print(f"{'─'*50}")
    feat_path = os.path.join(os.path.dirname(__file__), "..", "data", "xcad", "node_feat_algo.npy")
    if os.path.exists(feat_path):
        feat_algo = np.load(feat_path)  # [N_ALGO, 13]
        correct_mask = (rank == 1)
        wrong_mask   = (rank > 1)

        correct_heads = heads[correct_mask]
        wrong_heads   = heads[wrong_mask]

        # clamp
        N_ALGO = feat_algo.shape[0]
        correct_heads = correct_heads[correct_heads < N_ALGO]
        wrong_heads   = wrong_heads[wrong_heads < N_ALGO]

        if len(correct_heads) > 0 and len(wrong_heads) > 0:
            cf = feat_algo[correct_heads]
            wf = feat_algo[wrong_heads]

            feat_names = [
                "avg_plan_cpu", "std_plan_cpu", "min_plan_cpu", "max_plan_cpu",
                "avg_plan_mem", "std_plan_mem", "min_plan_mem", "max_plan_mem",
                "avg_plan_gpu", "std_plan_gpu", "min_plan_gpu", "max_plan_gpu",
                "instance_count"
            ]
            print(f"   (特征来自 node_feat_algo.npy, 经 log1p+z-score 标准化)")
            print(f"   {'维度':<22} {'预测对均值':>12} {'预测错均值':>12} {'差值':>12}")
            print(f"   {'-'*58}")
            for i, name in enumerate(feat_names):
                c_mean = cf[:, i].mean()
                w_mean = wf[:, i].mean()
                diff   = c_mean - w_mean
                print(f"   {name:<22} {c_mean:>12.4f} {w_mean:>12.4f} {diff:>12.4f}")
        else:
            print("   (一组为空, 无法计算)")
    else:
        print(f"   (特征文件不存在: {feat_path})")

    # ── 6. SPOT-CHECK: 随机抽 5 条 test r1 边, 打印 score 向量 ──
    print(f"\n{'─'*50}")
    print("6. SPOT-CHECK: 随机 5 条 test r1 边的 score 向量 (GPU entities + top1)")
    print(f"{'─'*50}")

    # 重新推理一次, 收集所有 r1 正向边的 score (无 filter, 只看原始 score 分布)
    current_history = [snap for snap in history_list[-args.test_history_len:]]

    spot_r1_triples = []  # [(head, rel, tail)]
    for time_idx, test_snap in enumerate(test_list):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                         for g in current_history]
        with torch.no_grad():
            evolve_embs, _, r_emb, _, _ = model.forward(history_glist, None, use_cuda)
            embedding = evolve_embs[-1]
            test_triples = torch.LongTensor(test_snap).to(device)
            inverse_triples = test_triples[:, [2, 1, 0]].clone()
            inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
            all_triples = torch.cat([test_triples, inverse_triples])
            r1_mask = (all_triples[:, 1] == 1)
            r1_t = all_triples[r1_mask]
            for j in range(len(r1_t)):
                spot_r1_triples.append((r1_t[j,0].item(), r1_t[j,1].item(), r1_t[j,2].item()))
        current_history.pop(0)
        current_history.append(test_snap)

    total_r1 = len(spot_r1_triples)
    if total_r1 >= 5:
        rng = np.random.RandomState(args.seed)
        sample_idx = rng.choice(total_r1, 5, replace=False)
        sample_trips = [spot_r1_triples[i] for i in sample_idx]

        # batch decode these 5 with final embedding
        with torch.no_grad():
            history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                             for g in [snap for snap in history_list[-args.test_history_len:]]]
            evolve_embs, _, r_emb, _, _ = model.forward(history_glist, None, use_cuda)
            embedding = evolve_embs[-1]
            sample_ts = torch.LongTensor([[h,r,t] for h,r,t in sample_trips]).to(device)
            batch_score = model.decoder_ob.forward(embedding, r_emb, sample_ts, mode="test")

        gpu_entity_ids = [1, 2, 3, 4, 5]  # GPU type local entity IDs

        print(f"   (从 {total_r1} 条 r1 中随机抽取 5 条, 无 filter, score 来自最后一个 test snapshot 的 embedding)")
        for i in range(len(sample_trips)):
            head = sample_trips[i][0]
            real_tail = sample_trips[i][2]
            scores = batch_score[i].cpu().numpy()
            gpu_scores = {eid: scores[eid] for eid in gpu_entity_ids}
            top1 = scores.argmax()
            top1_label = GPU_TYPE_LABELS.get(top1, f"实体{top1}")
            rank_this = 1 + (scores > scores[real_tail]).sum()
            real_label = GPU_TYPE_LABELS.get(real_tail, f"entity_{real_tail}")

            print(f"\n   样本 {i+1}: head={head}, real_tail={real_tail}({real_label})")
            print(f"   score 向量 (in 5 GPU entities):")
            for eid in gpu_entity_ids:
                markers = []
                if eid == real_tail:
                    markers.append("← 真实")
                if eid == top1:
                    markers.append("← TOP1")
                marker_str = "  " + ", ".join(markers) if markers else ""
                print(f"     {GPU_TYPE_LABELS.get(eid,'?'):>8}(eid={eid}): {gpu_scores[eid]:.4f}{marker_str}")
            print(f"   top1={top1}({top1_label}), correct rank={rank_this}")
    else:
        print(f"   (r1 边不足 5 条, 跳过)")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
