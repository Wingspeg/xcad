#!/usr/bin/env python3
"""
sklearn classifier comparison: 用 node_feat_algo.npy (13维资源画像)
预测 r1_suits 的算法→GPU type (1-5).

两种切分模式:
  --split-by-algo off (默认): 按 tau 切分 (和 RE-GCN 主实验一致)
  --split-by-algo on:         按算法 ID 严格切分 (80% 算法→train, 20%→test)

训练: 逻辑回归 + 随机森林.
输出: 总准确率 / per-GPU 准确率 / macro avg / 对比"永远猜众数"基线.
"""
import argparse
import numpy as np
import os

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_r1_data(txt_path, feat_algo, return_heads=False):
    """从 train/valid/test.txt 提取 r1_suits 边 (rel==1).
    返回 X[N,13], y[N].
    return_heads=True 时额外返回 head 算法 id 数组.
    """
    raw = np.loadtxt(txt_path, dtype=np.int32)
    # raw: [N, 4] columns = s, r, o, tau
    r1_mask = raw[:, 1] == 1
    r1 = raw[r1_mask]
    heads = r1[:, 0]   # algorithm entity ID
    tails = r1[:, 2]   # GPU type entity ID

    N_ALGO = feat_algo.shape[0]
    valid = heads < N_ALGO
    X = feat_algo[heads[valid]]
    y = tails[valid]
    if return_heads:
        return X, y, heads[valid]
    return X, y


def macro_accuracy(y_true, y_pred):
    per_class = []
    for c in sorted(set(y_true)):
        mask = (y_true == c)
        per_class.append(accuracy_score(y_true[mask], y_pred[mask]))
    return np.mean(per_class)


def evaluate(X_train, y_train, X_test, y_test, args, split_label):
    """训练 LR + RF, 打印评测指标."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}

    # --- 逻辑回归 ---
    lr = LogisticRegression(max_iter=1000, random_state=args.seed)
    lr.fit(X_train_s, y_train)
    results["LogisticRegression"] = lr.predict(X_test_s)

    # --- 随机森林 ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=args.seed, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    results["RandomForest"] = rf.predict(X_test_s)
    rf_model = rf  # save for feature importance

    # ── 基线 ──
    train_majority = np.bincount(y_train).argmax()
    test_majority  = np.bincount(y_test).argmax()
    guess_train_maj = np.full_like(y_test, train_majority)
    guess_test_maj  = np.full_like(y_test, test_majority)

    base_train_maj_acc = accuracy_score(y_test, guess_train_maj)
    base_train_maj_mac = macro_accuracy(y_test, guess_train_maj)
    base_test_maj_acc  = accuracy_score(y_test, guess_test_maj)
    base_test_maj_mac  = macro_accuracy(y_test, guess_test_maj)

    print(f"\n{'='*60}")
    print(f"  基线: 永远猜众数  [{split_label}]")
    print(f"{'='*60}")
    print(f"  [训练集众数=type {train_majority}] 总准确率={base_train_maj_acc*100:.2f}%  "
          f"macro={base_train_maj_mac*100:.2f}%")
    print(f"  [测试集众数=type {test_majority}]  总准确率={base_test_maj_acc*100:.2f}%  "
          f"macro={base_test_maj_mac*100:.2f}%")

    CLASS_NAMES = {1: "T4", 2: "MISC", 3: "P100", 4: "V100", 5: "V100M32"}
    feat_names = [
        "avg_plan_cpu","std_plan_cpu","min_plan_cpu","max_plan_cpu",
        "avg_plan_mem","std_plan_mem","min_plan_mem","max_plan_mem",
        "avg_plan_gpu","std_plan_gpu","min_plan_gpu","max_plan_gpu",
        "instance_count"
    ]

    for name, y_pred in results.items():
        acc = accuracy_score(y_test, y_pred)
        mac = macro_accuracy(y_test, y_pred)

        print(f"\n{'─'*60}")
        print(f"  {name}  [{split_label}]")
        print(f"{'─'*60}")
        print(f"  总准确率:  {acc*100:.2f}%")
        print(f"  macro avg: {mac*100:.2f}%")
        print(f"\n  Per-GPU 准确率:")
        for c in sorted(set(y_test)):
            mask = (y_test == c)
            c_acc = accuracy_score(y_test[mask], y_pred[mask])
            c_label = CLASS_NAMES.get(c, f"type{c}")
            n = mask.sum()
            print(f"    type {c} ({c_label:<8}): {c_acc*100:6.2f}%  (n={n:>5})")

        print(f"\n  对比基线:")
        print(f"    总 acc 提升 vs 猜训练众数 (type {train_majority}): "
              f"{acc - base_train_maj_acc:+.2%}")
        print(f"    总 acc 提升 vs 猜测试众数 (type {test_majority}): "
              f"{acc - base_test_maj_acc:+.2%}")
        print(f"    macro 提升 vs 猜众数 (20%): {mac - 0.20:+.2%}")

    # ── 特征重要性 (RF) ──
    imp = rf_model.feature_importances_
    print(f"\n{'─'*60}")
    print(f"  RF 特征重要性 (Top-13)  [{split_label}]")
    print(f"{'─'*60}")
    order = np.argsort(imp)[::-1]
    for i, idx in enumerate(order):
        print(f"  {i+1:>2}. {feat_names[idx]:<20} {imp[idx]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-path", default="../data/xcad/node_feat_algo.npy",
                        help="node_feat_algo.npy 路径")
    parser.add_argument("--data-dir", default="../data/xcad",
                        help="train/valid/test.txt 所在目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-algo", action="store_true", default=False,
                        help="按算法 ID 严格切分 (80% 算法→train, 20%→test), "
                             "而非按 tau 切分")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    feat_path = os.path.join(os.path.dirname(__file__), args.feat_path)

    # ── 加载特征 ──
    feat_algo = np.load(feat_path).astype(np.float64)
    N_ALGO = feat_algo.shape[0]
    print(f"[feat] shape={feat_algo.shape}, NaN={np.isnan(feat_algo).sum()}")
    print(f"       min={feat_algo.min():.4f}, max={feat_algo.max():.4f}, "
          f"mean={feat_algo.mean():.6f}")

    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path  = os.path.join(data_dir, "test.txt")

    # ════════════════════════════════════════════════════
    # 模式 A: 按 tau 切分 (默认, 与原版一致)
    # ════════════════════════════════════════════════════
    if not args.split_by_algo:
        print(f"\n{'#'*60}")
        print(f"  # 切分模式: 按 tau (默认)")
        print(f"{'#'*60}")

        X_tr, y_tr = load_r1_data(train_path, feat_algo)
        X_va, y_va = load_r1_data(valid_path, feat_algo)
        X_te, y_te = load_r1_data(test_path,  feat_algo)

        # 合并 train+valid 作为训练集
        X_train = np.concatenate([X_tr, X_va], axis=0)
        y_train = np.concatenate([y_tr, y_va], axis=0)
        X_test  = X_te
        y_test  = y_te

        print(f"\n[data] train+val: {len(X_train)} edges")
        for c in sorted(set(y_train)):
            print(f"  GPU type {c}: {(y_train==c).sum()}")
        print(f"[data] test:      {len(X_test)} edges")
        for c in sorted(set(y_test)):
            print(f"  GPU type {c}: {(y_test==c).sum()}")

        evaluate(X_train, y_train, X_test, y_test, args, split_label="tau-split")

    # ════════════════════════════════════════════════════
    # 模式 B: 按算法 ID 严格切分
    # ════════════════════════════════════════════════════
    else:
        print(f"\n{'#'*60}")
        print(f"  # 切分模式: 按算法 ID (80/20)")
        print(f"{'#'*60}")

        # 加载全部 r1 边 (train+valid+test)
        all_X_list, all_y_list, all_h_list = [], [], []
        for p in [train_path, valid_path, test_path]:
            X_, y_, h_ = load_r1_data(p, feat_algo, return_heads=True)
            all_X_list.append(X_)
            all_y_list.append(y_)
            all_h_list.append(h_)

        all_X = np.concatenate(all_X_list, axis=0)
        all_y = np.concatenate(all_y_list, axis=0)
        all_h = np.concatenate(all_h_list, axis=0)

        # 按算法 ID 分组
        unique_algos = np.unique(all_h)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(unique_algos)

        n_train_algos = int(len(unique_algos) * 0.8)
        train_algos = set(unique_algos[:n_train_algos])

        train_mask = np.array([h in train_algos for h in all_h])
        X_train = all_X[train_mask]
        y_train = all_y[train_mask]
        X_test  = all_X[~train_mask]
        y_test  = all_y[~train_mask]

        n_test_algos = len(unique_algos) - n_train_algos
        print(f"\n[algo-split] 总算法数={len(unique_algos)}")
        print(f"  训练算法: {n_train_algos} 个 ({n_train_algos/len(unique_algos)*100:.1f}%)")
        print(f"  测试算法: {n_test_algos} 个 ({n_test_algos/len(unique_algos)*100:.1f}%)")
        print(f"\n[data] train: {len(X_train)} edges")
        for c in sorted(set(y_train)):
            print(f"  GPU type {c}: {(y_train==c).sum()}")
        print(f"[data] test:  {len(X_test)} edges")
        for c in sorted(set(y_test)):
            print(f"  GPU type {c}: {(y_test==c).sum()}")

        # 验证 test 算法不在 train 中出现
        train_algos_from_edges = set(all_h[train_mask])
        test_algos_from_edges  = set(all_h[~train_mask])
        overlap = train_algos_from_edges & test_algos_from_edges
        print(f"\n[algo-split] train/test 算法重叠: {len(overlap)}  (应为 0)")

        evaluate(X_train, y_train, X_test, y_test, args, split_label="algo-split")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
