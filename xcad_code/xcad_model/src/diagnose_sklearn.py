#!/usr/bin/env python3
"""
sklearn classifier comparison: 用 node_feat_algo.npy (13维资源画像)
预测 r1_suits 的算法→GPU type (1-5).

两种切分模式:
  --split-by-algo off (默认): 按 tau 切分 (和 RE-GCN 主实验一致)
  --split-by-algo on:         按算法 ID 严格切分 (80% 算法→train, 20%→test)

训练: 逻辑回归 + 随机森林 + MLP(11-19 起).
输出: 总准确率 / per-GPU 准确率 / macro avg / 对比"永远猜众数"基线.

在默认的两个分类器之外, 本脚本另有三处可选行为（默认参数下与首次发布时的读数逐位一致）:

  --class-weight {none,balanced}   sklearn 拟合时的类别加权, 默认 none.
                                   none 即 sklearn 默认(class_weight=None), 用来核对加权口径与
                                   未加权参照读数之间是否只差在类别权重上;
                                   balanced = sklearn 的 n/(K·c), 与我们模型侧 inv 档同一口径思路.
  每个分类器额外打印【预测类边际】+ 逐类精确率（0 成本, 不改任何计算, 只多读一遍 y_pred）.
                                   类边际用来区分"稀有类 recall 为零"到底是打分不足还是输出塌缩.
  MLP 正式基线:
      MLPClassifier((64,32), relu, alpha=1e-4, max_iter=500, random_state=seed)
      —— 配置固定、不调参、不重试, 加权口径与 LR/RF 同参数.
      ⚠ 实装说明: MLPClassifier 在 sklearn 1.8 没有 class_weight 形参 ⇒ 平衡口径走
         fit(sample_weight=compute_sample_weight("balanced"))，与 class_weight="balanced"
         同一公式 n/(K·n_j)（就是 [fit] 行打印的那组乘数），只是实现层替换.
      只在 results 里【追加】一项，拟合顺序在 LR/RF 之后 ⇒ 两者读数应逐位不动，
      由 G-REPRO 门禁核对，不是假设. --skip-mlp 可退回只有 LR/RF 的旧形态.
"""
import argparse
import numpy as np
import os

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


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


# 两套编号都登记: 旧局部 id(0..5) 与全局四段 id(124885..124890)
_GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
CLASS_NAMES = {}
for _i, _n in enumerate(_GPU_TYPES):
    CLASS_NAMES[_i] = _n
    CLASS_NAMES[124885 + _i] = _n


def pred_marginal(y_pred):
    """模型 top-1 实际输出了多少条各类 —— 区分"稀有类 recall 为零"是打分不足还是输出塌缩的读数。"""
    vals, cnts = np.unique(y_pred, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, cnts)}


def evaluate(X_train, y_train, X_test, y_test, args, split_label):
    """训练 LR + RF（默认再追加 MLP）, 打印评测指标."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results = {}

    # --class-weight none 时 sklearn 收到 class_weight=None，与改动前逐位等价
    cw = None if args.class_weight == "none" else args.class_weight
    print(f"\n[fit] class_weight={args.class_weight} -> sklearn 参数 {cw!r}  "
          f"seed={args.seed}  train={len(X_train)}  test={len(X_test)}")
    if cw is not None:
        _cnt = np.bincount(y_train)
        _nz = np.nonzero(_cnt)[0]
        _k = len(_nz)
        print("[fit] balanced 期望乘数 n/(K*c) = " + "  ".join(
            "%s:%.4f" % (CLASS_NAMES[c], len(y_train) / (_k * _cnt[c])) for c in _nz))

    # --- 逻辑回归 ---
    lr = LogisticRegression(max_iter=1000, random_state=args.seed, class_weight=cw)
    lr.fit(X_train_s, y_train)
    results["LogisticRegression"] = lr.predict(X_test_s)

    # --- 随机森林 ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=args.seed, n_jobs=-1, class_weight=cw)
    rf.fit(X_train_s, y_train)
    results["RandomForest"] = rf.predict(X_test_s)
    rf_model = rf  # save for feature importance

    # --- MLP: 静态画像第三个分类器(与 LR/RF 同一份 13 维特征、同一套切分) ---
    # 配置固定如下, 不调参、不重试。
    # 放在 LR/RF 之后拟合: 三者各自带 random_state, 不吃全局 RNG, 故前两个读数逐位不动
    # (由 G-REPRO 门禁验证)。
    if not args.skip_mlp:
        # 实测 sklearn 1.8 的 MLPClassifier 没有 class_weight 形参（传了会 TypeError），
        # 但 fit() 收 sample_weight ⇒ 用 sklearn 自己的 compute_sample_weight("balanced")
        # 走同一条公式 w_j = n/(K·n_j)（与 LR/RF 的 class_weight="balanced" 完全一致，
        # 就是上面 [fit] 行打印的那组乘数）。这是实现层替换，不是新增调参自由度。
        sw = None if cw is None else compute_sample_weight(class_weight="balanced", y=y_train)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                            alpha=1e-4, max_iter=500, random_state=args.seed)
        mlp.fit(X_train_s, y_train, sample_weight=sw)
        results["MLP"] = mlp.predict(X_test_s)
        # 只打印跨版本一定存在的属性: losses_/n_params_ 在 sklearn 1.8 的 adam 下都不存在
        # (实测 AttributeError) ⇒ 用 coefs_ 推结构、用 n_iter_ 报收敛轮数。
        _arch = "x".join(str(c.shape[1]) for c in mlp.coefs_)
        print(f"[fit] MLP hidden=({_arch})  n_iter={mlp.n_iter_}  "
              f"{'sample_weight=balanced(同 class_weight 公式 n/(K·n_j))' if sw is not None else 'sample_weight=None'}")

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

    # 同时登记两套编号: 旧局部 id(0..5) 与全局四段 id(124885..124890), 打印时都能命中
    # （CLASS_NAMES 已提到模块级，pred_marginal / [fit] 行也要用）
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
        print(f"\n  Per-GPU 准确率 (= recall) 与精确率:")
        for c in sorted(set(y_test)):
            mask = (y_test == c)
            c_rec = accuracy_score(y_test[mask], y_pred[mask])
            pmask = (y_pred == c)                      # 精确率必须在【预测为该类的行】上算
            c_pre = accuracy_score(y_test[pmask], y_pred[pmask]) if pmask.any() else float("nan")
            c_label = CLASS_NAMES.get(c, f"type{c}")
            n = mask.sum()
            print(f"    type {c} ({c_label:<8}): recall={c_rec*100:6.2f}%  "
                  f"precision={c_pre*100:6.2f}%  (n={n:>5})")

        pm = pred_marginal(y_pred)
        print("  [marginal] top-1 预测类计数: " + "  ".join(
            "%s=%d" % (CLASS_NAMES.get(v, "type%d" % v), n) for v, n in sorted(pm.items()))
              + "   (真值 " + "  ".join(
                  "%s=%d" % (CLASS_NAMES.get(int(c), "type%d" % c), int((y_test == c).sum()))
                  for c in sorted(set(y_test))) + ")")

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
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="sklearn 拟合时的类别加权。none=不传 class_weight，与加该参数之前的日志逐位等价；"
                             "balanced=按 n/(K*c) 加权（无 cap，比模型侧 inv 档 cap5 更强），"
                             "用于核对加权口径与未加权参照读数的差是否只来自类别权重")
    parser.add_argument("--split-by-algo", action="store_true", default=False,
                        help="按算法 ID 严格切分 (80%% 算法→train, 20%%→test), "
                             "而非按 tau 切分")
    parser.add_argument("--skip-mlp", action="store_true", default=False,
                        help="只跑 LR+RF（复现两分类器时期的既有日志时用；默认不跳过）")
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
