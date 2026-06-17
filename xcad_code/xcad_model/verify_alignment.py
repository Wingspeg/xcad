"""
对齐验证: 加载 npy + 算 weights_per_snap + 跑所有断言 (不训练, 不依赖 dgl)
跑法: cd xcad_code/xcad_model && python3 verify_alignment.py

本文件只内联 attach_weights_to_snapshots / split_by_time 这两个纯 numpy 函数,
不 import dgl; 但语义与 rgcn/utils.py 中实现字段级一致。
"""
import os
import numpy as np
import pandas as pd


# === split_by_time (内联, 与 rgcn/utils.py 同) ===
def split_by_time(data):
    snapshot_list = []
    snapshot = []
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
    return snapshot_list


# === attach_weights_to_snapshots (内联, 与 rgcn/utils.py 同) ===
def attach_weights_to_snapshots(data_with_tau, weights_npy, snapshot_list=None):
    assert len(data_with_tau) == len(weights_npy), (
        f"data_with_tau {len(data_with_tau)} != weights_npy {len(weights_npy)}"
    )
    if isinstance(weights_npy, np.ndarray) and weights_npy.dtype != np.float32:
        weights_npy = weights_npy.astype(np.float32, copy=False)

    snapshot_weights = []
    current_weights = []
    latest_t = 0
    for i in range(len(data_with_tau)):
        t = data_with_tau[i][3]
        if latest_t != t:
            if len(current_weights):
                snapshot_weights.append(np.array(current_weights, dtype=np.float32).copy())
            current_weights = []
            latest_t = t
        current_weights.append(weights_npy[i])
    if len(current_weights) > 0:
        snapshot_weights.append(np.array(current_weights, dtype=np.float32).copy())

    if snapshot_list is not None:
        assert len(snapshot_weights) == len(snapshot_list), (
            f"桶数 {len(snapshot_weights)} != snapshot 数 {len(snapshot_list)}"
        )
        for idx, (sw, sp) in enumerate(zip(snapshot_weights, snapshot_list)):
            assert len(sw) == sp.shape[0], (
                f"snapshot[{idx}] {sp.shape[0]} != weight 桶 {len(sw)}"
            )
    return snapshot_weights


# === 主流程 ===
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'xcad')


def load_txt_4col(name):
    arr = []
    with open(os.path.join(DATA_DIR, name)) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 4:
                arr.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
    return np.array(arr, dtype=np.int64)


def main():
    print('=' * 78)
    print('STEP 1: 加载 train/valid/test.txt (4 列)')
    print('=' * 78)
    data_train = load_txt_4col('train.txt')
    data_valid = load_txt_4col('valid.txt')
    data_test  = load_txt_4col('test.txt')
    print(f'  data.train: shape={data_train.shape}, tau range=[{data_train[:,3].min()}, {data_train[:,3].max()}]')
    print(f'  data.valid: shape={data_valid.shape}, tau range=[{data_valid[:,3].min()}, {data_valid[:,3].max()}]')
    print(f'  data.test:  shape={data_test.shape},  tau range=[{data_test[:,3].min()}, {data_test[:,3].max()}]')
    print(f'  data.train tau 升序? 前 3 = {data_train[:3, 3].tolist()}, 后 3 = {data_train[-3:, 3].tolist()}')
    print(f'  data.valid tau 升序? 前 3 = {data_valid[:3, 3].tolist()}, 后 3 = {data_valid[-3:, 3].tolist()}')
    print(f'  data.test  tau 升序? 前 3 = {data_test[:3, 3].tolist()}, 后 3 = {data_test[-3:, 3].tolist()}')

    print()
    print('=' * 78)
    print('STEP 2: 加载 edge_weights_{train,valid,test}.npy')
    print('=' * 78)
    w_train = np.load(os.path.join(DATA_DIR, 'edge_weights_train.npy'))
    w_valid = np.load(os.path.join(DATA_DIR, 'edge_weights_valid.npy'))
    w_test  = np.load(os.path.join(DATA_DIR, 'edge_weights_test.npy'))
    print(f'  edge_weights_train.npy: shape={w_train.shape}, dtype={w_train.dtype}, min/max/mean={w_train.min():.4f}/{w_train.max():.4f}/{w_train.mean():.4f}')
    print(f'  edge_weights_valid.npy: shape={w_valid.shape}, min/max/mean={w_valid.min():.4f}/{w_valid.max():.4f}/{w_valid.mean():.4f}')
    print(f'  edge_weights_test.npy:  shape={w_test.shape},  min/max/mean={w_test.min():.4f}/{w_test.max():.4f}/{w_test.mean():.4f}')

    assert len(data_train) == len(w_train), f'train: 4col {len(data_train)} != npy {len(w_train)}'
    assert len(data_valid) == len(w_valid), f'valid: 4col {len(data_valid)} != npy {len(w_valid)}'
    assert len(data_test)  == len(w_test),  f'test:  4col {len(data_test)}  != npy {len(w_test)}'
    print(f'  ✓ 4col 行数 与 npy 长度 完全一致 (train/valid/test)')

    print()
    print('=' * 78)
    print('STEP 3: split_by_time 切 snapshot_list')
    print('=' * 78)
    train_list = split_by_time(data_train)
    valid_list = split_by_time(data_valid)
    test_list  = split_by_time(data_test)
    print(f'  train_list: {len(train_list)} snapshots, 边数 (min/max/mean) = '
          f'{min(s.shape[0] for s in train_list)}/{max(s.shape[0] for s in train_list)}/{int(np.mean([s.shape[0] for s in train_list]))}')
    print(f'  valid_list: {len(valid_list)} snapshots, 边数 (min/max/mean) = '
          f'{min(s.shape[0] for s in valid_list)}/{max(s.shape[0] for s in valid_list)}/{int(np.mean([s.shape[0] for s in valid_list]))}')
    print(f'  test_list:  {len(test_list)} snapshots, 边数 (min/max/mean) = '
          f'{min(s.shape[0] for s in test_list)}/{max(s.shape[0] for s in test_list)}/{int(np.mean([s.shape[0] for s in test_list]))}')

    print()
    print('=' * 78)
    print('STEP 4: attach_weights_to_snapshots (内置逐桶断言已过)')
    print('=' * 78)
    train_wps = attach_weights_to_snapshots(data_train, w_train, snapshot_list=train_list)
    valid_wps = attach_weights_to_snapshots(data_valid, w_valid, snapshot_list=valid_list)
    test_wps  = attach_weights_to_snapshots(data_test,  w_test,  snapshot_list=test_list)
    print(f'  train_weights_per_snap: {len(train_wps)} 桶, 边数 (min/max/mean) = '
          f'{min(len(w) for w in train_wps)}/{max(len(w) for w in train_wps)}/{int(np.mean([len(w) for w in train_wps]))}')
    print(f'  valid_weights_per_snap: {len(valid_wps)} 桶, 边数 (min/max/mean) = '
          f'{min(len(w) for w in valid_wps)}/{max(len(w) for w in valid_wps)}/{int(np.mean([len(w) for w in valid_wps]))}')
    print(f'  test_weights_per_snap:  {len(test_wps)}  桶, 边数 (min/max/mean) = '
          f'{min(len(w) for w in test_wps)}/{max(len(w) for w in test_wps)}/{int(np.mean([len(w) for w in test_wps]))}')

    print()
    print('=' * 78)
    print('STEP 5: 全局对齐断言 (桶数 = snapshot 数, sum 桶边数 = npy 长度)')
    print('=' * 78)
    assert len(train_wps) == len(train_list), f'train: 桶数 {len(train_wps)} != snapshot 数 {len(train_list)}'
    assert len(valid_wps) == len(valid_list), f'valid: 桶数 {len(valid_wps)} != snapshot 数 {len(valid_list)}'
    assert len(test_wps)  == len(test_list),  f'test:  桶数 {len(test_wps)}  != snapshot 数 {len(test_list)}'
    print(f'  ✓ len 桶数 = snapshot 数  (train/valid/test): {len(train_wps)}/{len(valid_wps)}/{len(test_wps)}')

    s_train = sum(len(w) for w in train_wps)
    s_valid = sum(len(w) for w in valid_wps)
    s_test  = sum(len(w) for w in test_wps)
    assert s_train == len(w_train), f'train: 各桶 sum {s_train} != npy len {len(w_train)}'
    assert s_valid == len(w_valid), f'valid: 各桶 sum {s_valid} != npy len {len(w_valid)}'
    assert s_test  == len(w_test),  f'test:  各桶 sum {s_test}  != npy len {len(w_test)}'
    print(f'  ✓ 各桶 sum = npy 长度      (train/valid/test): {s_train}/{s_valid}/{s_test} = '
          f'{len(w_train)}/{len(w_valid)}/{len(w_test)}')

    print()
    print('=' * 78)
    print('STEP 6: 抽样 5 个 snapshot, 对照 4col 源数据')
    print('=' * 78)
    import random
    random.seed(0)
    sample_idx = sorted(random.sample(range(len(train_list)), 5))
    cum = 0
    for idx in sample_idx:
        snap = train_list[idx]
        wps  = train_wps[idx]
        n_snap = snap.shape[0]
        n_wps  = len(wps)
        data_rows = data_train[cum:cum+n_snap]
        print(f'  snapshot[{idx}]: 边数 {n_snap}, weight 桶边数 {n_wps}, 4col cum=[{cum}, {cum+n_snap})')
        for k in [0, n_snap - 1]:
            row = data_rows[k]
            print(f'    data_train[{cum+k}] = (s={row[0]:>5}, r={row[1]}, o={row[2]:>5}, t={row[3]:>3})  weight={wps[k]:.6f}')
        cum += n_snap

    print()
    print('=' * 78)
    print('STEP 7: 抽样 r1 边 验证 npy weight = parquet success_rate')
    print('=' * 78)
    PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'outputs', 'edges', 'r1_suits_edges.parquet')
    pq = pd.read_parquet(PARQUET)
    print(f'  parquet: {len(pq)} rows, src unique={pq["src"].nunique()}, tau range={pq["tau"].min()}-{pq["tau"].max()}')

    ent2id = {}
    with open(os.path.join(DATA_DIR, 'entity2id.txt')) as f:
        for line in f:
            h, i = line.strip().split('\t')
            ent2id[h] = int(i)
    id2ent = {v: k for k, v in ent2id.items()}

    GPU_TYPES = ["CPU", "T4", "MISC", "P100", "V100", "V100M32"]
    algo_ids = {pk: idx for idx, pk in enumerate(pq['src'].unique().tolist())}
    print(f'  algo_ids size: {len(algo_ids)}, gpu_pool_ids: {{"CPU":0,"T4":1,"MISC":2,"P100":3,"V100":4,"V100M32":5}}')

    print('  抽 npy 前 10 个 r1 边 (rel=1), 用 (s, o, t) 查 parquet success_rate:')
    found = 0
    for i, row in enumerate(data_train):
        if row[1] == 1 and found < 10:
            s, r, o, t = row
            w_npy = w_train[i]
            s_hash = id2ent.get(int(s))
            if 0 <= int(o) < len(GPU_TYPES):
                gpu_str = GPU_TYPES[int(o)]
            else:
                gpu_str = '?'
            if s_hash:
                matches = pq[(pq['src'] == s_hash) & (pq['dst'] == gpu_str) & (pq['tau'] == int(t))]
                if len(matches):
                    sr_pq = float(matches['success_rate'].iloc[0])
                    match = '✓' if abs(sr_pq - w_npy) < 1e-5 else '✗'
                    print(f'    {match} data_train[{i}] (s={s}, r=1, o={o}, t={t})  '
                          f'npy={w_npy:.6f}  parquet_success_rate={sr_pq:.6f}  '
                          f'(diff={abs(sr_pq - w_npy):.2e})')
                    found += 1

    print()
    print('=' * 78)
    print('全部对齐断言通过 ✓')
    print('=' * 78)


if __name__ == '__main__':
    main()
