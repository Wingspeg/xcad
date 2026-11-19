#!/usr/bin/env python3
"""probe_maas_base_conflict.py -- model_id->base_id 的多值程度（决定 base 共享表征怎么写）。

export_maas_base_map.py 的函数性断言挂了：1,935 个 model_id 对多个 base_id。
这一版量清楚三件事，为的是选实现而不是找借口：
  (1) 冲突占比：按 distinct model_id 和按 test 边分别算；
  (2) 冲突形状：每个冲突 model 的 base 个数分布 + 最热 base 的份额（是"99/1 的脏数据"
      还是"真的两个爹"）；
  (3) 时间性：同一 model 的 base 是否随 day 变化（若 day 内稳定，则 base 可当
      (model, day) 的属性；若跨天也变，说明 trace 本身重分配过）。
只读 model_id/base_id/day，不读 gpu_type/cluster_id。
"""
import glob
import os

import numpy as np
import pandas as pd

paths = [p for p in sorted(glob.glob(os.path.expanduser(
    "~/xcad/datasets/maas/instance_daily/part_*.parquet"))) if "slim" not in os.path.basename(p)]
raw = pd.concat([pd.read_parquet(p, columns=["day", "model_id", "base_id"]) for p in paths],
                ignore_index=True)
print(f"[raw] rows={len(raw)}")

m = raw.dropna(subset=["model_id", "base_id"])
m = m[m.model_id.astype("string").str.strip().ne("")
      & m.base_id.astype("string").str.strip().ne("")]
m = m[["model_id", "base_id", "day"]].drop_duplicates()
print(f"[dedup] (model,base,day) 行={len(m)}  distinct model={m.model_id.nunique()}"
      f"  distinct base={m.base_id.nunique()}")

nb = m.groupby("model_id").base_id.nunique()
print(f"[1] model 对 >1 base 的个数={int((nb>1).sum())} / {len(nb)}"
      f" = {(nb>1).mean():.4%}   max base/model={int(nb.max())}")
print("    #base per model:", nb.value_counts().sort_index().head(8).to_dict())

# 最热 base 份额
top_share = (m.groupby(["model_id", "base_id"]).size().rename("n").reset_index()
             .groupby("model_id").n.max() / m.groupby("model_id").size())
conf_models = set(nb[nb > 1].index)
sh = top_share[top_share.index.isin(conf_models)]
print(f"[2] 冲突 model 里最热 base 的 day-份额：中位={sh.median():.4f} "
      f"p10={sh.quantile(0.1):.4f} p90={sh.quantile(0.9):.4f} 份额>=0.95 的 model={int((sh>=0.95).sum())}")

# 时间性：同一 model 在同一天内是否有多个 base
nd = m.groupby(["model_id", "day"]).base_id.nunique()
print(f"[3] (model,day) 内多 base 的比例={float((nd>1).mean()):.4%}  n_cell={len(nd)}")
cross = nb[nb > 1]
span = m[m.model_id.isin(cross.index)].groupby("model_id").day.agg(["min", "max", "nunique"])
print("    冲突 model 的时间跨度（天）中位=", float(span["nunique"].median()),
      " 最大=", int(span["nunique"].max()))

# 与磁盘图对齐：test 边里落在冲突 head 上的比例
data_dir = os.path.expanduser("~/xcad/xcad_code/xcad_model/data/maas")
id2name = {}
with open(os.path.join(data_dir, "entity2id.txt")) as f:
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        if len(p) == 2 and p[0] not in ("head", "relation"):
            id2name[int(p[1])] = p[0]
rows = []
with open(os.path.join(data_dir, "test.txt")) as f:
    for ln in f:
        p = ln.split()
        if len(p) >= 4 and int(p[1]) == 0:
            rows.append((int(p[0]), int(p[3])))
tt = pd.DataFrame(rows, columns=["head_id", "tau"])
tt["name"] = tt.head_id.map(id2name)
per_day = m.groupby(["model_id", "day"]).base_id.agg(list)
tab_mod = m.groupby("model_id").base_id.nunique()
tt["nbase"] = tt.name.map(tab_mod)
tt["multi"] = tt.nbase.fillna(0) > 1
# 该 (model, 快照对应 day) 当天是否多 base
days = np.sort(m.day.unique())
tau2day = {i + 1: int(d) for i, d in enumerate(days)}
tt["day"] = tt.tau.map(tau2day)
key = list(zip(tt.name, tt.day))
tt["multi_same_day"] = [len(per_day.get(k, [])) > 1 if k in per_day.index else False for k in key] \
    if False else pd.Series([len(per_day.get(k, [])) > 1 for k in zip(tt.name, tt.day)], index=tt.index)
print(f"[4] test 边={len(tt)}  head 无 base 记录={int(tt.nbase.isna().sum())}"
      f" ({tt.nbase.isna().mean():.4%})  落在跨期多 base 的 head={tt.multi.mean():.4%}"
      f"  当天即多 base={tt.multi_same_day.mean():.4%}")
hist_rows = pd.read_csv(os.path.join(data_dir, "train.txt"), sep=r"\s+", header=None,
                        usecols=[0, 3], names=["h", "tau"])
hh = set(hist_rows.h.map(id2name))
new_head = ~tt.name.isin(hh)
print(f"[5] 全新 head 边={int(new_head.sum())} ({new_head.mean():.4%})  其中 base 多值="
      f"{float(tt.loc[new_head, 'multi'].mean()):.4%}  base 缺失="
      f"{float(tt.loc[new_head, 'nbase'].isna().mean()):.4%}")
