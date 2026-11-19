"""Philly: which machine population carries the 8-device observation."""
import os
import pandas as pd

R = os.path.expanduser("~/xcad/datasets/clean")
dv = pd.read_parquet(f"{R}/philly/devices.parquet")
pdv = pd.read_parquet(f"{R}/philly/placement_devices.parquet")
cnt = dv.groupby("machine_gid", observed=True).size()
used = set(pdv["machine_gid"])
g = dv.groupby("machine_gid", observed=True)["gpu_label"].first()
ev = dv.groupby("machine_gid", observed=True)["evidence"].agg(
    lambda s: "|".join(sorted(set(s.astype(str)))))
print("device rows", len(dv), " machines", len(cnt), " carrying placements", len(used))
print("count dist ALL    :", cnt.value_counts().sort_index().to_dict())
print("count dist USED   :", cnt[cnt.index.isin(used)].value_counts().to_dict())
print("count dist UNUSED :", cnt[~cnt.index.isin(used)].value_counts().to_dict())
print("\ntier dist USED    :", g[g.index.isin(used)].value_counts().to_dict())
print("tier dist UNUSED  :", g[~g.index.isin(used)].value_counts().to_dict())
print("\nevidence values   :", dv["evidence"].value_counts(dropna=False).to_dict())
print("evidence by tier  :")
print(dv.groupby(["gpu_label", "evidence"], observed=True).size().to_string())
print("\nUSED machines, tier x device-count:")
u = pd.DataFrame({"tier": g, "n": cnt, "used": cnt.index.isin(used)})
print(u[u["used"]].groupby(["tier", "n"], observed=True).size().to_string())
