"""
Convert xCAD graph edges to ATISE format.
ATISE format:
  - entity2id.txt: name\tid
  - relation2id.txt: name\tid
  - train.txt: head\trelation\ttail\ttimestamp (YYYY-MM-DD)
  - valid.txt, test.txt: same

xCAD tau values (1-69) are mapped to dates starting from 2020-07-01.
"""
import pandas as pd
import os
from datetime import datetime, timedelta

BASE_DATE = "2020-07-01"

def tau_to_date(tau):
    if tau <= 0:
        return None
    d = datetime.strptime(BASE_DATE, "%Y-%m-%d") + timedelta(days=tau - 1)
    return d.strftime("%Y-%m-%d")

edge_files = {
    "placement": "xcad_code/outputs/edges/placement_edges.parquet",
    "r1_suits": "xcad_code/outputs/edges/r1_suits_edges.parquet",
    "r2_requires": "xcad_code/outputs/edges/r2_requires_edges.parquet",
    "r3_drives": "xcad_code/outputs/edges/r3_drives_edges.parquet",
}

all_edges = []
for rel, path in edge_files.items():
    df = pd.read_parquet(path)
    # filter out tau=-1 and null nodes
    df = df[(df['tau'] > 0) & (df['src'].notna()) & (df['dst'].notna())]
    df = df.copy()
    df['relation'] = rel
    all_edges.append(df[['src', 'dst', 'tau', 'relation']])

edges = pd.concat(all_edges, ignore_index=True)
print(f"Total edges (clean): {len(edges)}")

# Clean temporal split matching worklog: train[1,49] val[50,55] test[56,61]
train_df = edges[(edges['tau'] >= 1) & (edges['tau'] <= 49)].copy()
val_df   = edges[(edges['tau'] >= 50) & (edges['tau'] <= 55)].copy()
test_df  = edges[(edges['tau'] >= 56) & (edges['tau'] <= 61)].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Build mappings
all_nodes = set(edges['src'].unique()) | set(edges['dst'].unique())
all_relations = sorted(edges['relation'].unique().tolist())

entity2id = {node: i for i, node in enumerate(sorted(all_nodes, key=str))}
relation2id = {rel: i for i, rel in enumerate(all_relations)}

print(f"Entities: {len(entity2id)}, Relations: {len(relation2id)}")

OUT_DIR = "baselines/ATISE/xcad"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{OUT_DIR}/entity2id.txt", "w") as f:
    for node, eid in sorted(entity2id.items(), key=lambda x: x[1]):
        f.write(f"{node}\t{eid}\n")

with open(f"{OUT_DIR}/relation2id.txt", "w") as f:
    for rel, rid in sorted(relation2id.items(), key=lambda x: x[1]):
        f.write(f"{rel}\t{rid}\n")

def write_triples(df, path):
    count = 0
    with open(path, "w") as f:
        for _, row in df.iterrows():
            date_str = tau_to_date(row['tau'])
            if date_str is None:
                continue
            f.write(f"{row['src']}\t{row['relation']}\t{row['dst']}\t{date_str}\n")
            count += 1
    return count

n_train = write_triples(train_df, f"{OUT_DIR}/train.txt")
n_val   = write_triples(val_df,   f"{OUT_DIR}/valid.txt")
n_test  = write_triples(test_df,  f"{OUT_DIR}/test.txt")

print(f"\nWritten to {OUT_DIR}/:")
print(f"  entity2id.txt: {len(entity2id)} entities")
print(f"  relation2id.txt: {len(relation2id)} relations")
print(f"  train.txt: {n_train} triples")
print(f"  valid.txt: {n_val} triples")
print(f"  test.txt: {n_test} triples")
