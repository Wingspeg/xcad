#!/usr/bin/env python
"""R2-6 (5)：旗舰 checkpoint 的逐张量参数量分解（只读、CPU、0 GPU）。

正文 §5.8「参数与显存的绑定维度」与答复信 R2-6 (5) 的 *Model size* 条目引用的
每一个数字都由本脚本产出：总参数量、实体嵌入表、13 维画像缓冲、点1 关系基每层
（基矩阵 B·d² + 关系系数 2M·B）与文件字节数。

用法（服务器）：
    ~/xcad/xcad_code/.venv/bin/python \
      ~/xcad/xcad_code/scripts/param_dump_r26.py \
      > ~/xcad/xcad_code/outputs/reports/r26_param_breakdown/param_dump.txt

判读线（写在代码里，防事后凑数）：
  * 总 numel 必须 = 31,330,560（正文表 7 引用的参数量）。
  * msg_basis 每张量必须 = B·d² = 80,000，rel_msg_coeff 每张量必须 = 2M·B = 16。
  * 载入的必须是旗舰件本身（文件名以 -compat-seed42 结尾且不含 -memprobe）：
    同目录下 `xcad_v2-...-seed42-memprobe`（125,351,836 B）是 §4.2 的显存探针臂，
    与旗舰件（125,351,502 B）相差 334 B，不可混引。
"""

import collections
import os
import sys

import torch

CKPT_DIR = os.path.expanduser("~/xcad/xcad_code/xcad_model/models")
FLAGSHIP = (
    "xcad-uvrgcn-convtranse-ly2-dilate1-his6-weight:1.0-discount:1.0"
    "-angle:10-dp0.4|0.4|0.4|0.4-gpu0-het2-nonf-nodf-compat-seed42"
)
EXPECTED_TOTAL = 31_330_560
EXPECTED_BASIS = 80_000   # B * d * d, B=2, d=200
EXPECTED_COEFF = 16       # 2M * B, 2M=8 relations, B=2


def bucket(name: str) -> str:
    if "msg_basis" in name:
        return "msg_basis"
    if "rel_msg_coeff" in name:
        return "rel_msg_coeff"
    if "dynamic_emb" in name:
        return "dynamic_emb"
    if "compat_feat" in name:
        return "compat_feat"
    return "other"


def main() -> int:
    path = os.path.join(CKPT_DIR, FLAGSHIP)
    if "-memprobe" in FLAGSHIP:
        print("REFUSING: 这是显存探针臂，不是旗舰件", file=sys.stderr)
        return 2
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    total = sum(v.numel() for v in state.values())
    groups: collections.Counter = collections.Counter()
    for k, v in state.items():
        groups[bucket(k)] += v.numel()

    print("checkpoint :", FLAGSHIP)
    print("bytes      :", os.path.getsize(path))
    print("total numel:", total)
    for k, v in groups.most_common():
        print(f"  {k:16s} {v:>10d}  {100.0 * v / total:5.2f}%")
    print("-- hetero tensors --")
    layers = set()
    for k, v in state.items():
        if "msg_basis" in k or "rel_msg_coeff" in k:
            print(f"  {k}  shape={tuple(v.shape)}  numel={v.numel()}")
            layers.add(k.split(".")[2])          # rgcn.layers.<idx>.<name>
    n_layers = len(layers) or 1
    hetero = groups["msg_basis"] + groups["rel_msg_coeff"]
    print(f"layers with hetero basis : {sorted(layers)}")
    print(f"hetero extra total       : {hetero}")
    print(f"per-layer hetero extra   : {hetero // n_layers}")

    fails = []
    if n_layers != 2:
        fails.append(f"层数 {n_layers} != 2（正文写「两层共 160,032」）")
    if hetero // n_layers != EXPECTED_BASIS + EXPECTED_COEFF:
        fails.append(f"每层 {hetero // n_layers} != {EXPECTED_BASIS + EXPECTED_COEFF}")
    if total != EXPECTED_TOTAL:
        fails.append(f"total {total} != {EXPECTED_TOTAL}")
    if groups["msg_basis"] != 2 * EXPECTED_BASIS:
        fails.append(f"msg_basis {groups['msg_basis']} != 2x{EXPECTED_BASIS}")
    if groups["rel_msg_coeff"] != 2 * EXPECTED_COEFF:
        fails.append(f"rel_msg_coeff {groups['rel_msg_coeff']} != 2x{EXPECTED_COEFF}")
    print("GATES      :", "PASS" if not fails else "FAIL " + "; ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
