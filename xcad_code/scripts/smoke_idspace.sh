#!/bin/bash
# smoke_idspace.sh  (v2, 精简版)
# id 空间修复(局部编号 xcad -> 全局四段编号 xcad_v2)的最小对照 smoke
#
# 目的:
#   1) 验证 main.py 的 [id-space] 自动判定在两套数据上分别给出 LEGACY / GLOBAL;
#   2) 验证 compat 偏置列范围 + 辅监督标签改按 gpu_base 之后两套都能跑通;
#   3) 给出 base / compat 两配置下, 新旧编号的方向性指标差。
#
# 边界: CPU smoke(服务器 GPU 驱动当前不可用), 隐藏维/epoch 远小于论文配置,
#       绝对数值不可与论文表格比较, 只做两臂横比。
# 耗时: 主要花在评估(6 个快照 × ~1.5 万三元组 × 12.5 万候选), ~11 min/臂。

set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1

PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/smoke_idspace
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

COMMON=(--gpu -1 --encoder uvrgcn --decoder convtranse
        --n-hidden 32 --n-layers 1 --n-bases 8
        --train-history-len 1 --test-history-len 1
        --n-epochs 2 --evaluate-every 1
        --entity-prediction --seed 42)

run() {
  local tag="$1"; shift
  echo "=== [$tag] $* ===  start=$(date -u +%H:%M:%S)"
  /usr/bin/time -f "WALL_${tag}=%e s" "$PY" main.py "$@" > "$OUT/$tag.log" 2>&1
  echo "--- [$tag] exit=$? done=$(date -u +%H:%M:%S) ---"
  grep -aE "\[id-space\]|MRR \(filter_ent\)|r1_suits|GPU 段覆盖不全|compat_aux_loss|Hits \(filter_ent\) @ 1" "$OUT/$tag.log" | tail -12
}

for DS in xcad xcad_v2; do
  run "base_${DS}"   -d "$DS" "${COMMON[@]}"
  run "compat_${DS}" -d "$DS" "${COMMON[@]}" --compat --compat-lambda 5.0 --compat-aux-weight 0.5
done

echo "ALL_DONE $(date -u +%H:%M:%S)"
