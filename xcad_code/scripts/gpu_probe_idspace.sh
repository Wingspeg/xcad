#!/bin/bash
# gpu_probe_idspace.sh — 花 ~2 分钟确认 GPU 路径在两套编号上都能跑通(先看 [id-space]/compat/macro 自检)
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/gpu_probe
mkdir -p "$OUT"

COMMON=(--gpu 0 --encoder uvrgcn --decoder convtranse
        --n-hidden 32 --n-layers 1 --n-bases 8
        --train-history-len 1 --test-history-len 1
        --n-epochs 2 --evaluate-every 1
        --entity-prediction --seed 42)

for DS in xcad xcad_v2; do
  echo "=== probe compat on $DS  start=$(date -u +%H:%M:%S) ==="
  "$PY" main.py -d "$DS" "${COMMON[@]}" --compat --compat-lambda 5.0 --compat-aux-weight 0.5 \
      > "$OUT/probe_${DS}.log" 2>&1
  echo "exit=$? done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/probe_${DS}.log" | grep -aE "\[id-space\]|Is cuda available|compat_aux_loss|覆盖不全|Traceback|Error|error" | head -8
  tr '\r' '\n' < "$OUT/probe_${DS}.log" | grep -aE "^ *(T4|MISC|P100|V100|V100M32):|^ *CPU:|Per-GPU|MRR \(filter_ent\)|r1_suits" | tail -10
done
echo "PROBE_DONE $(date -u +%H:%M:%S)"
