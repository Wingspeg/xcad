#!/bin/bash
# run_maas_base_emb.sh -- screening S20 的修法：base 共享表征，单臂。
#
# 只改一件事：head 表征 = dynamic_emb[id] + base_emb[base_of[id]]（--base-emb）。
# 超参、输出层、训练目标、负采样、评测口径全部与 S18 的 base 臂逐字相同，所以读数和
#   base 0.1472 / entity_memory 0.7515 / fleet 0.3473 / tail_prior 0.1896
# 是同一把尺子上的直接对照，不需要再跑一遍旧臂。
#
# BAR（在看到任何数字之前写死，沿用 S17.6 + S20.4 的合并天花板）：
#   复发层 H@1 >= 0.3473  -> 超越编队构成 = 修法生效
#   复发层 H@1 <  0.1896  -> 连常数基线都没过 = 停止调参，问题不在表征
#   全新 head 层 H@1 >  0.1651(tail_prior) 且往 0.2980(base_hist 闭式天花板)靠
# 另附机制检查：base_emb 是否真的学到了"父节点 -> 卡型"的分离（脚本见文末）。
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/maas_arms
mkdir -p "$OUT"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=${EPOCHS:-30}
EVAL_EVERY=${EVAL_EVERY:-5}
ARM_TIMEOUT=${ARM_TIMEOUT:-900}
TAG=${TAG:-baseemb}
# 两条 base 条件化路（同一份超参、同一个 seed，只差这一面旗）：
#   --base-emb  : 空白可学习表（§21，已跑）
#   --base-feat : 注入 base 的历史卡型直方图（§21.5 的 (B')，本轮）
EXTRA=${EXTRA:---base-emb}

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs "$EPOCHS" --evaluate-every "$EVAL_EVERY" --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42 -d maas --probe-rel 0)

echo "[$TAG] start=$(date -u +%H:%M:%S)"
timeout "$ARM_TIMEOUT" "$PY" main.py "${PAPER[@]}" $EXTRA \
    --dump-preds "$OUT/$TAG.pt" > "$OUT/$TAG.log" 2>&1
rc=$?
echo "[$TAG] exit=$rc done=$(date -u +%H:%M:%S)"
tr '\r' '\n' < "$OUT/$TAG.log" | grep -aE "base-emb|base-prior|head-prior|\[prior\]|id-space|Traceback|OutOfMemory|RuntimeError" | head -8
exit $rc
