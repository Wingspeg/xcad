#!/bin/bash
# run_maas_prior_ts.sh -- screening §26 (H): 把输出层先验表换成时序口径, 串行跑在一张 3090 上。
#
# 改动只有一处: --base-prior/--head-prior 读的表从"train+valid 全历史静态表"换成
# "第 t 层只统计 ts<t 的边"的时序表(export_maas_base_prior_ts.py)。
# probe_maas_prior_ts_wiring.py 已在 CPU 上证过(T3): 所有 test 层的注入结果与静态注入【逐位相同】
# => test 侧评测条件一个字没变, 臂间 test 读数差只能来自"模型训练时看到了什么"。
#
# 判线(§25.8 在看到任何数字之前写死, 不许改):
#   全新 head 层 H@1 >= 0.2877(§23 单通道读数) —— 修好"第二通道把 base 信任度压下去"的倒退
#   全体       H@1 >= 0.3580(闭式 add_hist 靶子) —— 学习版至少要打过非学习版
#   复发层     H@1 >= 0.7337(= 0.7450 − 0.0113, §25 读数减一个噪声带) —— 不许回退
#   机制线     "两通道都答错"子集上的 H@1 >= 0.0700(§24.6b 0.0439 → §25 0.0451)
# 另: 实现自检的"train/test head 可用率差 <= 0.02"在 exporter 里量到 +0.0492, 已按事实记录为未达标。
#
# 阶段(顺序执行, 后一阶段 gated 在前一阶段成功上):
#   0 回归对照: 用 §24 的原旗标 + --test 复用 headprior checkpoint, 读数必须与 headprior.log 逐位相同
#     => 证明"把选表/选 alpha 的逻辑塞进函数"没有动静态口径一个 bit(不占训练时间, ~1 分钟)
#   1 冒烟: EPOCHS=6(必须 >=2, 否则 val 不跑 => 没有 best checkpoint => FileNotFoundError)
#   2 臂 A: --base-prior --head-prior --prior-ts              (对照 §24: 只差口径这一个变量)
#   3 臂 B: 臂 A + --prior-gate                               (对照 §25: 只差口径这一个变量)
#   4 臂 A2: 臂 A 换 seed 43                                   (量 run-to-run 噪声, 判线阈值 0.002 需要它)
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/maas_arms
DR=~/xcad/xcad_code/outputs/reports
mkdir -p "$OUT" "$DR"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=${EPOCHS:-30}
EVAL_EVERY=${EVAL_EVERY:-5}
ARM_TIMEOUT=${ARM_TIMEOUT:-1800}

paper () {   # $1 = seed, $2 = epochs
    echo "--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100"
    echo "--dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4"
    echo "--train-history-len 6 --test-history-len 20 --n-epochs $2 --evaluate-every $EVAL_EVERY"
    echo "--lr 0.001 --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10"
    echo "--seed $1 -d maas --probe-rel 0"
}

# arm <TAG> <EXTRA 旗标> <seed> <epochs> [额外的 main.py 参数]
arm () {
    local tag=$1 extra=$2 seed=$3 ep=$4 more=${5:-}
    # shellcheck disable=SC2086
    timeout "$ARM_TIMEOUT" "$PY" main.py $(paper "$seed" "$ep") $extra $more \
        --dump-preds "$OUT/$tag.pt" > "$OUT/$tag.log" 2>&1
    local rc=$?
    echo "[$tag] exit=$rc epochs=$ep seed=$seed extra='$extra' $more"
    tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "prior-ts|prior-gate|head-prior|base-prior|Traceback|Error|OutOfMemory" | head -6
    return $rc
}

MAIN="$DR/run_maas_prior_ts.log"
{
echo "==== §26 (H) 时序口径先验 chain start=$(date -u '+%F %H:%M:%S') ===="

echo "--- [0] 回归对照: §24 旗标 + --test 复用 checkpoint ---"
arm headprior_recheck "--base-prior --head-prior" 42 "$EPOCHS" "--test"
rc0=$?
if [ "$rc0" -ne 0 ]; then
    echo "[0] 回归对照失败(rc=$rc0), 但它是【复用老 checkpoint】, 不拦后面的新臂"
fi
echo "[0] 与 §24 逐位对照(应完全一致):"
for f in headprior headprior_recheck; do
    echo "  $f: $(tr '\r' '\n' < "$OUT/$f.log" | grep -aE '边 \(head,tail\) 在 history 里' | tail -1)"
done

echo "--- [1] 冒烟: EPOCHS=6 ---"
if arm priorts_smoke "--base-prior --head-prior --prior-ts" 42 6; then
    echo "--- [2] 臂 A: 时序口径双通道(两个全局 alpha) ---"
    arm priorts "--base-prior --head-prior --prior-ts" 42 "$EPOCHS"
    echo "--- [3] 臂 B: 臂 A + 按行信任门 ---"
    arm priortsgate "--base-prior --head-prior --prior-ts --prior-gate" 42 "$EPOCHS"
    echo "--- [4] 臂 A2: 臂 A 第二 seed(噪声带) ---"
    arm priorts_s43 "--base-prior --head-prior --prior-ts" 43 "$EPOCHS"
else
    echo "!!! 冒烟没过, 三臂不开 !!!"
fi
echo "==== chain done=$(date -u '+%F %H:%M:%S') ===="
} 2>&1 | tee "$MAIN"
