#!/bin/bash
# run_xcad_head_prior.sh -- screening §29 (D): 把 §23/§24/§26 的输出层先验配方搬回主数据集 xcad_v2。
#
# 唯一的实装新增 = src/data/export_xcad_head_prior.py 导的两张表:
#     head_prior.npy     (124891, 5)   静态口径(= §23/§24)
#     head_prior_ts.npy  (62, 124891, 5) 时序口径(= §26, 第 t 层只含 ts<t 的 rel1 边)
#   打分侧一行没改: --probe-rel 1 自己推出偏置列段 [124886,124890)。
#
# exporter 自检已过(ALL CHECKS OK, 0 GPU): test 段两口径逐位相同; 自泄漏行 0;
#   可用率 fit 0.0835 / test 0.0864(差 -0.0029)与 §28 探针一致; 静态口径给 +0.9136 的可用率差
#   且对 97,246 条训练行 100.00% 命中本行 gold  => 正式臂必须用 --prior-ts。
#
# 判线(§29.2 在看到任何 GPU 数字之前写死, 不许改):
#   ① 修复线   复发层 H@1 >= 0.90         (模型 0.6979 / 离线属性 0.8045 / 查表 0.9990)
#   ② 追平线   复发层 H@1 >= 0.997        (过了要说"基本就是那张表", 与⑥一起报)
#   ③ 属性线   全体   H@1 >= 0.889659     (必须打过纯属性分类器)
#   ④ 不回退   全体 >= 0.884843-0.01 且 全新层 >= 0.902525-0.01  (该层表恒 0, 是接线自检)
#   ⑤ 天花板   全体 >= 0.910865-0.01 = 0.901  (手工拼装"复发查表+全新模型"; 不许拿 ORACLE 0.9182 当线)
#   ⑥ 成色     复发层与闭式查表 top-1 同率 >= 0.95, 且"表答错/表为 0"两档分别报数
#   ⑦ 噪声     seed 42 vs 43: 复发层差 >= 0.03、全体/全新 >= 0.01 才允许写"谁比谁好"
#   ⑧ 停止线   复发层 < 0.804496 => (D) 判负, v2020 只保留已发表读数, 不开第三臂
#
# 阶段(串行, 后一阶段 gated 在前一阶段成功上):
#   0 回归对照: 论文配置旗标(不加先验) + --test 复用已发表 checkpoint, 四档读数必须逐位相同
#   1 冒烟: EPOCHS=6(必须 > evaluate-every=5, 否则 val 不跑 => 没有 best checkpoint => FileNotFoundError)
#   2 臂 A: --head-prior --prior-ts                     seed 42(正式读数)
#   3 臂 A2: 同上 seed 43(噪声带, 判线⑦要用)
#   4 臂 S: --head-prior(静态口径, 不加 --prior-ts) seed 42 —— §28.2-③ 的机制对照, 不设新判线
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/xcad_v2_arms
DR=~/xcad/xcad_code/outputs/reports
PUB=~/xcad/xcad_code/xcad_model/src/strata_idspace2/joint_xcad_v2.log
mkdir -p "$OUT" "$DR"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=${EPOCHS:-30}
EVAL_EVERY=${EVAL_EVERY:-5}
ARM_TIMEOUT=${ARM_TIMEOUT:-2400}

paper () {   # $1 = seed, $2 = epochs —— 与 rerun_idspace_paper.sh 的 PAPER+JOINT 逐字相同
    echo "--gpu 0 -d xcad_v2 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2"
    echo "--n-bases 100 --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4"
    echo "--train-history-len 6 --test-history-len 20 --n-epochs $2 --evaluate-every $EVAL_EVERY"
    echo "--lr 0.001 --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10"
    echo "--hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5"
    echo "--seed $1 --probe-rel 1"
}

# arm <TAG> <EXTRA 旗标> <seed> <epochs> [额外的 main.py 参数]
arm () {
    local tag=$1 extra=$2 seed=$3 ep=$4 more=${5:-}
    # shellcheck disable=SC2086
    timeout "$ARM_TIMEOUT" "$PY" main.py $(paper "$seed" "$ep") $extra $more \
        --dump-preds "$OUT/$tag.pt" > "$OUT/$tag.log" 2>&1
    local rc=$?
    echo "[$tag] exit=$rc epochs=$ep seed=$seed extra='$extra' $more"
    tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "\[prior\]|\[prior-ts\]|\[head-prior\]|\[id-space\]|Traceback|Error|OutOfMemory" | head -8
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/  gpu_after: /"
    return $rc
}

MAIN="$DR/run_xcad_head_prior.log"
{
echo "==== §29 (D) xcad_v2 先验臂 chain start=$(date -u '+%F %H:%M:%S') ===="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  gpu_before: /'

echo "--- [0] 回归对照: 论文配置 + --test 复用已发表 checkpoint(不加任何先验旗标) ---"
arm joint_recheck "" 42 "$EPOCHS" "--test"
rc0=$?
echo "[0] 逐位对照(全体 / 复发 / 全新 H@1 + r1 MRR):"
for f in "$PUB" "$OUT/joint_recheck.log"; do
    echo "  $(basename "$f"):"
    tr '\r' '\n' < "$f" | grep -aE "H@1=|rel_type 1" | sed 's/^/      /' | head -6
    tr '\r' '\n' < "$f" | grep -aE "复发 head|全新 head" | head -2 | sed 's/^/      /'
done
if [ "$rc0" -ne 0 ]; then
    echo "!!! [0] 回归对照没跑通(rc=$rc0) —— 默认路径被污染, (D) 的数字全部作废, 不开新臂 !!!"
    exit 1
fi

echo "--- [1] 冒烟: 时序口径先验, EPOCHS=6 ---"
if arm headprior_ts_smoke "--head-prior --prior-ts" 42 6; then
    echo "--- [2] 臂 A: 时序口径 head 先验, seed 42(正式读数) ---"
    arm headprior_ts "--head-prior --prior-ts" 42 "$EPOCHS"
    echo "--- [3] 臂 A2: 同上, seed 43(噪声带) ---"
    arm headprior_ts_s43 "--head-prior --prior-ts" 43 "$EPOCHS"
    echo "--- [4] 臂 S: 静态口径 head 先验(§28.2-③ 机制对照, 无新判线) ---"
    arm headprior_static "--head-prior" 42 "$EPOCHS"
else
    echo "!!! 冒烟没过, 正式臂不开 !!!"
fi
echo "==== chain done=$(date -u '+%F %H:%M:%S') ===="
} 2>&1 | tee "$MAIN"
