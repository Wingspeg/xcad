#!/bin/bash
# rerun_baselines_idspace.sh —— 把主表在【全局四段编号】上重做一遍，并保证每行同源
#
# 背景（docs/reviews/2026-11-12_id_space_defect_record.md §5.2/§7）：
#   v1 主表里 RE-GCN / CEN / RETIA 三行跑在 baselines/data/xcad 上，
#   而那份边文件就是【旧编号】（r1 的 tail 写成 1..5，与 placement 段撞号，md5 已核）。
#   同一配置换到 data/xcad_v2 后 baseline 侧读数塌了（r1 H@1 0.4432 -> 0.0117），
#   所以三行外部基线必须重跑；ours 侧几乎不动（0.884167 -> 0.884843）。
#
# 三段：
#   A 门禁（便宜，test-only 复用 6 月 checkpoint）：旧编号上逐位复现 v1 读数，
#     同一命令跑两遍判"评测是否确定性"；顺带首次实测基线在旧编号下的 per-GPU macro
#     （v1 表里 macro=20.0 是推断值，没有实测行）。
#   B 主臂：三个基线用与 6 月【完全相同】的超参，只把 -d xcad 换成 -d xcad_v2，
#     训练臂末尾自带 final test，全部指标取自同一个 run（同源）。先各跑 1 epoch 冒烟。
#   C ours：joint(点1+点2) 在 xcad_v2 上补 seed 123 / 2024，
#     恢复主表"三随机种子均值"的说法（seed 42 已有）。
#
# 顺序很重要，不要并发：单进程占 11.6GB 以上，两张臂并发直接 OOM（2026-11-12 已踩）。
# 环境：baselines 的 dgl 需要 pip 版 nvidia/*/lib 进 LD_LIBRARY_PATH，
#       否则 import dgl 报 libcusparse.so.12 not found（11-16 起复现，本轮已实测修好）。
set -u

BL=~/xcad/baselines
XC=~/xcad/xcad_code/xcad_model
OUT=~/xcad/outputs/reports/baselines_idspace
mkdir -p "$OUT"

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

stampt() { date -u +%H:%M:%S; }

# run_base <tag> <RE-GCN|CEN|RETIA> <args...>  —— 外部基线用 baselines 的 venv
run_base() {
  local tag="$1"; shift
  local proj="$1"; shift
  echo "[$tag] start=$(stampt) cwd=$proj"
  ( cd "$BL/$proj/src" || exit 1
    export LD_LIBRARY_PATH="$(ls -d "$BL"/.venv/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
    "$BL/.venv/bin/python3" main.py "$@" > "$OUT/$tag.log" 2>&1 )
  local rc=$?
  echo "[$tag] exit=$rc done=$(stampt)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "\[id-space\]|覆盖不全|Macro|Traceback|memory|Error" | head -6
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/  gpu_after: /"
}

# run_ours <tag> <args...> —— ours 用 xcad_code 的 venv
run_ours() {
  local tag="$1"; shift
  echo "[$tag] start=$(stampt)"
  ( cd "$XC/src" || exit 1
    # 与 seed42 那次(rerun_idspace_paper.sh)逐字一致，连显存分配策略都不换
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ~/xcad/xcad_code/.venv/bin/python main.py "$@" > "$OUT/$tag.log" 2>&1 )
  local rc=$?
  echo "[$tag] exit=$rc done=$(stampt)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "\[id-space\]|覆盖不全|Macro|Traceback|memory|Error" | head -6
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/  gpu_after: /"
}

# ── 三个基线的超参（逐字抄自 6 月日志的 Namespace，不含 -d / --test / --n-epochs）──
REGCN=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
       --train-history-len 3 --test-history-len 3
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --weight 0.5 --task-weight 0.7 --angle 10 --discount 1.0 --self-loop --layer-norm
       --weight-decay 1e-4 --lr 0.001 --entity-prediction)
CEN=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
     --start-history-len 2 --train-history-len 3 --test-history-len 3 --dilate-len 1
     --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
     --self-loop --layer-norm --entity-prediction --lr 0.001 --ft_lr 0.001 --norm_weight 0.1)
RETIA=(--gpu 0 --encoder uvrgcn --decoder convtranse --n-hidden 200 --n-layers 2 --n-bases 100
       --train-history-len 2 --test-history-len 2
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --self-loop --layer-norm --task-weight 0.7 --norm_weight 1.0
       --weight 1 --discount 1 --angle 10 --lr 0.001)

# ═════════════════════  A. 门禁：旧编号 test-only ×2  ═════════════════════
for R in a b; do
  run_base "gate_regcn_$R" RE-GCN "${REGCN[@]}" -d xcad --test
  run_base "gate_cen_$R"   CEN    "${CEN[@]}"   -d xcad --test 2
done
# RETIA 的 --test-test 走 continual_test，要的是 <name>-flr<ft_lr>-norm<norm>-<tc> 这种
# 微调试点（6 月那份不存在，11-17 的读数本来就来自 --cur-train 训练臂）；
# 所以 RETIA 的门禁 = 旧编号上重跑被引子那次 --cur-train 30 epoch（~15 分钟）。
run_base "gate_retia_curtrain" RETIA "${RETIA[@]}" -d xcad --cur-train --n-epochs 30 --evaluate-every 5
echo "PHASE_A_DONE $(stampt)"

# ═════════════════════  B. 主臂：三个基线换到 xcad_v2  ═════════════════════
# 冒烟（1 epoch）先行：只为确认新数据目录不在建图/建模阶段炸，checkpoint 会被正式臂覆盖。
run_base smoke_regcn RE-GCN "${REGCN[@]}" -d xcad_v2 --n-epochs 1 --evaluate-every 1
run_base smoke_cen   CEN    "${CEN[@]}"   -d xcad_v2 --test -1 --n-epochs 1 --evaluate-every 1
run_base smoke_retia RETIA  "${RETIA[@]}" -d xcad_v2 --cur-train --n-epochs 1 --evaluate-every 1
echo "PHASE_B_SMOKE_DONE $(stampt)"

# RE-GCN：被引子那次 = n-epochs 150 / evaluate-every 5（regcn_xcad_main_v2.log）
run_base v2_regcn RE-GCN "${REGCN[@]}" -d xcad_v2 --n-epochs 150 --evaluate-every 5
# CEN：pretrain(h=2) -> curriculum/蒸馏(h=3) -> 最终 test（两段 + 一次 test-only 复核）
run_base v2_cen_pretrain   CEN "${CEN[@]}" -d xcad_v2 --test -1 --n-epochs 30 --evaluate-every 1
run_base v2_cen_curriculum CEN "${CEN[@]}" -d xcad_v2 --test  0 --n-epochs 30 --evaluate-every 1
run_base v2_cen_test       CEN "${CEN[@]}" -d xcad_v2 --test  2
# RETIA：cur-train 30 epoch（retia_xcad_r1.log）+ test-only 复核
run_base v2_retia_curtrain RETIA "${RETIA[@]}" -d xcad_v2 --cur-train --n-epochs 30 --evaluate-every 5
echo "PHASE_B_DONE $(stampt)"

# ═════════════════════  C. ours：xcad_v2 上补 seed 123 / 2024  ═════════════════════
PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10)
JOINT=(--hetero --n-msg-basis 2 --compat --compat-lambda 5.0 --compat-aux-weight 0.5)
for S in 123 2024; do
  run_ours "v2_joint_s${S}" -d xcad_v2 "${PAPER[@]}" --seed "$S" "${JOINT[@]}"
done
echo "ALL_DONE $(stampt)"
