#!/bin/bash
# run_maas_arms.sh -- the three arms on the second dataset (Alibaba xMaaS maas).
#
# Why these three arms: R1-1's named experiment (2) asks for a scenario where the
# cross-time cohort is *constructible*, and maas is the only candidate that also
# separates the two competition channels (docs/reviews/
# 2026-11-16_second_dataset_screening.md S16/S17).  Same hyper-parameters as the
# MIT / xcad_v2 arms, so the only thing that changed is the dataset.
#   base    : PAPER config (relation GRU evolution ON)
#   hetero  : per-relation basis decomposition (generic, does not need a GPU segment)
#   static  : named ablation 1 (--no-rel-evolution), GRU -> static relation embedding
# Every arm carries --probe-rel 0, which routes main.py through generic_strata:
# the same test rows, the same filtered protocol, plus the entity_memory /
# tail_prior closed-form baselines per stratum.
#
# DECISION BAR, fixed before any GPU number was seen (S17.6):
#   recurrent stratum H@1 >= 0.7631  -> beats the memorisation ceiling
#   recurrent stratum H@1 >= 0.3473  -> beats the fleet/cluster-modal baseline
#   below 0.3473 or static >= base   -> maas only supports "cohort is constructible"
#
# Budget: serial arms on one 3090, each killed at ARM_TIMEOUT (default 900 s = the
# agreed 15 min/arm).  Smoke a run first:  EPOCHS=1 EVAL_EVERY=1 ARMS=base ...
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
ARMS=${ARMS:-base hetero static}

PAPER=(--gpu 0 --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs "$EPOCHS" --evaluate-every "$EVAL_EVERY" --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42 -d maas --probe-rel 0)

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  timeout "$ARM_TIMEOUT" "$PY" main.py "${PAPER[@]}" \
      --dump-preds "$OUT/$tag.pt" "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Traceback|OutOfMemory|ERROR|Error" | head -3
}

for a in $ARMS; do
  case "$a" in
    base)   run base ;;
    hetero) run hetero --hetero --n-msg-basis 2 ;;
    static) run static --no-rel-evolution ;;
  esac
done
echo "ALL_DONE $(date -u +%H:%M:%S)"
