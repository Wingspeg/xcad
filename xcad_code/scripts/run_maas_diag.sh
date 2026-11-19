#!/bin/bash
# run_maas_diag.sh -- typed-candidate-set diagnostic on maas (screening S18.4-B).
#
# Question it answers: is "the model loses to the closed-form baselines on maas"
# caused by the candidate space being 80,287 entities wide (of which 30 are card
# types), or does the model genuinely carry no discrimination signal?  We narrow
# the test-stage entity ranking to the tail type segment [80257, 80286] (30 card
# types; inverse rows use the head segment [0, 80257)) and re-score the SAME three
# checkpoints from run_maas_arms.sh.  No training happens here, so a pass is one
# evaluation over 121,371 test edges.
#
# The closed-form baselines inside generic_strata get the same segment
# (candidate_universe), so model / entity_memory / tail_prior stay one ruler.
#
# ORDER MATTERS: the first pass is a REGRESSION with no --ent-segment.  It must
# reproduce tonight's numbers (base: all-test H@1 0.126299, recurrent stratum
# 0.147218).  If it does not, the predict() refactor broke the main-table path and
# the segment passes below would be meaningless.
set -u
cd ~/xcad/xcad_code/xcad_model/src || exit 1
PY=~/xcad/xcad_code/.venv/bin/python
OUT=~/xcad/xcad_code/xcad_model/src/maas_arms
mkdir -p "$OUT"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PASS_TIMEOUT=${PASS_TIMEOUT:-900}
SEG=${SEG:-"80257 80286"}

PAPER=(--gpu 0 --test --encoder uvrgcn --decoder convtranse
       --n-hidden 200 --n-layers 2 --n-bases 100
       --dropout 0.4 --input-dropout 0.4 --hidden-dropout 0.4 --feat-dropout 0.4
       --train-history-len 6 --test-history-len 20
       --n-epochs 30 --evaluate-every 5 --lr 0.001
       --entity-prediction --dilate-len 1 --weight 1 --discount 1 --angle 10
       --seed 42 -d maas --probe-rel 0)

TIMECMD=()
if [ -x /usr/bin/time ]; then TIMECMD=(/usr/bin/time -v); fi

run() {
  local tag="$1"; shift
  echo "[$tag] start=$(date -u +%H:%M:%S)"
  if [ ${#TIMECMD[@]} -gt 0 ]; then
    "${TIMECMD[@]}" -o "$OUT/$tag.time" \
      timeout "$PASS_TIMEOUT" "$PY" main.py "${PAPER[@]}" \
        --dump-preds "$OUT/$tag.pt" "$@" > "$OUT/$tag.log" 2>&1
  else
    timeout "$PASS_TIMEOUT" "$PY" main.py "${PAPER[@]}" \
        --dump-preds "$OUT/$tag.pt" "$@" > "$OUT/$tag.log" 2>&1
  fi
  local rc=$?
  echo "[$tag] exit=$rc done=$(date -u +%H:%M:%S)"
  grep -a "Maximum resident" "$OUT/$tag.time" 2>/dev/null
  tr '\r' '\n' < "$OUT/$tag.log" | grep -aE "Traceback|OutOfMemory|ERROR|Error" | head -3
}

# 1) regression: base arm, full-entity ruler (must match tonight's base numbers)
run reg_base

# 2) typed candidate set, three arms
run seg_base   --ent-segment $SEG
run seg_hetero --ent-segment $SEG --hetero --n-msg-basis 2
run seg_static --ent-segment $SEG --no-rel-evolution

echo "ALL_DONE $(date -u +%H:%M:%S)"
