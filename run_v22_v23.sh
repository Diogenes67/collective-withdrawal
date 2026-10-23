#!/bin/bash
# v22 (unlearning methods at a matched flag rate, MIMIC-IV, loads saved models) then v23 (strict eICU protocol,
# one fresh process per consortium x seed, as v20), then the v23 summary and bootstrap.
cd ~/mimic
if pgrep -f "exp_v2[0-3]|run_v2" | grep -v $$ > /dev/null; then echo "another experiment is running; wait for it to finish"; exit 1; fi
PY=/opt/anaconda3/bin/python3
echo "== v22"; OMP_NUM_THREADS=4 $PY exp_v22_unlearning_flags.py || exit 1
echo "== v23 runs"
for i in 0 1 2 3 4 5 6 7 8 9; do for s in 0 1 2; do OMP_NUM_THREADS=4 IDX=$i SEED=$s SUMMARY=0 $PY exp_v23_eicu_strict.py || exit 1; done; done
echo "== v23 summary"; OMP_NUM_THREADS=4 $PY exp_v23_eicu_strict.py
echo "v22 v23 done"
