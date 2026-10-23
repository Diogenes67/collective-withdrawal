#!/bin/bash
# v24 (symmetric pooled comparator, MIMIC-IV, one process) then v25 (symmetric pooled comparator, eICU, one fresh process per
# consortium x seed, as v20/v23), then the v25 summary and bootstrap. Waits for nothing: refuses to start if v20-v25 is running.
cd ~/mimic
if pgrep -f "exp_v2[0-5]|run_v2" | grep -v $$ > /dev/null; then echo "another experiment is running; wait for it to finish"; exit 1; fi
PY=/opt/anaconda3/bin/python3
echo "== v24"; OMP_NUM_THREADS=4 $PY exp_v24_symmetric_mimic.py || exit 1
echo "== v25 runs"
for i in 0 1 2 3 4 5 6 7 8 9; do for s in 0 1 2; do OMP_NUM_THREADS=4 IDX=$i SEED=$s SUMMARY=0 $PY exp_v25_symmetric_eicu.py || exit 1; done; done
echo "== v25 summary"; OMP_NUM_THREADS=4 $PY exp_v25_symmetric_eicu.py
echo "v24 v25 done"
