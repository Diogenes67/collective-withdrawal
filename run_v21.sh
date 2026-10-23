#!/bin/bash
# v21 driver: one fresh process per seed (avoids the MPS slowdown seen in v20), then the summary and bootstrap.
cd ~/mimic
if pgrep -f "exp_v20|run_v20|exp_v21" > /dev/null; then echo "another experiment is running; wait for it to finish"; exit 1; fi
for s in 0 1 2; do OMP_NUM_THREADS=4 SEED=$s SUMMARY=0 /opt/anaconda3/bin/python3 exp_v21_sisa.py; done
OMP_NUM_THREADS=4 /opt/anaconda3/bin/python3 exp_v21_sisa.py
echo "v21 done"
