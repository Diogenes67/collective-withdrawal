#!/bin/sh
# Overnight run: seven more training seeds (3 to 9) for the MIMIC-IV chain behind the withdrawal and confinement contrasts,
# then v29 on all ten seeds. Every stage is resumable (finished runs and cached checkpoints are skipped) and SUMMARY=0 keeps
# the published three-seed result files (summary.csv, intervals.json, provenance.csv) untouched. Start with
#   cd ~/mimic && nohup caffeinate -i sh run_seeds_overnight.sh > /dev/null 2>&1 &
# and follow results/seeds_overnight.log. Rerun the same command to resume after an interruption.
cd ~/mimic || exit 1
P=/opt/anaconda3/bin/python3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOG=results/seeds_overnight.log
stage() {
  echo "=== $(date '+%F %T') $1" >> $LOG; shift
  "$@" >> $LOG 2>&1 || { echo "=== $(date '+%F %T') FAILED (see above); fix and rerun this script to resume" >> $LOG; exit 1; }
}
export SEEDS=3,4,5,6,7,8,9 SUMMARY=0
stage "v30 seed core: pooled model, exact retrain, random-set retrain, seeds 3 to 9" $P exp_v30_seed_core.py
stage "v16 strict bases" $P exp_v16_strict.py
stage "v17 cross-fitting fold models" $P exp_v17_crossfit.py
stage "v19 protocol components" $P exp_v19_protocol.py
stage "v24 symmetric comparator" $P exp_v24_symmetric_mimic.py
unset SUMMARY
export SEEDS=0,1,2,3,4,5,6,7,8,9 TAG=_s10 PART=mimic
stage "v29 on ten seeds" $P exp_v29_seeds_equivalence.py
echo "=== $(date '+%F %T') all done" >> $LOG
