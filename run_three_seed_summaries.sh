#!/bin/sh
# Recompute the v20, v23, v25 and v26 summaries, and then v37, on seeds 0 to 2 only (24 Nov 2026).
# The Air rerun ran these scripts for seeds 0 to 9, and their summaries read every run on disk, so their intervals now
# pool ten seeds. The Supplementary Appendix reports three seeds. Runs and predictions for seeds 3 to 9 are moved to
# results/_hold_s3_s9 for the duration and moved back at the end; the ten-seed summary files are kept as *_s10 copies.
# Start:  cd ~/mimic && nohup caffeinate -i sh run_three_seed_summaries.sh > /dev/null 2>&1 &
# Follow: tail -f ~/mimic/results/three_seed_summaries.log
cd ~/mimic || exit 1
if pgrep -f "exp_v[0-9]" > /dev/null; then echo "another experiment is running; wait for it to finish"; exit 1; fi
P=/opt/anaconda3/bin/python3; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ROOT=$HOME/mimic
LOG=results/three_seed_summaries.log; H=results/_hold_s3_s9
echo "=== $(date '+%F %T') start" >> $LOG
restore() {
  for d in v20_eicu_protocol v23_eicu_strict v25_symmetric_eicu; do
    for sub in runs preds; do [ -d $H/$d/$sub ] && mv $H/$d/$sub/* results/$d/$sub/ 2>/dev/null; done
  done
  echo "=== $(date '+%F %T') seeds 3 to 9 moved back" >> $LOG
}
trap restore EXIT
for d in v20_eicu_protocol v23_eicu_strict v25_symmetric_eicu v26_eicu_table2 v37_exact_p; do
  for f in intervals.json summary.csv summary.txt all.json multiplicity.csv table_S19.md; do
    [ -e results/$d/$f ] && [ ! -e results/$d/${f%.*}_s10.${f##*.} ] && cp results/$d/$f results/$d/${f%.*}_s10.${f##*.}
  done
done
for d in v20_eicu_protocol v23_eicu_strict v25_symmetric_eicu; do
  for sub in runs preds; do mkdir -p $H/$d/$sub; for f in results/$d/$sub/*__s[3-9].*; do [ -e "$f" ] && mv "$f" $H/$d/$sub/; done; done
done
echo "=== $(date '+%F %T') seeds 3 to 9 held; runs left: v20 $(ls results/v20_eicu_protocol/runs | wc -l), v23 $(ls results/v23_eicu_strict/runs | wc -l), v25 $(ls results/v25_symmetric_eicu/runs | wc -l)" >> $LOG
run() { echo "=== $(date '+%F %T') $1" >> $LOG; shift; "$@" >> $LOG 2>&1 || { echo "=== $(date '+%F %T') FAILED" >> $LOG; exit 1; }; }
run "v20 summary, seeds 0 to 2" $P exp_v20_eicu_protocol.py
run "v23 summary, seeds 0 to 2" $P exp_v23_eicu_strict.py
run "v25 summary, seeds 0 to 2" $P exp_v25_symmetric_eicu.py
for k in 0 1 2 3 4 combine; do run "v26 chunk $k" env CHUNK=$k $P exp_v26_eicu_table2.py; done
run "v37 exact p-values" env FAMILY=v9 $P exp_v37_exact_p.py
echo "=== $(date '+%F %T') all done" >> $LOG
