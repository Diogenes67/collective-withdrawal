#!/bin/sh
# One fresh process per (consortium, seed): finished runs are skipped, so this resumes. Summary and bootstrap at the end.
cd ~/mimic
P=/opt/anaconda3/bin/python3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for C in region/Midwest region/Northeast region/South region/West teaching/f teaching/t "bedsize/<100" "bedsize/100_-_249" "bedsize/250_-_499" "bedsize/>=_500"; do
  for S in 0 1 2; do
    ONLY="$C" SEED=$S SUMMARY=0 $P exp_v20_eicu_protocol.py >> results/v20_1.log 2>&1
  done
done
$P exp_v20_eicu_protocol.py >> results/v20_1.log 2>&1
