#!/bin/sh
# Rerun on the Air of every analysis that used Colab (18 Nov 2026, PhysioNet DUA). Colab ran v7.2 and v12 (13-14 Oct),
# eICU seeds 3-9 (v7.2, v20, v23, v25, v39, v29) and the extended tuning grid v38 (31 Oct - 1 Nov). Every eICU result
# downstream of the Colab v7.2 checkpoints is rebuilt here; the MIMIC-IV chain ran on the Air and is not retrained
# (its summaries are recomputed with the same stored random streams, so MIMIC-IV numbers reproduce).
# Start:   cd ~/mimic && nohup caffeinate -i sh run_air_rerun.sh > /dev/null 2>&1 &
# Follow:  tail -f ~/mimic/results/air_rerun.log      Resume after any stop: run the same start command again.
cd ~/mimic || exit 1
if pgrep -f "exp_v[0-9]" > /dev/null; then echo "another experiment is running; wait for it to finish"; exit 1; fi
P=/opt/anaconda3/bin/python3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ROOT=$HOME/mimic
LOG=results/air_rerun.log
stage() {
  key=$(printf '%s' "$1" | cksum | cut -d' ' -f1)
  if [ -f results/.done_$key ]; then echo "=== $(date '+%F %T') skip (done): $1" >> $LOG; return; fi
  echo "=== $(date '+%F %T') $1" >> $LOG; shift
  "$@" >> $LOG 2>&1 || { echo "=== $(date '+%F %T') FAILED (see above); fix and rerun this script to resume" >> $LOG; exit 1; }
  touch results/.done_$key
}
archive() {
  A=results/_colab_superseded_20261118; mkdir -p $A/partial
  for d in model_cache_eicu_es v7_eicu_consortium_es v7_eicu_consortium_es_results.zip v12_eicu_federated_es \
           v12_eicu_federated_es_results.zip v18_landmark v20_eicu_protocol v23_eicu_strict v25_symmetric_eicu \
           v26_eicu_table2 v35_landmark v38_tuned_unlearning_ext colab_v9 v29_seeds_equivalence_eicu_s10 v39_eicu_seed_core; do
    [ -e results/$d ] && mv results/$d $A/
  done
  # mixed MIMIC-IV/eICU folders: keep a full copy, then move only the eICU caches aside so the scripts rebuild them
  for d in v13_intervals_1 v27_churn v29_seeds_equivalence v31_fold_manifests v32_pooled_full_adapt v36_others \
           v42_strict_penalty v34_isolation v37_exact_p v43_isolation_adjusted; do
    [ -e results/$d ] && [ ! -e $A/partial/$d ] && cp -R results/$d $A/partial/$d
  done
  mkdir -p $A/eicu_caches/v13 $A/eicu_caches/v27 $A/eicu_caches/v29 $A/eicu_caches/v31 $A/eicu_caches/v32/runs \
           $A/eicu_caches/v32/preds $A/eicu_caches/v36 $A/eicu_caches/v42/runs $A/eicu_caches/v42/preds
  for f in preds_eicu.npz membership_eicu.json extra_chunk0.npz extra_chunk1.npz extra_chunk2.npz extra_chunk3.npz \
           extra_chunk4.npz extra_contrasts.json; do [ -e results/v13_intervals_1/$f ] && mv results/v13_intervals_1/$f $A/eicu_caches/v13/; done
  for f in rows_eicu.npz per_group_eicu.csv; do [ -e results/v27_churn/$f ] && mv results/v27_churn/$f $A/eicu_caches/v27/; done
  [ -e results/v29_seeds_equivalence/draws_eicu.npz ] && mv results/v29_seeds_equivalence/draws_eicu.npz $A/eicu_caches/v29/
  [ -e results/v31_fold_manifests/folds/eicu ] && mv results/v31_fold_manifests/folds/eicu $A/eicu_caches/v31/
  for f in results/v32_pooled_full_adapt/runs/*; do case $(basename "$f") in community_*) ;; *) mv "$f" $A/eicu_caches/v32/runs/;; esac; done
  for f in results/v32_pooled_full_adapt/preds/*; do case $(basename "$f") in community_*) ;; *) mv "$f" $A/eicu_caches/v32/preds/;; esac; done
  for f in results/v36_others/draws_eicu_*; do [ -e "$f" ] && mv "$f" $A/eicu_caches/v36/; done
  for f in results/v42_strict_penalty/runs/eicu__* ; do [ -e "$f" ] && mv "$f" $A/eicu_caches/v42/runs/; done
  for f in results/v42_strict_penalty/preds/eicu__* ; do [ -e "$f" ] && mv "$f" $A/eicu_caches/v42/preds/; done
  echo "archived to $A"
}
CONS='region/Midwest region/Northeast region/South region/West teaching/f teaching/t bedsize/<100 bedsize/100_-_249 bedsize/250_-_499 bedsize/>=_500'
stage "air: archive Colab-derived results" archive
# eICU centralised chain, ten seeds
stage "air: v7.2 eICU consortium withdrawal, seeds 0 to 9" env SEEDS=0,1,2,3,4,5,6,7,8,9 $P exp_v7_eicu_consortium_v72.py
stage "air: v12 eICU federation, three seeds" $P exp_v12_eicu_federated_v122.py
for s in 0 1 2 3 4 5 6 7 8 9; do
  for C in region/Midwest region/Northeast region/South region/West teaching/f teaching/t "bedsize/<100" "bedsize/100_-_249" "bedsize/250_-_499" "bedsize/>=_500"; do
    stage "air: v20 protocol $C seed $s" env ONLY="$C" SEED=$s SUMMARY=0 $P exp_v20_eicu_protocol.py
  done
  for i in 0 1 2 3 4 5 6 7 8 9; do
    stage "air: v23 strict base consortium $i seed $s" env IDX=$i SEED=$s SUMMARY=0 $P exp_v23_eicu_strict.py
    stage "air: v25 symmetric comparator consortium $i seed $s" env IDX=$i SEED=$s SUMMARY=0 $P exp_v25_symmetric_eicu.py
  done
done
stage "air: v20 summary" $P exp_v20_eicu_protocol.py
stage "air: v23 summary" $P exp_v23_eicu_strict.py
stage "air: v25 summary" $P exp_v25_symmetric_eicu.py
stage "air: v23 strict base on other patients" env OTHERS=1 $P exp_v23_eicu_strict.py
# intervals and contrasts
stage "air: v13 membership" env MEMBERSHIP_ONLY=1 PART=both $P exp_v13_intervals.py
stage "air: v13 intervals" env PART=both $P exp_v13_intervals.py
stage "air: v13b matched flags" env PART=both $P exp_v13b_matched_flags.py
for k in 0 1 2 3 4 combine; do stage "air: v13c extra contrasts chunk $k" env CHUNK=$k $P exp_v13c_extra_contrasts.py; done
for pop in landmark all; do for k in 0 1 2 3 4 combine; do stage "air: v18 $pop chunk $k" env POP=$pop CHUNK=$k $P exp_v18_landmark.py; done; done
for k in 0 1 2 3 4 combine; do stage "air: v26 chunk $k" env CHUNK=$k $P exp_v26_eicu_table2.py; done
stage "air: v27 other patients' predictions" env PART=both $P exp_v27_churn.py
stage "air: v32 full pooled comparator" env PART=both $P exp_v32_pooled_full_adapt.py
stage "air: v42 complete-pipeline penalty" $P exp_v42_strict_penalty.py
stage "air: v31 fold manifests" env PART=both $P exp_v31_fold_manifests.py
stage "air: v35 landmark" $P exp_v35_landmark.py
stage "air: v36 other patients' AUROC" $P exp_v36_others.py
stage "air: v39 eICU prediction cache, seeds 3 to 9" env SEEDS=3,4,5,6,7,8,9 $P exp_v39_eicu_seed_preds.py
stage "air: v29 three seeds" $P exp_v29_seeds_equivalence.py
stage "air: v29 ten eICU seeds" env PART=eicu ESEEDS=0,1,2,3,4,5,6,7,8,9 TAG=_eicu_s10 $P exp_v29_seeds_equivalence.py
stage "air: v34 isolation regression" $P exp_v34_isolation.py
stage "air: v43 adjusted isolation regression" $P exp_v43_isolation_adjusted.py
# MIMIC-IV tuned unlearning (extended grid). One process per group: a single process over all 25 groups ran out of MPS
# memory at c13 on 21 Nov 2026. Finished runs are skipped, so the last v38 line only writes the summary.
for c in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
  stage "air: v38 extended grid, group c$c" env GRID=ext SEEDS=0,1,2 SEEDS_TUNE=0 ONLY_GROUPS=c$c SUMMARY=0 $P exp_v38_tuned_unlearning.py
done
stage "air: v38 tuned unlearning, extended grid" env GRID=ext SEEDS=0,1,2 SEEDS_TUNE=0 $P exp_v38_tuned_unlearning.py
stage "air: v38m removal-matched comparison" $P exp_v38m_matched.py
stage "air: v37 exact p-values" env FAMILY=v9 $P exp_v37_exact_p.py
echo "=== $(date '+%F %T') all done; compare with results/_colab_superseded_20261118" >> $LOG
