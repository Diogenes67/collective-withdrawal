"""export_cluster_assign.py. Writes the 25 clustered groups exactly as fitted on this machine (the v5 clustering) to
derived/cluster_assign_v5.csv (subject_id, cluster) so that Colab runs use the same groups as the v5 checkpoints, and checks
every group's training and held-out patient counts and random set against the v5 run records. Run on the Mac with the
Python that ran v5: /opt/anaconda3/bin/python3 export_cluster_assign.py"""
import os, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
ROOT = os.environ.get("ROOT", os.path.expanduser("~/mimic")); V5 = f"{ROOT}/results/v5_partitions"
META = ["subject_id", "hadm_id", "admittime", "dischtime", "anchor_year_group", "admission_type", "admission_location", "discharge_location", "insurance", "marital_status", "race_group", "readmit30"]
df = pd.read_parquet(f"{ROOT}/derived/cohort_latest.parquet"); feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float)
for c in [c for c in feat if c.startswith("lab_")]:
    X[c + "_missing"] = X[c].isna().astype(float); X[c] = X[c].fillna(X[c].median())
X = X.values.astype(np.float32); sid = df.subject_id.values
rng = np.random.RandomState(0); pids = df.subject_id.unique(); rng.shuffle(pids); test_p = set(pids[: int(0.2 * len(pids))])
is_test = df.subject_id.isin(test_p).values; is_train = ~is_test
mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-6; X = (X - mu) / sd
pat_mean = pd.DataFrame(X).groupby(sid).mean(); train_pats = np.array(sorted(set(sid[is_train])))
km = KMeans(n_clusters=25, n_init=3, random_state=0).fit(pat_mean.loc[train_pats].values); assign = pd.Series(km.predict(pat_mean.values), index=pat_mean.index)
bad = 0
for c in range(25):
    mem = np.isin(sid, assign.index[assign == c].values); r = json.load(open(f"{V5}/runs/community_trainfit__c{c:02d}__s0.json"))
    ok = r["n_train_patients"] == len(set(sid[is_train & mem])) and r["n_test_patients"] == len(set(sid[is_test & mem])); bad += not ok
    print(f"c{c:02d}: train {len(set(sid[is_train & mem]))} (v5 {r['n_train_patients']}), held-out {len(set(sid[is_test & mem]))} (v5 {r['n_test_patients']}) {'ok' if ok else 'MISMATCH'}")
if bad: raise SystemExit(f"{bad} groups differ from v5; not written")
out = f"{ROOT}/derived/cluster_assign_v5.csv"; pd.DataFrame(dict(subject_id=assign.index, cluster=assign.values)).to_csv(out, index=False)
print("written", out, len(assign), "patients")
