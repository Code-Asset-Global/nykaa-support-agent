"""calibrate.py - Task 4: similarity naapkar 'I don't know' threshold tay karo."""
import json
from rag import retrieve
from queries import CALIB_IN_SCOPE, CALIB_OUT_OF_SCOPE

thresholds = {}
for coll in ["kb_fixed", "kb_sentence"]:
    print(f"\n===== {coll} =====")
    in_sims, out_sims = [], []
    print("IN-SCOPE (top-1 cosine similarity):")
    for q in CALIB_IN_SCOPE:
        s = retrieve(q, coll, k=1)[0]["similarity"]
        in_sims.append(s)
        print(f"  {s:.4f}  | {q}")
    print("OUT-OF-SCOPE:")
    for q in CALIB_OUT_OF_SCOPE:
        s = retrieve(q, coll, k=1)[0]["similarity"]
        out_sims.append(s)
        print(f"  {s:.4f}  | {q}")
    min_in, max_out = min(in_sims), max(out_sims)
    print(f"min in-scope = {min_in:.4f}, max out-of-scope = {max_out:.4f}")
    if min_in > max_out:
        thr = round((min_in + max_out) / 2, 3)
        print(f"Clusters alag hain -> threshold = midpoint = {thr}")
    else:
        thr = round(max_out + 0.01, 3)
        print(f"WARNING: clusters overlap. Temporary threshold = {thr}. Result mujhe dikhao.")
    thresholds[coll] = thr

with open("threshold.json", "w") as f:
    json.dump(thresholds, f, indent=2)
print("\nSaved threshold.json:", thresholds)