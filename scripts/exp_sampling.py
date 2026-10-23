"""E4b: does certified sampling matter?  A thin vertical post (8 mm x 8 mm)
is placed on the swept path of a transit move of each verified plan.
Checkers (kinematic geometry only, to isolate sampling): 10 fixed samples
per move, certified conservative advancement, and a dense reference
(500 samples per move)."""
import copy
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsdt.robot import ur5e_class_arm
from vsdt.scene import load_benchmark, Box
from vsdt.verifier import Verifier, VerifierConfig

R = os.path.join(os.path.dirname(__file__), "..", "results")
KIN = dict(check_torque=False, check_precision=False, check_contact=False, deflected_geometry=False)


def job(rec):
    scenes = {s.name: s for s in load_benchmark(os.path.join(R, "benchmark.json"))}
    scene = scenes[rec["scene"]]
    ref = Verifier(ur5e_class_arm(), cfg=VerifierConfig(certified=False, n_fixed_samples=500, **KIN))
    r0 = ref.verify(rec["plan"], scene)
    # pick a TCP sample in the middle of the longest move and put a post there
    tr = r0.trace
    steps = np.array(tr["step"])
    if len(steps) == 0:
        return None
    k = max(set(steps.tolist()), key=lambda s: (steps == s).sum())
    idx = np.where(steps == k)[0]
    # choose a sample between two of the 10 fixed samples (worst case for coarse sampling)
    i = idx[int(len(idx) * 0.15)]
    p = np.array(tr["tcp"][i])
    sc = copy.deepcopy(scene)
    sc.obstacles.append(Box("thin_post", [round(p[0] - 0.004, 4), round(p[1] - 0.004, 4), round(p[2] - 0.02, 4)],
                            [round(p[0] + 0.004, 4), round(p[1] + 0.004, 4), round(p[2] + 0.02, 4)], "post"))
    out = {"scene": rec["scene"], "tier": rec["tier"], "step": int(k)}
    for name, cfg in (("fixed10", VerifierConfig(certified=False, n_fixed_samples=10, **KIN)),
                      ("fixed50", VerifierConfig(certified=False, n_fixed_samples=50, **KIN)),
                      ("certified", VerifierConfig(certified=True, **KIN)),
                      ("reference", VerifierConfig(certified=False, n_fixed_samples=500, **KIN))):
        r = Verifier(ur5e_class_arm(), cfg=cfg).verify(rec["plan"], sc)
        out[name] = bool(any(v.kind == "collision" and v.data.get("obstacle") == "thin_post" for v in r.violations))
    return out


if __name__ == "__main__":
    recs = [json.loads(l) for l in open(os.path.join(R, "e5_planning.jsonl"))]
    valid = [r for r in recs if r["mode"] == "variable" and r["ok"] and r["plan"]["feasible"]]
    sel = []
    for t in (1, 2, 3):        # same stratified subset as E4
        sel += sorted([r for r in valid if r["tier"] == t], key=lambda r: r["scene"])[:15]
    valid = sel
    with Pool(2) as pool:
        res = [r for r in pool.map(job, valid) if r is not None]
    with open(os.path.join(R, "e4b_sampling.json"), "w") as f:
        json.dump(res, f)
    real = [r for r in res if r["reference"]]
    print(len(res), "plans,", len(real), "real collisions")
    for k in ("fixed10", "fixed50", "certified"):
        print(k, "recall", np.mean([r[k] for r in real]), "false alarms", sum(r[k] and not r["reference"] for r in res))
