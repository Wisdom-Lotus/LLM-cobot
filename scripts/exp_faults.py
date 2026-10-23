"""E4: verifier soundness by fault injection.

Faults are injected into plans that the full verifier accepted in E5
(variable mode).  Ground truth for each faulty plan is a brute-force
reference check: the full verifier with dense fixed sampling (150
samples per move) -- a fault counts only if the reference flags it.

Compared checkers
  kinematic   : reach + joint limits + collision on the commanded
                configuration with 10 fixed samples per move, no torque,
                precision or contact checks (the original manuscript)
  kin_dense   : as above but certified (conservative advancement)
  no_contact  : full model but without the ISO/TS 15066 contact check
  rigid_model : full checks but the impact model assumes a rigid joint
                (ISO formula with configuration-dependent mass)
  full        : proposed verifier
"""
import copy
import json


def _np_default(o):
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsdt.robot import ur5e_class_arm
from vsdt.scene import load_benchmark, Box
from vsdt.verifier import Verifier, VerifierConfig

R = os.path.join(os.path.dirname(__file__), "..", "results")
SAFETY = {"collision", "human_collision", "torque", "contact_speed", "keep_out", "joint_limit", "precision",
          "semantic", "reach", "task"}


def checkers():
    return {
        "kinematic": VerifierConfig(check_torque=False, check_precision=False, check_contact=False,
                                    deflected_geometry=False, certified=False, n_fixed_samples=10),
        "kin_certified": VerifierConfig(check_torque=False, check_precision=False, check_contact=False,
                                        deflected_geometry=False, certified=True),
        "no_contact": VerifierConfig(check_contact=False),
        "full": VerifierConfig(),
        "reference": VerifierConfig(certified=False, n_fixed_samples=150),
    }


def moves(plan):
    return [i for i, s in enumerate(plan["steps"]) if s["op"] == "move"]


def inject(plan, scene, kind, rng):
    """Return (faulty_plan, scene_used) or None if not applicable."""
    p = copy.deepcopy(plan)
    st = p["steps"]
    mv = moves(p)
    if kind == "obstacle_intrusion":
        # first transit way-point driven 3 cm into the source station
        k = mv[0]
        src = next(o for o in scene.obstacles if o.id == "source_station")
        cx, cy = (src.lo[0] + src.hi[0]) / 2, (src.lo[1] + src.hi[1]) / 2
        st[k]["target"] = [round(cx + 0.1, 3), round(cy + 0.1, 3), round(src.hi[2] - 0.03, 3)]
        st[k].pop("offset", None)
        return p, scene
    if kind == "thin_obstacle":
        # add a 6 mm plate across the longest transit move
        sc = copy.deepcopy(scene)
        k = mv[len(mv) // 2]
        v = Verifier(ur5e_class_arm())
        try:
            a = v._resolve_target(st[k - 1] if st[k - 1]["op"] == "move" else st[mv[0]], sc)
        except Exception:
            return None
        b = v._resolve_target(st[k], sc)
        mid = 0.5 * (a + b)
        d = b - a
        if abs(d[0]) > abs(d[1]):
            lo, hi = [mid[0] - 0.003, mid[1] - 0.6, mid[2] - 0.6], [mid[0] + 0.003, mid[1] + 0.6, mid[2] + 0.6]
        else:
            lo, hi = [mid[0] - 0.6, mid[1] - 0.003, mid[2] - 0.6], [mid[0] + 0.6, mid[1] + 0.003, mid[2] + 0.6]
        sc.obstacles.append(Box("thin_guard", [round(x, 4) for x in lo], [round(x, 4) for x in hi], "guard plate"))
        return p, sc
    if kind == "under_stiffness":
        # softest setting on every loaded move
        changed = False
        held = False
        for s_ in st:
            if s_["op"] == "grasp":
                held = True
            if s_["op"] == "release":
                held = False
            if s_["op"] == "move" and held:
                s_["stiffness"] = 0.0
                changed = True
        return (p, scene) if changed else None
    if kind == "overspeed_near_human":
        if not scene.human:
            return None
        for i in mv:
            st[i]["speed"] = 1.0
        return p, scene
    if kind == "rigid_assumption":
        # plan tuned for soft joints but executed stiff near the operator
        if not scene.human:
            return None
        for i in mv:
            st[i]["stiffness"] = 1.0
            st[i]["speed"] = min(1.0, st[i].get("speed", 0.5) * 1.6)
        return p, scene
    if kind == "wrong_item":
        for s_ in st:
            if s_["op"] == "grasp":
                s_["item"] = "item_2"
        return p, scene
    if kind == "place_offset":
        k = [i for i in mv if i < len(st) - 1 and st[i + 1]["op"] == "release"]
        if not k:
            return None
        t = st[k[0]]["target"]
        if isinstance(t, str):
            return None
        st[k[0]]["target"] = [t[0] + 0.12, t[1], t[2]]
        return p, scene
    raise ValueError(kind)


FAULTS = ["obstacle_intrusion", "thin_obstacle", "under_stiffness", "overspeed_near_human",
          "rigid_assumption", "wrong_item", "place_offset"]


def job(args):
    rec, seed = args
    scenes = {s.name: s for s in load_benchmark(os.path.join(R, "benchmark.json"))}
    scene = scenes[rec["scene"]]
    rng = np.random.default_rng(seed)
    vers = {name: Verifier(ur5e_class_arm(), cfg=cfg) for name, cfg in checkers().items()}
    outs = []
    for kind in FAULTS:
        res = inject(rec["plan"], scene, kind, rng)
        if res is None:
            continue
        plan, sc = res
        out = {"scene": rec["scene"], "tier": rec["tier"], "fault": kind}
        for name, v in vers.items():
            t0 = time.time()
            r = v.verify(plan, sc)
            out[name] = {"flag": not r.ok, "kinds": r.kinds, "time": time.time() - t0}
        outs.append(out)
    return outs


if __name__ == "__main__":
    recs = [json.loads(l) for l in open(os.path.join(R, "e5_planning.jsonl"))]
    valid = [r for r in recs if r["mode"] == "variable" and r["ok"] and r["plan"]["feasible"]]
    # stratified subset: up to N_PER_TIER plans per tier (deterministic)
    N_PER_TIER = int(os.environ.get("N_PER_TIER", 15))
    sel = []
    for t in (1, 2, 3):
        vt = sorted([r for r in valid if r["tier"] == t], key=lambda r: r["scene"])
        sel += vt[:N_PER_TIER]
    out_path = os.path.join(R, "e4_faults.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            done.add(json.loads(l)["scene"])
    jobs = [(r, i) for i, r in enumerate(sel) if r["scene"] not in done]
    print(len(valid), "valid plans;", len(sel), "selected;", len(jobs), "to run", flush=True)
    t0 = time.time()
    with Pool(2) as pool, open(out_path, "a") as f:
        for n, outs in enumerate(pool.imap_unordered(job, jobs), 1):
            for o in outs:
                f.write(json.dumps(o, default=_np_default) + "\n")
            f.flush()
            print(n, "plans done", f"{time.time()-t0:.0f}s", flush=True)
