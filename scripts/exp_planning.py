"""E5: 150-scene benchmark with the algorithmic planners.

  rigid    : stiffness fixed at maximum, speed reduced near the operator
  soft     : stiffness fixed at minimum
  variable : per-joint stiffness scheduling + speed (proposed repair loop)

Every planner output is verified by the full digital twin.  Writes
results/benchmark.json (the scenes; also used by the LLM harness) and
results/e5_planning.jsonl.
"""
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
from vsdt.scene import benchmark, save_benchmark, load_benchmark
from vsdt.verifier import Verifier, VerifierConfig
from vsdt.planners import RepairPlanner

R = os.path.join(os.path.dirname(__file__), "..", "results")
BENCH = os.path.join(R, "benchmark.json")
OUT = os.path.join(R, "e5_planning.jsonl")


def run(args):
    idx, mode = args
    scenes = load_benchmark(BENCH)
    s = scenes[idx]
    if mode == "rigid_robot":     # conventional rigid joints (harmonic drive only)
        v = Verifier(ur5e_class_arm(), cfg=VerifierConfig(rigid_K=3.1e4))
        planner = RepairPlanner(v, "rigid")
    else:
        v = Verifier(ur5e_class_arm())
        planner = RepairPlanner(v, mode)
    t0 = time.time()
    plan, rep, hist = planner.plan(s)
    return {
        "scene": s.name, "tier": s.tier, "feasible_gt": s.expected_feasible, "mode": mode, "task_kind": s.task_kind,
        "ok": rep.ok, "task_success": rep.task_success, "declared_infeasible": rep.declared_infeasible,
        "iterations": len(hist), "cycle_time": rep.cycle_time, "energy": rep.modulation_energy,
        "min_human_distance": None if not np.isfinite(rep.min_human_distance) else rep.min_human_distance,
        "max_contact_ratio": rep.max_contact_ratio, "kinds": rep.kinds,
        "violations": [v_.to_dict() for v_ in rep.violations[:10]],
        "first_kinds": hist[0].kinds if hist else [],
        "plan": plan, "wall_time": time.time() - t0,
    }


if __name__ == "__main__":
    if not os.path.exists(BENCH):
        save_benchmark(benchmark(), BENCH)
    # one verifier per worker: order jobs scene-major so memo caches are reused
    scenes = load_benchmark(BENCH)
    modes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["variable", "rigid", "rigid_robot", "soft"]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line); done.add((r["scene"], r["mode"]))
    jobs = [(i, m) for i, s in enumerate(scenes) for m in modes if (s.name, m) not in done]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with Pool(2) as pool, open(OUT, "a") as f:
        for n, r in enumerate(pool.imap_unordered(run, jobs), 1):
            f.write(json.dumps(r, default=_np_default) + "\n"); f.flush()
            print(f"[{n}/{len(jobs)}] {r['scene']} {r['mode']} ok={r['ok']} it={r['iterations']} "
                  f"T={r['cycle_time']:.1f}s {r['wall_time']:.0f}s elapsed={time.time()-t0:.0f}s", flush=True)
