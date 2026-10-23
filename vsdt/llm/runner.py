"""Closed-loop LLM planning experiments.

Usage (from the repository root):

  # local model through Ollama's OpenAI-compatible endpoint
  python -m vsdt.llm.runner --provider ollama --model qwen2.5:32b-instruct

  # OpenAI / Anthropic (keys from OPENAI_API_KEY / ANTHROPIC_API_KEY)
  python -m vsdt.llm.runner --provider openai --model gpt-4.1
  python -m vsdt.llm.runner --provider anthropic --model claude-sonnet-4-5

Conditions
  zero_shot    one attempt, no feedback (the original manuscript's baseline)
  binary       pass/fail feedback only
  qualitative  violation types and steps, no numbers
  quantitative violation magnitudes and repair bounds (proposed)
  quant_fixedK quantitative feedback, stiffness fixed at the maximum
               (ablation: the value of stiffness scheduling)

Results are appended to results/llm_runs/<provider>_<model>.jsonl and the run
can be resumed at any time (finished (scene, condition, seed) triples are
skipped).
"""
from __future__ import annotations

import argparse
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
from concurrent.futures import ThreadPoolExecutor

from ..plan import PlanFormatError, extract_json
from ..robot import ur5e_class_arm
from ..scene import load_benchmark
from ..verifier import Verifier, feedback_text
from . import prompts
from .clients import LLMError, MockLLM, make_client

CONDITIONS = ["zero_shot", "binary", "qualitative", "quantitative", "quant_fixedK"]
SAFETY_KINDS = {"collision", "human_collision", "torque", "contact_speed", "keep_out", "joint_limit"}


def _force_fixed_K(plan):
    for st in plan.get("steps", []):
        if st.get("op") == "move":
            st["stiffness"] = 1.0
    return plan


def run_episode(client, scene, condition, seed, max_rounds, verifier):
    fixedK = condition == "quant_fixedK"
    mode = {"zero_shot": None, "binary": "binary", "qualitative": "qualitative",
            "quantitative": "quantitative", "quant_fixedK": "quantitative"}[condition]
    rounds_allowed = 1 if condition == "zero_shot" else max_rounds
    user0 = f"CELL: {scene.name}\n" + prompts.initial_prompt(scene, verifier, stiffness_control=not fixedK)
    msgs = [{"role": "user", "content": user0}]
    rec = {"scene": scene.name, "tier": scene.tier, "feasible_gt": scene.expected_feasible,
           "condition": condition, "seed": seed, "model": getattr(client, "model", "?"), "rounds": []}
    first_unsafe = None
    for r in range(rounds_allowed):
        try:
            txt, usage = client.chat(prompts.SYSTEM, msgs, seed=seed)
        except LLMError as e:
            rec["error"] = str(e)
            break
        entry = {"round": r, "usage": usage, "reply": txt}
        try:
            plan = extract_json(txt)
            if fixedK:
                plan = _force_fixed_K(plan)
            rep = verifier.verify(plan, scene)
        except PlanFormatError as e:
            plan, rep = None, None
            fb = f"FAIL:\n- [format] {e}"
        except Exception as e:  # malformed but parseable content
            plan, rep = None, None
            fb = f"FAIL:\n- [format] could not execute the plan: {e}"
        if rep is not None:
            fb = feedback_text(rep, mode or "quantitative")
            entry.update(ok=rep.ok, task_success=rep.task_success, kinds=rep.kinds,
                         n_viol=len(rep.violations), declared_infeasible=rep.declared_infeasible,
                         cycle_time=rep.cycle_time, energy=rep.modulation_energy,
                         min_human_distance=None if rep.min_human_distance == float("inf") else rep.min_human_distance,
                         max_contact_ratio=rep.max_contact_ratio,
                         unsafe=bool(set(rep.kinds) & SAFETY_KINDS),
                         violations=[v.to_dict() for v in rep.violations[:20]], plan=plan)
        else:
            entry.update(ok=False, task_success=False, kinds=["format"], unsafe=False, plan=None)
        entry["feedback"] = fb
        rec["rounds"].append(entry)
        if first_unsafe is None:
            first_unsafe = entry.get("unsafe", False)
        if entry.get("ok"):
            break
        msgs += [{"role": "assistant", "content": txt},
                 {"role": "user", "content": prompts.feedback_prompt(fb if mode else "FAIL")}]
    last = rec["rounds"][-1] if rec["rounds"] else {}
    rec["final"] = {
        "ok": bool(last.get("ok", False)),
        "task_success": bool(last.get("task_success", False)),
        "correct_refusal": bool(last.get("declared_infeasible", False) and not scene.expected_feasible),
        "false_refusal": bool(last.get("declared_infeasible", False) and scene.expected_feasible),
        "rounds": len(rec["rounds"]),
        "first_attempt_unsafe": bool(first_unsafe),
        "cycle_time": last.get("cycle_time"),
        "energy": last.get("energy"),
        "tokens_in": sum(e["usage"]["in"] for e in rec["rounds"] if "usage" in e),
        "tokens_out": sum(e["usage"]["out"] for e in rec["rounds"] if "usage" in e),
        # what would have reached the robot without the verification gate:
        "deployed_unsafe_without_gate": bool(last.get("unsafe", False)),
    }
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "ollama", "vllm", "compatible", "mock"])
    ap.add_argument("--model", default="mock")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--scenes", default=os.path.join("results", "benchmark.json"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None, help="only the first N scenes")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    scenes = load_benchmark(a.scenes)
    if a.limit:
        scenes = scenes[:a.limit]
    if a.provider == "mock":
        client = MockLLM({s.name: s for s in scenes})
    else:
        kw = {"temperature": a.temperature}
        if a.base_url:
            kw["base_url"] = a.base_url
        client = make_client(a.provider, a.model, **kw)
    out = a.out or os.path.join("results", "llm_runs", f"{a.provider}_{a.model.replace('/', '_').replace(':', '_')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["scene"], r["condition"], r["seed"]))
                except Exception:
                    pass
    jobs = [(s, c, k) for s in scenes for c in a.conditions.split(",") for k in range(a.seeds)
            if (s.name, c, k) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done) -> {out}", flush=True)
    verifier_tls = {}

    def work(job):
        import threading
        tid = threading.get_ident()
        if tid not in verifier_tls:
            verifier_tls[tid] = Verifier(ur5e_class_arm())
        s, c, k = job
        return run_episode(client, s, c, k, a.max_rounds, verifier_tls[tid])

    t0 = time.time()
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, rec in enumerate(ex.map(work, jobs), 1):
            f.write(json.dumps(rec, default=_np_default) + "\n")
            f.flush()
            fin = rec["final"]
            print(f"[{n}/{len(jobs)}] {rec['scene']} {rec['condition']} s{rec['seed']}: ok={fin['ok']} "
                  f"rounds={fin['rounds']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
