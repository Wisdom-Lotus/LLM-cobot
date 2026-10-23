"""Aggregate LLM runs into the paper's table and statistics.

  python -m vsdt.llm.analyze results/llm_runs/*.jsonl

Writes results/llm_summary.json and results/llm_table.tex (included by the
manuscript).  Statistics: Wilson 95 % intervals for proportions; exact
McNemar test (paired by scene and seed) of each condition against
'quantitative'; Holm correction across the comparisons.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from math import comb, sqrt

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1, running)
    return adj


def load(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


def summarize(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["model"], r["condition"])].append(r)
    out = {}
    for (m, c), rs in sorted(by.items()):
        feas = [r for r in rs if r["feasible_gt"]]
        infe = [r for r in rs if not r["feasible_gt"]]
        succ = sum(r["final"]["ok"] for r in rs)
        out[f"{m}|{c}"] = {
            "model": m, "condition": c, "n": len(rs),
            "success": wilson(succ, len(rs)),
            "success_feasible": wilson(sum(r["final"]["ok"] for r in feas), len(feas)),
            "correct_refusal": wilson(sum(r["final"]["correct_refusal"] for r in infe), len(infe)),
            "false_refusal": wilson(sum(r["final"]["false_refusal"] for r in feas), len(feas)),
            "first_attempt_unsafe": wilson(sum(r["final"]["first_attempt_unsafe"] for r in rs), len(rs)),
            "unsafe_without_gate": wilson(sum(r["final"]["deployed_unsafe_without_gate"] for r in rs), len(rs)),
            "rounds_mean": float(np.mean([r["final"]["rounds"] for r in rs])),
            "cycle_time_median": float(np.median([r["final"]["cycle_time"] for r in rs if r["final"]["ok"]
                                                  and r["final"]["cycle_time"]] or [np.nan])),
            "tokens_mean": float(np.mean([r["final"]["tokens_in"] + r["final"]["tokens_out"] for r in rs])),
            "by_tier": {t: wilson(sum(r["final"]["ok"] for r in rs if r["tier"] == t),
                                  sum(1 for r in rs if r["tier"] == t)) for t in (1, 2, 3)},
        }
    # paired tests vs quantitative
    tests = {}
    for m in sorted({r["model"] for r in recs}):
        ref = {(r["scene"], r["seed"]): r["final"]["ok"] for r in recs
               if r["model"] == m and r["condition"] == "quantitative"}
        names, ps = [], []
        for c in sorted({r["condition"] for r in recs if r["model"] == m} - {"quantitative"}):
            other = {(r["scene"], r["seed"]): r["final"]["ok"] for r in recs
                     if r["model"] == m and r["condition"] == c}
            keys = ref.keys() & other.keys()
            b = sum(ref[k] and not other[k] for k in keys)
            cc = sum(other[k] and not ref[k] for k in keys)
            names.append(c); ps.append(mcnemar_exact(b, cc))
            tests[f"{m}|{c}"] = {"n_pairs": len(keys), "quant_only": b, "other_only": cc, "p": ps[-1]}
        if ps:
            for n_, pa in zip(names, holm(np.array(ps))):
                tests[f"{m}|{n_}"]["p_holm"] = float(pa)
    return out, tests


def latex_table(out, tests):
    fmt = lambda w: "--" if np.isnan(w[0]) else f"{100*w[0]:.1f} [{100*w[1]:.0f}, {100*w[2]:.0f}]"
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Condition & Success (\%) & Unsafe 1st plan (\%) & Rounds & $p_\mathrm{Holm}$ \\",
             r"\midrule"]
    for key, s in out.items():
        t = tests.get(key, {})
        p = t.get("p_holm")
        lines.append(f"{s['model']} & {s['condition'].replace('_', ' ')} & {fmt(s['success'])} & "
                     f"{fmt(s['first_attempt_unsafe'])} & {s['rounds_mean']:.2f} & "
                     f"{'--' if p is None else ('<0.001' if p < 1e-3 else f'{p:.3f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv=None):
    argv = argv or sys.argv[1:]
    paths = [p for a in argv for p in glob.glob(a)] or glob.glob("results/llm_runs/*.jsonl")
    recs = [r for r in load(paths) if r.get("model") != "mock"]
    if not recs:
        print("no (non-mock) runs found"); return
    out, tests = summarize(recs)
    with open("results/llm_summary.json", "w") as f:
        json.dump({"summary": out, "tests": tests}, f, indent=1)
    with open("results/llm_table.tex", "w") as f:
        f.write(latex_table(out, tests))
    for k, s in out.items():
        print(k, "success %.3f [%.3f, %.3f]" % s["success"], "rounds %.2f" % s["rounds_mean"])
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    main()
