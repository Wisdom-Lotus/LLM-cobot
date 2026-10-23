"""Write paper/numbers.tex and the result tables from results/*.

Every number quoted in the manuscript is a macro defined here, so the text
cannot drift from the code output.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(os.path.dirname(__file__), "..", "results")
P = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(P, exist_ok=True)
M = {}


def f(x, n=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def pct(x, n=0):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{n}f}"


# ------------------------------------------------------------------ E1
e1 = json.load(open(os.path.join(R, "e1_vssea.json")))
M["Kmin"] = f(e1["range"]["K_joint_min"], 1)
M["Kmax"] = f"{e1['range']['K_joint_max']:.0f}"
M["Kratio"] = f"{e1['range']['ratio']:.0f}"
M["KvsamMax"] = f"{e1['range']['K_vsam_max_timo']:.0f}"
M["FEerrMaxPct"] = f(100 * max(abs(x) for x in e1["stiffness"]["fe_vs_timo_rel"]), 2)
fe = e1["stiffness"]["fe_mesh_convergence_x0.03"]
for key, name in zip(["nx30_ny1", "nx60_ny2", "nx120_ny4", "nx240_ny6"], ["feA", "feB", "feC", "feD"]):
    M[name] = f(fe[key] / 1000, 1)
from vsdt.vssea import VSAMParams, tau_capacity, tau_max_stress, qd_max_stress
p = VSAMParams()
M["taucapSoft"] = f(float(tau_capacity(p, p.x_max)), 1)
M["qdmaxSoft"] = f"{np.rad2deg(float(qd_max_stress(p, p.x_max))):.0f}"
xs = np.linspace(p.x_min, p.x_max, 60)
drop = 1 - tau_capacity(p, xs) / tau_max_stress(p, xs)
M["nlCapDropPct"] = f"{100 * drop.max():.0f}"
nl = e1["nonlinear"]
M["nlErrStiffPct"] = f(100 * nl["0.04"]["rel_err_at_qmax"], 1)
M["nlErrSoftPct"] = f"{100 * nl['0.17']['rel_err_at_qmax']:.0f}"
m = e1["modulation"]
M["tmod"] = f(m["t_full_range"], 2)
M["modrate"] = f"{m['rate_Nm_per_rad_per_s']:.0f}"
M["Eeq"] = f(m["E_eq_full_range"], 3)
M["Eload"] = f(m["E_loaded"]["0.020"], 2)
M["qdLoadDeg"] = f(np.rad2deg(0.02), 2)
M["tauMone"] = "100"

# ------------------------------------------------------------------ E2/E3
e23 = json.load(open(os.path.join(R, "e2_e3.json")))
wo, ws = e23["workspace_original"], e23["workspace"]
M["reachOrig"] = f(wo["max_reach"], 1)
M["feasOrigPct"] = pct(wo["frac_torque_feasible_2.5kg"])
M["maxTauOrig"] = f"{wo['max_tau_per_joint_2.5kg'][1]:.0f}"
M["reachStudy"] = f(ws["max_reach"], 2)
M["feasStudyPct"] = pct(ws["frac_torque_feasible_2.5kg"])
M["maxTauStudy"] = f"{ws['max_tau_per_joint_2.5kg'][1]:.0f}"
M["softOkStudyPct"] = pct(ws["frac_all_joints_soft_ok_2.5kg"])
ik = e23["ik"]
M["ikSuccPct"] = pct(ik["analytic"]["success"])
M["ikItMed"] = f"{ik['analytic']['median_iter']:.0f}"
M["ikTimeA"] = f"{ik['analytic']['time_ms']:.0f}"
M["ikTimeFD"] = f"{ik['fd']['time_ms']:.0f}"
M["ikSpeedup"] = f(ik["fd"]["time_ms"] / ik["analytic"]["time_ms"], 1)
c = e23["contact"]
K = np.array(c["K_joint"])
iK, iR = int(np.searchsorted(K, c["K_vssea_range"][1])), int(np.searchsorted(K, c["K_hd"]))
cv = c["curves"]
M["chestSoft"] = f(cv["chest"]["vmax"][0])
M["chestStiff"] = f(cv["chest"]["vmax"][iK])
M["chestRigid"] = f(cv["chest"]["vmax"][iR])
M["chestGainPct"] = f"{100 * (cv['chest']['vmax'][iK] / cv['chest']['vmax'][iR] - 1):.0f}"
M["handClampSoft"] = f(cv["hand"]["vmax_clamped"][0])
M["handClampStiff"] = f(cv["hand"]["vmax_clamped"][iK])
M["handClampRigid"] = f(cv["hand"]["vmax_clamped"][iR])
M["armClampSoft"] = f(cv["lower_arm"]["vmax_clamped"][0])
M["armClampStiff"] = f(cv["lower_arm"]["vmax_clamped"][iK])
M["mlRep"] = f(cv["chest"]["m_l"], 1)
M["mmRep"] = f(cv["chest"]["m_m"], 1)
M["isoTSchest"] = f(cv["chest"]["iso_TS_formula"])

# secondary impacts: systematic sweep over realistic reduced models
from vsdt.safety import ImpactModel, ISO_BODY
IM = ImpactModel()
rng = np.random.default_rng(3)
worst = 0.0
for _ in range(200):
    reg = rng.choice(["chest", "abdomen", "back", "thigh"])
    k, mH = ISO_BODY[reg][2], ISO_BODY[reg][3]
    ml, mm, Kc = 10 ** rng.uniform(0.3, 1.3), 10 ** rng.uniform(0, 1.3), 10 ** rng.uniform(1.5, 4.3)
    g, g1 = IM.gain(mm, ml, Kc, k, mH), IM.gain(mm, ml, Kc, k, mH, first_phase_only=True)
    worst = max(worst, g / g1 - 1)
M["secondaryMax"] = f"{100 * worst:.0f}"

# lookup conservativeness
from vsdt.contact import gain_lookup
under, over = 0, []
for _ in range(300):
    reg = rng.choice(["chest", "hand", "lower_arm", "abdomen", "thigh", "neck", "upper_arm", "back"])
    ml, mm, Kc = 10 ** rng.uniform(-0.3, 3), 10 ** rng.uniform(-0.5, 3), 10 ** rng.uniform(1.2, 6.5)
    k, mH = ISO_BODY[reg][2], ISO_BODY[reg][3]
    g, gl = IM.gain(mm, ml, Kc, k, mH), gain_lookup(reg, ml, mm, Kc)
    under += gl < g
    over.append(gl / g - 1)
M["lookupUnder"] = str(under)
M["lookupOverMed"] = f(100 * np.median(over), 1)

# ------------------------------------------------------------------ E5
e5p = os.path.join(R, "e5_planning.jsonl")
NAMES = {"rigid_robot": "Rigid-joint robot", "rigid": "VSSEA, fixed stiff",
         "soft": "VSSEA, fixed soft", "variable": "VSSEA, scheduled (ours)"}
if os.path.exists(e5p):
    recs = [json.loads(l) for l in open(e5p)]
    by = defaultdict(dict)
    for r in recs:
        by[r["scene"]][r["mode"]] = r
    modes = [mm for mm in ["rigid_robot", "rigid", "soft", "variable"] if any(mm in d for d in by.values())]
    nfeas = sum(1 for d in by.values() if next(iter(d.values()))["feasible_gt"])
    ninf = len(by) - nfeas
    M["nInfeasible"] = str(ninf)
    rows = []
    stats = {}
    for mm in modes:
        rs = [d[mm] for d in by.values() if mm in d]
        feas = [r for r in rs if r["feasible_gt"]]
        inf = [r for r in rs if not r["feasible_gt"]]
        succ = np.mean([r["ok"] for r in feas]) if feas else np.nan
        refu = np.mean([r["ok"] for r in inf]) if inf else np.nan
        ct = [r["cycle_time"] for r in feas if r["ok"]]
        ratio = [r["max_contact_ratio"] for r in feas if r["ok"] and r["min_human_distance"] is not None]
        tiers = [np.mean([r["ok"] for r in feas if r["tier"] == t]) for t in (1, 2, 3)]
        stats[mm] = dict(succ=succ, refu=refu, ct=np.median(ct) if ct else np.nan,
                         ratio=np.median(ratio) if ratio else np.nan, tiers=tiers,
                         energy=np.median([r["energy"] for r in feas if r["ok"]] or [np.nan]),
                         first=defaultdict(int))
        for r in feas:
            for kd in r["first_kinds"]:
                stats[mm]["first"][kd] += 1
        rows.append(f"{NAMES[mm]} & {pct(succ)} & {' / '.join(pct(t) for t in tiers)} & {pct(refu)} & "
                    f"{f(stats[mm]['ct'], 1)} & {f(stats[mm]['ratio'])} \\\\")
    # paired comparisons on cells solved by both
    def paired(a, b):
        both = [s for s, d in by.items() if a in d and b in d and d[a]["ok"] and d[b]["ok"]
                and d[a]["feasible_gt"]]
        if not both:
            return np.nan, np.nan, 0
        ta = np.array([by[s][a]["cycle_time"] for s in both]); tb = np.array([by[s][b]["cycle_time"] for s in both])
        ra = np.array([by[s][a]["max_contact_ratio"] for s in both if by[s][a]["min_human_distance"] is not None])
        rb = np.array([by[s][b]["max_contact_ratio"] for s in both if by[s][b]["min_human_distance"] is not None])
        return float(np.median(ta / tb)), (float(np.median(ra / rb)) if len(ra) and len(rb) == len(ra) else np.nan), len(both)
    tab = ["\\begin{tabular}{lccccc}", "\\toprule",
           "Planner & Solved (\\%) & T1 / T2 / T3 (\\%) & Refusals (\\%) & Cycle (s) & ISO util. \\\\", "\\midrule"]
    tab += rows + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(P, "planning_table.tex"), "w").write("\n".join(tab))
    if "variable" in stats:
        M["varSucc"] = pct(stats["variable"]["succ"])
    for mm, key in (("rigid", "rigidSucc"), ("soft", "softSucc"), ("rigid_robot", "rigidRobotSucc")):
        if mm in stats:
            M[key] = pct(stats[mm]["succ"])
    tr_vr, rr_vr, n_vr = paired("variable", "rigid")
    tr_rr, rr_rr, n_rr = paired("rigid", "rigid_robot")
    M["ctRatioVarStiff"] = f(tr_vr)
    M["nPairedVar"] = str(n_vr)
    M["utilVSSEA"] = f(stats.get("rigid", {}).get("ratio", np.nan))
    M["utilRigid"] = f(stats.get("rigid_robot", {}).get("ratio", np.nan))
    M["utilRatioRR"] = f(rr_rr)
    nsched = 0
    for s, d in by.items():
        if "variable" in d and d["variable"]["ok"]:
            pl = d["variable"]["plan"]
            if any(isinstance(st.get("stiffness"), list) and min(st["stiffness"]) < 0.99
                   for st in pl.get("steps", []) if st.get("op") == "move"):
                nsched += 1
    M["nSoftened"] = str(nsched)
    softfirst = stats.get("soft", {}).get("first", {})
    M["softTorquePct"] = pct(softfirst.get("torque", 0) / max(nfeas, 1))
    json.dump({k: {kk: (vv if not isinstance(vv, defaultdict) else dict(vv)) for kk, vv in v.items()}
               for k, v in stats.items()}, open(os.path.join(R, "e5_summary.json"), "w"), indent=1, default=float)
    # cells where the elastic arm was faster than the rigid-joint robot
    faster, sched_faster = [], []
    for s_, d in by.items():
        if all(mm in d and d[mm]["ok"] for mm in ("rigid_robot", "rigid", "variable")) and d["rigid"]["feasible_gt"]:
            best = min(d["rigid"]["cycle_time"], d["variable"]["cycle_time"])
            if best < d["rigid_robot"]["cycle_time"] - 0.01:
                faster.append(1 - best / d["rigid_robot"]["cycle_time"])
            if d["variable"]["cycle_time"] < d["rigid"]["cycle_time"] - 0.01:
                sched_faster.append(1 - d["variable"]["cycle_time"] / d["rigid"]["cycle_time"])
    M["nBound"] = str(len(faster))
    M["boundGainMed"] = f"{100 * np.median(faster):.0f}" if faster else "0"
    M["boundGainMax"] = f"{100 * max(faster):.0f}" if faster else "0"
    M["nSchedFaster"] = str(len(sched_faster))
    M["schedGainMax"] = f"{100 * max(sched_faster):.0f}" if sched_faster else "0"
    t3 = [d for d in by.values() if d["variable"]["tier"] == 3 and d["variable"]["feasible_gt"]]
    M["tThreeSolved"] = str(sum(d["variable"]["ok"] for d in t3))
    M["tThreeN"] = str(len(t3))
    fk = defaultdict(int)
    for d in t3:
        if not d["variable"]["ok"]:
            for kd in d["variable"]["kinds"]:
                fk[kd] += 1
    M["tThreeUnsolved"] = str(sum(1 for d in t3 if not d["variable"]["ok"]))
    M["tThreeFailGeom"] = str(sum(1 for d in t3 if not d["variable"]["ok"] and
                                  set(d["variable"]["kinds"]) & {"collision", "human_collision", "keep_out"}))
    M["refusedOK"] = pct(stats["variable"]["refu"])
    M["planningSummary"] = (
        "The rigid-joint robot and the fixed-stiff and scheduled VSSEA planners solved the same cells: all cells of "
        "tiers~1 and~2 and \\tThreeSolved\\ of the \\tThreeN\\ feasible tier-3 cells. They also refused all "
        "\\nInfeasible\\ infeasible cells. Of the \\tThreeUnsolved\\ unsolved tier-3 cells, \\tThreeFailGeom\\ failed on geometry "
        "(collisions with the operator's hands or arms, or the head keep-out) that the template-based repair could not "
        "resolve; these are the cells where a stronger planner, such as an LLM using the quantitative feedback, can "
        "add value. Fixed minimum stiffness solved no cell: a torque violation appeared in the first plan of "
        "\\softTorquePct\\% of the feasible cells and could not be removed without stiffening. "
        "Contact limits bound the speed in \\nBound\\ tier-3 cells. There, the elastic arm was faster than the rigid-joint "
        "robot by a median of \\boundGainMed\\% (up to \\boundGainMax\\%), because its higher admissible speeds near the "
        "operator could be used. Everywhere else the joint-speed limit of $\\pi/2$~rad/s, not safety, set the cycle "
        "time, and the cycles were identical. Stiffness scheduling shortened the cycle in \\nSchedFaster\\ cell "
        "(by \\schedGainMax\\%): with modulation costing about \\tmod~s, softening rarely pays at this actuator speed.")
else:
    for k in ("varSucc", "rigidSucc", "softSucc", "nInfeasible", "planningSummary"):
        M[k] = "\\todo{E5}"
    open(os.path.join(P, "planning_table.tex"), "w").write("\\todo{run scripts/exp\\_planning.py}")

# ------------------------------------------------------------------ E4
e4p = os.path.join(R, "e4_faults.jsonl")
if os.path.exists(e4p):
    recs = [json.loads(l) for l in open(e4p)]
    ch = ["kinematic", "kin_certified", "no_contact", "full"]
    lab = {"kinematic": "Kin.\\ 10", "kin_certified": "Kin.\\ cert.",
           "no_contact": "No contact", "full": "Full"}
    faults = sorted({r["fault"] for r in recs})
    real = [r for r in recs if r["reference"]["flag"]]
    M["nFaultPlans"] = str(len({r["scene"] for r in recs}))
    M["nFaultsReal"] = str(len(real))
    hdr = "Fault & $n$ & " + " & ".join(lab[c] for c in ch) + " \\\\"
    rows = []
    for fl in faults:
        rr = [r for r in real if r["fault"] == fl]
        if not rr:
            continue
        rows.append(fl.replace("_", " ") + f" & {len(rr)} & " +
                    " & ".join(pct(np.mean([r[c]["flag"] for r in rr])) for c in ch) + " \\\\")
    tot = " & ".join(pct(np.mean([r[c]["flag"] for r in real])) for c in ch)
    rows.append("\\midrule\nAll & " + str(len(real)) + " & " + tot + " \\\\")
    agree = np.mean([r["full"]["flag"] == r["reference"]["flag"] for r in recs])
    tab = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccccc}", "\\toprule", hdr, "\\midrule"] + rows + \
          ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(P, "faults_table.tex"), "w").write("\n".join(tab))
    M["fullRecall"] = pct(np.mean([r["full"]["flag"] for r in real]), 1)
    M["kinRecall"] = pct(np.mean([r["kinematic"]["flag"] for r in real]), 1)
    M["kincRecall"] = pct(np.mean([r["kin_certified"]["flag"] for r in real]), 1)
    M["noContactRecall"] = pct(np.mean([r["no_contact"]["flag"] for r in real]), 1)
    M["fullAgreePct"] = pct(agree, 1)
    M["fullFalseAlarm"] = str(sum(r["full"]["flag"] and not r["reference"]["flag"] for r in recs))
    tf = np.median([r["full"]["time"] for r in recs])
    M["verifyTimeMed"] = f(tf, 1)
    M["timeKinTen"] = f(np.median([r["kinematic"]["time"] for r in recs]), 1)
    M["timeKinCert"] = f(np.median([r["kin_certified"]["time"] for r in recs]), 1)
    M["faultSummary"] = (
        "The full verifier agreed with the dense reference on \\fullAgreePct\\% of the faulty plans and flagged "
        "\\fullRecall\\% of the real faults, with \\fullFalseAlarm\\ alarms the reference did not raise. "
        "A kinematics-only twin of the kind used in the original manuscript caught \\kinRecall\\% "
        "(\\kincRecall\\% with certified sampling). Removing the contact check lowered recall to "
        "\\noContactRecall\\%. The median verification time was \\verifyTimeMed~s per plan.")
else:
    for k in ("nFaultPlans", "fullRecall", "kinRecall", "faultSummary"):
        M[k] = "\\todo{E4}"
    open(os.path.join(P, "faults_table.tex"), "w").write("\\todo{run scripts/exp\\_faults.py}")

# ------------------------------------------------------------------ E4b
e4b = os.path.join(R, "e4b_sampling.json")
if os.path.exists(e4b):
    res = json.load(open(e4b))
    real = [r for r in res if r["reference"]]
    M["nPost"] = str(len(res))
    M["nPostReal"] = str(len(real))
    for k, name in (("fixed10", "postTen"), ("fixed50", "postFifty"), ("certified", "postCert")):
        M[name] = pct(np.mean([r[k] for r in real]) if real else float("nan"))
    M["postCertFA"] = str(sum(r["certified"] and not r["reference"] for r in res))
else:
    for k in ("nPost", "nPostReal", "postTen", "postFifty", "postCert", "postCertFA"):
        M[k] = "\\todo{E4b}"

with open(os.path.join(P, "numbers.tex"), "w") as fh:
    fh.write("% generated by scripts/export_numbers.py -- do not edit\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
for k, v in M.items():
    if len(str(v)) < 40:
        print(k, v)
