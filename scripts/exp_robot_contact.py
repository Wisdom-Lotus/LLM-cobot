"""E2 (arm kinematics, IK, gravity-feasible workspace) and
E3 (stiffness-dependent ISO/TS 15066 speed limits)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsdt.robot import paper_arm, ur5e_class_arm
from vsdt import ik
from vsdt.safety import ISO_BODY, ImpactModel, ClampModel, transient_limit, iso_vmax, cartesian_quantities
from vsdt.verifier import Verifier

R = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(7)
out = {}


def workspace(arm, v, tag, N=20000):
    Q = rng.uniform(-np.pi, np.pi, (N, 6))
    P = np.array([arm.fk(q)[:3, 3] for q in Q])
    tau0 = np.array([np.abs(arm.gravity_torque(q, 0.0)) for q in Q])
    tau25 = np.array([np.abs(arm.gravity_torque(q, 2.5)) for q in Q])
    capmax = v.tau_cap(np.array([v.Kmax]))[0]
    capmin = v.tau_cap(np.array([v.Kmin]))[0]
    feas0, feas25 = (tau0 <= capmax).all(1), (tau25 <= capmax).all(1)
    r = np.linalg.norm(P, axis=1)
    edges = np.linspace(0, r.max(), 11)
    res = {
        "max_reach": float(r.max()), "z_min": float(P[:, 2].min()), "z_max": float(P[:, 2].max()),
        "frac_torque_feasible_no_payload": float(feas0.mean()),
        "frac_torque_feasible_2.5kg": float(feas25.mean()),
        "frac_all_joints_soft_ok_2.5kg": float((tau25 <= capmin).all(1).mean()),
        "max_tau_per_joint_2.5kg": tau25.max(0).round(1).tolist(),
        "reach_edges": edges.tolist(),
        "feasible_by_reach_2.5kg": [float(feas25[(r >= a) & (r < b)].mean()) if np.any((r >= a) & (r < b)) else None
                                    for a, b in zip(edges[:-1], edges[1:])],
    }
    np.savez_compressed(os.path.join(R, f"e2_workspace_{tag}.npz"), P=P[:6000], feas25=feas25[:6000], tau25=tau25[:6000])
    return res


arm0 = paper_arm()
out["workspace_original"] = workspace(arm0, Verifier(arm0), "original")
arm = ur5e_class_arm()
v = Verifier(arm)
out["workspace"] = workspace(arm, v, "study")

# ------------------------------------------------------------- E2 IK
ikres = {}
targets = []
while len(targets) < 200:
    q = rng.uniform(-np.pi, np.pi, 6)
    T = arm.fk(q)
    if T[2, 2] < -0.2:      # roughly downward tool
        targets.append((T[:3, 3], T[:3, 2]))
for jac in ("analytic", "fd"):
    ok, its, tt, errs = 0, [], 0.0, []
    for i, (p, z) in enumerate(targets):
        t0 = time.time()
        qs, c, it, e = ik.solve(arm, p, z, jac=jac, rng=np.random.default_rng(i))
        tt += time.time() - t0
        ok += c; its.append(it); errs.append(e)
    ikres[jac] = {"success": ok / len(targets), "mean_iter": float(np.mean(its)),
                  "median_iter": float(np.median(its)), "time_ms": 1000 * tt / len(targets),
                  "max_err": float(np.max(errs))}
out["ik"] = ikres

# ------------------------------------------------------------- E3 contact
M = ImpactModel()
# representative configuration: arm reaching 1.1 m forward at bench height,
# contact at the wrist module moving horizontally
q_rep = ik.solve_all(arm, np.array([0.6, 0.0, 0.05]), seeds=v._posture_seeds(np.array([0.6, 0.0, 0.05])))[0]
Ts = arm.frames_fk(q_rep)
CONTACT_FRAME = 4                      # wrist-1 module
pt = Ts[CONTACT_FRAME][:3, 3]
u = np.array([0.0, 1.0, 0.0])
Ks = np.logspace(np.log10(v.Kmin), 6, 30)
CM = ClampModel()
curves = {}
for region in ("hand", "lower_arm", "upper_arm", "chest", "abdomen", "back", "thigh", "neck"):
    k, mH = ISO_BODY[region][2], ISO_BODY[region][3]
    vm, vm_first, vcl, Kcs = [], [], [], []
    for K in Ks:
        m_l, m_m, K_c = cartesian_quantities(arm, q_rep, pt, CONTACT_FRAME, u, np.full(6, K))
        vm.append(transient_limit(region) / M.gain(m_m, m_l, K_c, k, mH))
        vm_first.append(transient_limit(region) / M.gain(m_m, m_l, K_c, k, mH, first_phase_only=True))
        vcl.append(CM.vmax(region, m_l, K_c) if region in ("hand", "lower_arm", "upper_arm") else None)
        Kcs.append(K_c)
    m_l, m_m, _ = cartesian_quantities(arm, q_rep, pt, CONTACT_FRAME, u, np.full(6, v.Kmax))
    M_total = sum(m for m, _, _ in arm.point_masses(q_rep, Ts, 0.0))
    curves[region] = {
        "vmax": vm, "vmax_first_impact_only": vm_first, "vmax_clamped": vcl, "K_c": Kcs,
        "iso_rigid_config": iso_vmax(region, m_l + m_m),
        "iso_TS_formula": iso_vmax(region, M_total / 2 + 2.5),
        "m_l": m_l, "m_m": m_m, "M_total": M_total,
    }
out["contact"] = {"K_joint": Ks.tolist(), "curves": curves, "q_rep": q_rep.tolist(),
                  "K_vssea_range": [v.Kmin, v.Kmax], "K_hd": v.p.K_hd}

# force time histories (chest) for soft vs stiff
def history(m_m, m_l, K_c, k, mH, T=0.6, n=6000):
    from scipy.integrate import solve_ivp
    c = 2 * 0.05 * np.sqrt(K_c * m_m * m_l / (m_m + m_l))
    def f(t, y):
        xm, xl, xh, vm_, vl, vh = y
        Fc = k * max(0, xl - xh); Fs = K_c * (xm - xl) + c * (vm_ - vl)
        return [vm_, vl, vh, -Fs / m_m, (Fs - Fc) / m_l, Fc / mH]
    s = solve_ivp(f, (0, T), [0, 0, 0, 1, 1, 0], t_eval=np.linspace(0, T, n), max_step=5e-5, rtol=1e-8)
    return s.t.tolist(), (k * np.maximum(0, s.y[1] - s.y[2])).tolist()

k, mH = ISO_BODY["chest"][2], ISO_BODY["chest"][3]
hist = {}
for name, K in (("soft", v.Kmin), ("medium", float(v.level_to_K(0.5))), ("stiff", v.Kmax)):
    m_l, m_m, K_c = cartesian_quantities(arm, q_rep, pt, CONTACT_FRAME, u, np.full(6, K))
    hist[name] = dict(zip(("t", "F"), history(m_m, m_l, K_c, k, mH)))
    hist[name]["K_c"] = K_c
out["chest_history"] = hist

with open(os.path.join(R, "e2_e3.json"), "w") as f:
    json.dump(out, f)
print(json.dumps(out["workspace"], indent=1))
print(json.dumps(out["ik"], indent=1))
for reg, c in curves.items():
    print(reg, "soft %.2f stiff %.2f  ISO(rigid cfg) %.2f ISO(TS M/2) %.2f  first-impact soft %.2f" %
          (c["vmax"][0], c["vmax"][-1], c["iso_rigid_config"], c["iso_TS_formula"], c["vmax_first_impact_only"][0]),
          "m_l %.2f m_m %.2f" % (c["m_l"], c["m_m"]),
          "clamped soft/stiff/K=1e6:", None if c["vmax_clamped"][0] is None else
          (round(c["vmax_clamped"][0], 3), round(c["vmax_clamped"][int(np.searchsorted(Ks, v.Kmax))], 3), round(c["vmax_clamped"][-1], 3)),
          "transient at 1e6: %.2f" % c["vmax"][-1])
