"""E1: VSAM model verification and characterisation.

Outputs results/e1_vssea.json and figures/e1_*.pdf
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsdt.vssea import (VSAMParams, kb_eb, kb_timo, fe_plane, K_vsam, K_joint, K_range,
                        tau_max_stress, qd_max_stress, torque_curve_elastica, elastica,
                        modulation_time, modulation_energy, x_from_K)
from vsdt.verifier import Verifier
from vsdt.robot import ur5e_class_arm

R = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(R, exist_ok=True)
p = VSAMParams()
out = {"params": {k: getattr(p, k) for k in p.__dataclass_fields__}}

# --- stiffness vs roller position: EB, Timoshenko, plane-stress FE
xs_fe = np.array([0.03, 0.045, 0.06, 0.08, 0.10, 0.13, 0.17])
fe = np.array([fe_plane(p, x, nx=120, ny=4) for x in xs_fe])
fe_conv = {f"nx{nx}_ny{ny}": fe_plane(p, 0.03, nx=nx, ny=ny) for nx, ny in [(30, 1), (60, 2), (120, 4), (240, 6)]}
fe_ext = fe_plane(p, 0.06, nx=160, ny=4, extend_to_L=True)
out["stiffness"] = {
    "x": xs_fe.tolist(), "kb_eb": kb_eb(p, xs_fe).tolist(), "kb_timo": kb_timo(p, xs_fe).tolist(),
    "kb_fe": fe.tolist(), "fe_vs_timo_rel": (fe / kb_timo(p, xs_fe) - 1).tolist(),
    "fe_vs_eb_rel": (fe / kb_eb(p, xs_fe) - 1).tolist(),
    "fe_mesh_convergence_x0.03": fe_conv,
    "fe_full_length_vs_cut_x0.06": [fe_ext, float(fe_plane(p, 0.06, nx=120, ny=4))],
}
Kmin, Kmax = K_range(p)
out["range"] = {"K_joint_min": Kmin, "K_joint_max": Kmax, "ratio": Kmax / Kmin,
                "K_vsam_max_timo": float(K_vsam(p, p.x_min)), "K_vsam_max_eb": float(K_vsam(p, p.x_min, "eb")),
                "K_hd": p.K_hd}

# --- torque capacity vs stiffness
xs = np.linspace(p.x_min, p.x_max, 60)
out["capacity"] = {"x": xs.tolist(), "K_joint": K_joint(p, xs).tolist(),
                   "tau_max": tau_max_stress(p, xs).tolist(),
                   "qd_max_deg": np.rad2deg(qd_max_stress(p, xs)).tolist()}

# --- nonlinear torque-deflection (elastica) vs linear
nl = {}
for x in (0.04, 0.08, 0.17):
    qmax = float(qd_max_stress(p, x))
    qd = np.linspace(0, qmax, 15)
    tau, fax, sig = torque_curve_elastica(p, x, qd)
    lin = K_vsam(p, x, "eb") * qd
    nl[f"{x:.2f}"] = {"qd": qd.tolist(), "tau_elastica": tau.tolist(), "tau_linear": lin.tolist(),
                      "F_axial": fax.tolist(), "sigma_root_MPa": (sig / 1e6).tolist(),
                      "rel_err_at_qmax": float(lin[-1] / tau[-1] - 1),
                      "stress_elastica_at_qmax_MPa": float(sig[-1] / 1e6)}
out["nonlinear"] = nl

# --- stiffness modulation
out["modulation"] = {
    "t_full_range": modulation_time(p, p.x_max, p.x_min),
    "E_eq_full_range": modulation_energy(p, p.x_max, p.x_min, 0.0),
    "E_loaded": {f"{q:.3f}": modulation_energy(p, p.x_max, p.x_min, q) for q in (0.005, 0.01, 0.02)},
    "rate_Nm_per_rad_per_s": (Kmax - Kmin) / modulation_time(p, p.x_max, p.x_min),
}
# linear axial-force approximation used inside the verifier vs elastica
v = Verifier(ur5e_class_arm())
cmp = []
for x in (0.05, 0.1, 0.15):
    for q in (0.005, 0.01):
        e = elastica(p, x, p.r * q)
        P = kb_timo(p, x) * p.r * q
        cmp.append({"x": x, "qd": q, "F_ax_elastica": p.n * e["F_axial"],
                    "F_ax_linear": p.n * P * 1.5 * p.r * q / x})
out["axial_force_check"] = cmp

with open(os.path.join(R, "e1_vssea.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("range", "modulation")}, indent=1))
print("FE vs Timoshenko:", np.round(out["stiffness"]["fe_vs_timo_rel"], 4))
print("FE mesh:", fe_conv, "full-length vs cut:", out["stiffness"]["fe_full_length_vs_cut_x0.06"])
for k, d in nl.items():
    print("x", k, "linear/elastica-1 at qmax", round(d["rel_err_at_qmax"], 4), "sigma", round(d["stress_elastica_at_qmax_MPa"]))
print(cmp)
