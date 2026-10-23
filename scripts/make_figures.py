"""Figures for the manuscript (vector PDF, IEEE column width)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W1, W2 = 3.5, 7.16


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".pdf"))
    fig.savefig(os.path.join(F, name + ".png"), dpi=200)
    plt.close(fig)


def fig_vssea():
    d = json.load(open(os.path.join(R, "e1_vssea.json")))
    p = d["params"]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.2))
    # (a) stiffness vs roller position
    a = ax[0, 0]
    s = d["stiffness"]
    xs = np.linspace(p["x_min"], p["x_max"], 100)
    from vsdt.vssea import VSAMParams, K_vsam, K_joint
    P = VSAMParams()
    a.semilogy(xs * 1000, K_vsam(P, xs, "eb"), color=C[0], label="Euler-Bernoulli (VSAM)")
    a.semilogy(xs * 1000, K_joint(P, xs), color=C[1], ls="--", label="Joint (VSAM + harmonic drive)")
    a.semilogy(np.array(s["x"]) * 1000, P.n * np.array(s["kb_fe"]) * P.r ** 2, "o", color=C[2],
               mfc="white", label="Plane-stress FE (VSAM)")
    a.set_xlabel("Roller position $x$ (mm)"); a.set_ylabel("Stiffness (Nm/rad)")
    a.legend(loc="upper right"); a.set_title("(a) Stiffness vs. roller position", loc="left")
    # (b) capacity vs stiffness
    b = ax[0, 1]
    cap = d["capacity"]
    from vsdt.vssea import tau_capacity
    x = np.array(cap["x"])
    b.semilogx(cap["K_joint"], cap["tau_max"], color=C[0], label="Linear beam")
    b.semilogx(cap["K_joint"], tau_capacity(P, x), color=C[1], ls="--", label="Elastica (used by verifier)")
    b.axhline(100, color=INK2, lw=0.8, ls=":")
    b.text(25, 103, "M1 limit 100 Nm", color=INK2, fontsize=7)
    b.set_xlabel("Joint stiffness $K$ (Nm/rad)"); b.set_ylabel("Torque capacity (Nm)")
    b.legend(loc="center right"); b.set_title("(b) Stress-limited torque capacity", loc="left")
    # (c) torque-deflection
    c = ax[1, 0]
    for i, (k, v) in enumerate(d["nonlinear"].items()):
        q = np.rad2deg(v["qd"])
        c.plot(q, v["tau_elastica"], color=C[i], label=f"$x$={float(k)*1000:.0f} mm elastica")
        c.plot(q, v["tau_linear"], color=C[i], ls=":", lw=1.0)
    c.set_xlabel("Spring deflection $q_d$ (deg)"); c.set_ylabel("Torque (Nm)")
    c.legend(loc="center right", title="dotted: linear", title_fontsize=7)
    c.set_title("(c) Torque-deflection up to the stress limit", loc="left")
    # (d) modulation energy
    e = ax[1, 1]
    m = d["modulation"]
    qs = [0.0] + [float(k) for k in m["E_loaded"]]
    Es = [m["E_eq_full_range"]] + list(m["E_loaded"].values())
    e.bar(range(len(qs)), Es, color=C[0], width=0.6)
    for i, E_ in enumerate(Es):
        e.text(i, E_ * 1.15 + 0.005, f"{E_:.2f} J", ha="center", fontsize=7, color=INK)
    e.set_xticks(range(len(qs)), [f"{np.rad2deg(q):.2f}°" for q in qs])
    e.set_yscale("log"); e.set_ylim(0.01, 3)
    e.set_xlabel("Spring deflection during soft$\\rightarrow$stiff change")
    e.set_ylabel("M2 work (J)")
    e.set_title(f"(d) Modulation energy (full range in {m['t_full_range']:.2f} s)", loc="left")
    fig.tight_layout()
    save(fig, "fig_vssea")


def fig_contact():
    d = json.load(open(os.path.join(R, "e2_e3.json")))["contact"]
    K = np.array(d["K_joint"])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    a = ax[0]
    regs = [("hand", C[0]), ("lower_arm", C[1]), ("chest", C[2]), ("thigh", C[6])]
    for r, col in regs:
        cv = d["curves"][r]
        a.semilogx(K, cv["vmax"], color=col, label=r.replace("_", " ") + " (free)")
        if cv["vmax_clamped"][0] is not None:
            a.semilogx(K, cv["vmax_clamped"], color=col, ls="--", label=r.replace("_", " ") + " (clamped)")
        a.plot([K[-1] * 1.25], [cv["iso_rigid_config"]], marker="<", color=col, ms=5, clip_on=False)
    lo, hi = d["K_vssea_range"]
    a.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
    a.text(np.sqrt(lo * hi), 0.03, "VSSEA range", ha="center", fontsize=7, color=INK2)
    a.set_ylim(0, None)
    a.set_xlabel("Joint stiffness, all joints (Nm/rad)"); a.set_ylabel("Admissible approach speed (m/s)")
    a.set_title("(a) ISO/TS 15066 speed limit vs. stiffness", loc="left")
    a.legend(loc="center left", bbox_to_anchor=(0.30, 0.66), ncol=2, fontsize=6, frameon=True,
             facecolor="white", framealpha=0.9, edgecolor="none")
    b = ax[1]
    iK = int(np.searchsorted(K, hi))
    iR = int(np.searchsorted(K, d["K_hd"]))
    cases = [("hand", "vmax_clamped", "hand\nclamped"), ("lower_arm", "vmax_clamped", "lower arm\nclamped"),
             ("chest", "vmax", "chest\nfree"), ("thigh", "vmax", "thigh\nfree")]
    w = 0.26
    for j, (lab, idx, col) in enumerate((("Rigid joint (harmonic drive only)", iR, C[1]), ("VSSEA stiffest", iK, C[6]),
                                         ("VSSEA softest", 0, C[0]))):
        vals = [d["curves"][r][key][idx] for r, key, _ in cases]
        bars = b.bar(np.arange(len(cases)) + (j - 1) * w, vals, width=w - 0.02, color=col, label=lab)
        for x_, v_ in zip(np.arange(len(cases)) + (j - 1) * w, vals):
            b.text(x_, v_ + 0.015, f"{v_:.2f}", ha="center", fontsize=6, color=INK)
    b.set_xticks(range(len(cases)), [c[2] for c in cases])
    b.set_ylabel("Admissible approach speed (m/s)")
    b.set_ylim(0, 0.92)
    b.set_title("(b) Rigid vs. VSSEA settings", loc="left")
    b.legend(loc="upper left", fontsize=6, ncol=1, frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")
    fig.tight_layout()
    save(fig, "fig_contact")


def fig_workspace():
    w = json.load(open(os.path.join(R, "e2_e3.json")))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    for a, tag, title in ((ax[0], "original", "(a) Original 9-frame chain"), (ax[1], "study", "(b) UR5e-class chain (this study)")):
        d = np.load(os.path.join(R, f"e2_workspace_{tag}.npz"))
        P, f = d["P"], d["feas25"]
        r = np.linalg.norm(P[:, :2], axis=1)
        a.scatter(r[f], P[f, 2], s=1.0, color=C[0], label="holdable with 2.5 kg", rasterized=True)
        a.scatter(r[~f], P[~f, 2], s=1.0, color=C[1], label="needs > 100 Nm", rasterized=True)
        ws = w["workspace_original" if tag == "original" else "workspace"]
        a.set_title(f"{title}: {100*ws['frac_torque_feasible_2.5kg']:.0f}% holdable", loc="left")
        a.set_xlabel("Radial distance (m)"); a.set_ylabel("Height $z$ (m)")
        a.legend(loc="lower left", markerscale=4, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    save(fig, "fig_workspace")


def fig_planning():
    path = os.path.join(R, "e5_planning.jsonl")
    if not os.path.exists(path):
        return
    recs = [json.loads(l) for l in open(path)]
    modes = [("rigid_robot", "Rigid-joint robot", C[1]), ("rigid", "VSSEA fixed stiff", C[6]),
             ("soft", "VSSEA fixed soft", C[3]), ("variable", "VSSEA scheduled", C[0])]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    a = ax[0]
    w = 0.2
    for i, (m, lab, col) in enumerate(modes):
        rates = []
        for t in (1, 2, 3):
            rs = [r for r in recs if r["mode"] == m and r["tier"] == t and r["feasible_gt"]]
            rates.append(100 * np.mean([r["ok"] for r in rs]) if rs else np.nan)
        xs_ = np.arange(3) + (i - 1.5) * w
        a.bar(xs_, rates, width=w - 0.02, color=col, label=lab)
        for x_, r_ in zip(xs_, rates):
            a.text(x_, r_ + 2, f"{r_:.0f}", ha="center", fontsize=6, color=INK)
    a.set_xticks(range(3), ["Tier 1", "Tier 2", "Tier 3"])
    a.set_ylabel("Feasible cells solved (%)"); a.set_ylim(0, 135)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_title("(a) Verified plans", loc="left")
    a.legend(loc="upper center", ncol=2, fontsize=6)
    b = ax[1]
    byk = {(r["scene"], r["mode"]): r for r in recs}
    cells = []
    for (s_, m), r in byk.items():
        if m != "rigid_robot" or not r["ok"] or not r["feasible_gt"]:
            continue
        o, v = byk.get((s_, "rigid")), byk.get((s_, "variable"))
        if o and v and o["ok"] and v["ok"] and min(o["cycle_time"], v["cycle_time"]) < r["cycle_time"] - 0.01:
            cells.append((s_, r["cycle_time"], o["cycle_time"], v["cycle_time"]))
    cells.sort(key=lambda c: c[2] / c[1])
    x = np.arange(len(cells))
    for j, (lab, col) in enumerate((("Rigid-joint robot", C[1]), ("VSSEA fixed stiff", C[6]), ("VSSEA scheduled", C[0]))):
        b.bar(x + (j - 1) * 0.26, [c[1 + j] for c in cells], width=0.24, color=col, label=lab)
    b.set_xticks(x, [c[0].replace("cell_3_", "") for c in cells], fontsize=6)
    b.set_xlabel("Tier-3 cells where the contact limit bound the speed")
    b.set_ylabel("Cycle time (s)")
    b.set_title("(b) Cycle time where contact limits bind", loc="left")
    b.set_ylim(0, 31)
    b.legend(loc="upper center", ncol=3, fontsize=6)
    fig.tight_layout()
    save(fig, "fig_planning")


if __name__ == "__main__":
    which = sys.argv[1:] or ["vssea", "contact", "workspace", "planning"]
    for w in which:
        globals()["fig_" + w]()
        print("ok", w)
