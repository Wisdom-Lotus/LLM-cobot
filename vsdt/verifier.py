"""Stiffness-aware digital-twin verifier.

Executes a candidate plan on the kinematic/quasi-static twin of the
VSSEA-driven arm and returns every constraint violation together with its
magnitude and a repair bound.  Checks (all can be toggled for ablations):

  C0  format / semantic consistency (schema, ids, gripper state)
  C1  reachability (IK) and joint limits
  C2  collision of links / carried item with obstacles, other items and
      the operator; self-collision.  Certified by conservative
      advancement on the *link-side* (deflected) configuration.
  C3  joint torque <= min(M1 torque limit, VSAM stress-limited capacity at
      the commanded stiffness)
  C4  task precision at grasp/release (deflection-induced TCP error)
  C5  joint velocity limit (via time parameterisation)
  C6  ISO/TS 15066 transient-contact speed limit near the operator,
      using the stiffness-dependent impact model; keep-out for head
  C7  task completion (item delivered within tolerance)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
import numpy as np

from . import ik as ikmod
from .contact import vmax_lookup, vmax_clamp_lookup
from .geometry import closest_seg_box, closest_seg_seg, batch_seg_box, batch_seg_seg
from .plan import PlanFormatError, validate
from .robot import Arm
from .scene import Scene, region_for_contact
from .vssea import VSAMParams, K_range, x_from_K, tau_max_stress, tau_capacity, K_joint


@dataclass
class VerifierConfig:
    check_torque: bool = True
    check_precision: bool = True
    check_contact: bool = True
    deflected_geometry: bool = True     # collision on link-side configuration
    certified: bool = True              # conservative advancement (else fixed N samples)
    n_fixed_samples: int = 10           # used when certified=False
    d_safe: float = 0.01                # m, clearance to obstacles
    d_prox: float = 0.45                # m, contact-check activation distance
    d_head: float = 0.30                # m, keep-out around skull/face
    tau_m1: float = 100.0               # Nm, M1 continuous torque limit
    w_max: float = np.pi / 2            # rad/s, joint speed at speed=1
    a_max: float = 3.0                  # rad/s^2
    tol_grasp: float = 0.010            # m, TCP error at grasp
    tol_place_xy: float = 0.05          # m
    tol_place_z: float = 0.015          # m, max drop height at release
    t_grip: float = 0.5                 # s, gripper actuation time
    max_violations_reported: int = 12
    comp_residual: float = 0.10         # residual static deflection after feed-forward
    rigid_K: float | None = None        # baseline: conventional rigid joint (harmonic drive only)


@dataclass
class Violation:
    kind: str
    step: int
    message: str
    magnitude: float = 0.0
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"kind": self.kind, "step": self.step, "message": self.message,
                "magnitude": round(float(self.magnitude), 4), **self.data}


@dataclass
class Report:
    ok: bool
    task_success: bool
    violations: list
    cycle_time: float = 0.0
    modulation_energy: float = 0.0
    min_human_distance: float = np.inf
    max_contact_ratio: float = 0.0       # max over samples of v_u / v_max
    declared_infeasible: bool = False
    trace: dict = field(default_factory=dict)

    @property
    def kinds(self):
        return sorted({v.kind for v in self.violations})


class Verifier:
    def __init__(self, arm: Arm, vsam: VSAMParams | None = None, cfg: VerifierConfig | None = None):
        self.arm, self.p = arm, (vsam or VSAMParams())
        self.cfg = cfg or VerifierConfig()
        self.Kmin, self.Kmax = K_range(self.p)
        if self.cfg.rigid_K is not None:
            self.Kmin, self.Kmax = self.cfg.rigid_K * 0.999, self.cfg.rigid_K
        # per-joint bound on distance from joint axis to any distal point
        L = [float(np.hypot(f.d, f.a)) for f in arm.frames]
        jidx = arm.joint_frame_index()
        self.R = np.array([sum(L[i:]) + 0.25 for i in jidx])   # +tool/item
        xs = np.linspace(self.p.x_min, self.p.x_max, 400)
        Ks = K_joint(self.p, xs)                       # decreasing in x
        self._lK = np.log(Ks[::-1])
        self._cap = tau_capacity(self.p, xs)[::-1]
        self._move_memo, self._ik_memo = {}, {}
        sr = np.random.default_rng(12345)
        self._seeds = [sr.uniform(arm.q_min, arm.q_max) for _ in range(10)]

    # ---------------------------------------------------------- helpers
    def level_to_K(self, s):
        s = np.clip(np.asarray(s, float), 0, 1)
        return self.Kmin * (self.Kmax / self.Kmin) ** s

    def K_to_level(self, K):
        return np.log(np.asarray(K) / self.Kmin) / np.log(self.Kmax / self.Kmin)

    def tau_cap(self, K):
        """Torque capacity per joint at joint stiffness K (vector): the
        smaller of the M1 limit and the stress-limited VSAM capacity
        (geometrically exact beam model, interpolated)."""
        if self.cfg.rigid_K is not None:
            return np.full(np.atleast_1d(K).shape, self.cfg.tau_m1)
        lk = np.log(np.clip(np.atleast_1d(K), self.Kmin, self.Kmax))
        return np.minimum(np.interp(lk, self._lK, self._cap), self.cfg.tau_m1)

    def deflected(self, q, K, payload, qdd=None):
        """Link-side configuration and joint torque.

        The controller feeds forward the static spring deflection
        tau_g/K (standard gravity compensation for elastic joints), so only
        a residual fraction ``comp_residual`` of the static deflection
        remains (stiffness/payload model error); inertial deflection
        M*qdd/K is not compensated."""
        ql = q.copy()
        tau = np.zeros(6)
        for _ in range(3):
            tg = self.arm.gravity_torque(ql, payload)
            ti = self.arm.mass_matrix(ql, payload) @ qdd if qdd is not None and np.any(qdd) else 0.0
            tau = tg + ti
            ql = q - (self.cfg.comp_residual * tg + ti) / K
        return ql, tau

    def _resolve_target(self, st, scene: Scene):
        t = st["target"]
        if isinstance(t, str):
            if any(i.id == t for i in scene.items):
                p = np.array(scene.item(t).grasp_point)
            elif any(l.id == t for l in scene.locations):
                p = np.array(scene.location(t).point, float)
            else:
                raise PlanFormatError(f"unknown target id '{t}'")
        else:
            p = np.array(t, float)
        if "offset" in st:
            p = p + np.array(st["offset"], float)
        return p

    @staticmethod
    def _trapezoid(dist, vmax, amax):
        if dist <= vmax ** 2 / amax:
            T = 2 * np.sqrt(dist / amax)
            return T, np.sqrt(dist * amax), T / 2
        return dist / vmax + vmax / amax, vmax, vmax / amax

    @staticmethod
    def _profile(t, T, vpk, ta, dist):
        """position fraction, velocity, acceleration (scalar) of trapezoid."""
        if dist <= 0:
            return 0.0, 0.0, 0.0
        a = vpk / ta
        if t < ta:
            s, v, acc = 0.5 * a * t * t, a * t, a
        elif t < T - ta:
            s, v, acc = 0.5 * a * ta * ta + vpk * (t - ta), vpk, 0.0
        else:
            tr = T - t
            s, v, acc = dist - 0.5 * a * tr * tr, a * tr, -a
        return s / dist, v / dist, acc / dist

    # ----------------------------------------------------- geometry
    def _robot_capsules(self, q, held=None):
        Ts = self.arm.frames_fk(q)
        caps = []
        for a, b, i in self.arm.segments(q, Ts):
            r = 0.03 if i == len(Ts) - 1 else self.arm.link_radius
            caps.append((a, b, r, i))
        held_box = None
        if held is not None:
            tcp = Ts[-1][:3, 3]
            sz = np.array(held.size)
            c = tcp - np.array([0, 0, sz[2] / 2])
            held_box = (c - sz / 2 + np.array([0, 0, 0.004]), c + sz / 2)
        return Ts, caps, held_box

    def _clearances(self, q, scene: Scene, held, item_pos, phase_item_ids):
        """Return (min clearance incl. margin, collision records,
        human-proximity records, frames).  Vectorised over all pairs."""
        cfg = self.cfg
        Ts, caps, held_box = self._robot_capsules(q, held)
        nT = len(Ts)
        ids, LO, HI = [], [], []
        for o in scene.obstacles:
            ids.append(o.id); LO.append(o.lo); HI.append(o.hi)
        for it in scene.items:
            if held is not None and it.id == held.id:
                continue
            c = np.array(it.center) if item_pos is None else item_pos[it.id]
            sz = np.array(it.size)
            ids.append(it.id); LO.append(c - sz / 2); HI.append(c + sz / 2)
        LO, HI = np.array(LO, float), np.array(HI, float)
        A0 = np.array([c[0] for c in caps]); A1 = np.array([c[1] for c in caps])
        rad = np.array([c[2] for c in caps]); fid = np.array([c[3] for c in caps])
        cmin, cols, prox = np.inf, [], []
        # links vs boxes
        D, _ = batch_seg_box(A0, A1, LO, HI)
        gap = D - rad[:, None]
        mask = np.ones_like(gap, bool)
        for j, bid in enumerate(ids):
            if bid == "pedestal":
                mask[fid <= 2, j] = False
            if bid in phase_item_ids:
                mask[fid == nT - 1, j] = False
        g = np.where(mask, gap - cfg.d_safe, np.inf)
        if g.size:
            cmin = min(cmin, float(g.min()))
            for i, j in zip(*np.where(g < 0)):
                cols.append(("link", int(fid[i]), ids[j], float(gap[i, j])))
        # self collision (segments at least 3 apart)
        Ds, _, _ = batch_seg_seg(A0, A1, A0, A1)
        gs = Ds - rad[:, None] - rad[None, :]
        n = len(caps)
        iu = np.array([(x, y) for x in range(n) for y in range(x + 3, n)])
        if len(iu):
            vals = gs[iu[:, 0], iu[:, 1]]
            cmin = min(cmin, float(vals.min()))
            for (x, y), val in zip(iu, vals):
                if val < 0:
                    cols.append(("self", int(fid[x]), f"link{int(fid[y])}", float(val)))
        # carried item vs boxes (AABB-AABB)
        if held_box is not None:
            hlo, hhi = held_box
            keep = np.array([bid not in phase_item_ids for bid in ids])
            gapv = np.maximum(np.maximum(LO - hhi, hlo - HI), 0)
            dd = np.linalg.norm(gapv, axis=1)
            cc = np.where(keep, dd - cfg.d_safe * 0.5, np.inf)
            cmin = min(cmin, float(cc.min()))
            for j in np.where(cc < 0)[0]:
                cols.append(("item", -1, ids[j], float(dd[j])))
            hc = 0.5 * (hlo + hhi)
            A0 = np.vstack([A0, hc]); A1 = np.vstack([A1, hc])
            rad = np.append(rad, 0.5 * float(np.linalg.norm(hhi - hlo)))
            fid = np.append(fid, nT - 1)
        # human
        if scene.human:
            B0 = np.array([c.a for c in scene.human], float)
            B1 = np.array([c.b for c in scene.human], float)
            Rh = np.array([c.radius for c in scene.human])
            Dh, C1, C2 = batch_seg_seg(A0, A1, B0, B1)
            gh = Dh - rad[:, None] - Rh[None, :]
            cmin = min(cmin, float(gh.min() - cfg.d_safe))
            for i, j in zip(*np.where(gh < cfg.d_prox)):
                hcap = scene.human[j]
                if gh[i, j] < 0:
                    cols.append(("human", int(fid[i]), hcap.id, float(gh[i, j])))
                u = (C2[i, j] - C1[i, j]) / max(Dh[i, j], 1e-9)
                prox.append(dict(cap=hcap, gap=float(gh[i, j]), point=C1[i, j], u=u, frame=int(fid[i])))
        return cmin, cols, prox, Ts

    # --------------------------------------------------------- verify
    def verify(self, plan: dict, scene: Scene, q_start=None) -> Report:
        cfg = self.cfg
        V: list[Violation] = []
        try:
            validate(plan)
        except PlanFormatError as e:
            return Report(False, False, [Violation("format", -1, str(e))])
        if not plan["feasible"]:
            return Report(not scene.expected_feasible, False, [], declared_infeasible=True)

        q = self.arm.q_home.copy() if q_start is None else np.array(q_start, float)
        K = np.full(6, self.Kmax)
        held, item_pos = None, {it.id: np.array(it.center, float) for it in scene.items}
        t_total, E_mod = 0.0, 0.0
        min_hd, max_ratio = np.inf, 0.0
        rng = np.random.default_rng(0)
        trace = {"t": [], "q": [], "K": [], "tcp": [], "step": [], "payload": []}
        last_target = None
        last_released = None
        scene_sig = self._scene_sig(scene)

        def add(v: Violation):
            V.append(v)

        for k, st in enumerate(plan["steps"]):
            op = st["op"]
            if op == "grasp":
                t_total += cfg.t_grip
                if held is not None:
                    add(Violation("semantic", k, f"step {k}: gripper already holds {held.id}"))
                    continue
                if st["item"] not in item_pos:
                    add(Violation("semantic", k, f"step {k}: unknown item '{st['item']}'"))
                    continue
                it = scene.item(st["item"])
                gp = item_pos[it.id] + np.array([0, 0, it.size[2] / 2])
                ql, _ = self.deflected(q, K, 0.0) if cfg.check_precision else (q, None)
                tcp = self.arm.fk(ql)[:3, 3]
                err = float(np.linalg.norm(tcp - gp))
                if err > cfg.tol_grasp:
                    sv = self._stiffness_for_precision(q, 0.0, gp, cfg.tol_grasp)
                    add(Violation("precision", k,
                                  f"step {k}: grasp of {it.id} failed, TCP (link side) is {err*1000:.1f} mm from the grasp "
                                  f"point (tolerance {cfg.tol_grasp*1000:.0f} mm)." + sv,
                                  err - cfg.tol_grasp, {"tcp": tcp.round(4).tolist(), "grasp_point": gp.round(4).tolist()}))
                    continue
                held = it
                continue
            if op == "release":
                t_total += cfg.t_grip
                if held is None:
                    add(Violation("semantic", k, f"step {k}: release with empty gripper"))
                    continue
                ql, _ = self.deflected(q, K, held.mass) if cfg.check_precision else (q, None)
                tcp = self.arm.fk(ql)[:3, 3]
                bottom = tcp[2] - held.size[2]
                # supporting surface below the item
                support = FLOOR_Z = -1.0
                for o in scene.obstacles:
                    if o.lo[0] <= tcp[0] <= o.hi[0] and o.lo[1] <= tcp[1] <= o.hi[1]:
                        support = max(support, o.hi[2])
                drop = bottom - support
                item_pos[held.id] = np.array([tcp[0], tcp[1], support + held.size[2] / 2])
                if drop > cfg.tol_place_z:
                    add(Violation("precision", k, f"step {k}: {held.id} released {drop*1000:.0f} mm above the "
                                                  f"support surface (max {cfg.tol_place_z*1000:.0f} mm); lower the TCP "
                                                  f"by {drop*1000:.0f} mm.", drop - cfg.tol_place_z))
                if drop < -0.004:
                    add(Violation("collision", k, f"step {k}: {held.id} penetrates the support surface by "
                                                  f"{-drop*1000:.0f} mm; raise the TCP by {-drop*1000:.0f} mm.", -drop))
                last_released = held.id
                held = None
                continue

            # ---------------------------------------------- move
            try:
                p_goal = self._resolve_target(st, scene)
            except PlanFormatError as e:
                add(Violation("semantic", k, f"step {k}: {e}"))
                continue
            speed = float(st.get("speed", 0.5))
            s_new = st.get("stiffness", None)
            K_new = K if s_new is None else (self.level_to_K(s_new) * np.ones(6))
            # stiffness modulation (performed at the start pose before moving)
            if np.any(np.abs(np.log(K_new / K)) > 1e-6):
                payload = 0.0 if held is None else held.mass
                tmod, emod = self._modulation_cost(q, K, K_new, payload)
                t_total += tmod
                E_mod += emod
                # torque capacity must hold during and after the change
                K = K_new
            payload_t = 0.0 if held is None else held.mass
            q_goal, conv, err = self._select_ik(p_goal, q, K, payload_t, scene, held, item_pos, st, plan, k)
            if not conv:
                add(Violation("reach", k, f"step {k}: target {np.round(p_goal,3).tolist()} is not reachable with the "
                                          f"tool pointing down (residual {err:.3f}).", err))
                continue
            # choose the joint-space-closest equivalent (no wrap: limits are +-pi)
            dq = q_goal - q
            dist = float(np.max(np.abs(dq)))
            vmax, amax = speed * cfg.w_max, cfg.a_max * max(speed, 0.2)
            T, vpk, ta = self._trapezoid(dist, vmax, amax) if dist > 1e-9 else (0.0, 0.0, 0.0)
            payload = 0.0 if held is None else held.mass
            phase_items = set()
            if held is not None:
                phase_items.add(held.id)
            # allow tool contact with an item that is the target of the next grasp
            nxt = plan["steps"][k + 1] if k + 1 < len(plan["steps"]) else None
            if nxt is not None and nxt.get("op") == "grasp":
                phase_items.add(nxt.get("item", ""))
            prv = plan["steps"][k - 1] if k > 0 else None
            if prv is not None and prv.get("op") in ("grasp", "release"):
                # leaving a surface: the item may still touch its support and
                # the tool may still touch the item it just released
                pref = held.id if (held is not None) else last_released
                if pref is not None:
                    phase_items.add(pref)
                    c0 = item_pos[pref]
                    for o in scene.obstacles:
                        if o.lo[0] <= c0[0] <= o.hi[0] and o.lo[1] <= c0[1] <= o.hi[1]:
                            phase_items.add(o.id)
            # supports of the held item at the place location
            if held is not None and nxt is not None and nxt.get("op") == "release":
                for o in scene.obstacles:
                    if o.lo[0] <= p_goal[0] <= o.hi[0] and o.lo[1] <= p_goal[1] <= o.hi[1]:
                        phase_items.add(o.id)
            # ---- memoised move check (repair loops re-verify unchanged moves)
            sig = (scene_sig, tuple(np.round(q, 7)), tuple(np.round(q_goal, 7)), round(speed, 6),
                   tuple(np.round(np.log(K), 7)), None if held is None else held.id,
                   tuple(sorted(phase_items)), tuple((kk, tuple(np.round(vv, 5))) for kk, vv in sorted(item_pos.items())))
            memo = self._move_memo.get(sig)
            if memo is not None:
                for (kind, msg, mag, data) in memo["viol"]:
                    add(Violation(kind, k, re.sub(r"^step \d+:", f"step {k}:", msg), mag, dict(data)))
                min_hd = min(min_hd, memo["min_hd"]); max_ratio = max(max_ratio, memo["max_ratio"])
                for (tt, qq, KK, cc) in memo["trace"]:
                    trace["t"].append(t_total + tt); trace["q"].append(qq); trace["K"].append(KK); trace["tcp"].append(cc)
                    trace["step"].append(k); trace["payload"].append(payload)
                t_total += T
                q = q_goal
                last_target = p_goal
                continue
            min_hd_loc, max_ratio_loc, trace_loc = np.inf, 0.0, []
            # samples
            step_viol = {}
            # Conservative advancement: no robot point moves farther than
            # Vb*h in a time step h, so a step no longer than clearance/Vb
            # cannot skip a collision (20 % added for deflection changes).
            wj = np.abs(dq) / max(dist, 1e-12) * vpk
            Vb = 1.2 * float(self.R @ wj) + 1e-6
            h_min, h_max = max(T / 2000, 1e-4), max(T / 25, 1e-3)
            fixed = None if cfg.certified else list(np.linspace(0, T, cfg.n_fixed_samples))
            t = 0.0
            while True:
                if fixed is not None:
                    if not fixed:
                        break
                    t = fixed.pop(0)
                fr, fv, fa = self._profile(t, T, vpk, ta, dist)
                qc = q + dq * fr
                qd = dq * fv
                qdd = dq * fa
                ql, tau = self.deflected(qc, K, payload, qdd) if (cfg.deflected_geometry or cfg.check_torque) \
                    else (qc, None)
                qgeom = ql if cfg.deflected_geometry else qc
                # C1 joint limits
                lim = np.maximum(qc - self.arm.q_max, self.arm.q_min - qc)
                if np.any(lim > 0):
                    j = int(np.argmax(lim))
                    step_viol.setdefault("joint_limit", Violation(
                        "joint_limit", k, f"step {k}: joint {j+1} exceeds its limit by {np.rad2deg(lim[j]):.1f} deg.",
                        float(lim[j])))
                # C3 torque
                if cfg.check_torque and tau is not None:
                    cap = self.tau_cap(K)
                    over = np.abs(tau) - cap
                    if np.any(over > 0):
                        j = int(np.argmax(over))
                        need = self._min_level_for_torque(abs(tau[j]))
                        v0 = step_viol.get("torque")
                        if v0 is None or over[j] > v0.magnitude:
                            msg = (f"step {k}: joint {j+1} needs {abs(tau[j]):.1f} Nm but holds only {cap[j]:.1f} Nm at "
                                   f"stiffness level {float(self.K_to_level(K[j])):.2f}.")
                            msg += (f" Raise joint {j+1} stiffness to >= {need:.2f}." if need is not None else
                                    f" This exceeds the actuator limit ({cfg.tau_m1:.0f} Nm) at any stiffness; "
                                    f"change the arm posture/path or reduce the payload.")
                            step_viol["torque"] = Violation("torque", k, msg, float(over[j]),
                                                            {"joint": j + 1, "required_level": need})
                # C2 + C6
                cmin, cols, prox, Ts = self._clearances(qgeom, scene, held, item_pos, phase_items)
                for c in cols:
                    kind = "human_collision" if c[0] == "human" else "collision"
                    if kind not in step_viol or self.cfg.d_safe - c[3] > step_viol[kind].magnitude:
                        what = {"link": f"link {c[1]}", "self": f"link {c[1]}", "item": "carried item",
                                "human": f"link {c[1]}"}[c[0]]
                        gapmm = c[3] * 1000
                        desc = (f"penetrates it by {-gapmm:.0f} mm" if gapmm < 0 else
                                f"comes within {gapmm:.0f} mm (required clearance {self.cfg.d_safe*1000:.0f} mm)")
                        step_viol[kind] = Violation(kind, k, f"step {k}: {what} collides with {c[2]}: {desc} at "
                                                             f"t={t:.2f} s of the move; add a via point or change the "
                                                             f"approach height.",
                                                    self.cfg.d_safe - c[3], {"obstacle": c[2]})
                if prox:
                    min_hd_loc = min(min_hd_loc, min(p["gap"] for p in prox))
                if cfg.check_contact and prox:
                    scache = self._sample_cache(qc, payload)
                    clamp_memo = {}
                    for pr in prox:
                        region = region_for_contact(pr["cap"], pr["u"], np.array(scene.human_facing))
                        if region in ("skull", "face"):
                            if pr["gap"] < cfg.d_head:
                                v0 = step_viol.get("keep_out")
                                mag = cfg.d_head - pr["gap"]
                                if v0 is None or mag > v0.magnitude:
                                    step_viol["keep_out"] = Violation(
                                        "keep_out", k, f"step {k}: robot passes {pr['gap']*100:.0f} cm from the "
                                                       f"operator's head (keep-out {cfg.d_head*100:.0f} cm).", mag)
                            continue
                        vu = self._approach_speed(qc, qd, pr, scache)
                        if vu <= 0.005:          # < 5 mm/s: no transient impact
                            continue
                        m_l, m_m, K_c = self._effective(qc, pr, K, payload, scache)
                        ck = pr["cap"].id
                        if ck not in clamp_memo:
                            clamp_memo[ck] = self._is_clamped(pr, scene)
                        clamped = clamp_memo[ck]
                        vmax_c = self._vmax(region, m_l, m_m, K_c, clamped)
                        ratio = vu / max(vmax_c, 1e-9)
                        max_ratio_loc = max(max_ratio_loc, ratio)
                        if ratio > 1.0:
                            v0 = step_viol.get("contact_speed")
                            if v0 is None or ratio - 1 > v0.magnitude:
                                sp_ok = speed / ratio
                                hint_txt, hint_lv = self._soft_hint(qc, qd, pr, K, payload, region, clamped, scache)
                                step_viol["contact_speed"] = Violation(
                                    "contact_speed", k,
                                    f"step {k}: approach speed {vu:.2f} m/s toward the operator's {region.replace('_',' ')}"
                                    + (" (which could be clamped against a surface)" if clamped else "")
                                    + f" exceeds the ISO/TS 15066 limit {vmax_c:.2f} m/s for the current stiffness "
                                    f"(link-side mass {m_l:.1f} kg, Cartesian stiffness {K_c:.0f} N/m). Use speed <= "
                                    f"{sp_ok:.2f} for this move"
                                    + hint_txt + ".",
                                    ratio - 1.0, {"region": region, "clamped": clamped,
                                                  "max_speed_param": round(sp_ok, 3), "soft_levels": hint_lv})
                trace_loc.append((t, qgeom.tolist(), K.tolist(), Ts[-1][:3, 3].tolist()))
                if fixed is None:
                    if t >= T:
                        break
                    h = h_max / 4 if cols else float(np.clip(cmin / Vb, h_min, h_max))
                    if prox:
                        h = min(h, h_max / 3)
                    t = min(t + h, T)
            # C5: speeds are bounded by construction (speed <= 1)
            for v in step_viol.values():
                add(v)
            if len(self._move_memo) > 20000:
                self._move_memo.clear()
            self._move_memo[sig] = {"viol": [(x.kind, x.message, x.magnitude, dict(x.data)) for x in step_viol.values()],
                                    "min_hd": min_hd_loc, "max_ratio": max_ratio_loc, "trace": trace_loc}
            min_hd = min(min_hd, min_hd_loc); max_ratio = max(max_ratio, max_ratio_loc)
            for (tt, qq, KK, cc) in trace_loc:
                trace["t"].append(t_total + tt); trace["q"].append(qq); trace["K"].append(KK); trace["tcp"].append(cc)
                trace["step"].append(k); trace["payload"].append(payload)
            t_total += T
            q = q_goal
            last_target = p_goal

        # C7 task completion
        ok_task = False
        tgt = scene.task.get("item")
        if tgt in item_pos and held is None:
            loc = scene.location(scene.task["target"])
            pos = item_pos[tgt]
            dxy = float(np.linalg.norm(pos[:2] - np.array(loc.point[:2])))
            ok_task = dxy <= cfg.tol_place_xy and abs(pos[2] - (loc.surface_z + scene.item(tgt).size[2] / 2)) < 0.02
            if not ok_task:
                V.append(Violation("task", len(plan["steps"]),
                                   f"{tgt} ended {dxy*100:.0f} cm from the centre of {loc.id} "
                                   f"(tolerance {cfg.tol_place_xy*100:.0f} cm) or on the wrong surface.", dxy))
        else:
            V.append(Violation("task", len(plan["steps"]), f"{tgt} was not delivered (still held or never grasped)."))
        if not scene.expected_feasible:
            V.append(Violation("task", -1, "the task is infeasible for this robot; the plan should declare it."))
        safety_kinds = {"collision", "human_collision", "torque", "contact_speed", "keep_out", "joint_limit"}
        ok = len(V) == 0
        return Report(ok, ok_task and ok, V, t_total, E_mod, min_hd, max_ratio, trace=trace)

    # ---------------------------------------------------- sub-routines
    @staticmethod
    def _scene_sig(scene):
        import hashlib, json as _json
        d = {"o": [(o.id, o.lo, o.hi) for o in scene.obstacles], "h": [(c.a, c.b, c.radius) for c in scene.human],
             "i": [(i.id, i.center, i.size, i.mass) for i in scene.items], "n": scene.name}
        return hashlib.md5(_json.dumps(d, sort_keys=True, default=float).encode()).hexdigest()

    def _select_ik(self, p_goal, q, K, payload, scene, held, item_pos, st, plan, k):
        """Controller-side IK branch selection: among the solutions reached
        from the current configuration and a fixed set of posture seeds,
        take the collision-free one closest (max-norm) to the current
        configuration; if none is collision-free, the closest one."""
        key = (self._scene_sig(scene), tuple(np.round(p_goal, 7)), tuple(np.round(q, 7)),
               tuple(np.round(np.log(K), 7)), round(payload, 6), None if held is None else held.id,
               tuple((kk, tuple(np.round(vv, 5))) for kk, vv in sorted(item_pos.items())),
               (plan["steps"][k + 1].get("op"), plan["steps"][k + 1].get("item")) if k + 1 < len(plan["steps"]) else None)
        if key in self._ik_memo:
            return self._ik_memo[key]
        res = self._select_ik_raw(p_goal, q, K, payload, scene, held, item_pos, st, plan, k)
        if len(self._ik_memo) > 20000:
            self._ik_memo.clear()
        self._ik_memo[key] = res
        return res

    def _posture_seeds(self, p, q_now=None):
        """Elbow-up, wrist-down seeds facing the target (base angle chosen so
        that the arm, which extends along -x at q1 = 0, points at p)."""
        if not hasattr(self.arm, "q_home"):
            return []
        az = float(np.arctan2(p[1], p[0]))
        q1 = (az - np.pi + self.arm.q_home[0] + np.pi) % (2 * np.pi) - np.pi
        if q_now is not None:   # equivalent base angle closest to the current one
            q1 = q1 + 2 * np.pi * np.round((q_now[0] - q1) / (2 * np.pi))
        out = []
        for q2, q3 in ((-1.9, 1.9), (-1.3, 1.6), (-0.9, 1.2)):
            s = self.arm.q_home.copy(); s[0] = q1; s[1] = q2; s[2] = q3; s[3] = -np.pi / 2 - (q2 + q3)
            out.append(np.clip(s, self.arm.q_min, self.arm.q_max))
        return out

    def _select_ik_raw(self, p_goal, q, K, payload, scene, held, item_pos, st, plan, k):
        seeds = [q] + self._posture_seeds(p_goal, q) + list(self._seeds)
        sols = ikmod.solve_all(self.arm, p_goal, seeds=seeds)
        if not sols:
            qb, ok, _, err = ikmod.solve(self.arm, p_goal, q0=q, rng=np.random.default_rng(0), restarts=6)
            return qb, ok, err
        phase = set()
        nxt = plan["steps"][k + 1] if k + 1 < len(plan["steps"]) else None
        if nxt is not None and nxt.get("op") == "grasp":
            phase.add(nxt.get("item", ""))
        if held is not None:
            phase.add(held.id)
            for o in scene.obstacles:
                if o.lo[0] <= p_goal[0] <= o.hi[0] and o.lo[1] <= p_goal[1] <= o.hi[1]:
                    phase.add(o.id)
        # leaving a surface: the support/released item may still be touched
        prv = plan["steps"][k - 1] if k > 0 else None
        if prv is not None and prv.get("op") in ("grasp", "release"):
            pref = held.id if held is not None else prv.get("item")
            for iid, c0 in item_pos.items():
                if pref is None or iid == pref or np.linalg.norm(c0[:2] - p_goal[:2]) < 0.05:
                    phase.add(iid)
                    for o in scene.obstacles:
                        if o.lo[0] <= c0[0] <= o.hi[0] and o.lo[1] <= c0[1] <= o.hi[1]:
                            phase.add(o.id)
        sols = sorted(sols, key=lambda s: float(np.max(np.abs(s - q))))
        for s in sols:
            ok = True
            for f in np.linspace(0, 1, 13)[1:]:
                qq = q + (s - q) * f
                _, cols, _, _ = self._clearances(qq, scene, held, item_pos, phase)
                if any(c[0] != "human" or True for c in cols):
                    if cols:
                        ok = False
                        break
            if ok:
                return s, True, 0.0
        return sols[0], True, 0.0

    def _sample_times(self, q, dq, T, vpk, ta, dist, scene, held, phase_items, K, payload, item_pos=None):
        if T <= 0:
            return [0.0]
        if not self.cfg.certified:
            return list(np.linspace(0, T, self.cfg.n_fixed_samples))
        # conservative advancement: displacement of any robot point over
        # [t, t+h] is bounded by sum_j R_j |dq_j| * (fraction moved); we add
        # 20 % for the change of elastic deflection along the path.
        w = np.abs(dq) / dist * vpk                 # joint speeds at cruise
        Vb = 1.2 * float(self.R @ w) + 1e-6
        ts, t = [0.0], 0.0
        h_min, h_max = T / 2000, max(T / 25, 1e-3)
        while t < T:
            fr, _, _ = self._profile(t, T, vpk, ta, dist)
            qc = q + dq * fr
            ql = self.deflected(qc, K, payload)[0] if self.cfg.deflected_geometry else qc
            cmin, cols, prox, _ = self._clearances(ql, scene, held, item_pos, phase_items)
            if cols:
                h = h_max / 4
            else:
                h = np.clip(cmin / Vb, h_min, h_max)
            if prox:   # near the operator use dense sampling for speed checks
                h = min(h, h_max / 3)
            t = min(t + h, T)
            ts.append(t)
        return ts

    def _sample_cache(self, q, payload):
        Ts = self.arm.frames_fk(q)
        return {"q": q, "Ts": Ts,
                "Ml_inv": np.linalg.inv(self.arm.mass_matrix(q, payload, Ts=Ts)),
                "Mr_inv": np.linalg.inv(self.arm.mass_matrix(q, payload, include_rotor=True, Ts=Ts))}

    def _approach_speed(self, q, qd, pr, cache=None):
        Ts = cache["Ts"] if cache else None
        Jp = self.arm.jacobian(q, Ts=Ts, point=pr["point"], upto=pr["frame"])[:3]
        return float((Jp @ qd) @ pr["u"])

    def _effective(self, q, pr, K, payload, cache=None):
        from .safety import cartesian_quantities
        cache = cache or self._sample_cache(q, payload)
        return cartesian_quantities(self.arm, q, pr["point"], pr["frame"], pr["u"], K,
                                    Ts=cache["Ts"], Ml_inv=cache["Ml_inv"], Mr_inv=cache["Mr_inv"])

    def _vmax(self, region, m_l, m_m, K_c, clamped):
        v = vmax_lookup(region, m_l, m_m, K_c)
        if clamped:
            v = min(v, vmax_clamp_lookup(region, m_l, K_c))
        return v

    def _is_clamped(self, pr, scene) -> bool:
        """The body part can be pinned if a fixed surface lies behind it
        (along the approach direction) within its radius + 10 cm."""
        cap = pr["cap"]
        a, b = np.array(cap.a), np.array(cap.b)
        p = pr["point"] + pr["u"] * pr["gap"]
        ab = b - a
        t = 0.0 if np.allclose(ab, 0) else float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
        c = a + ab * t
        X = c[None, :] + pr["u"][None, :] * np.linspace(0, cap.radius + 0.10, 12)[:, None]
        LO = np.array([o.lo for o in scene.obstacles], float)
        HI = np.array([o.hi for o in scene.obstacles], float)
        inside = np.all((X[:, None, :] >= LO[None]) & (X[:, None, :] <= HI[None]), axis=-1)
        return bool(inside.any())

    def _soft_hint(self, q, qd, pr, K, payload, region, clamped=False, cache=None):
        """Would softening the joints (keeping them torque feasible) raise
        the admissible speed?  Returns a hint string."""
        tau = np.abs(self.arm.gravity_torque(q, payload, Ts=cache["Ts"] if cache else None))
        K2 = K.copy()
        for j in range(6):
            need = self._min_level_for_torque(tau[j] * 1.15 + 1.0)
            if need is not None:
                K2[j] = min(K[j], self.level_to_K(need))
        if np.allclose(K2, K):
            return "", None
        m_l, m_m, K_c = self._effective(q, pr, K, payload, cache)
        v1 = self._vmax(region, m_l, m_m, K_c, clamped)
        m_l, m_m, K_c = self._effective(q, pr, K2, payload, cache)
        v2 = self._vmax(region, m_l, m_m, K_c, clamped)
        if v2 < v1 * 1.05:
            return "", None
        levels = np.round(self.K_to_level(K2), 2).tolist()
        return f", or set stiffness levels to {levels} (admissible speed then {v2:.2f} m/s)", levels

    def _min_level_for_torque(self, tau):
        """Smallest stiffness level whose capacity >= tau (None if none)."""
        if self.cfg.rigid_K is not None:
            return 0.0 if tau <= self.cfg.tau_m1 else None
        cap_max = float(min(self._cap[-1], self.cfg.tau_m1))
        if tau > cap_max:
            return None
        if self._cap[0] >= tau:
            return 0.0
        i = int(np.searchsorted(self._cap, tau))      # capacity increases with K
        lk = np.interp(tau, self._cap[i - 1:i + 1], self._lK[i - 1:i + 1])
        lvl = float(self.K_to_level(np.exp(lk)))
        return round(min(1.0, lvl + 0.005), 3)

    def _stiffness_for_precision(self, q, payload, gp, tol):
        for lvl in np.linspace(0, 1, 21):
            Kt = self.level_to_K(np.full(6, lvl))
            ql, _ = self.deflected(q, Kt, payload)
            if np.linalg.norm(self.arm.fk(ql)[:3, 3] - gp) <= tol:
                return f" Gravity deflection is the cause; use stiffness level >= {lvl:.2f} for the approach move."
        return " Correct the target position (the error is not caused by deflection alone)."

    def _modulation_cost(self, q, K0, K1, payload):
        from .vssea import modulation_time
        tau = np.abs(self.arm.gravity_torque(q, payload))
        t, E = 0.0, 0.0
        for j in range(6):
            if abs(np.log(K1[j] / K0[j])) < 1e-6:
                continue
            x0, x1 = x_from_K(self.p, K0[j]), x_from_K(self.p, K1[j])
            t = max(t, modulation_time(self.p, x0, x1))   # joints modulate in parallel
            E += self._mod_energy_linear(x0, x1, tau[j])
        return t, E

    def _mod_energy_linear(self, x0, x1, tau, n=30):
        """Carriage work with the linear-beam axial reaction
        F_ax = n_s * P * phi, phi = 1.5 delta / x (slope at the load point)."""
        from .vssea import K_vsam, kb_timo
        p = self.p
        xs = np.linspace(x0, x1, n)
        F = []
        for x in xs:
            qd = tau / K_vsam(p, x)
            delta = p.r * qd
            P = kb_timo(p, x) * delta
            Fax = p.n * P * 1.5 * delta / x
            Ffr = p.mu_roll * (p.n * p.preload + p.n * P)
            F.append(max(Ffr + Fax * np.sign(x0 - x1), 0.0))
        return float(abs(np.trapezoid(F, xs)) / p.screw_eff)


def feedback_text(rep: Report, mode: str = "quantitative", limit: int = 12) -> str:
    """Natural-language feedback for the planner.

    mode: 'binary' | 'qualitative' | 'quantitative'
    """
    if rep.ok:
        return "PASS"
    if mode == "binary":
        return "FAIL: the plan violates at least one constraint. Revise the plan."
    lines = []
    seen = set()
    for v in rep.violations[:limit]:
        if mode == "qualitative":
            key = (v.kind, v.step)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- step {v.step}: {v.kind.replace('_', ' ')} violation")
        else:
            lines.append(f"- [{v.kind}] {v.message}")
    more = len(rep.violations) - limit
    if more > 0:
        lines.append(f"- ... and {more} more")
    return "FAIL:\n" + "\n".join(lines)
