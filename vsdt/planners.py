"""Non-LLM planners used as baselines and as the algorithmic reference.

* ``template_plan``   : fixed pick-and-place skeleton from the task spec
* ``RepairPlanner``   : verifier-in-the-loop repair using the structured
                        violation data (the same information the LLM gets
                        as quantitative feedback).  Variants:
    - mode="variable": per-joint stiffness scheduling (proposed)
    - mode="rigid"   : stiffness fixed at the maximum (rigid-robot PFL
                       practice: only speed is reduced near the operator)
    - mode="soft"    : stiffness fixed at the minimum
"""
from __future__ import annotations

import copy
import numpy as np

from .scene import Scene
from .verifier import Verifier, Report


def _arc(p0, p1, z, step_deg=25.0, avoid=None):
    """Way-points on an arc around the base from p0 to p1 at height z.
    The arc direction avoids the azimuth ``avoid`` (operator) if given."""
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    r0, r1 = np.hypot(*p0[:2]), np.hypot(*p1[:2])
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi          # shortest direction
    if avoid is not None:
        rel = (avoid - a0) % (2 * np.pi)
        inside = (0 < rel < d) if d > 0 else (d < rel - 2 * np.pi < 0)
        if inside:
            d = d - np.sign(d) * 2 * np.pi                  # go the other way round
    n = max(1, int(np.ceil(abs(np.rad2deg(d)) / step_deg)))
    pts = []
    for i in range(1, n):
        f = i / n
        a, r = a0 + f * d, r0 + f * (r1 - r0)
        pts.append([round(r * np.cos(a), 3), round(r * np.sin(a), 3), round(z, 3)])
    return pts


def template_plan(scene: Scene, lift: float = 0.15, transit_speed: float = 0.6,
                  fine_speed: float = 0.2, stiffness=1.0, z_clear: float = 0.30) -> dict:
    """Pick-and-place skeleton: rise to a safe height, follow an arc around
    the base (away from the operator), descend vertically."""
    it = scene.item(scene.task["item"])
    loc = scene.location(scene.task["target"])
    hz = it.size[2]
    place = [loc.point[0], loc.point[1], loc.surface_z + hz]
    gp = list(it.grasp_point)
    tops = [o.hi[2] for o in scene.obstacles if o.kind not in ("beam",)]
    z_safe = max(max(tops) + z_clear, gp[2] + lift, place[2] + lift)
    avoid = None
    if scene.human:
        c = np.mean([np.array(h.a) for h in scene.human], axis=0)
        avoid = float(np.arctan2(c[1], c[0]))
    S = stiffness
    mv = lambda t, sp, **kw: {"op": "move", "target": t, "speed": sp, "stiffness": S, **kw}
    above_src = [gp[0], gp[1], z_safe]
    above_dst = [place[0], place[1], z_safe]
    steps = [mv(above_src, transit_speed)]
    steps += [mv(gp[:2] + [gp[2] + lift], fine_speed) ] if z_safe - gp[2] > lift + 0.02 else []
    steps += [mv(it.id, fine_speed), {"op": "grasp", "item": it.id}, mv(above_src, fine_speed)]
    steps += [mv(w, transit_speed) for w in _arc(above_src, above_dst, z_safe, avoid=avoid)]
    steps += [mv(above_dst, transit_speed)]
    steps += [mv(place[:2] + [place[2] + lift], fine_speed)] if z_safe - place[2] > lift + 0.02 else []
    steps += [mv(place, fine_speed), {"op": "release"}, mv(above_dst, fine_speed)]
    return {"feasible": True, "reason": "", "steps": steps}


class RepairPlanner:
    def __init__(self, verifier: Verifier, mode: str = "variable", max_iter: int = 12):
        self.v, self.mode, self.max_iter = verifier, mode, max_iter

    def _init_plan(self, scene):
        s0 = {"variable": 1.0, "rigid": 1.0, "soft": 0.0}[self.mode]
        return template_plan(scene, stiffness=s0)

    def _levels(self, st):
        s = st.get("stiffness", 1.0)
        return list(np.full(6, s)) if np.isscalar(s) else list(s)

    def plan(self, scene: Scene):
        plan = self._init_plan(scene)
        history = []
        for it in range(self.max_iter):
            rep = self.v.verify(plan, scene)
            history.append(rep)
            if rep.ok:
                break
            changed = self.repair(plan, rep, scene)
            if not changed:
                break
        if not rep.ok:
            # declare infeasible only if a hard actuator limit is the cause
            hard = any(v.kind == "torque" and v.data.get("required_level") is None for v in rep.violations)
            if hard:
                plan = {"feasible": False, "reason": "payload/reach exceeds actuator capability", "steps": []}
                rep = self.v.verify(plan, scene)
                history.append(rep)
        if self.mode in ("variable", "rigid") and rep.ok and plan["feasible"]:
            plan, rep = self.soften(plan, scene, rep)
            history.append(rep)
        return plan, rep, history

    # ------------------------------------------------------------ repair
    def repair(self, plan, rep: Report, scene) -> bool:
        changed = False
        steps = plan["steps"]
        for v in rep.violations:
            k = v.step
            if v.kind == "torque" and 0 <= k < len(steps) and steps[k]["op"] == "move":
                need = v.data.get("required_level")
                if need is None or self.mode in ("rigid",):
                    continue
                if self.mode == "soft":
                    continue
                lv = self._levels(steps[k])
                j = v.data["joint"] - 1
                if lv[j] < need:
                    lv[j] = min(1.0, need + 0.03)
                    steps[k]["stiffness"] = lv
                    changed = True
            elif v.kind == "precision" and 0 <= k < len(steps):
                # stiffen the approach move preceding the grasp/release
                kk = k - 1
                while kk >= 0 and steps[kk]["op"] != "move":
                    kk -= 1
                if kk >= 0:
                    if "released" in v.message or "penetrates" in v.message:
                        dz = v.magnitude + (0.012 if "released" in v.message else 0.004)
                        sign = -1 if "released" in v.message else 1
                        t = steps[kk]["target"]
                        if isinstance(t, list):
                            steps[kk]["target"] = [t[0], t[1], t[2] + sign * (v.magnitude + 0.006)]
                            changed = True
                        continue
                    lv = self._levels(steps[kk])
                    if self.mode != "soft" and min(lv) < 1.0:
                        steps[kk]["stiffness"] = 1.0
                        changed = True
                    else:
                        # deflection compensation (feed-forward offset)
                        tcp = np.array(v.data.get("tcp", [0, 0, 0]))
                        gp = np.array(v.data.get("grasp_point", [0, 0, 0]))
                        off = np.array(steps[kk].get("offset", [0, 0, 0]), float) + (gp - tcp)
                        steps[kk]["offset"] = off.round(4).tolist()
                        changed = True
            elif v.kind == "contact_speed" and 0 <= k < len(steps) and steps[k]["op"] == "move":
                lvls = v.data.get("soft_levels")
                if self.mode == "variable" and lvls is not None:
                    cur = self._levels(steps[k])
                    newl = [min(a, b) for a, b in zip(cur, lvls)]
                    if any(b < a - 1e-6 for a, b in zip(cur, newl)):
                        steps[k]["stiffness"] = [round(x, 3) for x in newl]
                        changed = True
                        continue
                sp = v.data.get("max_speed_param", 0.1)
                new = max(0.02, round(0.92 * sp, 3))
                if steps[k].get("speed", 0.5) > new:
                    steps[k]["speed"] = new
                    changed = True
            elif v.kind in ("collision", "keep_out", "human_collision") and 0 <= k < len(steps) \
                    and steps[k]["op"] == "move":
                if steps[k].get("_via", 0) >= 3:
                    continue
                prev = self._prev_point(plan, k, scene)
                tgt = self.v._resolve_target(steps[k], scene)
                lift = 0.08 * (steps[k].get("_via", 0) + 1)
                if prev is None:
                    continue
                via = (np.array(prev) + tgt) / 2
                via[2] = min(max(prev[2], tgt[2]) + lift, 0.55)
                if v.kind in ("keep_out",):
                    via[2] = max(prev[2], tgt[2]) + 0.05
                    via[:2] *= 0.6    # pull toward the robot base, away from the operator
                new = copy.deepcopy(steps[k])
                new["target"] = via.round(3).tolist()
                new.pop("offset", None)
                new["_via"] = steps[k].get("_via", 0) + 1
                steps[k]["_via"] = new["_via"]
                steps.insert(k, new)
                return True  # indices shift: re-verify
        return changed

    def _prev_point(self, plan, k, scene):
        for kk in range(k - 1, -1, -1):
            if plan["steps"][kk]["op"] == "move":
                return self.v._resolve_target(plan["steps"][kk], scene)
        return np.array(self.v.arm.fk(self.v.arm.q_home)[:3, 3])

    # ----------------------------------------------- stiffness lowering
    def soften(self, plan, scene, rep):
        """Speed/stiffness pass on a verified plan.  Transit moves are
        grouped into blocks (maximal runs of moves not followed by a
        grasp/release); each block is raised to full speed and, in
        'variable' mode, set to the minimum torque-feasible stiffness
        (30 % torque margin, per joint) computed from the verified
        trajectory.  Contact violations are then repaired inside the block
        (softer joints if that helps, otherwise lower speed).  A block
        change is kept only if the plan still verifies and the cycle time
        decreases."""
        best_plan, best_rep = plan, rep
        steps = plan["steps"]
        blocks, cur = [], []
        for k, st in enumerate(steps):
            nxt = steps[k + 1]["op"] if k + 1 < len(steps) else None
            if st["op"] == "move" and nxt not in ("grasp", "release"):
                cur.append(k)
            else:
                if cur:
                    blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
        for blk in blocks:
            options = [False, True] if self.mode == "variable" else [False]
            for soften in options:
                cand = self._try_block(best_plan, best_rep, blk, scene, soften)
                if cand is not None and cand[1].cycle_time < best_rep.cycle_time - 1e-3:
                    best_plan, best_rep = cand
        # precision approaches (followed by grasp/release): speed only,
        # stiffness unchanged, bounded by the contact limits
        for k, st in enumerate(steps):
            nxt = steps[k + 1]["op"] if k + 1 < len(steps) else None
            if st["op"] == "move" and nxt in ("grasp", "release"):
                cand = self._try_block(best_plan, best_rep, [k], scene, False)
                if cand is not None and cand[1].cycle_time < best_rep.cycle_time - 1e-3:
                    best_plan, best_rep = cand
        return best_plan, best_rep

    def _try_block(self, base_plan, base_rep, blk, scene, soften):
        cand = copy.deepcopy(base_plan)
        if soften:
            need = self._block_levels(base_rep, blk)
            if need is None:
                return None
            for k in blk:
                cand["steps"][k]["stiffness"] = need
        for k in blk:
            cand["steps"][k]["speed"] = 1.0
        r = None
        for _ in range(8):
            r = self.v.verify(cand, scene)
            if r.ok:
                return cand, r
            bad = [v for v in r.violations if v.step in blk]
            if not bad or len(bad) < len(r.violations) or \
                    any(v.kind not in ("contact_speed", "torque") for v in bad):
                return None
            for v in bad:
                cs = cand["steps"][v.step]
                if v.kind == "contact_speed" and soften and v.data.get("soft_levels") is not None \
                        and any(b < a - 1e-6 for a, b in zip(self._levels(cs), v.data["soft_levels"])):
                    new = [round(min(a, b), 3) for a, b in zip(self._levels(cs), v.data["soft_levels"])]
                    for k in blk:      # keep the block uniform (no extra modulation)
                        cand["steps"][k]["stiffness"] = [round(min(a, b), 3) for a, b in
                                                         zip(self._levels(cand["steps"][k]), new)]
                elif v.kind == "contact_speed":
                    cs["speed"] = max(0.02, round(0.92 * v.data["max_speed_param"], 3))
                elif v.kind == "torque" and v.data.get("required_level") is not None:
                    j = v.data["joint"] - 1
                    for k in blk:
                        lv2 = list(self._levels(cand["steps"][k]))
                        lv2[j] = min(1.0, max(lv2[j], v.data["required_level"] + 0.03))
                        cand["steps"][k]["stiffness"] = lv2
        return None

    def _block_levels(self, rep, blk):
        """Minimum per-joint stiffness level carrying 1.3x the gravity
        torque (+2 Nm) at every verified sample of the block."""
        idx = [i for i, s in enumerate(rep.trace.get("step", [])) if s in blk]
        if not idx:
            return None
        need = np.zeros(6)
        arm = self.v.arm
        for i in idx[::3]:
            q = np.array(rep.trace["q"][i])
            tau = np.abs(arm.gravity_torque(q, rep.trace["payload"][i])) * 1.3 + 2.0
            for j in range(6):
                lvl = self.v._min_level_for_torque(tau[j])
                if lvl is None:
                    return None
                need[j] = max(need[j], lvl)
        return np.round(need, 3).tolist()

    def _torque_levels(self, plan, k, scene):
        """Required stiffness level per joint for move k (gravity at the
        start and end configurations, 30 % margin)."""
        from . import ik
        v = self.v
        # find held payload at step k
        held = None
        for st in plan["steps"][:k]:
            if st["op"] == "grasp":
                held = scene.item(st["item"])
            elif st["op"] == "release":
                held = None
        payload = 0.0 if held is None else held.mass
        prev = self._prev_point(plan, k, scene)
        tgt = v._resolve_target(plan["steps"][k], scene)
        need = np.zeros(6)
        for p in (prev, 0.5 * (prev + tgt), tgt):
            q, ok, _, _ = ik.solve(v.arm, p, restarts=3)
            if not ok:
                continue
            tau = np.abs(v.arm.gravity_torque(q, payload)) * 1.3 + 2.0
            for j in range(6):
                lvl = v._min_level_for_torque(tau[j])
                if lvl is None:
                    return None
                need[j] = max(need[j], lvl)
        return need
