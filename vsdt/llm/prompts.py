"""Prompt construction for the LLM planner."""
from __future__ import annotations

import json
import numpy as np

from ..scene import Scene
from ..verifier import Verifier
from ..safety import ISO_BODY, transient_limit

PLAN_FORMAT = """Return ONLY one JSON object (no prose outside it) with this structure:
{
  "feasible": true | false,          // false if the task cannot be done safely by this robot
  "reason": "<short text>",
  "steps": [
    {"op": "move", "target": [x, y, z] | "<item id>" | "<location id>",
     "offset": [dx, dy, dz],          // optional, added to the target (metres)
     "speed": <0.01..1.0>,            // fraction of the max joint speed (pi/2 rad/s)
     "stiffness": <0..1> | [s1, s2, s3, s4, s5, s6]   // joint stiffness level(s), see table
    },
    {"op": "grasp", "item": "<item id>"},
    {"op": "release"}
  ]
}
Semantics: a target given as an item id means that item's grasp point (centre of its top face);
a location id means its place point on the support surface. The tool always points straight down and
the TCP is the centre of the gripper fingertips. To place an item, the TCP must be at
(support surface z + item height). Stiffness changes are executed before the move starts
(this costs time, about 1 s for the full range). Speeds and stiffness persist only for the step in
which they are given (default speed 0.5; default stiffness = previous)."""


def capability_text(v: Verifier, stiffness_control: bool = True) -> str:
    lv = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    K = v.level_to_K(lv)
    cap = v.tau_cap(K)
    rows = "\n".join(f"  level {l:.2f}: K = {k:7.0f} Nm/rad, torque capacity {c:5.1f} Nm"
                     for l, k, c in zip(lv, K, cap))
    iso = "\n".join(
        f"  {r.replace('_', ' ')}: " + (f"transient force limit {transient_limit(r):.0f} N"
                                         if transient_limit(r) else "contact not permitted (keep-out)")
        for r in ISO_BODY)
    a = v.arm
    s = f"""ROBOT
- 6 revolute joints driven by variable-stiffness series elastic actuators (VSSEA), payload up to
  {a.payload:.1f} kg nominal, reach about 0.85 m from the shoulder (UR5e-class kinematics), base at the origin on a pedestal (top z = -0.02).
- Joint limits +-180 deg; the tool points straight down for every target. Max joint speed pi/2 rad/s at speed = 1.0.
- Each actuator holds at most {v.cfg.tau_m1:.0f} Nm. The elastic element limits the torque further
  depending on the stiffness level:
{rows}
  Softer joints decouple the motor inertia and lower the impact force on a person, but they
  carry less torque and deflect more. Gravity deflection is compensated by the controller
  (residual about {v.cfg.comp_residual*100:.0f} %); inertial deflection is not.
SAFETY RULES (checked by the digital twin)
- Keep >= {v.cfg.d_safe*1000:.0f} mm clearance between robot/carried item and obstacles.
- Near the operator (closer than {v.cfg.d_prox:.2f} m) the approach speed toward each body region must
  stay below the ISO/TS 15066 transient-contact limit, which depends on the robot effective mass
  and joint stiffness. Body-region limits:
{iso}
- Keep >= {v.cfg.d_head*100:.0f} cm from the operator's head.
- Grasp: TCP within {v.cfg.tol_grasp*1000:.0f} mm of the grasp point. Release: item bottom at most
  {v.cfg.tol_place_z*1000:.0f} mm above the support surface, item centre within {v.cfg.tol_place_xy*100:.0f} cm
  of the place point.
"""
    if not stiffness_control:
        s += "- NOTE: in this configuration the stiffness is fixed at level 1.0; omit the 'stiffness' field.\n"
    return s


SYSTEM = """You are the motion-planning module of an industrial collaborative robot cell.
You translate an operator instruction into an executable, safe pick-and-place plan for a
digital twin that verifies it before deployment. Be precise with coordinates (metres).
Think about clearance over obstacles, payload torque, and the operator's position.
If the task cannot be done safely or within the actuator limits, set "feasible": false."""


def initial_prompt(scene: Scene, v: Verifier, stiffness_control=True) -> str:
    return (capability_text(v, stiffness_control) + "\nWORLD MODEL (JSON)\n" +
            json.dumps(scene.world_model(), indent=1) +
            f"\n\nINSTRUCTION\n{scene.instruction}\n\n{PLAN_FORMAT}")


def feedback_prompt(feedback: str) -> str:
    return ("The digital twin rejected the plan.\n" + feedback +
            "\nReturn a corrected complete plan as one JSON object in the same format.")
