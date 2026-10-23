"""Verifier behaviour on hand-built cases (needs results/*_table.npz)."""
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsdt.robot import ur5e_class_arm
from vsdt.scene import generate_scene, Box
from vsdt.verifier import Verifier, VerifierConfig
from vsdt.planners import RepairPlanner

HAVE_TABLES = all(os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", f))
                  for f in ("gain_table.npz", "clamp_table.npz"))
pytestmark = pytest.mark.skipif(not HAVE_TABLES, reason="contact tables not built")


@pytest.fixture(scope="module")
def solved():
    v = Verifier(ur5e_class_arm())
    for seed in range(20):
        s = generate_scene(2000 + seed, 1)
        plan, rep, _ = RepairPlanner(v, "variable").plan(s)
        if rep.ok and plan["feasible"]:
            return v, s, plan
    pytest.skip("no solvable tier-1 scene found")


def test_valid_plan_passes(solved):
    v, s, plan = solved
    assert v.verify(plan, s).ok


def test_detects_wall_in_path(solved):
    v, s, plan = solved
    s2 = copy.deepcopy(s)
    tcp = np.array(v.arm.fk(v.arm.q_home)[:3, 3])
    # a large wall through the middle of the workspace intersects any transfer
    s2.obstacles.append(Box("wall", [-3, -0.003, -1.0], [3, 0.003, 3], "wall"))
    s2.obstacles.append(Box("wall2", [-0.003, -3, -1.0], [0.003, 3, 3], "wall"))
    assert not v.verify(plan, s2).ok


def test_detects_soft_joints_under_load(solved):
    v, s, plan = solved
    p2 = copy.deepcopy(plan)
    for st in p2["steps"]:
        if st["op"] == "move":
            st["stiffness"] = 0.0
    r = v.verify(p2, s)
    assert "torque" in r.kinds or "precision" in r.kinds


def test_detects_wrong_item(solved):
    v, s, plan = solved
    p2 = copy.deepcopy(plan)
    for st in p2["steps"]:
        if st["op"] == "grasp":
            st["item"] = "item_2"
    assert not v.verify(p2, s).ok


def test_infeasible_declaration_scored(solved):
    v, s, _ = solved
    r = v.verify({"feasible": False, "reason": "x", "steps": []}, s)
    assert not r.ok   # the scene is feasible, so refusing is wrong
