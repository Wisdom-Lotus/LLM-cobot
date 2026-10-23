"""Plan representation exchanged between the planner (LLM) and the
digital-twin verifier."""
from __future__ import annotations

import json
import re
import jsonschema

PLAN_SCHEMA = {
    "type": "object",
    "required": ["feasible", "steps"],
    "properties": {
        "feasible": {"type": "boolean"},
        "reason": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op": {"enum": ["move", "grasp", "release"]},
                    "target": {"oneOf": [
                        {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        {"type": "string"}]},
                    "offset": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "speed": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                    "stiffness": {"oneOf": [
                        {"type": "number", "minimum": 0, "maximum": 1},
                        {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                         "minItems": 6, "maxItems": 6}]},
                    "item": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
    },
}


class PlanFormatError(Exception):
    pass


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ``` fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = m.group(1) if m else None
    if cand is None:
        start = text.find("{")
        if start < 0:
            raise PlanFormatError("no JSON object found in the reply")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i + 1]
                    break
    if cand is None:
        raise PlanFormatError("unterminated JSON object")
    try:
        return json.loads(cand)
    except json.JSONDecodeError as e:
        raise PlanFormatError(f"invalid JSON: {e}")


def validate(plan: dict) -> None:
    try:
        jsonschema.validate(plan, PLAN_SCHEMA)
    except jsonschema.ValidationError as e:
        raise PlanFormatError(f"schema violation at {list(e.path)}: {e.message}")
    for k, st in enumerate(plan["steps"]):
        if st["op"] == "move" and "target" not in st:
            raise PlanFormatError(f"step {k}: 'move' requires 'target'")
        if st["op"] == "grasp" and "item" not in st:
            raise PlanFormatError(f"step {k}: 'grasp' requires 'item'")
