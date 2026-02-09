# src/orchestrator/schema.py
from __future__ import annotations

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mode", "meshing", "repair", "exposure"],
    "properties": {
        "mode": {"type": "string", "enum": ["preview", "accurate", "balanced"]},
        "meshing": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_edge", "angle_deg", "tolerance"],
            "properties": {
                "max_edge": {"type": "number", "minimum": 0.0},
                "angle_deg": {"type": "number", "minimum": 0.0, "maximum": 90.0},
                "tolerance": {"type": "number", "minimum": 0.0},
            },
        },
        "repair": {
            "type": "object",
            "additionalProperties": False,
            "required": ["remove_degenerate", "fix_normals"],
            "properties": {
                "remove_degenerate": {"type": "boolean"},
                "fix_normals": {"type": "boolean"},
            },
        },
        "exposure": {
            "type": "object",
            "additionalProperties": False,
            "required": ["rays", "samples_per_face", "threshold_ratio", "soft_area"],
            "properties": {
                "rays": {"type": "integer", "minimum": 16, "maximum": 2000000},
                "samples_per_face": {"type": "integer", "minimum": 1, "maximum": 128},
                "threshold_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "soft_area": {"type": "boolean"},
            },
        },
    },
}

DEFAULT_PLAN = {
    "mode": "balanced",
    "meshing": {"max_edge": 0.15, "angle_deg": 15.0, "tolerance": 0.01},
    "repair": {"remove_degenerate": True, "fix_normals": True},
    "exposure": {"rays": 200000, "samples_per_face": 2, "threshold_ratio": 0.6, "soft_area": True},
}
