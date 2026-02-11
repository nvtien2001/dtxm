# src/planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from jsonschema import validate, ValidationError

from .schema import PLAN_SCHEMA, DEFAULT_PLAN
from .ollama_client import OllamaCfg, chat_json

SYSTEM_PROMPT = (
    "Bạn là kỹ sư CAD/mesh. Chỉ trả về JSON hợp lệ, KHÔNG giải thích, KHÔNG markdown.\n"
    "Schema:\n"
    "{\n"
    '  "mode": "preview|accurate|balanced",\n'
    '  "meshing": {"max_edge": float, "angle_deg": float, "tolerance": float},\n'
    '  "repair": {"remove_degenerate": bool, "fix_normals": bool},\n'
    '  "exposure": {"rays": int, "samples_per_face": int, "threshold_ratio": float, "soft_area": bool}\n'
    "}\n"
    "Quy tắc:\n"
    "- Chỉ JSON, không code fence.\n"
    "- Ưu tiên offline + GPU, tránh tham số cực lớn nếu không cần.\n"
)

@dataclass
class PlanResult:
    plan: Dict[str, Any]
    ok: bool
    error: str = ""

def plan_from_metrics(cfg: OllamaCfg, metrics: dict) -> PlanResult:
    user = {
        "goal": "tính exposed area xi mạ offline",
        "metrics": metrics,
        "constraints": {"offline": True, "prefer_gpu": True, "two_pass": True},
    }

    try:
        plan = chat_json(cfg, SYSTEM_PROMPT, user, temperature=0.1)
        validate(instance=plan, schema=PLAN_SCHEMA)
        return PlanResult(plan=plan, ok=True)
    except (ValidationError, KeyError, ValueError) as e:
        return PlanResult(plan=DEFAULT_PLAN.copy(), ok=False, error=f"Invalid plan JSON/schema: {e}")
    except Exception as e:
        return PlanResult(plan=DEFAULT_PLAN.copy(), ok=False, error=f"Ollama error: {e}")
