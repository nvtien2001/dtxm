# src/pipeline.py
from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ============================================================
# 0) Ensure we import the built pybind11 module (.pyd) correctly
# ============================================================

BINDIR = os.getenv("BREPMESH_BINDIR")
if not BINDIR:
    here = os.path.abspath(os.path.dirname(__file__))
    BINDIR = os.path.abspath(os.path.join(here, "..", "brepmesh", "build", "out", "Release"))

if BINDIR not in sys.path:
    sys.path.insert(0, BINDIR)

import brepmesh  # noqa: E402

# Self-check để bắt nhầm module ngay lập tức
if not hasattr(brepmesh, "mesh_file_3dm"):
    raise ImportError(
        "Đang import nhầm module 'brepmesh' (không có mesh_file_3dm). "
        f"Imported from: {getattr(brepmesh, '__file__', None)}. "
        f"sys.path[0:8]={sys.path[:8]}"
    )

# ============================================================
# 1) Config models
# ============================================================

@dataclass
class TwoPassConfig:
    # Pass 1 (preview)
    preview_max_edge: float = 0.50
    preview_angle_deg: float = 25.0
    preview_tolerance: float = 0.05

    # Pass 2 (accurate default - nếu Ollama không dùng)
    accurate_max_edge: float = 0.15
    accurate_angle_deg: float = 15.0
    accurate_tolerance: float = 0.02

    # Ollama
    use_ollama: bool = True
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_timeout_sec: int = 120

    # Exposure (placeholder - bạn sẽ thay bằng thuật toán exposed area thật)
    exposure_rays: int = 2000
    exposure_samples_per_face: int = 1
    exposure_threshold_ratio: float = 0.50


# ============================================================
# 2) Mesh wrapper + sanity checks
# ============================================================

def mesh_with_params(path_3dm: str, max_edge: float, angle_deg: float, tolerance: float):
    """Call brepmesh.mesh_file_3dm and return MeshResult from pybind."""
    if not os.path.isfile(path_3dm):
        raise FileNotFoundError(path_3dm)

    r = brepmesh.mesh_file_3dm(path_3dm, float(max_edge), float(angle_deg), float(tolerance))

    nv = len(r.vertices_xyz) // 3
    nt = len(r.faces_tri) // 3

    if len(r.vertices_xyz) % 3 != 0:
        raise ValueError(f"vertices_xyz length not multiple of 3: {len(r.vertices_xyz)}")
    if len(r.faces_tri) % 3 != 0:
        raise ValueError(f"faces_tri length not multiple of 3: {len(r.faces_tri)}")

    if nv == 0 or nt == 0:
        msg = getattr(r, "message", "")
        raise ValueError(f"Empty mesh: nv={nv}, nt={nt}. MeshResult.message={msg!r}")

    # faces index range check (FAST)
    # 1) Quick min/max check
    fmin = min(r.faces_tri) if r.faces_tri else 0
    fmax = max(r.faces_tri) if r.faces_tri else -1
    if fmin < 0 or fmax >= nv:
        raise ValueError(f"Mesh has out-of-range indices: min={fmin}, max={fmax}, nv={nv}")

    # 2) Optional sampling check (avoid O(nt) full scan)
    # sample a few thousand tris at most
    if nt > 200_000:
        step = max(1, nt // 5000)   # ~5000 samples
    else:
        step = max(1, nt // 2000)   # ~2000 samples

    bad = 0
    for t in range(0, nt, step):
        i = t * 3
        a, b, c = r.faces_tri[i], r.faces_tri[i + 1], r.faces_tri[i + 2]
        if not (0 <= a < nv and 0 <= b < nv and 0 <= c < nv):
            bad += 1
            if bad > 5:
                break
    if bad:
        raise ValueError(f"Mesh seems corrupted: sampled bad triangles={bad}, nv={nv}, nt={nt}")

    return r


# ============================================================
# 3) Metrics from preview mesh (for Ollama planning)
# ============================================================

def compute_preview_metrics(mesh_result) -> Dict[str, Any]:
    """Compute light-weight metrics from preview mesh for planning."""
    v = mesh_result.vertices_xyz
    f = mesh_result.faces_tri
    nv = len(v) // 3
    nt = len(f) // 3

    xs = v[0::3]
    ys = v[1::3]
    zs = v[2::3]
    bb = {
        "min": [float(min(xs)), float(min(ys)), float(min(zs))],
        "max": [float(max(xs)), float(max(ys)), float(max(zs))],
    }
    diag = ((bb["max"][0] - bb["min"][0]) ** 2 + (bb["max"][1] - bb["min"][1]) ** 2 + (bb["max"][2] - bb["min"][2]) ** 2) ** 0.5

    return {
        "nv": int(nv),
        "nt": int(nt),
        "bbox": bb,
        "bbox_diag": float(diag),
        "preview": {"note": "metrics are from preview mesh"},
    }


# ============================================================
# 4) Ollama planner (local HTTP)
# ============================================================

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return s

def ollama_plan(model: str, metrics: Dict[str, Any], url: str, timeout_sec: int = 120) -> Dict[str, Any]:
    """
    Ask Ollama to return a STRICT JSON plan.
    Requires: pip install requests
    """
    import requests  # local import

    # Schema cứng, yêu cầu chỉ JSON
    system = (
        "Bạn là kỹ sư CAD/mesh. CHỈ trả về 1 JSON object đúng schema, KHÔNG giải thích.\n"
        "Schema:\n"
        "{\n"
        "  \"mode\": \"preview\" | \"accurate\",\n"
        "  \"meshing\": {\"max_edge\": number, \"angle_deg\": number, \"tolerance\": number},\n"
        "  \"repair\": {\"remove_degenerate\": bool, \"fix_normals\": bool},\n"
        "  \"exposure\": {\"rays\": int, \"samples_per_face\": int, \"threshold_ratio\": number}\n"
        "}\n"
        "Ràng buộc:\n"
        "- max_edge > 0; tolerance > 0; angle_deg trong [5..45].\n"
        "- Số phải hợp lý theo bbox_diag (kích thước mẫu).\n"
        "Nếu không chắc, chọn conservative: max_edge ~ bbox_diag/200..bbox_diag/500.\n"
    )

    user = {"goal": "tính exposed area xi mạ offline", "metrics": metrics}

    r = requests.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=timeout_sec,
    )
    r.raise_for_status()

    content = r.json()["message"]["content"]
    content = _strip_code_fences(content)

    return json.loads(content)


def _safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _clamp_plan(plan: Dict[str, Any], metrics: Dict[str, Any], cfg: TwoPassConfig) -> Dict[str, Any]:
    """
    Clamp plan parameters to safe ranges based on bbox_diag.
    Tránh Ollama đề xuất quá mịn => nổ triangles / chậm.
    """
    diag = float(metrics.get("bbox_diag") or 0.0)
    if diag <= 0:
        return plan

    # Heuristic ranges
    # max_edge: [diag/5000 .. diag/20] (tuỳ size model)
    min_edge = max(diag / 5000.0, 0.005)   # không nhỏ quá
    max_edge = max(diag / 20.0, min_edge)

    # tolerance: [max_edge/200 .. max_edge/2]
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    m = plan.get("meshing", {}) if isinstance(plan.get("meshing", {}), dict) else {}
    pe = float(m.get("max_edge", cfg.accurate_max_edge))
    pa = float(m.get("angle_deg", cfg.accurate_angle_deg))
    pt = float(m.get("tolerance", cfg.accurate_tolerance))

    pe = clamp(pe, min_edge, max_edge)
    pa = clamp(pa, 5.0, 45.0)
    pt = clamp(pt, max(pe / 200.0, 1e-6), max(pe / 2.0, 1e-6))

    plan["meshing"] = {"max_edge": pe, "angle_deg": pa, "tolerance": pt}

    ex = plan.get("exposure", {}) if isinstance(plan.get("exposure", {}), dict) else {}
    rays = int(ex.get("rays", cfg.exposure_rays))
    spf = int(ex.get("samples_per_face", cfg.exposure_samples_per_face))
    thr = float(ex.get("threshold_ratio", cfg.exposure_threshold_ratio))

    # clamp exposure
    rays = int(clamp(rays, 200, 200000))
    spf = int(clamp(spf, 1, 8))
    thr = clamp(thr, 0.05, 0.95)

    plan["exposure"] = {"rays": rays, "samples_per_face": spf, "threshold_ratio": thr}

    rep = plan.get("repair", {}) if isinstance(plan.get("repair", {}), dict) else {}
    plan["repair"] = {
        "remove_degenerate": bool(rep.get("remove_degenerate", True)),
        "fix_normals": bool(rep.get("fix_normals", False)),
    }

    if plan.get("mode") not in ("preview", "accurate"):
        plan["mode"] = "accurate"

    return plan


# ============================================================
# 5) Exposed-area placeholder
# ============================================================

def exposure_preview_stub(mesh_result, rays: int, samples_per_face: int, threshold_ratio: float) -> Dict[str, Any]:
    """
    Stub: trả số liệu giả để bạn test pipeline.
    """
    nt = len(mesh_result.faces_tri) // 3
    exposed_ratio = 0.6
    return {
        "exposed_ratio": float(exposed_ratio),
        "estimated_exposed_faces": int(nt * exposed_ratio),
        "rays": int(rays),
        "samples_per_face": int(samples_per_face),
        "threshold_ratio": float(threshold_ratio),
    }


# ============================================================
# 6) Two-pass orchestrator
# ============================================================

def run_two_pass(path_3dm: str, cfg: TwoPassConfig) -> Dict[str, Any]:
    """
    Pass 1: preview mesh -> metrics -> (optional) ollama plan
    Pass 2: accurate mesh using plan -> compute exposure (stub now)
    """
    t0 = time.time()
    ollama_error: Optional[str] = None

    # -------- Pass 1: preview mesh
    preview_mesh = mesh_with_params(
        path_3dm,
        max_edge=cfg.preview_max_edge,
        angle_deg=cfg.preview_angle_deg,
        tolerance=cfg.preview_tolerance,
    )
    metrics = compute_preview_metrics(preview_mesh)
    exposure_preview = exposure_preview_stub(
        preview_mesh,
        rays=cfg.exposure_rays,
        samples_per_face=cfg.exposure_samples_per_face,
        threshold_ratio=cfg.exposure_threshold_ratio,
    )

    plan: Optional[Dict[str, Any]] = None
    if cfg.use_ollama:
        try:
            raw_plan = ollama_plan(
                model=cfg.ollama_model,
                metrics={**metrics, "exposure_preview": exposure_preview},
                url=cfg.ollama_url,
                timeout_sec=cfg.ollama_timeout_sec,
            )
            plan = _clamp_plan(raw_plan, metrics, cfg)
        except Exception as e:
            plan = None
            ollama_error = f"{type(e).__name__}: {e}"

    # -------- Decide pass 2 params
    if plan:
        max_edge = float(_safe_get(plan, ("meshing", "max_edge"), cfg.accurate_max_edge))
        angle_deg = float(_safe_get(plan, ("meshing", "angle_deg"), cfg.accurate_angle_deg))
        tolerance = float(_safe_get(plan, ("meshing", "tolerance"), cfg.accurate_tolerance))

        rays = int(_safe_get(plan, ("exposure", "rays"), cfg.exposure_rays))
        samples_per_face = int(_safe_get(plan, ("exposure", "samples_per_face"), cfg.exposure_samples_per_face))
        threshold_ratio = float(_safe_get(plan, ("exposure", "threshold_ratio"), cfg.exposure_threshold_ratio))
    else:
        max_edge = cfg.accurate_max_edge
        angle_deg = cfg.accurate_angle_deg
        tolerance = cfg.accurate_tolerance

        rays = cfg.exposure_rays
        samples_per_face = cfg.exposure_samples_per_face
        threshold_ratio = cfg.exposure_threshold_ratio

    # -------- Pass 2: accurate mesh
    accurate_mesh = mesh_with_params(path_3dm, max_edge=max_edge, angle_deg=angle_deg, tolerance=tolerance)

    exposure_accurate = exposure_preview_stub(
        accurate_mesh, rays=rays, samples_per_face=samples_per_face, threshold_ratio=threshold_ratio
    )

    dt = time.time() - t0

    return {
        "path": path_3dm,
        "preview": {
            "meshing": {
                "max_edge": cfg.preview_max_edge,
                "angle_deg": cfg.preview_angle_deg,
                "tolerance": cfg.preview_tolerance,
            },
            "metrics": metrics,
            "exposure_preview": exposure_preview,
            "nv": len(preview_mesh.vertices_xyz) // 3,
            "nt": len(preview_mesh.faces_tri) // 3,
        },
        "plan": plan,
        "accurate": {
            "meshing": {"max_edge": max_edge, "angle_deg": angle_deg, "tolerance": tolerance},
            "exposure": exposure_accurate,
            "nv": len(accurate_mesh.vertices_xyz) // 3,
            "nt": len(accurate_mesh.faces_tri) // 3,
        },
        "ollama_error": ollama_error,
        "timing_sec": float(dt),
        "brepmesh_module_file": getattr(brepmesh, "__file__", None),
        "bindir": BINDIR,
    }
