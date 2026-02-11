# src/pipeline.py
from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .area import triangles_area_sum
from .exposure import ExposureCfg, compute_exposed_area

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
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_timeout_sec: int = 120

    # Exposure
    exposure_rays: int = 2000
    exposure_samples_per_face: int = 1
    exposure_threshold_ratio: float = 0.50
    exposure_soft_area: bool = True
    exposure_max_points: int = 200_000
    exposure_seed: int = 0

    # DEBUG
    debug_print_mesh_fingerprint: bool = True
    debug_extreme_params_test: bool = False  # bật True để test cache mesher


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
    fmin = min(r.faces_tri) if r.faces_tri else 0
    fmax = max(r.faces_tri) if r.faces_tri else -1
    if fmin < 0 or fmax >= nv:
        raise ValueError(f"Mesh has out-of-range indices: min={fmin}, max={fmax}, nv={nv}")

    # Optional sampling check (avoid O(nt) full scan)
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


def _mesh_to_numpy(mesh_result) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MeshResult packed arrays -> numpy arrays:
    - vertices: (N,3) float64
    - faces: (M,3) int64
    """
    v = np.asarray(mesh_result.vertices_xyz, dtype=np.float64)
    f = np.asarray(mesh_result.faces_tri, dtype=np.int64)
    vertices = v.reshape((-1, 3))
    faces = f.reshape((-1, 3))
    return vertices, faces


# ============================================================
# 2.5) Debug fingerprint (để bắt cache/same-mesh)
# ============================================================

def mesh_fingerprint(m) -> Dict[str, Any]:
    v = m.vertices_xyz
    f = m.faces_tri
    nv = len(v) // 3
    nt = len(f) // 3

    def _sum_slice(arr, a, b):
        if not arr:
            return 0.0
        a = max(0, a)
        b = min(len(arr), b)
        return float(sum(arr[a:b]))

    return {
        "nv": int(nv),
        "nt": int(nt),
        "v_sum_head": _sum_slice(v, 0, min(3000, len(v))),  # ~1000 verts
        "v_sum_tail": _sum_slice(v, max(0, len(v) - 3000), len(v)),
        "f_sum_head": _sum_slice(f, 0, min(3000, len(f))),  # ~1000 tris
        "f_sum_tail": _sum_slice(f, max(0, len(f) - 3000), len(f)),
    }


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
    diag = ((bb["max"][0] - bb["min"][0]) ** 2 +
            (bb["max"][1] - bb["min"][1]) ** 2 +
            (bb["max"][2] - bb["min"][2]) ** 2) ** 0.5

    return {
        "nv": int(nv),
        "nt": int(nt),
        "bbox": bb,
        "bbox_diag": float(diag),
        "preview": {"note": "metrics are from preview mesh"},
    }


def compute_surface_metrics(mesh_result) -> Dict[str, Any]:
    """Compute surface metrics from mesh: total_area."""
    vertices, faces = _mesh_to_numpy(mesh_result)
    total_area = triangles_area_sum(vertices, faces)
    return {"total_area": float(total_area)}


def compute_exposure_metrics(
    mesh_result,
    total_area: float,
    cfg: TwoPassConfig,
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute exposed area using ray casting (offline).
    override keys:
      rays, samples_per_face, threshold_ratio, soft_area, max_points, seed
    """
    o = override or {}
    exp_cfg = ExposureCfg(
        rays=int(o.get("rays", cfg.exposure_rays)),
        samples_per_face=int(o.get("samples_per_face", cfg.exposure_samples_per_face)),
        threshold_ratio=float(o.get("threshold_ratio", cfg.exposure_threshold_ratio)),
        soft_area=bool(o.get("soft_area", cfg.exposure_soft_area)),
        max_points=int(o.get("max_points", cfg.exposure_max_points)),
        seed=int(o.get("seed", cfg.exposure_seed)),
    )

    vertices, faces = _mesh_to_numpy(mesh_result)
    return compute_exposed_area(vertices, faces, total_area=total_area, cfg=exp_cfg)


# ============================================================
# 4) Ollama planner (local HTTP)
# ============================================================

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return s


def ollama_plan(model: str, metrics: Dict[str, Any], url: str, timeout_sec: int = 120) -> Dict[str, Any]:
    """
    Ask Ollama to return a STRICT JSON plan.
    Requires: pip install requests
    """
    import requests  # local import

    system = (
        "Bạn là kỹ sư CAD/mesh. CHỈ trả về 1 JSON object đúng schema, KHÔNG giải thích.\n"
        "Schema:\n"
        "{\n"
        "  \"mode\": \"preview\" | \"accurate\",\n"
        "  \"meshing\": {\"max_edge\": number, \"angle_deg\": number, \"tolerance\": number},\n"
        "  \"repair\": {\"remove_degenerate\": bool, \"fix_normals\": bool},\n"
        "  \"exposure\": {\"rays\": int, \"samples_per_face\": int, \"threshold_ratio\": number, \"soft_area\": bool}\n"
        "}\n"
        "Ràng buộc:\n"
        "- max_edge > 0; tolerance > 0; angle_deg trong [5..45].\n"
        "- samples_per_face nên nhỏ (1..4) nếu nt lớn.\n"
        "- rays hợp lý theo độ phức tạp.\n"
        "Nếu không chắc, chọn conservative.\n"
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

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # max_edge: [diag/5000 .. diag/20]
    min_edge = max(diag / 5000.0, 0.005)
    max_edge = max(diag / 20.0, min_edge)

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
    soft = bool(ex.get("soft_area", cfg.exposure_soft_area))

    rays = int(clamp(rays, 200, 200000))
    spf = int(clamp(spf, 1, 8))
    thr = clamp(thr, 0.05, 0.95)

    plan["exposure"] = {
        "rays": rays,
        "samples_per_face": spf,
        "threshold_ratio": thr,
        "soft_area": soft,
    }

    rep = plan.get("repair", {}) if isinstance(plan.get("repair", {}), dict) else {}
    plan["repair"] = {
        "remove_degenerate": bool(rep.get("remove_degenerate", True)),
        "fix_normals": bool(rep.get("fix_normals", False)),
    }

    if plan.get("mode") not in ("preview", "accurate"):
        plan["mode"] = "accurate"

    return plan


# ============================================================
# 5) Two-pass orchestrator
# ============================================================

def run_two_pass(path_3dm: str, cfg: TwoPassConfig) -> Dict[str, Any]:
    """
    Pass 1: preview mesh -> metrics -> preview area/exposure -> (optional) ollama plan
    Pass 2: accurate mesh using plan -> total_area + exposed_area
    """
    t0 = time.time()
    ollama_error: Optional[str] = None

    # -------- Pass 1: preview mesh
    print("[PASS1 preview] meshing params:",
          cfg.preview_max_edge, cfg.preview_angle_deg, cfg.preview_tolerance, flush=True)

    preview_mesh = mesh_with_params(
        path_3dm,
        max_edge=cfg.preview_max_edge,
        angle_deg=cfg.preview_angle_deg,
        tolerance=cfg.preview_tolerance,
    )

    if cfg.debug_print_mesh_fingerprint:
        fp1 = mesh_fingerprint(preview_mesh)
        print("[PASS1 preview] fingerprint:", fp1, flush=True)
    else:
        fp1 = None

    metrics = compute_preview_metrics(preview_mesh)
    preview_surface = compute_surface_metrics(preview_mesh)

    # preview exposure: dùng ít rays/points hơn để nhanh
    preview_exposure = compute_exposure_metrics(
        preview_mesh,
        total_area=float(preview_surface["total_area"]),
        cfg=cfg,
        override={
            "rays": int(min(512, cfg.exposure_rays)),
            "samples_per_face": int(min(2, cfg.exposure_samples_per_face)),
            "threshold_ratio": float(cfg.exposure_threshold_ratio),
            "soft_area": bool(cfg.exposure_soft_area),
            "max_points": int(min(50_000, cfg.exposure_max_points)),
            "seed": int(cfg.exposure_seed),
        },
    )

    # optional: extreme params test for cache detection
    if cfg.debug_extreme_params_test:
        print("\n[DEBUG] Force extreme params test (detect cache/ignore-params)", flush=True)
        mA = mesh_with_params(path_3dm, max_edge=10.0, angle_deg=45.0, tolerance=0.5)
        mB = mesh_with_params(path_3dm, max_edge=0.2, angle_deg=5.0, tolerance=0.01)
        fpA = mesh_fingerprint(mA)
        fpB = mesh_fingerprint(mB)
        print("[DEBUG] fpA:", fpA, flush=True)
        print("[DEBUG] fpB:", fpB, flush=True)
        print("[DEBUG] A==B ?", fpA == fpB, flush=True)
        print("[DEBUG] Done extreme test\n", flush=True)

    plan: Optional[Dict[str, Any]] = None
    if cfg.use_ollama:
        try:
            raw_plan = ollama_plan(
                model=cfg.ollama_model,
                metrics={**metrics, "preview_surface": preview_surface, "preview_exposure": preview_exposure},
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
        soft_area = bool(_safe_get(plan, ("exposure", "soft_area"), cfg.exposure_soft_area))
    else:
        max_edge = cfg.accurate_max_edge
        angle_deg = cfg.accurate_angle_deg
        tolerance = cfg.accurate_tolerance

        rays = cfg.exposure_rays
        samples_per_face = cfg.exposure_samples_per_face
        threshold_ratio = cfg.exposure_threshold_ratio
        soft_area = cfg.exposure_soft_area

    print("[PASS2 accurate] meshing params:", max_edge, angle_deg, tolerance, flush=True)

    # -------- Pass 2: accurate mesh
    accurate_mesh = mesh_with_params(path_3dm, max_edge=max_edge, angle_deg=angle_deg, tolerance=tolerance)

    if cfg.debug_print_mesh_fingerprint:
        fp2 = mesh_fingerprint(accurate_mesh)
        print("[PASS2 accurate] fingerprint:", fp2, flush=True)
        if fp1 is not None:
            print("[COMPARE] same_nv_nt=",
                  (fp1["nv"] == fp2["nv"] and fp1["nt"] == fp2["nt"]),
                  "same_sums=",
                  (fp1["v_sum_head"] == fp2["v_sum_head"] and fp1["v_sum_tail"] == fp2["v_sum_tail"] and
                   fp1["f_sum_head"] == fp2["f_sum_head"] and fp1["f_sum_tail"] == fp2["f_sum_tail"]),
                  flush=True)

    accurate_surface = compute_surface_metrics(accurate_mesh)
    accurate_exposure = compute_exposure_metrics(
        accurate_mesh,
        total_area=float(accurate_surface["total_area"]),
        cfg=cfg,
        override={
            "rays": int(rays),
            "samples_per_face": int(samples_per_face),
            "threshold_ratio": float(threshold_ratio),
            "soft_area": bool(soft_area),
            "max_points": int(cfg.exposure_max_points),
            "seed": int(cfg.exposure_seed),
        },
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
            "surface": preview_surface,
            "exposure": preview_exposure,
            "nv": len(preview_mesh.vertices_xyz) // 3,
            "nt": len(preview_mesh.faces_tri) // 3,
        },
        "plan": plan,
        "accurate": {
            "meshing": {"max_edge": max_edge, "angle_deg": angle_deg, "tolerance": tolerance},
            "surface": accurate_surface,
            "exposure": accurate_exposure,
            "nv": len(accurate_mesh.vertices_xyz) // 3,
            "nt": len(accurate_mesh.faces_tri) // 3,
        },
        "ollama_error": ollama_error,
        "timing_sec": float(dt),
        "brepmesh_module_file": getattr(brepmesh, "__file__", None),
        "bindir": BINDIR,
    }
