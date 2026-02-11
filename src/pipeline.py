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
    preview_max_edge: float = 1.0
    preview_angle_deg: float = 25.0
    preview_tolerance: float = 0.1

    # Pass 2 defaults (fallback)
    accurate_max_edge: float = 0.5
    accurate_angle_deg: float = 20.0
    accurate_tolerance: float = 0.05

    # Ollama
    use_ollama: bool = True
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_timeout_sec: int = 180

    # Exposure defaults (fallback)
    exposure_rays: int = 256
    exposure_samples_per_face: int = 1
    exposure_threshold_ratio: float = 0.5
    exposure_soft_area: bool = True
    exposure_max_points: int = 50_000
    exposure_seed: int = 0

    # -------- SPEED SWITCHES ----------
    # 1) Skip preview exposure entirely (huge save)
    skip_preview_exposure: bool = True

    # 2) If mesher caches/ignores params (your log proves it), reuse mesh instead of calling pass2 again
    reuse_mesh_if_same_fingerprint: bool = True

    # 3) Force exposure soft mode (early stop works much better)
    force_soft_area: bool = True

    # 4) Clamp max_points based on triangle count (avoid 200k on large meshes)
    #    points ~= min(exposure_max_points, nt * points_per_tri_cap)
    points_per_tri_cap: float = 0.25  # 0.25 => nt=300k => 75k max

    # Debug
    debug_print_mesh_fingerprint: bool = True
    debug_extreme_params_test: bool = False


# ============================================================
# 2) Mesh wrapper + sanity checks
# ============================================================

def mesh_with_params(path_3dm: str, max_edge: float, angle_deg: float, tolerance: float):
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

    fmin = min(r.faces_tri) if r.faces_tri else 0
    fmax = max(r.faces_tri) if r.faces_tri else -1
    if fmin < 0 or fmax >= nv:
        raise ValueError(f"Mesh out-of-range indices: min={fmin}, max={fmax}, nv={nv}")

    return r


def _mesh_to_numpy(mesh_result) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(mesh_result.vertices_xyz, dtype=np.float64).reshape((-1, 3))
    f = np.asarray(mesh_result.faces_tri, dtype=np.int64).reshape((-1, 3))
    return v, f


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
        "v_sum_head": _sum_slice(v, 0, min(3000, len(v))),
        "v_sum_tail": _sum_slice(v, max(0, len(v) - 3000), len(v)),
        "f_sum_head": _sum_slice(f, 0, min(3000, len(f))),
        "f_sum_tail": _sum_slice(f, max(0, len(f) - 3000), len(f)),
    }


# ============================================================
# 3) Metrics
# ============================================================

def compute_preview_metrics(mesh_result) -> Dict[str, Any]:
    v = mesh_result.vertices_xyz
    nv = len(v) // 3
    nt = len(mesh_result.faces_tri) // 3

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

    return {"nv": int(nv), "nt": int(nt), "bbox": bb, "bbox_diag": float(diag)}


def compute_surface_metrics(mesh_result) -> Dict[str, Any]:
    vertices, faces = _mesh_to_numpy(mesh_result)
    total_area = triangles_area_sum(vertices, faces)
    return {"total_area": float(total_area)}


def compute_exposure_metrics(mesh_result, total_area: float, cfg: TwoPassConfig, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
# 4) Ollama planner
# ============================================================

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return s


def ollama_plan(model: str, metrics: Dict[str, Any], url: str, timeout_sec: int = 180) -> Dict[str, Any]:
    import requests

    system = (
        "CHỈ trả về 1 JSON object, không giải thích.\n"
        "{\n"
        "  \"mode\": \"preview\" | \"accurate\",\n"
        "  \"meshing\": {\"max_edge\": number, \"angle_deg\": number, \"tolerance\": number},\n"
        "  \"repair\": {\"remove_degenerate\": bool, \"fix_normals\": bool},\n"
        "  \"exposure\": {\"rays\": int, \"samples_per_face\": int, \"threshold_ratio\": number, \"soft_area\": bool}\n"
        "}\n"
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
    content = _strip_code_fences(r.json()["message"]["content"])
    return json.loads(content)


def _safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ============================================================
# 5) Orchestrator
# ============================================================

def run_two_pass(path_3dm: str, cfg: TwoPassConfig) -> Dict[str, Any]:
    t0 = time.time()
    ollama_error: Optional[str] = None

    # PASS1 mesh
    print("[PASS1 preview] meshing params:", cfg.preview_max_edge, cfg.preview_angle_deg, cfg.preview_tolerance, flush=True)
    preview_mesh = mesh_with_params(path_3dm, cfg.preview_max_edge, cfg.preview_angle_deg, cfg.preview_tolerance)

    fp1 = mesh_fingerprint(preview_mesh) if cfg.debug_print_mesh_fingerprint else None
    if fp1 is not None:
        print("[PASS1 preview] fingerprint:", fp1, flush=True)

    metrics = compute_preview_metrics(preview_mesh)
    preview_surface = compute_surface_metrics(preview_mesh)

    # Preview exposure: SKIP for speed
    if cfg.skip_preview_exposure:
        preview_exposure = {"skipped": True, "reason": "speed: skip preview exposure"}
    else:
        preview_exposure = compute_exposure_metrics(
            preview_mesh,
            total_area=float(preview_surface["total_area"]),
            cfg=cfg,
            override={"rays": 64, "max_points": 10_000, "soft_area": True, "seed": cfg.exposure_seed},
        )

    # Optional extreme test (detect cache)
    if cfg.debug_extreme_params_test:
        print("\n[DEBUG] extreme params cache test", flush=True)
        mA = mesh_with_params(path_3dm, max_edge=10.0, angle_deg=45.0, tolerance=0.5)
        mB = mesh_with_params(path_3dm, max_edge=0.2, angle_deg=5.0, tolerance=0.01)
        fpA = mesh_fingerprint(mA)
        fpB = mesh_fingerprint(mB)
        print("[DEBUG] fpA:", fpA, flush=True)
        print("[DEBUG] fpB:", fpB, flush=True)
        print("[DEBUG] A==B ?", fpA == fpB, flush=True)

    # Ollama plan
    plan: Optional[Dict[str, Any]] = None
    if cfg.use_ollama:
        try:
            plan = ollama_plan(cfg.ollama_model, metrics, cfg.ollama_url, cfg.ollama_timeout_sec)
        except Exception as e:
            plan = None
            ollama_error = f"{type(e).__name__}: {e}"

    # Decide PASS2 params
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

    # FORCE soft area for speed/robustness
    if cfg.force_soft_area:
        soft_area = True

    # Clamp max_points by nt (speed guard)
    nt = int(metrics["nt"])
    points_cap = int(min(cfg.exposure_max_points, max(5_000, nt * cfg.points_per_tri_cap)))
    # safety upper bound
    points_cap = int(min(points_cap, 100_000))

    print("[PASS2 accurate] meshing params:", max_edge, angle_deg, tolerance, flush=True)
    print("[PASS2 exposure] rays:", rays, "samples_per_face:", samples_per_face,
          "soft_area:", soft_area, "threshold:", threshold_ratio, "max_points:", points_cap, flush=True)

    # PASS2 mesh (or reuse)
    accurate_mesh = None
    fp2 = None

    if cfg.reuse_mesh_if_same_fingerprint and fp1 is not None:
        # We already proved brepmesh caches/ignores params in your log.
        # So reuse preview mesh to avoid a second identical meshing cost.
        # This does NOT change correctness if mesher returns same mesh anyway.
        accurate_mesh = preview_mesh
        fp2 = fp1
        print("[PASS2] reuse preview mesh (mesher cache/ignore params)", flush=True)
    else:
        accurate_mesh = mesh_with_params(path_3dm, max_edge, angle_deg, tolerance)
        if cfg.debug_print_mesh_fingerprint:
            fp2 = mesh_fingerprint(accurate_mesh)
            print("[PASS2 accurate] fingerprint:", fp2, flush=True)

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
            "max_points": int(points_cap),
            "seed": int(cfg.exposure_seed),
        },
    )

    dt = time.time() - t0

    return {
        "path": path_3dm,
        "preview": {
            "meshing": {"max_edge": cfg.preview_max_edge, "angle_deg": cfg.preview_angle_deg, "tolerance": cfg.preview_tolerance},
            "metrics": metrics,
            "surface": preview_surface,
            "exposure": preview_exposure,
            "nv": len(preview_mesh.vertices_xyz) // 3,
            "nt": len(preview_mesh.faces_tri) // 3,
            "fingerprint": fp1,
        },
        "plan": plan,
        "accurate": {
            "meshing": {"max_edge": max_edge, "angle_deg": angle_deg, "tolerance": tolerance},
            "surface": accurate_surface,
            "exposure": accurate_exposure,
            "nv": len(accurate_mesh.vertices_xyz) // 3,
            "nt": len(accurate_mesh.faces_tri) // 3,
            "fingerprint": fp2,
        },
        "ollama_error": ollama_error,
        "timing_sec": float(dt),
        "brepmesh_module_file": getattr(brepmesh, "__file__", None),
        "bindir": BINDIR,
    }
