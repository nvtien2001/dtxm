# src/exposure.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .area import triangles_area


@dataclass
class ExposureCfg:
    rays: int = 256
    samples_per_face: int = 1
    threshold_ratio: float = 0.5
    soft_area: bool = True
    max_points: int = 50_000
    seed: int = 0
    eps_scale: float = 1e-6

    # perf
    chunk_points: int = 2000
    batch_rays: int = 64
    adaptive_stages: Tuple[int, ...] = (32, 128, 256)

    soft_stop_low: float = 0.03
    soft_stop_high: float = 0.97

    hard_stop_margin: float = 0.0
    return_stats: bool = True

    # debug
    debug_print_intersector: bool = True


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _build_tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    helper = np.zeros_like(n)
    use_z = np.abs(n[:, 2]) < 0.9
    helper[use_z] = np.array([0.0, 0.0, 1.0])
    helper[~use_z] = np.array([0.0, 1.0, 0.0])

    t = np.cross(helper, n)
    t = _unit(t)
    b = np.cross(n, t)
    b = _unit(b)
    return t, b


def _cosine_weighted_hemisphere_dirs(rng: np.random.Generator, n: np.ndarray, batch: int) -> np.ndarray:
    k = n.shape[0]
    u1 = rng.random((k, batch))
    u2 = rng.random((k, batch))

    r = np.sqrt(u1)
    theta = 2.0 * math.pi * u2

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(np.maximum(0.0, 1.0 - u1))

    t, b = _build_tangent_basis(n)
    dirs = (t[:, None, :] * x[:, :, None]) + (b[:, None, :] * y[:, :, None]) + (n[:, None, :] * z[:, :, None])
    return _unit(dirs)


def _sample_points_on_faces(
    rng: np.random.Generator,
    vertices: np.ndarray,
    faces: np.ndarray,
    samples_per_face: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    m = faces.shape[0]
    if m == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    areas = triangles_area(vertices, faces)
    total_target = int(m * max(1, samples_per_face))
    p = min(total_target, int(max_points))
    p = max(p, 1)

    if p == total_target:
        face_ids = np.repeat(np.arange(m, dtype=np.int64), samples_per_face)
    else:
        prob = areas / max(float(areas.sum()), 1e-12)
        face_ids = rng.choice(m, size=p, replace=True, p=prob)

    tri = vertices[faces[face_ids]]
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]

    u = rng.random(len(face_ids))
    v = rng.random(len(face_ids))
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    pts = v0 + (v1 - v0) * u[:, None] + (v2 - v0) * v[:, None]
    return pts.astype(np.float64, copy=False), face_ids.astype(np.int64, copy=False)


def _pick_intersector(mesh) -> Tuple[Any, str]:
    try:
        intersector = mesh.ray
        _ = intersector.intersects_any(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        )
        return intersector, "auto"
    except Exception:
        from trimesh.ray.ray_triangle import RayMeshIntersector as TriIntersector
        return TriIntersector(mesh), "triangle"


def _adaptive_stages(rays_max: int, stages: Tuple[int, ...]) -> Tuple[int, ...]:
    s = [int(x) for x in stages if int(x) > 0]
    s = sorted(set(s))
    s = [x for x in s if x <= rays_max]
    if not s or s[-1] != rays_max:
        s.append(rays_max)
    return tuple(s)


def compute_exposed_area(vertices: np.ndarray, faces: np.ndarray, total_area: float, cfg: ExposureCfg) -> Dict[str, Any]:
    if total_area <= 0.0 or faces.size == 0 or vertices.size == 0:
        return {
            "exposed_ratio": 0.0,
            "exposed_area": 0.0,
            "points": 0,
            "rays": int(cfg.rays),
            "soft_area": bool(cfg.soft_area),
            "threshold_ratio": float(cfg.threshold_ratio),
            "accel": "none",
            "note": "empty mesh or total_area<=0",
        }

    try:
        import trimesh
    except Exception as e:
        raise RuntimeError("Thiếu dependency trimesh. Cài: pip install trimesh") from e

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    intersector, accel = _pick_intersector(mesh)

    if cfg.debug_print_intersector:
        print("[RAY] intersector:", type(intersector), "accel:", accel, flush=True)

    rng = np.random.default_rng(int(cfg.seed))

    points, face_ids = _sample_points_on_faces(
        rng=rng,
        vertices=vertices,
        faces=faces,
        samples_per_face=int(cfg.samples_per_face),
        max_points=int(cfg.max_points),
    )
    pcount = int(points.shape[0])
    if pcount == 0:
        return {
            "exposed_ratio": 0.0,
            "exposed_area": 0.0,
            "points": 0,
            "rays": int(cfg.rays),
            "soft_area": bool(cfg.soft_area),
            "threshold_ratio": float(cfg.threshold_ratio),
            "accel": accel,
            "note": "no sample points",
        }

    tri = vertices[faces[face_ids]]
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    n = _unit(np.cross(v1 - v0, v2 - v0))

    bbox = mesh.bounds
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    eps = max(diag * float(cfg.eps_scale), 1e-9)
    origins0 = points + n * eps

    rays_max = max(1, int(cfg.rays))
    stages = _adaptive_stages(rays_max, cfg.adaptive_stages)
    batch_rays = max(1, int(cfg.batch_rays))
    chunk_points = max(1, int(cfg.chunk_points))

    miss_ratio = np.zeros((pcount,), dtype=np.float64)
    rays_used = np.zeros((pcount,), dtype=np.int32)

    for i0 in range(0, pcount, chunk_points):
        i1 = min(pcount, i0 + chunk_points)
        o = origins0[i0:i1]
        nn = n[i0:i1]
        k = o.shape[0]

        hit_counts = np.zeros((k,), dtype=np.int32)
        used = np.zeros((k,), dtype=np.int32)
        active = np.ones((k,), dtype=bool)

        prev_stage = 0
        for stage in stages:
            need = stage - prev_stage
            prev_stage = stage
            if need <= 0:
                continue

            remaining = need
            while remaining > 0:
                b = min(batch_rays, remaining)
                remaining -= b

                idx = np.nonzero(active)[0]
                if idx.size == 0:
                    break

                o_act = o[idx]
                n_act = nn[idx]

                dirs = _cosine_weighted_hemisphere_dirs(rng, n_act, b)
                dirs_flat = dirs.reshape((-1, 3))
                origins_flat = np.repeat(o_act, repeats=b, axis=0)

                try:
                    hits_flat = intersector.intersects_any(origins_flat, dirs_flat)
                except Exception:
                    hits_flat = intersector.intersects_any(
                        np.ascontiguousarray(origins_flat),
                        np.ascontiguousarray(dirs_flat),
                    )

                hits = hits_flat.reshape((-1, b))
                hit_counts[idx] += hits.sum(axis=1).astype(np.int32)
                used[idx] += b

                miss = 1.0 - (hit_counts.astype(np.float64) / np.maximum(used.astype(np.float64), 1.0))

                if cfg.soft_area:
                    done = (miss <= float(cfg.soft_stop_low)) | (miss >= float(cfg.soft_stop_high))
                    active[done] = False
                else:
                    thr = float(cfg.threshold_ratio)
                    miss_count = used - hit_counts
                    remaining_possible = rays_max - used
                    best = (miss_count + remaining_possible) / np.maximum((used + remaining_possible), 1.0)
                    worst = (miss_count + 0) / np.maximum((used + remaining_possible), 1.0)
                    margin = float(cfg.hard_stop_margin)
                    done = (best < (thr - margin)) | (worst >= (thr + margin))
                    active[done] = False

            if not active.any():
                break

        miss = 1.0 - (hit_counts.astype(np.float64) / np.maximum(used.astype(np.float64), 1.0))
        miss_ratio[i0:i1] = miss
        rays_used[i0:i1] = used

    if cfg.soft_area:
        exposed_ratio = float(np.mean(miss_ratio))
    else:
        thr = float(cfg.threshold_ratio)
        exposed_ratio = float(np.mean((miss_ratio >= thr).astype(np.float64)))

    exposed_ratio = max(0.0, min(1.0, exposed_ratio))
    exposed_area = float(total_area * exposed_ratio)

    out = {
        "exposed_ratio": exposed_ratio,
        "exposed_area": exposed_area,
        "points": int(pcount),
        "rays": int(cfg.rays),
        "soft_area": bool(cfg.soft_area),
        "threshold_ratio": float(cfg.threshold_ratio),
        "accel": accel,
        "eps": float(eps),
        "bbox_diag": float(diag),
        "note": "monte-carlo hemisphere ray casting (batch+adaptive)",
    }

    if cfg.return_stats:
        out["avg_rays_used"] = float(np.mean(rays_used))
        out["min_rays_used"] = int(np.min(rays_used))
        out["max_rays_used"] = int(np.max(rays_used))
        out["early_stop_ratio"] = float(np.mean(rays_used < int(cfg.rays)))

    return out
