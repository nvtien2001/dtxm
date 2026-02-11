# src/exposure.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .area import triangles_area


@dataclass
class ExposureCfg:
    rays: int = 2000
    samples_per_face: int = 1
    threshold_ratio: float = 0.5
    soft_area: bool = True
    max_points: int = 200_000
    seed: int = 0

    # offset để tránh self-hit
    eps_scale: float = 1e-6


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _build_tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build tangent (t) and bitangent (b) for each normal n.
    n: (K,3) unit

    Returns:
        t: (K,3)
        b: (K,3)
    """
    # choose helper axis far from n to avoid degeneracy
    helper = np.zeros_like(n)
    use_z = np.abs(n[:, 2]) < 0.9
    helper[use_z] = np.array([0.0, 0.0, 1.0])
    helper[~use_z] = np.array([0.0, 1.0, 0.0])

    t = np.cross(helper, n)
    t = _unit(t)
    b = np.cross(n, t)
    b = _unit(b)
    return t, b


def _cosine_weighted_hemisphere(rng: np.random.Generator, n: np.ndarray) -> np.ndarray:
    """
    Cosine-weighted hemisphere sampling around normal n.
    n: (K,3) unit
    Returns directions: (K,3) unit
    """
    k = n.shape[0]
    u1 = rng.random(k)
    u2 = rng.random(k)

    r = np.sqrt(u1)
    theta = 2.0 * math.pi * u2

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(np.maximum(0.0, 1.0 - u1))

    # local -> world
    t, b = _build_tangent_basis(n)
    d = (t * x[:, None]) + (b * y[:, None]) + (n * z[:, None])
    return _unit(d)


def _sample_points_on_faces(
    rng: np.random.Generator,
    vertices: np.ndarray,
    faces: np.ndarray,
    samples_per_face: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points on triangles by barycentric sampling.

    Returns:
        points: (P,3)
        face_ids: (P,) indices into faces
    """
    m = faces.shape[0]
    if m == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    areas = triangles_area(vertices, faces)
    total_target = int(m * max(1, samples_per_face))

    # cap points to avoid memory blow-up
    p = min(total_target, int(max_points))
    if p <= 0:
        p = 1

    if p == total_target:
        # sample each face samples_per_face times (may still be big but already <= max_points)
        face_ids = np.repeat(np.arange(m, dtype=np.int64), samples_per_face)
    else:
        # area-weighted face sampling
        prob = areas / max(float(areas.sum()), 1e-12)
        face_ids = rng.choice(m, size=p, replace=True, p=prob)

    # barycentric sampling
    tri = vertices[faces[face_ids]]  # (P,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]

    u = rng.random(len(face_ids))
    v = rng.random(len(face_ids))

    # Turk 1990: reflect if u+v>1
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    pts = v0 + (v1 - v0) * u[:, None] + (v2 - v0) * v[:, None]
    return pts.astype(np.float64, copy=False), face_ids.astype(np.int64, copy=False)


def compute_exposed_area(
    vertices: np.ndarray,
    faces: np.ndarray,
    total_area: float,
    cfg: ExposureCfg,
) -> Dict[str, Any]:
    """
    Exposed area estimation by Monte Carlo ray casting.
    - Sample points on surface (area-weighted).
    - For each point, cast rays to hemisphere (cosine-weighted) along face normal.
    - exposed_ratio = average(miss_ratio).
      miss_ratio(point) = fraction of rays that DO NOT hit the mesh (unoccluded).

    Notes:
    - "soft_area=True": use continuous ratio
    - "soft_area=False": binary per-point: miss_ratio >= threshold_ratio => exposed
    """
    if total_area <= 0.0:
        return {
            "exposed_ratio": 0.0,
            "exposed_area": 0.0,
            "points": 0,
            "rays": int(cfg.rays),
            "soft_area": bool(cfg.soft_area),
            "threshold_ratio": float(cfg.threshold_ratio),
            "accel": "none",
            "note": "total_area<=0",
        }

    try:
        import trimesh
        from trimesh.ray.ray_triangle import RayMeshIntersector as TriIntersector
    except Exception as e:
        raise RuntimeError(
            "Thiếu dependency trimesh. Cài: pip install trimesh"
        ) from e

    # build mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # choose intersector (prefer embree if available via mesh.ray)
    accel = "auto"
    try:
        # trimesh will use embree if installed in some envs
        intersector = mesh.ray
        _ = intersector.intersects_any(np.array([[0, 0, 0.0]]), np.array([[1, 0, 0.0]]))
    except Exception:
        # fallback to pure triangle intersector
        intersector = TriIntersector(mesh)
        accel = "triangle"

    rng = np.random.default_rng(int(cfg.seed))

    # sample points
    points, face_ids = _sample_points_on_faces(
        rng=rng,
        vertices=vertices,
        faces=faces,
        samples_per_face=int(cfg.samples_per_face),
        max_points=int(cfg.max_points),
    )

    pcount = points.shape[0]
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

    # face normals
    tri = vertices[faces[face_ids]]  # (P,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    n = _unit(n)

    # epsilon offset
    bbox = mesh.bounds
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    eps = max(diag * float(cfg.eps_scale), 1e-9)
    origins0 = points + n * eps

    # cast rays in chunks to control memory
    rays_per_point = max(1, int(cfg.rays))
    chunk_points = 2000  # tune: bigger=less overhead, but more RAM
    miss_ratio = np.zeros((pcount,), dtype=np.float64)

    for i0 in range(0, pcount, chunk_points):
        i1 = min(pcount, i0 + chunk_points)
        o = origins0[i0:i1]
        nn = n[i0:i1]
        k = o.shape[0]

        # build all rays for this chunk
        # directions: (k*rays,3)
        # origins repeat: (k*rays,3)
        # Do in loops over rays for stability RAM: accumulate hit counts
        hit_counts = np.zeros((k,), dtype=np.int32)

        # loop rays, but vectorized per ray batch
        for _ in range(rays_per_point):
            d = _cosine_weighted_hemisphere(rng, nn)  # (k,3)
            # check occlusion
            try:
                hits = intersector.intersects_any(o, d)
            except Exception:
                # some trimesh ray implementations want contiguous arrays
                hits = intersector.intersects_any(np.ascontiguousarray(o), np.ascontiguousarray(d))
            hit_counts += hits.astype(np.int32)

        # miss = not hit
        miss = 1.0 - (hit_counts.astype(np.float64) / float(rays_per_point))
        miss_ratio[i0:i1] = miss

    if bool(cfg.soft_area):
        exposed_ratio = float(np.mean(miss_ratio))
    else:
        thr = float(cfg.threshold_ratio)
        exposed_ratio = float(np.mean((miss_ratio >= thr).astype(np.float64)))

    exposed_ratio = max(0.0, min(1.0, exposed_ratio))
    exposed_area = float(total_area * exposed_ratio)

    return {
        "exposed_ratio": exposed_ratio,
        "exposed_area": exposed_area,
        "points": int(pcount),
        "rays": int(cfg.rays),
        "soft_area": bool(cfg.soft_area),
        "threshold_ratio": float(cfg.threshold_ratio),
        "accel": accel,
        "eps": eps,
        "bbox_diag": diag,
        "note": "monte-carlo hemisphere ray casting",
    }
