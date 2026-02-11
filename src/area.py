# src/area.py
from __future__ import annotations

import numpy as np


def triangles_area_sum(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute total surface area of a triangle mesh.

    Args:
        vertices: (N,3) float
        faces: (M,3) int

    Returns:
        total area (float)
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (M,3), got {faces.shape}")
    if len(faces) == 0:
        return 0.0

    tri = vertices[faces]  # (M,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    return float(area.sum())


def triangles_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Per-triangle areas.

    Returns:
        (M,) float
    """
    tri = vertices[faces]
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)
