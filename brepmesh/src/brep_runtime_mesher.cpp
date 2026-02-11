#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "brep_runtime_mesher.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Tránh macro max/min bằng trick (std::max)(a,b) nếu cần,
// nhưng ta đã undef ở trên nên bình thường cũng ok.

static int clamp_i(int v, int lo, int hi)
{
  return (std::max)(lo, (std::min)(v, hi));
}

static void add_vertex(MeshResult& out, const ON_3dPoint& p)
{
  out.vertices_xyz.push_back((float)p.x);
  out.vertices_xyz.push_back((float)p.y);
  out.vertices_xyz.push_back((float)p.z);
}

static void add_tri(MeshResult& out, int a, int b, int c)
{
  if (a == b || b == c || a == c) return;
  out.faces_tri.push_back(a);
  out.faces_tri.push_back(b);
  out.faces_tri.push_back(c);
}

static bool eval_point(const ON_Surface* s, double u, double v, ON_3dPoint& p)
{
  if (!s) return false;
  return s->EvPoint(u, v, p) ? true : false;
}

static void estimate_uv_lengths(const ON_Surface* s,
                                const ON_Interval& du,
                                const ON_Interval& dv,
                                double& out_len_u,
                                double& out_len_v)
{
  out_len_u = 0.0;
  out_len_v = 0.0;
  if (!s) return;

  const double u0 = du.Min();
  const double u1 = du.Max();
  const double v0 = dv.Min();
  const double v1 = dv.Max();

  const double vm = 0.5 * (v0 + v1);
  ON_3dPoint pu0, pu1;
  if (eval_point(s, u0, vm, pu0) && eval_point(s, u1, vm, pu1))
    out_len_u = pu0.DistanceTo(pu1);

  const double um = 0.5 * (u0 + u1);
  ON_3dPoint pv0, pv1;
  if (eval_point(s, um, v0, pv0) && eval_point(s, um, v1, pv1))
    out_len_v = pv0.DistanceTo(pv1);

  // fallback nếu singular/invalid
  if (out_len_u <= 1e-12 || out_len_v <= 1e-12)
  {
    ON_BoundingBox bb;
    bb.Destroy();

    for (int iu = 0; iu <= 2; ++iu)
    for (int iv = 0; iv <= 2; ++iv)
    {
      const double u = u0 + (u1 - u0) * (iu / 2.0);
      const double v = v0 + (v1 - v0) * (iv / 2.0);
      ON_3dPoint p;
      if (eval_point(s, u, v, p))
        bb.Set(p, true);
    }

    if (bb.IsValid())
    {
      const ON_3dVector d = bb.Diagonal();
      const double diag = d.Length();
      out_len_u = (std::max)(out_len_u, diag * 0.5);
      out_len_v = (std::max)(out_len_v, diag * 0.5);
    }
  }
}

static bool mesh_face_uv_grid(const ON_BrepFace& face,
                              const RuntimeMeshConfig& cfg,
                              MeshResult& out)
{
  const ON_Surface* s = face.SurfaceOf();
  if (!s) return false;

  ON_Interval du = face.Domain(0);
  ON_Interval dv = face.Domain(1);
  if (!du.IsIncreasing() || !dv.IsIncreasing())
    return false;

  double len_u = 0.0, len_v = 0.0;
  estimate_uv_lengths(s, du, dv, len_u, len_v);

  const double max_edge = (cfg.max_edge > 0.0) ? cfg.max_edge : 0.2;

  int nu = (len_u > 1e-9) ? (int)std::ceil(len_u / max_edge) : cfg.uv_min_div;
  int nv = (len_v > 1e-9) ? (int)std::ceil(len_v / max_edge) : cfg.uv_min_div;

  nu = clamp_i(nu, cfg.uv_min_div, cfg.uv_max_div);
  nv = clamp_i(nv, cfg.uv_min_div, cfg.uv_max_div);

  const int grid_w = nu + 1;
  const int grid_h = nv + 1;

  std::vector<int> vidx((size_t)grid_w * (size_t)grid_h, -1);

  const double u0 = du.Min();
  const double u1 = du.Max();
  const double v0 = dv.Min();
  const double v1 = dv.Max();

  const int base_v = (int)(out.vertices_xyz.size() / 3);

  int created = 0;
  for (int j = 0; j < grid_h; ++j)
  {
    const double tv = (nv == 0) ? 0.0 : (double)j / (double)nv;
    const double v = v0 + (v1 - v0) * tv;

    for (int i = 0; i < grid_w; ++i)
    {
      const double tu = (nu == 0) ? 0.0 : (double)i / (double)nu;
      const double u = u0 + (u1 - u0) * tu;

      ON_3dPoint p;
      if (!eval_point(s, u, v, p))
        continue;

      const int idx = base_v + created;
      add_vertex(out, p);
      vidx[(size_t)j * (size_t)grid_w + (size_t)i] = idx;
      created++;
    }
  }

  int tris_added = 0;
  for (int j = 0; j < nv; ++j)
  {
    for (int i = 0; i < nu; ++i)
    {
      const int a = vidx[(size_t)j * (size_t)grid_w + (size_t)i];
      const int b = vidx[(size_t)j * (size_t)grid_w + (size_t)(i + 1)];
      const int c = vidx[(size_t)(j + 1) * (size_t)grid_w + (size_t)(i + 1)];
      const int d = vidx[(size_t)(j + 1) * (size_t)grid_w + (size_t)i];

      if (a < 0 || b < 0 || c < 0 || d < 0) continue;

      add_tri(out, a, b, c);
      add_tri(out, a, c, d);
      tris_added += 2;
    }
  }

  return tris_added > 0;
}

bool mesh_brep_runtime(const ON_Brep& brep,
                       const RuntimeMeshConfig& cfg,
                       MeshResult& out)
{
  bool any = false;

  const int face_count = brep.m_F.Count();
  for (int fi = 0; fi < face_count; ++fi)
  {
    const ON_BrepFace& face = brep.m_F[fi];
    if (mesh_face_uv_grid(face, cfg, out))
      any = true;
  }

  return any;
}
