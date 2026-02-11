#include "mesh_brep.h"
#include "brep_runtime_mesher.h"

#include <opennurbs.h>

static void append_mesh(const ON_Mesh &m, MeshResult &out)
{
  const int v0 = (int)(out.vertices_xyz.size() / 3);

  const ON_3fPointArray &V = m.m_V;
  out.vertices_xyz.reserve(out.vertices_xyz.size() + (size_t)V.Count() * 3);
  for (int i = 0; i < V.Count(); ++i)
  {
    out.vertices_xyz.push_back(V[i].x);
    out.vertices_xyz.push_back(V[i].y);
    out.vertices_xyz.push_back(V[i].z);
  }

  const ON_SimpleArray<ON_MeshFace> &F = m.m_F;
  out.faces_tri.reserve(out.faces_tri.size() + (size_t)F.Count() * 6);

  for (int fi = 0; fi < F.Count(); ++fi)
  {
    const ON_MeshFace &f = F[fi];
    const int a = v0 + f.vi[0];
    const int b = v0 + f.vi[1];
    const int c = v0 + f.vi[2];
    const int d = v0 + f.vi[3];

    if (f.vi[2] == f.vi[3])
      out.faces_tri.insert(out.faces_tri.end(), {a, b, c});
    else
      out.faces_tri.insert(out.faces_tri.end(), {a, b, c, a, c, d});
  }
}

static void append_cached_brep_mesh(const ON_Brep &brep, ON::mesh_type mt, MeshResult &out)
{
  ON_SimpleArray<const ON_Mesh *> meshes;
  const int rc = brep.GetMesh(mt, meshes);
  if (rc <= 0)
    return;

  for (int i = 0; i < meshes.Count(); ++i)
  {
    if (meshes[i])
      append_mesh(*meshes[i], out);
  }
}

void append_brep_mesh(const ON_Brep &brep,
                      double max_edge,
                      double angle_deg,
                      double tolerance,
                      MeshResult &out)
{
  const size_t before = out.faces_tri.size();

  // 1) Ưu tiên cached mesh (nếu file có sẵn render/analysis mesh)
  append_cached_brep_mesh(brep, ON::mesh_type::render_mesh, out);
  append_cached_brep_mesh(brep, ON::mesh_type::analysis_mesh, out);

  if (out.faces_tri.size() != before)
  {
    out.ok = true;
    return;
  }

  // 2) Runtime meshing (Option 2) - không dùng ON_Brep::CreateMesh (vì openNURBS free không link được)
  RuntimeMeshConfig cfg;
  cfg.max_edge = (max_edge > 0.0) ? max_edge : cfg.max_edge;
  cfg.angle_deg = angle_deg;
  cfg.tolerance = tolerance;
  cfg.enable_trim = false; // Phase 1: coarse

  const bool ok = mesh_brep_runtime(brep, cfg, out);
  if (ok)
  {
    out.ok = true;
    return;
  }

  out.ok = false;
  if (out.message.empty())
    out.message = "No cached brep mesh and runtime meshing failed.";
}
