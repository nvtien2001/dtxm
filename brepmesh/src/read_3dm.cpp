#include "read_3dm.h"
#include "mesh_brep.h"

#include "opennurbs.h"
#include "opennurbs_extensions.h"

#include <cstdio>

static void append_mesh(const ON_Mesh& m, MeshResult& out)
{
  const int v0 = (int)(out.vertices_xyz.size() / 3);

  const ON_3fPointArray& V = m.m_V;
  out.vertices_xyz.reserve(out.vertices_xyz.size() + (size_t)V.Count() * 3);
  for (int i = 0; i < V.Count(); ++i)
  {
    out.vertices_xyz.push_back(V[i].x);
    out.vertices_xyz.push_back(V[i].y);
    out.vertices_xyz.push_back(V[i].z);
  }

  const ON_SimpleArray<ON_MeshFace>& F = m.m_F;
  out.faces_tri.reserve(out.faces_tri.size() + (size_t)F.Count() * 6);
  for (int fi = 0; fi < F.Count(); ++fi)
  {
    const ON_MeshFace& f = F[fi];
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

static FILE* open_3dm(const std::string& path)
{
  return ON::OpenFile(path.c_str(), "rb");
}

MeshResult mesh_file_3dm(const std::string& path,
                         double max_edge,
                         double angle_deg,
                         double tolerance)
{
  MeshResult out;

  FILE* fp = open_3dm(path);
  if (!fp)
  {
    out.ok = false;
    out.message = "Cannot open file";
    return out;
  }

  ON_BinaryFile archive(ON::archive_mode::read3dm, fp);
  ONX_Model model;

  if (!model.IncrementalReadBegin(archive, true, 0, nullptr))
  {
    out.ok = false;
    out.message = "IncrementalReadBegin failed";
    ON::CloseFile(fp);
    return out;
  }

  for (;;)
  {
    ON_ModelComponentReference mg_ref;

    const bool ok = model.IncrementalReadModelGeometry(
      archive,
      true,  // manage component
      true,  // manage geometry
      true,  // manage attributes
      0,     // filter all
      mg_ref
    );

    if (!ok) break;
    if (mg_ref.IsEmpty()) break;

    const ON_ModelComponent* comp = mg_ref.ModelComponent();
    const ON_ModelGeometryComponent* mgc = ON_ModelGeometryComponent::Cast(comp);
    if (!mgc) continue;

    const ON_Geometry* geo = mgc->Geometry(nullptr);
    if (!geo) continue;

    // Mesh có sẵn trong file
    if (const ON_Mesh* mesh = ON_Mesh::Cast(geo))
    {
      append_mesh(*mesh, out);
      continue;
    }

    // Brep
    if (const ON_Brep* brep = ON_Brep::Cast(geo))
    {
      // append_brep_mesh sẽ cố lấy render/analysis mesh (cached) hoặc báo không có
      append_brep_mesh(*brep, max_edge, angle_deg, tolerance, out);
      continue;
    }

    // Extrusion -> Brep
    if (const ON_Extrusion* ext = ON_Extrusion::Cast(geo))
    {
      ON_Brep brep_ext;
      if (ext->BrepForm(&brep_ext))
        append_brep_mesh(brep_ext, max_edge, angle_deg, tolerance, out);
      continue;
    }

    // SubD: để sau (tuỳ version API openNURBS)
  }

  model.IncrementalReadFinish(archive, true, 0, nullptr);
  ON::CloseFile(fp);

  if (out.vertices_xyz.empty() || out.faces_tri.empty())
  {
    if (out.message.empty())
      out.message = "No mesh data produced (file may contain Brep without cached mesh).";
  }

  return out;
}
