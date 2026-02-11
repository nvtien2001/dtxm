#include "read_3dm.h"
#include "mesh_brep.h"

#include "opennurbs.h"
#include "opennurbs_extensions.h"

#include <cstdio>
#include <unordered_set>
#include <cstring>
#include <cstddef>
#include <cstdint>

// -------------------- UUID hash/equal --------------------

struct UuidHasher
{
  std::size_t operator()(const ON_UUID& u) const noexcept
  {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&u);
    std::size_t h = static_cast<std::size_t>(1469598103934665603ull);
    for (int i = 0; i < 16; ++i)
    {
      h ^= static_cast<std::size_t>(p[i]);
      h *= static_cast<std::size_t>(1099511628211ull);
    }
    return h;
  }
};

struct UuidEq
{
  bool operator()(const ON_UUID& a, const ON_UUID& b) const noexcept
  {
    return 0 == std::memcmp(&a, &b, 16);
  }
};

// -------------------- Mesh append helpers --------------------

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

static void append_mesh_xformed(const ON_Mesh& m, const ON_Xform& xf, MeshResult& out)
{
  ON_Mesh tmp(m);
  tmp.Transform(xf);
  append_mesh(tmp, out);
}

// -------------------- 3dm open --------------------

static FILE* open_3dm(const std::string& path)
{
  return ON::OpenFile(path.c_str(), "rb");
}

// -------------------- InstanceRef expansion --------------------

static bool append_instance_ref_geometry(
  const ONX_Model& model,
  const ON_InstanceRef& iref,
  const ON_Xform& parent_xf,
  double max_edge,
  double angle_deg,
  double tolerance,
  MeshResult& out,
  std::unordered_set<ON_UUID, UuidHasher, UuidEq>& visiting_idef,
  int depth
)
{
  if (depth > 32) return false;

  if (ON_UuidIsNil(iref.m_instance_definition_uuid))
    return false;

  const ON_UUID idef_id = iref.m_instance_definition_uuid;

  if (visiting_idef.find(idef_id) != visiting_idef.end())
    return false;

  visiting_idef.insert(idef_id);

  const ON_Xform xf = parent_xf * iref.m_xform;

  ON_ModelComponentReference idef_ref =
    model.ComponentFromId(ON_ModelComponent::Type::InstanceDefinition, idef_id);

  const ON_InstanceDefinition* idef =
    ON_InstanceDefinition::FromModelComponentRef(idef_ref, nullptr);

  if (!idef)
  {
    visiting_idef.erase(idef_id);
    return false;
  }

  const ON_SimpleArray<ON_UUID>& geom_ids = idef->InstanceGeometryIdList();

  for (unsigned int j = 0; j < geom_ids.UnsignedCount(); ++j)
  {
    ON_ModelComponentReference gref =
      model.ComponentFromId(ON_ModelComponent::Type::ModelGeometry, geom_ids[j]);

    const ON_ModelGeometryComponent* mgc =
      ON_ModelGeometryComponent::Cast(gref.ModelComponent());
    if (!mgc) continue;

    const ON_Geometry* geo = mgc->Geometry(nullptr);
    if (!geo) continue;

    if (const ON_InstanceRef* nested = ON_InstanceRef::Cast(geo))
    {
      append_instance_ref_geometry(model, *nested, xf,
                                  max_edge, angle_deg, tolerance,
                                  out, visiting_idef, depth + 1);
      continue;
    }

    if (const ON_Mesh* mesh = ON_Mesh::Cast(geo))
    {
      append_mesh_xformed(*mesh, xf, out);
      continue;
    }

    if (const ON_Brep* brep = ON_Brep::Cast(geo))
    {
      ON_Brep b(*brep);
      b.Transform(xf);
      append_brep_mesh(b, max_edge, angle_deg, tolerance, out);
      continue;
    }

    if (const ON_Extrusion* ext = ON_Extrusion::Cast(geo))
    {
      ON_Brep b;
      if (ext->BrepForm(&b))
      {
        b.Transform(xf);
        append_brep_mesh(b, max_edge, angle_deg, tolerance, out);
      }
      continue;
    }
  }

  visiting_idef.erase(idef_id);
  return true;
}

// -------------------- Main API --------------------

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

    if (const ON_InstanceRef* iref = ON_InstanceRef::Cast(geo))
    {
      std::unordered_set<ON_UUID, UuidHasher, UuidEq> visiting;
      ON_Xform I(1.0);
      append_instance_ref_geometry(model, *iref, I,
                                   max_edge, angle_deg, tolerance,
                                   out, visiting, 0);
      continue;
    }

    if (const ON_Mesh* mesh = ON_Mesh::Cast(geo))
    {
      append_mesh(*mesh, out);
      continue;
    }

    if (const ON_Brep* brep = ON_Brep::Cast(geo))
    {
      append_brep_mesh(*brep, max_edge, angle_deg, tolerance, out);
      continue;
    }

    if (const ON_Extrusion* ext = ON_Extrusion::Cast(geo))
    {
      ON_Brep brep_ext;
      if (ext->BrepForm(&brep_ext))
        append_brep_mesh(brep_ext, max_edge, angle_deg, tolerance, out);
      continue;
    }
  }

  model.IncrementalReadFinish(archive, true, 0, nullptr);
  ON::CloseFile(fp);

  if (out.vertices_xyz.empty() || out.faces_tri.empty())
  {
    if (out.message.empty())
      out.message = "No mesh data produced (Brep-only needs runtime mesher; or unresolved instances).";
  }

  return out;
}
