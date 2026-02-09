#pragma once
#include <string>
#include <vector>

struct MeshResult
{
  // vertices packed: [x0,y0,z0, x1,y1,z1, ...]
  std::vector<float> vertices_xyz;

  // triangles packed: [a,b,c, a,b,c, ...]
  std::vector<int> faces_tri;

  bool ok = true;
  std::string message;
};

MeshResult mesh_file_3dm(const std::string& path,
                         double max_edge,
                         double angle_deg,
                         double tolerance);
