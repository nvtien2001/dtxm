#pragma once
#include <opennurbs.h>
#include "read_3dm.h"

struct RuntimeMeshConfig
{
  // Đơn vị tuỳ theo model (thường là mm trong trang sức)
  double max_edge   = 0.2;    // target edge length (coarse)
  double tolerance  = 0.01;   // chưa dùng nhiều trong Phase 1
  double angle_deg  = 20.0;   // chưa dùng trong Phase 1

  int    uv_min_div = 8;      // tối thiểu subdivisions mỗi chiều
  int    uv_max_div = 512;    // tránh nổ triangles

  bool   enable_trim = false; // Phase 2 (chưa implement)
};

bool mesh_brep_runtime(const ON_Brep& brep,
                       const RuntimeMeshConfig& cfg,
                       MeshResult& out);
