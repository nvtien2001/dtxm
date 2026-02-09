#pragma once

#include "read_3dm.h"
#include "opennurbs.h"

// Tạo mesh cho 1 Brep và append vào MeshResult
// Nếu openNURBS của bạn không có mesher runtime, hàm sẽ cố lấy cached mesh trước.
void append_brep_mesh(const ON_Brep& brep,
                      double max_edge,
                      double angle_deg,
                      double tolerance,
                      MeshResult& out);
