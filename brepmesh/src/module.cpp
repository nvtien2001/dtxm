#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "read_3dm.h"

namespace py = pybind11;

PYBIND11_MODULE(brepmesh, m)
{
  m.doc() = "brepmesh: read 3dm and extract meshes (cached meshes + safe pipeline)";

  py::class_<MeshResult>(m, "MeshResult")
    .def(py::init<>())
    .def_readwrite("vertices_xyz", &MeshResult::vertices_xyz)
    .def_readwrite("faces_tri", &MeshResult::faces_tri)
    .def_readwrite("ok", &MeshResult::ok)
    .def_readwrite("message", &MeshResult::message);

  m.def("mesh_file_3dm",
        &mesh_file_3dm,
        py::arg("path"),
        py::arg("max_edge") = 0.0,
        py::arg("angle_deg") = 0.0,
        py::arg("tolerance") = 0.0);
}
