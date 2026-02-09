cmake_minimum_required(VERSION 3.20)
project(brepmesh LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) Find Python (đúng venv)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# 2) pybind11 (vendored)
add_subdirectory(third_party/pybind11)

# -------------------------
# ZLIB (from opennurbs/zlib)
# -------------------------
set(ZLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opennurbs/zlib")

file(GLOB ZLIB_SRC
  "${ZLIB_DIR}/*.c"
)

# đảm bảo có compress.c (bạn đã nói cần thêm)
# file(GLOB) đã lấy *.c nên compress.c sẽ vào.

add_library(zlibstatic STATIC ${ZLIB_SRC})
target_include_directories(zlibstatic PUBLIC "${ZLIB_DIR}")

# Copy zlib.lib vào đúng nơi openNURBS đang #pragma link tới
add_custom_command(TARGET zlibstatic POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:zlibstatic>
          "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opennurbs/zlib.lib"
)

if(MSVC)
  target_compile_definitions(zlibstatic PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(zlibstatic PRIVATE /W0)
endif()

# -------------------------
# openNURBS
# -------------------------
# Chỉ lấy *.cpp ở ROOT third_party/opennurbs (không recurse => không build examples/)
file(GLOB OPENNURBS_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opennurbs/*.cpp"
)

# loại bỏ precompiled header impl (không cần)
list(FILTER OPENNURBS_SRC EXCLUDE REGEX "opennurbs_precompiledheader\\.cpp$")

# Apple-only sources
if(WIN32)
  list(FILTER OPENNURBS_SRC EXCLUDE REGEX "opennurbs_apple_.*\\.cpp$")
endif()

add_library(opennurbs STATIC ${OPENNURBS_SRC})
target_include_directories(opennurbs PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opennurbs"
  "${ZLIB_DIR}"
)

target_compile_definitions(opennurbs PRIVATE
  ON_COMPILING_OPENNURBS
  UNICODE
  _UNICODE
  NOMINMAX
  # required by opennurbs_zlib.cpp
  OPENNURBS_INPUT_LIBS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/third_party/opennurbs\"
)

if(MSVC)
  target_compile_definitions(opennurbs PRIVATE
    _CRT_SECURE_NO_WARNINGS
    WIN32
    _WINDOWS
  )
  target_compile_options(opennurbs PRIVATE /W0)
endif()

target_link_libraries(opennurbs PRIVATE zlibstatic)

# Windows: PathFileExists / PathIsDirectory nằm trong Shlwapi
if(WIN32)
  target_link_libraries(opennurbs PRIVATE Shlwapi)
endif()

# -------------------------
# pybind module
# -------------------------
pybind11_add_module(brepmesh
  src/module.cpp
  src/read_3dm.cpp
  src/mesh_brep.cpp
)

target_include_directories(brepmesh PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opennurbs"
  "${ZLIB_DIR}"
)

target_link_libraries(brepmesh PRIVATE
  opennurbs
  Python::Module
)

set_target_properties(brepmesh PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
)
