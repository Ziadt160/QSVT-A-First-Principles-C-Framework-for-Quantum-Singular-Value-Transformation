cmake_minimum_required(VERSION 3.16)
project(qsp_angles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

# Eigen is header-only. Prefer a system install for fast/offline local builds;
# otherwise fetch it so `pip install .` and cibuildwheel work on any machine with
# no system Eigen (this is what makes the wheel portable).
#
# Overrides:
#   * Set -DFETCHCONTENT_SOURCE_DIR_EIGEN3=/path/to/eigen to use a local Eigen
#     checkout and skip the network fetch entirely (offline builds).
#   * A system Eigen found by find_package() is preferred and also skips the
#     fetch.
#
# The FetchContent path requires NETWORK ACCESS (clones Eigen 3.4.0 from GitLab).
find_package(Eigen3 3.3 NO_MODULE QUIET)
if(Eigen3_FOUND)
  message(STATUS "qsp-angles: using system Eigen3 (no network fetch).")
else()
  if(DEFINED FETCHCONTENT_SOURCE_DIR_EIGEN3 AND FETCHCONTENT_SOURCE_DIR_EIGEN3)
    message(STATUS
      "qsp-angles: system Eigen3 not found; using local Eigen from "
      "FETCHCONTENT_SOURCE_DIR_EIGEN3='${FETCHCONTENT_SOURCE_DIR_EIGEN3}' "
      "(no network fetch).")
  else()
    message(STATUS
      "qsp-angles: system Eigen3 not found; fetching Eigen 3.4.0 (header-only) "
      "over the network from gitlab.com/libeigen/eigen. This step REQUIRES "
      "network access. To build offline, install a system Eigen or set "
      "-DFETCHCONTENT_SOURCE_DIR_EIGEN3=/path/to/eigen.")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
endif()

pybind11_add_module(_core src/_core.cpp src/QspAngleSolver.cpp src/SymQspAngleSolver.cpp)
target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE Eigen3::Eigen)

if(NOT MSVC)
  target_compile_options(_core PRIVATE -O3)
endif()

install(TARGETS _core LIBRARY DESTINATION qsp_angles)
