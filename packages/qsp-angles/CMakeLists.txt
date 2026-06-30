cmake_minimum_required(VERSION 3.16)
project(qsp_angles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

# Eigen is header-only. Prefer a system install for fast local builds; otherwise
# fetch it so `pip install .` and cibuildwheel work on any machine with no
# system Eigen (this is what makes the wheel portable).
find_package(Eigen3 3.3 NO_MODULE QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "System Eigen3 not found; fetching Eigen 3.4.0 (header-only).")
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

pybind11_add_module(_core src/_core.cpp src/QspAngleSolver.cpp)
target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE Eigen3::Eigen)

if(NOT MSVC)
  target_compile_options(_core PRIVATE -O3)
endif()

install(TARGETS _core LIBRARY DESTINATION qsp_angles)
