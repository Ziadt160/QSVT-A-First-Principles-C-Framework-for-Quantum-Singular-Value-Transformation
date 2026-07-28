cmake_minimum_required(VERSION 3.16)
project(qsp_angles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

# The default build is STDLIB-ONLY C++17 (plus pybind11): no third-party C++
# dependency, no find_package, no network fetch. That is what makes the wheel
# build offline on a bare machine -- and it is the same property that lets the
# C ABI be embedded in a compiled stack (see capi/).
#
# The homotopy-continuation fallback (src/QspAngleSolver.cpp) is the one piece
# that needs Eigen, so it is OPT-IN:
#
#   cmake -DQSP_ANGLES_WITH_HOMOTOPY=ON ...      # adds `method="homotopy"`
#   pip install -C cmake.define.QSP_ANGLES_WITH_HOMOTOPY=ON .
#
# With it ON, a system Eigen is preferred; failing that Eigen 3.4.0 is fetched
# from GitLab, which REQUIRES NETWORK ACCESS. Set
# -DFETCHCONTENT_SOURCE_DIR_EIGEN3=/path/to/eigen to use a local checkout.
#
# The default solver (sym_qsp) is both faster and higher-degree than the
# fallback, so turning this off costs nothing in normal use.
option(QSP_ANGLES_WITH_HOMOTOPY
       "Build the Eigen-dependent homotopy-continuation fallback solver" OFF)

set(QSP_CORE_SOURCES src/_core.cpp src/SymQspAngleSolver.cpp)

if(QSP_ANGLES_WITH_HOMOTOPY)
  list(APPEND QSP_CORE_SOURCES src/QspAngleSolver.cpp)

  find_package(Eigen3 3.3 NO_MODULE QUIET)
  if(Eigen3_FOUND)
    message(STATUS "qsp-angles: homotopy ON, using system Eigen3 (no fetch).")
  else()
    if(DEFINED FETCHCONTENT_SOURCE_DIR_EIGEN3 AND FETCHCONTENT_SOURCE_DIR_EIGEN3)
      message(STATUS
        "qsp-angles: homotopy ON, system Eigen3 not found; using local Eigen "
        "from FETCHCONTENT_SOURCE_DIR_EIGEN3="
        "'${FETCHCONTENT_SOURCE_DIR_EIGEN3}' (no network fetch).")
    else()
      message(STATUS
        "qsp-angles: homotopy ON, system Eigen3 not found; fetching Eigen "
        "3.4.0 over the network from gitlab.com/libeigen/eigen. This REQUIRES "
        "network access. Build with -DQSP_ANGLES_WITH_HOMOTOPY=OFF (the "
        "default) for a dependency-free offline build.")
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
else()
  message(STATUS
    "qsp-angles: stdlib-only build (no Eigen, no network). Pass "
    "-DQSP_ANGLES_WITH_HOMOTOPY=ON to add the homotopy fallback solver.")
endif()

pybind11_add_module(_core ${QSP_CORE_SOURCES})
target_include_directories(_core PRIVATE src)

if(QSP_ANGLES_WITH_HOMOTOPY)
  target_compile_definitions(_core PRIVATE QSP_ANGLES_WITH_HOMOTOPY)
  target_link_libraries(_core PRIVATE Eigen3::Eigen)
endif()

if(NOT MSVC)
  target_compile_options(_core PRIVATE -O3)
endif()

install(TARGETS _core LIBRARY DESTINATION qsp_angles)
