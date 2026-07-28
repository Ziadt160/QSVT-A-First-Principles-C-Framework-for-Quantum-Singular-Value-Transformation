# C ABI for qsp-angles: a static + shared library and a pure-C example.
#
#   cmake -S . -B build && cmake --build build
#   ./build/example_c
#
# Downstream C/C++ projects can `find_package`-style consume this, or just link
# the static lib. Rust/Julia/Go bind via the C header (see README.md).
cmake_minimum_required(VERSION 3.16)
project(qsp_angles_c LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# NO THIRD-PARTY DEPENDENCIES -- deliberately.
#
# The C ABI wraps SymQspAngleSolver, which is stdlib-only C++17. There is no
# find_package() and no FetchContent here: this library must build on a machine
# with nothing installed but a C++17 compiler, offline. The Eigen-dependent
# homotopy fallback (QspAngleSolver) is NOT part of the ABI and is not compiled.
#
# Enforced by the `abi-no-third-party-deps` CI job, which compiles these sources
# with no -I flags at all.

set(QSP_SRC
  qsp_angles.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/SymQspAngleSolver.cpp)

# Static and shared flavors of the C ABI.
add_library(qsp_angles_c_static STATIC ${QSP_SRC})
add_library(qsp_angles_c_shared SHARED ${QSP_SRC})
foreach(t qsp_angles_c_static qsp_angles_c_shared)
  target_include_directories(${t} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  set_target_properties(${t} PROPERTIES OUTPUT_NAME qsp_angles_c)
endforeach()

# Pure-C example linked against the static lib (proves the C ABI).
add_executable(example_c example_c.c)
target_link_libraries(example_c PRIVATE qsp_angles_c_static)

enable_testing()
add_test(NAME example_c COMMAND example_c)
