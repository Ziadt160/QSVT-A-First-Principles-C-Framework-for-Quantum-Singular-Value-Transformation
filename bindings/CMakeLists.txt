# Python bindings (optional). Requires Python development headers
# (e.g. `python3-dev`) and pybind11 (`pip install pybind11`).
#
# Enable with: cmake -S . -B build -DBUILD_PYTHON=ON

# Locate a pip-installed pybind11 if the system package isn't present.
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_Interpreter_FOUND AND NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_pybind11_cmakedir)
        set(pybind11_DIR "${_pybind11_cmakedir}")
    endif()
endif()

find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(qsvt_native qsvt_py.cpp)
    target_link_libraries(qsvt_native PRIVATE qsvt_lib)
    message(STATUS "Python bindings enabled (module: qsvt_native).")
else()
    message(WARNING
        "BUILD_PYTHON=ON but pybind11 was not found. "
        "Install it with `pip install pybind11` (and python3-dev), or see "
        "bindings/README.md for a standalone build command.")
endif()
