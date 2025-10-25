cmake_minimum_required(VERSION 3.16)
project(QSVT_Project VERSION 0.1 LANGUAGES CXX)

include(FetchContent) # <-- ADDED THIS LINE

option(USE_CUDA "Enable CUDA-enabled Qrack interface at runtime/build" OFF)
option(BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Basic warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find CUDA libraries
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit")
else()
    message(WARNING "CUDA Toolkit not found. Linking will likely fail.")
endif()

# Try to find Qrack (create cmake/FindQrack.cmake if needed)
find_package(Qrack CONFIG QUIET)
if(Qrack_FOUND)
    message(STATUS "Found Qrack via find_package")
else()
    message(STATUS "Qrack not found by CMake. Provide cmake/FindQrack.cmake or set QRACK_INCLUDE_DIRS/QRACK_LIBRARIES.")
    if(NOT DEFINED QRACK_INCLUDE_DIRS)
        set(QRACK_INCLUDE_DIRS "" CACHE PATH "Qrack include directory")
    endif()
    if(NOT DEFINED QRACK_LIBRARIES)
        set(QRACK_LIBRARIES "" CACHE STRING "Qrack libraries")
    endif()
endif()

# If find_package didn't find Qrack, try common locations for headers/libs and create an imported target
if(NOT Qrack_FOUND)

    # Force the find commands to run again by clearing any cached -NOTFOUND
    unset(QRACK_INCLUDE_DIRS CACHE)
    unset(QRACK_LIBRARIES CACHE)

    find_path(QRACK_INCLUDE_DIRS
        NAMES qrack/hamiltonian.hpp  # Look for a file we know exists
        PATHS /usr/local/include     # Search in the main include path
              /usr/include
              /opt/local/include
    )
    
    find_library(QRACK_LIBRARIES
        NAMES qrack libqrack
        PATHS /usr/local/lib/qrack /usr/local/lib /usr/lib /opt/local/lib
        NO_DEFAULT_PATH
    )

    # Now, check if the find commands *actually* succeeded
    if(QRACK_INCLUDE_DIRS AND QRACK_LIBRARIES)
        message(STATUS "Found Qrack headers: ${QRACK_INCLUDE_DIRS}")
        message(STATUS "Found Qrack library: ${QRACK_LIBRARIES}")
        
        add_library(Qrack::Qrack UNKNOWN IMPORTED)
        set_target_properties(Qrack::Qrack PROPERTIES
            IMPORTED_LOCATION "${QRACK_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${QRACK_INCLUDE_DIRS}")
        
        # ADD THIS CHECK: If CUDA was found, make it a dependency of Qrack
        if(CUDAToolkit_FOUND)
            set_target_properties(Qrack::Qrack PROPERTIES
                INTERFACE_LINK_LIBRARIES "CUDA::cudart"
            )
        endif()

        set(Qrack_FOUND TRUE)
    else()
        message(STATUS "Qrack not found in common locations. You may set QRACK_INCLUDE_DIRS and QRACK_LIBRARIES manually.")
        # Print debug info on failure
        message(STATUS "Debug: QRACK_INCLUDE_DIRS = ${QRACK_INCLUDE_DIRS}")
        message(STATUS "Debug: QRACK_LIBRARIES = ${QRACK_LIBRARIES}")
    endif()
endif()

# Propagate CUDA flag as preprocessor definition
if(USE_CUDA)
    add_compile_definitions(HAVE_QRACK_CUDA)
    message(STATUS "Building with CUDA support enabled (HAVE_QRACK_CUDA defined). Ensure Qrack was built with CUDA.")
endif()

# Project layout
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install rules (minimal)
install(TARGETS qsvt_app qsvt_lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)