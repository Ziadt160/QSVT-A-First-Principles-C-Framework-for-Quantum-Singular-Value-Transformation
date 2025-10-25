## src/CMakeLists.txt - builds library and executable

# collect sources; exclude main.cpp from library
file(GLOB_RECURSE QSVT_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# remove main.cpp from library sources if present
list(FILTER QSVT_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

add_library(qsvt_lib ${QSVT_SOURCES})
target_include_directories(qsvt_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qsvt_lib PUBLIC Eigen3::Eigen)

# Link Qrack if available
if(TARGET Qrack::Qrack)
    target_link_libraries(qsvt_lib PUBLIC Qrack::Qrack)
else()
    if(QRACK_LIBRARIES)
        target_link_libraries(qsvt_lib PUBLIC ${QRACK_LIBRARIES})
    endif()
    if(QRACK_INCLUDE_DIRS)
        target_include_directories(qsvt_lib PUBLIC ${QRACK_INCLUDE_DIRS})
    endif()
endif()

# executable
set(MAIN_FILE "")
if(EXISTS ${CMAKE_SOURCE_DIR}/main.cpp)
    set(MAIN_FILE ${CMAKE_SOURCE_DIR}/main.cpp)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    set(MAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
endif()

if(MAIN_FILE)
    add_executable(qsvt_app ${MAIN_FILE})
    
    target_link_libraries(qsvt_app PRIVATE qsvt_lib)
    # allow defining runtime choice for CUDA interface
    target_compile_definitions(qsvt_app PRIVATE $<$<BOOL:${USE_CUDA}>:USE_QRACK_CUDA>)
else()
    message(WARNING "main.cpp not found; qsvt_app will not be created. Place main.cpp in project root or src/ to build the app.")
endif()
