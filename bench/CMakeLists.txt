# Benchmark harness executable. Uses only the pure-math parts of qsvt_lib
# (decomposition + QSP angle finding); it does not drive the simulator.
add_executable(qsvt_bench bench.cpp)
target_link_libraries(qsvt_bench PRIVATE qsvt_lib)
