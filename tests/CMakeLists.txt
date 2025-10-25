# This command tells CMake to download GoogleTest from GitHub
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG    v1.14.0  # <-- This was the line I fixed
)

# This command makes the GTest targets (like GTest::gtest_main) available
# It will only download if it's not already present
FetchContent_MakeAvailable(googletest)

# Now, we create our tests just like before
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(qsvt_tests ${TEST_SOURCES})

# We can link to GTest::gtest_main because FetchContent created it
target_link_libraries(qsvt_tests PRIVATE 
    qsvt_lib 
    GTest::gtest_main
    Qrack::Qrack
)

include(GoogleTest)
gtest_discover_tests(qsvt_tests)