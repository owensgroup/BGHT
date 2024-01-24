include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)


CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.14.0
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)
add_executable(gtest_map gtest_map.cu)
target_link_libraries(gtest_map PRIVATE bght::bght PRIVATE gtest)

enable_testing()
add_test(gtest_map gtest_map)
