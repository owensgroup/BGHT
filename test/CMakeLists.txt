include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)


CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.14.0
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)


function(add_bght_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE bght::bght PRIVATE gtest)
endfunction()

set(test_targets
    gtest_map
    iht_tests)

foreach(target ${test_targets})
  add_bght_test(${target} ${target}.cu)
endforeach()
