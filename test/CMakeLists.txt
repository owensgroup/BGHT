function(add_bght_test target_name source_file)
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES
                          FOLDER "tests")
    target_link_libraries(${target_name} PRIVATE bght::bght)
endfunction()

set(test_targets
    test_1cht
    test_bcht
    test_iht
    test_types
    test_allocator
)

foreach(target ${test_targets})
	add_bght_test(${target} ${target}.cu)
endforeach()


include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)


CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.14.0
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)
add_executable(gtest_map gtest_map.cu)
target_link_libraries(gtest_map PRIVATE bght::bght
								  PRIVATE gtest)

enable_testing()
add_test(gtest_map gtest_map)
