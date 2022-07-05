project (test)

add_executable(test_1cht test_1cht.cu)
set_target_properties(test_1cht PROPERTIES
								FOLDER "tests")
target_link_libraries(test_1cht PRIVATE bght
								  PRIVATE bght)


add_executable(test_bcht test_bcht.cu)
set_target_properties(test_bcht PROPERTIES
								FOLDER "tests")
target_link_libraries(test_bcht PRIVATE bght
								  PRIVATE bght)


add_executable(test_iht test_iht.cu)
set_target_properties(test_iht PROPERTIES
								  FOLDER "tests")
target_link_libraries(test_iht PRIVATE bght
									PRIVATE bght)

add_executable(test_types test_types.cu)
set_target_properties(test_types PROPERTIES
								FOLDER "tests")
target_link_libraries(test_types PRIVATE bght
								  PRIVATE bght)


add_executable(test_allocator test_allocator.cu)
set_target_properties(test_allocator PROPERTIES
								FOLDER "tests")
target_link_libraries(test_allocator PRIVATE bght
								  PRIVATE bght)


include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)


CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG release-1.11.0
  VERSION 1.11.0
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)
add_executable(gtest_map gtest_map.cu)
target_link_libraries(gtest_map PRIVATE bght
								  PRIVATE bght
								  PRIVATE gtest)

enable_testing()
add_test(gtest_map gtest_map)
