project (test)

add_executable(test_bcht test_bcht.cu)
set_target_properties(test_bcht PROPERTIES
								FOLDER "tests")
target_compile_definitions(test_bcht PRIVATE TEMPLATE_DEFINE="${template_define}")
target_link_libraries(test_bcht PRIVATE bght_compiler_flags
								  PRIVATE bght_lib)


add_executable(test_bcht_types test_bcht_types.cu)
set_target_properties(test_bcht_types PROPERTIES
								FOLDER "tests")
target_compile_definitions(test_bcht_types PRIVATE TEMPLATE_DEFINE="${template_define}")
target_link_libraries(test_bcht_types PRIVATE bght_compiler_flags
								  PRIVATE bght_lib)

