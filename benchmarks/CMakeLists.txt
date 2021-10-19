project (benchmarks)


#template_project bench
add_executable(benchmark_template_project benchmark_template_project.cu)
set_target_properties(benchmark_template_project PROPERTIES
								FOLDER "benchmarks")
# Defines
target_compile_definitions(benchmark_template_project PRIVATE TEMPLATE_DEFINE="${template_define}")
target_link_libraries(benchmark_template_project PRIVATE template_project_compiler_flags
								  PRIVATE template_project_lib)

