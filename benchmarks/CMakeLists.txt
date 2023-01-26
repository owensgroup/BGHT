project (benchmarks)

add_executable(probes_per_technique probes_per_technique.hip)
set_target_properties(probes_per_technique PROPERTIES
								FOLDER "benchmarks"
								LINKER_LANGUAGE HIP)
target_compile_definitions(probes_per_technique PRIVATE COUNT_PROBES)
target_link_libraries(probes_per_technique PRIVATE bght
								  PRIVATE bght)


add_executable(rates_per_technique rates_per_technique.hip)
set_target_properties(rates_per_technique PROPERTIES
								FOLDER "benchmarks"
								LINKER_LANGUAGE HIP)
target_link_libraries(rates_per_technique PRIVATE bght
								  PRIVATE bght)

add_executable(rates_per_technique_fixed_lf rates_per_technique_fixed_lf.hip)
set_target_properties(rates_per_technique_fixed_lf PROPERTIES
								FOLDER "benchmarks"
								LINKER_LANGUAGE HIP)
target_link_libraries(rates_per_technique_fixed_lf PRIVATE bght
								  PRIVATE bght)