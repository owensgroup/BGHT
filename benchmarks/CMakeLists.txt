project (benchmarks)

function(add_bght_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})
	set_target_properties(${target_name} PROPERTIES FOLDER "benchmarks")
    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE HIP)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
endfunction()

set(test_targets
	probes_per_technique
    rates_per_technique
	rates_per_technique_fixed_lf)

foreach(target ${test_targets})
  add_bght_test(${target} ${target}.cu)
endforeach()


target_compile_definitions(probes_per_technique PRIVATE COUNT_PROBES)
