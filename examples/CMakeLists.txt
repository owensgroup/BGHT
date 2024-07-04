function(add_bght_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE bght::bght)

    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE HIP)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
endfunction()

set(example_targets
    # array_example
    # custom_allocator_example
    1cht_example
    bcht_example
    # custom_types_example
    iht_example
    hash_join
    histogram
    )

foreach(target ${example_targets})
    add_bght_example(${target} ${target}.cu)
endforeach()