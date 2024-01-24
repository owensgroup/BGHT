function(add_tree_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE bght::bght)
endfunction()

set(example_targets
    array
    )

foreach(target ${example_targets})
    add_tree_example(${target} ${target}.cu)
endforeach()