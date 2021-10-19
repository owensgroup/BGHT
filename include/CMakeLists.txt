add_library(bght_lib INTERFACE)

file(GLOB_RECURSE   bght_src
                    "*.h"
                    "*.hpp"
                    "*.cuh"
                    "*.cu")
set(SOURCE_LIST ${bght_src})
target_sources(bght_lib INTERFACE ${bght_src})
target_include_directories(bght_lib INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")