file(GLOB_RECURSE   bght_src
                    "*.h"
                    "*.hpp"
                    "*.cuh"
                    "*.cu")
set(SOURCE_LIST ${bght_src})
target_sources(bght INTERFACE ${bght_src})
target_include_directories(bght INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")