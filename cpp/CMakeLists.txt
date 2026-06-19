# boltz.cpp — C++/Vulkan inference engine for BoltzGen
#
# Hard constraints (see ../CPP_VULKAN_PORT_PLAN.md):
#   * Only third-party dependency is ggml (git submodule, below).
#   * No package manager. Plain CMake primitives only.
#   * Everything else (parsing, featurization, I/O) is hand-rolled C++17 / STL.
#
# Develop/validate on ggml's CPU backend first; flip BOLTZCPP_VULKAN=ON once the
# custom kernels are validated and a Vulkan SDK is present.

cmake_minimum_required(VERSION 3.21)
project(boltzcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BOLTZCPP_VULKAN "Build ggml with the Vulkan backend" OFF)

# --- ggml (the one and only dependency) ---------------------------------------
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(BOLTZCPP_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

set(_ggml_dir ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
if(NOT EXISTS ${_ggml_dir}/CMakeLists.txt)
    message(FATAL_ERROR
        "ggml submodule missing. Run:\n"
        "  git submodule update --init --recursive cpp/third_party/ggml")
endif()
add_subdirectory(${_ggml_dir} EXCLUDE_FROM_ALL)

# --- boltzcpp engine library (hand-rolled) ------------------------------------
# Sources are added incrementally as phases land. For now the library is just the
# front-end/runtime skeleton; the smoke-test exe exercises the ggml wiring.
add_library(boltzcpp INTERFACE)
target_include_directories(boltzcpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(boltzcpp INTERFACE ggml)

# --- smoke test: proves ggml builds + runs on the CPU backend -----------------
add_executable(boltzcpp_smoke src/smoke.cpp)
target_link_libraries(boltzcpp_smoke PRIVATE boltzcpp)
