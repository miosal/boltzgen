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
# Sources are added incrementally as phases land.
add_library(boltzcpp STATIC
    src/residue_constraints.cpp
)
target_include_directories(boltzcpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(boltzcpp PUBLIC ggml)

# --- smoke test: proves ggml builds + runs on the CPU backend -----------------
add_executable(boltzcpp_smoke src/smoke.cpp)
target_link_libraries(boltzcpp_smoke PRIVATE boltzcpp)

# --- unit tests (hand-rolled harness, analogous to the pytest suite) ----------
enable_testing()

function(boltz_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${name} PRIVATE boltzcpp)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

boltz_add_test(test_residue_constraints)
boltz_add_test(test_inverse_fold_constraint_masks)
