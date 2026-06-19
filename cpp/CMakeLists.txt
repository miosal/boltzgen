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
    src/diffusion_schedule.cpp
    src/ggml_nn.cpp
    src/featurizer.cpp
    src/weight_store.cpp
    src/gnn.cpp
    src/gnn_layer.cpp
    src/cif.cpp
    src/ifold_model.cpp
    src/triangle_mult.cpp
    src/attention_pair_bias.cpp
    src/outer_product_mean.cpp
    src/triangle_attention.cpp
    src/pairformer.cpp
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
boltz_add_test(test_diffusion_schedule)
boltz_add_test(test_ggml_nn)
boltz_add_test(test_featurizer)
boltz_add_test(test_weight_store)
boltz_add_test(test_gnn)
boltz_add_test(test_gnn_layer)
boltz_add_test(test_python_gguf)
boltz_add_test(test_cif)
boltz_add_test(test_ifold)
boltz_add_test(test_triangle_mult)
boltz_add_test(test_attention_pair_bias)
boltz_add_test(test_outer_product_mean)
boltz_add_test(test_triangle_attention)
boltz_add_test(test_pairformer)

# Cross-language test needs to know where the python tools live.
target_compile_definitions(test_python_gguf PRIVATE
    BOLTZ_TOOLS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_compile_definitions(test_cif PRIVATE
    BOLTZ_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../example")

# --- inverse-folding CLI driver ----------------------------------------------
add_executable(boltzcpp_ifold src/ifold_main.cpp)
target_link_libraries(boltzcpp_ifold PRIVATE boltzcpp)

# Run the inverse-folding example as a ctest: generate a synthetic model with the
# stdlib python tool, then run the CLI on the real example structure. Skipped if
# python3 is unavailable.
find_program(PYTHON3 python3)
if(PYTHON3)
    set(_synth_gguf ${CMAKE_CURRENT_BINARY_DIR}/ifold_synth.gguf)
    execute_process(
        COMMAND ${PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_synth_ifold_gguf.py ${_synth_gguf}
        RESULT_VARIABLE _gen_rc)
    if(_gen_rc EQUAL 0)
        add_test(NAME ifold_example_1brs
            COMMAND boltzcpp_ifold ${_synth_gguf}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../example/inverse_folding/1brs.cif A)
    endif()
endif()
