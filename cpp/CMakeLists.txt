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
    src/ifold_real.cpp
    src/triangle_mult.cpp
    src/attention_pair_bias.cpp
    src/outer_product_mean.cpp
    src/triangle_attention.cpp
    src/pairformer.cpp
    src/mmcif_writer.cpp
    src/trunk.cpp
    src/diffusion_blocks.cpp
    src/diffusion_transformer.cpp
    src/diffusion_sampler.cpp
    src/atom_windowing.cpp
    src/atom_transformer.cpp
    src/rel_pos.cpp
    src/atom_featurizer.cpp
    src/input_embedder.cpp
    src/smiles.cpp
    src/conformer.cpp
    src/npy.cpp
    src/confidence.cpp
    src/affinity.cpp
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
boltz_add_test(test_mmcif_writer)
boltz_add_test(test_trunk)
boltz_add_test(test_diffusion_blocks)
boltz_add_test(test_diffusion_transformer)
boltz_add_test(test_diffusion_sampler)
boltz_add_test(test_diffusion_pipeline)
boltz_add_test(test_atom_windowing)
boltz_add_test(test_atom_transformer)
boltz_add_test(test_rel_pos)
boltz_add_test(test_atom_featurizer)
boltz_add_test(test_input_embedder)
boltz_add_test(test_smiles)
boltz_add_test(test_conformer)
boltz_add_test(test_npy)
boltz_add_test(test_confidence)
boltz_add_test(test_affinity)
boltz_add_test(test_parity)

# Numeric parity vs the PyTorch reference on REAL trained weights. The gguf +
# golden fixtures are derived from a HuggingFace-gated checkpoint and are not
# committed; when absent the test skips. Regenerate into the build dir with:
#   python3 tools/convert_ckpt_to_gguf.py <ckpt> ${BUILD}/boltzgen1_ifold.gguf --arch boltzgen-ifold
#   python3 tools/dump_golden.py <ckpt> ${BUILD}/golden_ifold
target_compile_definitions(test_parity PRIVATE
    BOLTZ_PARITY_GGUF="${CMAKE_CURRENT_BINARY_DIR}/boltzgen1_ifold.gguf"
    BOLTZ_PARITY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ifold")

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
                    ${CMAKE_CURRENT_SOURCE_DIR}/../example/inverse_folding/1brs.cif A
                    --out ${CMAKE_CURRENT_BINARY_DIR}/ifold_out.cif)
    endif()
endif()

# --- folding-style demonstration (trunk + diffusion + confidence) ------------
add_executable(boltzcpp_fold src/fold_demo.cpp)
target_link_libraries(boltzcpp_fold PRIVATE boltzcpp)
add_test(NAME fold_example_1brs
    COMMAND boltzcpp_fold ${CMAKE_CURRENT_SOURCE_DIR}/../example/inverse_folding/1brs.cif A
            --out ${CMAKE_CURRENT_BINARY_DIR}/fold_out.cif)

# --- design-style demonstration (real featurizers + trunk + diffusion) -------
add_executable(boltzcpp_design src/design_demo.cpp)
target_link_libraries(boltzcpp_design PRIVATE boltzcpp)
add_test(NAME design_example_1brs
    COMMAND boltzcpp_design ${CMAKE_CURRENT_SOURCE_DIR}/../example/inverse_folding/1brs.cif A
            --out ${CMAKE_CURRENT_BINARY_DIR}/design_out.cif)

# --- ligand featurization demo (SMILES -> conformer -> mmCIF) -----------------
add_executable(boltzcpp_ligand src/ligand_demo.cpp)
target_link_libraries(boltzcpp_ligand PRIVATE boltzcpp)
add_test(NAME ligand_example_aspirin
    COMMAND boltzcpp_ligand "CC(=O)Oc1ccccc1C(=O)O"
            --out ${CMAKE_CURRENT_BINARY_DIR}/ligand_out.cif)
target_compile_definitions(test_npy PRIVATE
    BOLTZ_TOOLS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tools")
