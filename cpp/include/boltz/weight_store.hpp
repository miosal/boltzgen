// gguf-backed weight store. Loads a converted BoltzGen checkpoint (.gguf) into
// a ggml_context and exposes tensors by name plus a few metadata accessors.
// This is the C++ side of the weight pipeline; the offline converter
// (tools/convert_ckpt_to_gguf.py) produces the .gguf from a PyTorch .ckpt.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace boltz {

class WeightStore {
public:
    // Loads `path`, allocating tensor data in an owned ggml_context.
    explicit WeightStore(const std::string& path);
    ~WeightStore();

    WeightStore(const WeightStore&) = delete;
    WeightStore& operator=(const WeightStore&) = delete;

    // Tensor by exact name; throws if absent.
    ggml_tensor* get(const std::string& name) const;
    // Tensor by name or nullptr if absent.
    ggml_tensor* try_get(const std::string& name) const;
    bool has(const std::string& name) const;

    int64_t n_tensors() const;

    // Metadata accessors (return nullopt if key absent).
    std::optional<std::string> meta_str(const std::string& key) const;
    std::optional<uint32_t> meta_u32(const std::string& key) const;

    ggml_context* ctx() const { return ctx_; }

private:
    ggml_context* ctx_ = nullptr;
    gguf_context* gguf_ = nullptr;
};

}  // namespace boltz
