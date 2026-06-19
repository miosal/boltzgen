#include "boltz/weight_store.hpp"

#include "gguf.h"

namespace boltz {

WeightStore::WeightStore(const std::string& path) {
    gguf_init_params params{};
    params.no_alloc = false;   // load tensor data into ctx_
    params.ctx = &ctx_;
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (gguf_ == nullptr || ctx_ == nullptr)
        throw std::runtime_error("WeightStore: failed to load gguf '" + path + "'");
}

WeightStore::~WeightStore() {
    if (gguf_ != nullptr) gguf_free(gguf_);
    if (ctx_ != nullptr) ggml_free(ctx_);
}

ggml_tensor* WeightStore::try_get(const std::string& name) const {
    return ggml_get_tensor(ctx_, name.c_str());
}

ggml_tensor* WeightStore::get(const std::string& name) const {
    ggml_tensor* t = try_get(name);
    if (t == nullptr) throw std::runtime_error("WeightStore: missing tensor '" + name + "'");
    return t;
}

bool WeightStore::has(const std::string& name) const { return try_get(name) != nullptr; }

int64_t WeightStore::n_tensors() const { return gguf_get_n_tensors(gguf_); }

std::optional<std::string> WeightStore::meta_str(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return std::nullopt;
    return std::string(gguf_get_val_str(gguf_, id));
}

std::optional<uint32_t> WeightStore::meta_u32(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return std::nullopt;
    return gguf_get_val_u32(gguf_, id);
}

}  // namespace boltz
