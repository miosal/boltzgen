#include "boltz/npy.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace boltz {
namespace {

std::string find_value(const std::string& header, const std::string& key) {
    const size_t k = header.find(key);
    if (k == std::string::npos) return "";
    size_t c = header.find(':', k);
    if (c == std::string::npos) return "";
    ++c;
    while (c < header.size() && (header[c] == ' ' || header[c] == '\'')) ++c;
    size_t e = c;
    while (e < header.size() && header[e] != ',' && header[e] != '}' &&
           header[e] != '\'' && header[e] != ')')
        ++e;
    return header.substr(c, e - c);
}

}  // namespace

NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_npy: cannot open '" + path + "'");

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("load_npy: bad magic in '" + path + "'");
    uint8_t ver_major = 0, ver_minor = 0;
    f.read(reinterpret_cast<char*>(&ver_major), 1);
    f.read(reinterpret_cast<char*>(&ver_minor), 1);
    if (ver_major != 1) throw std::runtime_error("load_npy: only npy v1.0 supported");

    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);  // little-endian
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    const std::string descr = find_value(header, "descr");
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": False") == std::string::npos)
        throw std::runtime_error("load_npy: only C-order supported");

    bool f64 = false;
    if (descr == "<f4") f64 = false;
    else if (descr == "<f8") f64 = true;
    else throw std::runtime_error("load_npy: unsupported dtype '" + descr + "'");

    // Parse shape tuple, e.g. "(2, 3,)" or "()".
    NpyArray a;
    const size_t lp = header.find('(');
    const size_t rp = header.find(')', lp);
    std::string dims = header.substr(lp + 1, rp - lp - 1);
    std::string cur;
    for (char ch : dims) {
        if (ch == ',') { if (!cur.empty()) { a.shape.push_back(std::stol(cur)); cur.clear(); } }
        else if (!std::isspace(static_cast<unsigned char>(ch))) cur += ch;
    }
    if (!cur.empty()) a.shape.push_back(std::stol(cur));

    long n = 1;
    for (long d : a.shape) n *= d;
    a.data.resize(n);
    if (f64) {
        std::vector<double> tmp(n);
        f.read(reinterpret_cast<char*>(tmp.data()), n * 8);
        for (long i = 0; i < n; ++i) a.data[i] = static_cast<float>(tmp[i]);
    } else {
        f.read(reinterpret_cast<char*>(a.data.data()), n * 4);
    }
    if (!f) throw std::runtime_error("load_npy: truncated data in '" + path + "'");
    return a;
}

DiffStats compare(const NpyArray& got, const NpyArray& golden) {
    DiffStats s;
    s.shapes_match = got.shape == golden.shape;
    const long n = std::min(got.size(), golden.size());
    s.n = n;
    double sum = 0;
    for (long i = 0; i < n; ++i) {
        const float d = std::fabs(got.data[i] - golden.data[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum += d;
    }
    s.mean_abs = n > 0 ? static_cast<float>(sum / n) : 0.0f;
    return s;
}

}  // namespace boltz
