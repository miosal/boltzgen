// Equivalence test for MLPAttnGNN.forward: the ggml-backed implementation is
// checked against a fully independent, from-scratch scalar reference that
// recomputes the layer with raw loops (its own linear/gelu/softmax). This
// validates the assembled wiring (gather, concat order, head broadcast, scatter
// aggregation, residuals, batchnorm) without any trained weights.
#include "boltz/gnn_layer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <vector>

using namespace boltz;
using namespace boltztest;

namespace {

void near(float got, float want, const std::string& what) {
    if (std::fabs(got - want) > 1e-4f)
        throw AssertFailure(what + " got " + std::to_string(got) + " want " +
                            std::to_string(want));
}

float geluf(float x) { return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f))); }

std::vector<float> linf(const std::vector<float>& x, const Linear& L) {
    std::vector<float> y(L.out);
    for (int o = 0; o < L.out; ++o) {
        float s = L.b[o];
        for (int i = 0; i < L.in; ++i) s += L.w[o * L.in + i] * x[i];
        y[o] = s;
    }
    return y;
}

Linear make_lin(int in, int out, float seed) {
    Linear L;
    L.in = in;
    L.out = out;
    L.w.resize((size_t)in * out);
    L.b.resize(out);
    for (int i = 0; i < in * out; ++i) L.w[i] = 0.1f * std::sin(seed + i * 0.7f);
    for (int o = 0; o < out; ++o) L.b[o] = 0.05f * std::cos(seed + o);
    return L;
}

Affine make_affine(int f, float seed) {
    Affine a;
    a.scale.resize(f);
    a.shift.resize(f);
    for (int i = 0; i < f; ++i) {
        a.scale[i] = 1.0f + 0.1f * std::sin(seed + i);
        a.shift[i] = 0.02f * std::cos(seed + i);
    }
    return a;
}

std::vector<float> apply_affine_ref(std::vector<float> x, const Affine& a) {
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * a.scale[i % a.scale.size()] + a.shift[i % a.shift.size()];
    return x;
}

}  // namespace

BOLTZ_TEST(mlp_attn_gnn_matches_scalar_reference) {
    const int node = 2, pair = 2, hidden = 2, heads = 2, N = 3;

    MLPAttnGNNWeights w;
    w.node_dim = node; w.pair_dim = pair; w.hidden_dim = hidden; w.num_heads = heads;
    w.edge_l1 = make_lin(2 * node + pair, hidden, 1.0f);
    w.edge_l2 = make_lin(hidden, pair, 2.0f);
    w.edge_bn = make_affine(pair, 3.0f);
    w.aw_l1 = make_lin(2 * node + pair, hidden, 4.0f);
    w.aw_l2 = make_lin(hidden, hidden, 5.0f);
    w.aw_l3 = make_lin(hidden, heads, 6.0f);
    w.av_l1 = make_lin(node + pair, hidden, 7.0f);
    w.av_l2 = make_lin(hidden, hidden, 8.0f);
    w.av_l3 = make_lin(hidden, node, 9.0f);
    w.out_l = make_lin(heads * node, node, 10.0f);
    w.out_bn = make_affine(node, 11.0f);
    w.ffn_l1 = make_lin(node, hidden, 12.0f);
    w.ffn_l2 = make_lin(hidden, node, 13.0f);
    w.ffn_bn = make_affine(node, 14.0f);

    std::vector<float> s = {0.1f, -0.2f,  0.3f, 0.4f,  -0.5f, 0.6f};  // [3,2]
    std::vector<int> src_i = {0, 1, 2, 1, 0};
    std::vector<int> dst_i = {0, 0, 1, 1, 2};
    const int E = (int)src_i.size();
    std::vector<float> z(E * pair);
    for (int i = 0; i < E * pair; ++i) z[i] = 0.05f * std::sin(0.3f + i);

    auto out = mlp_attn_gnn_forward(w, N, s, z, src_i, dst_i);

    // ---- independent scalar reference ----
    auto row = [](const std::vector<float>& m, int r, int d) {
        return std::vector<float>(m.begin() + r * d, m.begin() + (r + 1) * d);
    };
    std::vector<float> rs = s, rz = z;

    // edge update (uses pre-update s rows; updates z in place).
    std::vector<std::vector<float>> s_src(E), s_dst(E);
    for (int e = 0; e < E; ++e) { s_src[e] = row(rs, src_i[e], node); s_dst[e] = row(rs, dst_i[e], node); }
    for (int e = 0; e < E; ++e) {
        std::vector<float> in = s_src[e];
        in.insert(in.end(), s_dst[e].begin(), s_dst[e].end());
        for (int c = 0; c < pair; ++c) in.push_back(rz[e * pair + c]);
        std::vector<float> h = linf(in, w.edge_l1);
        for (float& v : h) v = geluf(v);
        std::vector<float> u = linf(h, w.edge_l2);
        u = apply_affine_ref(u, w.edge_bn);
        for (int c = 0; c < pair; ++c) rz[e * pair + c] += u[c];
    }

    // attn weights/values
    std::vector<std::vector<float>> aw(E), av(E);
    for (int e = 0; e < E; ++e) {
        std::vector<float> zin(rz.begin() + e * pair, rz.begin() + (e + 1) * pair);
        std::vector<float> awi = s_dst[e];
        awi.insert(awi.end(), s_src[e].begin(), s_src[e].end());
        awi.insert(awi.end(), zin.begin(), zin.end());
        std::vector<float> a = linf(awi, w.aw_l1); for (float& v : a) v = geluf(v);
        a = linf(a, w.aw_l2); for (float& v : a) v = geluf(v);
        aw[e] = linf(a, w.aw_l3);

        std::vector<float> avi = s_src[e];
        avi.insert(avi.end(), zin.begin(), zin.end());
        std::vector<float> vv = linf(avi, w.av_l1); for (float& v : vv) v = geluf(v);
        vv = linf(vv, w.av_l2); for (float& v : vv) v = geluf(v);
        av[e] = linf(vv, w.av_l3);
    }

    // softmax of aw over destination groups per head
    std::vector<std::vector<float>> aw_soft(E, std::vector<float>(heads));
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < heads; ++h) {
            float mx = -1e30f;
            for (int e = 0; e < E; ++e) if (dst_i[e] == n) mx = std::max(mx, aw[e][h]);
            float sum = 0;
            for (int e = 0; e < E; ++e) if (dst_i[e] == n) sum += std::exp(aw[e][h] - mx);
            for (int e = 0; e < E; ++e) if (dst_i[e] == n)
                aw_soft[e][h] = std::exp(aw[e][h] - mx) / (sum + 1e-10f);
        }
    }

    // aggregate + output linear + residual
    std::vector<std::vector<float>> agg(N, std::vector<float>(heads * node, 0.0f));
    for (int e = 0; e < E; ++e)
        for (int h = 0; h < heads; ++h)
            for (int c = 0; c < node; ++c)
                agg[dst_i[e]][h * node + c] += aw_soft[e][h] * av[e][c];
    for (int n = 0; n < N; ++n) {
        std::vector<float> o = linf(agg[n], w.out_l);
        o = apply_affine_ref(o, w.out_bn);
        for (int c = 0; c < node; ++c) rs[n * node + c] += o[c];
    }
    // FFN residual
    for (int n = 0; n < N; ++n) {
        std::vector<float> sn(rs.begin() + n * node, rs.begin() + (n + 1) * node);
        std::vector<float> h = linf(sn, w.ffn_l1); for (float& v : h) v = geluf(v);
        std::vector<float> o = linf(h, w.ffn_l2);
        o = apply_affine_ref(o, w.ffn_bn);
        for (int c = 0; c < node; ++c) rs[n * node + c] += o[c];
    }

    // compare
    for (int i = 0; i < N * node; ++i) near(out.s[i], rs[i], "s");
    for (int i = 0; i < E * pair; ++i) near(out.z[i], rz[i], "z");
}

int main() {
    std::printf("== test_gnn_layer ==\n");
    return boltztest::run_all();
}
