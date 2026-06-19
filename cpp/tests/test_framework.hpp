// Minimal hand-rolled test harness (no external deps). Mirrors the structure of
// the pytest suite: register named cases, assert, report pass/fail, exit nonzero
// on any failure.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace boltztest {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

struct AssertFailure : std::runtime_error {
    explicit AssertFailure(const std::string& m) : std::runtime_error(m) {}
};

inline void expect_true(bool cond, const std::string& what) {
    if (!cond) throw AssertFailure("expected true: " + what);
}

inline void expect_eq_f(float got, float want, const std::string& what) {
    if (std::fabs(got - want) > 1e-6f)
        throw AssertFailure(what + " — got " + std::to_string(got) + " want " +
                            std::to_string(want));
}

inline void expect_eq_i(long got, long want, const std::string& what) {
    if (got != want)
        throw AssertFailure(what + " — got " + std::to_string(got) + " want " +
                            std::to_string(want));
}

// Run `fn`; assert it throws and that the message contains `needle`.
inline void expect_throws_contains(const std::function<void()>& fn,
                                   const std::string& needle, const std::string& what) {
    try {
        fn();
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find(needle) == std::string::npos)
            throw AssertFailure(what + " — threw but message '" + msg +
                                "' lacks '" + needle + "'");
        return;
    }
    throw AssertFailure(what + " — expected throw containing '" + needle + "'");
}

inline int run_all() {
    int passed = 0, failed = 0;
    for (const Case& c : registry()) {
        try {
            c.fn();
            std::printf("  PASS  %s\n", c.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("  FAIL  %s\n        %s\n", c.name.c_str(), e.what());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed (%d total)\n", passed, failed,
                passed + failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace boltztest

#define BOLTZ_TEST(name)                                                       \
    static void name();                                                        \
    static ::boltztest::Registrar reg_##name(#name, name);                     \
    static void name()
