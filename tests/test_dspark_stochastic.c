/* test_dspark_stochastic — correctness of the CPU-side math behind DSpark's
 * stochastic (temperature > 0) speculative verification: dspark_token_prob
 * (temperature-scaled softmax probability of one token) and
 * dspark_sample_residual (Leviathan/Chen residual resample on rejection).
 *
 * These are pure float-array math with no GPU/session dependency, exposed
 * via the DS4_TEST_HOOKS-gated ds4_test_dspark_token_prob /
 * ds4_test_dspark_sample_residual wrappers. Compiles against the CPU-only
 * ds4_cpu_test_hooks.o build (same object test_engine_mgpu_placement uses),
 * so it runs on every platform including Darwin.
 *
 * The critical property under test (test_full_pipeline_matches_target) is
 * the algorithm's actual correctness guarantee: marginalizing over
 * "sample x~q, accept w.p. min(1,p(x)/q(x)), else resample from the
 * residual" must reproduce the target distribution p exactly. This is what
 * makes MTP/DSpark speculative sampling unbiased for any temperature > 0,
 * as opposed to naively resampling from p on rejection (which double-counts
 * mass and is measurably biased -- see the derivation this test encodes). */

#define DS4_TEST_HOOKS
#include "../ds4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } \
} while (0)

static void softmax(const float *logits, uint32_t n, float temperature, float *out) {
    float max_logit = -1e30f;
    for (uint32_t i = 0; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        out[i] = expf((logits[i] - max_logit) / temperature);
        sum += out[i];
    }
    for (uint32_t i = 0; i < n; i++) out[i] = (float)(out[i] / sum);
}

static void test_token_prob_matches_manual_softmax(void) {
    const uint32_t n = 4;
    const float logits[4] = {2.0f, 1.0f, 0.0f, -1.0f};
    float expected[4];
    softmax(logits, n, 1.0f, expected);

    for (uint32_t i = 0; i < n; i++) {
        float got = 0.0f;
        bool ok = ds4_test_dspark_token_prob(logits, n, 1.0f, (int)i, &got);
        CHECK(ok, "dspark_token_prob succeeds");
        CHECK(fabsf(got - expected[i]) < 1e-5f, "dspark_token_prob matches manual softmax");
    }

    float unused;
    CHECK(!ds4_test_dspark_token_prob(logits, n, 1.0f, -1, &unused), "negative token id rejected");
    CHECK(!ds4_test_dspark_token_prob(logits, n, 1.0f, (int)n, &unused), "out-of-range token id rejected");
    CHECK(!ds4_test_dspark_token_prob(NULL, n, 1.0f, 0, &unused), "null logits rejected");
}

static void test_residual_zero_when_p_equals_q(void) {
    const uint32_t n = 5;
    const float logits[5] = {2.0f, 1.0f, 0.0f, -1.0f, -2.0f};
    float scratch[5];
    /* p == q everywhere: residual is exactly zero, so the function must fall
     * back to argmax(p) deterministically -- there is nothing to reject
     * into, so every call (any rng state) should return the same token. */
    for (int seed = 0; seed < 20; seed++) {
        uint64_t rng = (uint64_t)(seed * 2654435761u + 12345u);
        int token = ds4_test_dspark_sample_residual(logits, logits, n, 1.0f,
                                                    &rng, scratch);
        CHECK(token == 0, "residual with p==q falls back to argmax(p)");
    }
}

/* p and q chosen so residual mass splits exactly 50/50 between tokens 0 and
 * 1, with tokens 2-4 having zero residual (q already dominates or ties
 * there). Verified by hand: softmax(p)=[.4368,.4368,.0591,.0591,.008],
 * softmax(q)=[.0591,.0591,.4368,.4368,.008]; residual=[.3777,.3777,0,0,0]
 * normalized to [.5,.5,0,0,0]. */
static const float g_p_logits[5] = {2.0f, 2.0f, 0.0f, 0.0f, -2.0f};
static const float g_q_logits[5] = {0.0f, 0.0f, 2.0f, 2.0f, -2.0f};

static void test_residual_statistical_distribution(void) {
    const uint32_t n = 5;
    float scratch[5];
    const int trials = 200000;
    int counts[5] = {0, 0, 0, 0, 0};
    uint64_t rng = 0x1234567890abcdefULL;
    for (int t = 0; t < trials; t++) {
        int token = ds4_test_dspark_sample_residual(g_p_logits, g_q_logits, n,
                                                     1.0f, &rng, scratch);
        CHECK(token >= 0 && token < (int)n, "residual token in range");
        if (token >= 0 && token < (int)n) counts[token]++;
    }
    const double expected[5] = {0.5, 0.5, 0.0, 0.0, 0.0};
    for (int i = 0; i < 5; i++) {
        const double freq = (double)counts[i] / trials;
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "residual token %d frequency %.4f matches expected %.2f",
                 i, freq, expected[i]);
        CHECK(fabs(freq - expected[i]) < 0.01, msg);
    }
}

/* Simple local RNG for the outer draws the test itself needs to make
 * (sampling x~q, and the accept/reject coin flip) -- unrelated to ds4's own
 * internal sample_rng_f32, which dspark_sample_residual calls internally on
 * the rng state we pass it. */
static uint64_t test_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}
static float test_rng_f32(uint64_t *state) {
    return (float)((test_rng_next(state) >> 40) & 0xffffffu) / 16777216.0f;
}

static int sample_from_probs(const float *probs, uint32_t n, uint64_t *rng) {
    float r = test_rng_f32(rng);
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        acc += probs[i];
        if (r <= acc) return (int)i;
    }
    return (int)n - 1;
}

/* The actual correctness guarantee under test: marginalizing over the
 * two-step process (draft x~q, accept w.p. min(1,p(x)/q(x)), else resample
 * from residual) must reproduce p exactly -- not merely "close to p", and
 * specifically NOT the biased scheme of resampling from plain p on reject
 * (see PR discussion: that scheme double-counts high-p/low-q tokens). */
static void test_full_pipeline_matches_target(void) {
    const uint32_t n = 5;
    float q_probs[5];
    softmax(g_q_logits, n, 1.0f, q_probs);
    float scratch[5];
    const int trials = 300000;
    int counts[5] = {0, 0, 0, 0, 0};
    uint64_t outer_rng = 0xdeadbeefcafef00dULL;
    uint64_t residual_rng = 0x0badc0ffee1234ULL;
    for (int t = 0; t < trials; t++) {
        int x = sample_from_probs(q_probs, n, &outer_rng);
        float px = 0.0f, qx = 0.0f;
        CHECK(ds4_test_dspark_token_prob(g_p_logits, n, 1.0f, x, &px),
              "target prob available");
        CHECK(ds4_test_dspark_token_prob(g_q_logits, n, 1.0f, x, &qx),
              "draft prob available");
        const float accept = qx > 0.0f ? (px / qx) : 0.0f;
        const float accept_ratio = accept > 1.0f ? 1.0f : accept;
        int final_token;
        if (test_rng_f32(&outer_rng) <= accept_ratio) {
            final_token = x;
        } else {
            final_token = ds4_test_dspark_sample_residual(
                    g_p_logits, g_q_logits, n, 1.0f, &residual_rng, scratch);
        }
        if (final_token >= 0 && final_token < (int)n) counts[final_token]++;
    }
    float p_probs[5];
    softmax(g_p_logits, n, 1.0f, p_probs);
    for (int i = 0; i < 5; i++) {
        const double freq = (double)counts[i] / trials;
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "full-pipeline marginal for token %d: %.4f vs target p=%.4f",
                 i, freq, p_probs[i]);
        /* Generous absolute+relative tolerance: this is a statistical smoke
         * test at finite N, not an exact-arithmetic check. */
        CHECK(fabs(freq - p_probs[i]) < 0.01 + 0.1 * p_probs[i], msg);
    }
}

int main(void) {
    test_token_prob_matches_manual_softmax();
    test_residual_zero_when_p_equals_q();
    test_residual_statistical_distribution();
    test_full_pipeline_matches_target();

    fprintf(stderr, "\ntest_dspark_stochastic: %d/%d checks passed (%d failed)\n",
            g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
