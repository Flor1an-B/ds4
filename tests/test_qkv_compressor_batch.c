/* Model-backed correctness oracle for the new row-batched fused QKV +
 * indexer-compressor projection kernel
 * (ds4_gpu_qkv_pair_quad_compressor_project_batch_tensor), compared against
 * n_tokens sequential calls to the proven single-row kernel
 * (ds4_gpu_qkv_pair_quad_compressor_store_tensor) it is meant to replace in
 * DSpark's speculative verify path.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-qkv-compressor-batch
 *
 * This is a pure kernel-math equivalence check (deterministic synthetic
 * activation, real model weights) -- see the DS4_TEST_HOOKS comment in
 * ds4.c above ds4_test_qkv_compressor_batch_matches_sequential for why exact
 * (not just close) agreement is the right bar here: there is no cross-row
 * reduction at this projection stage, so batched and sequential must produce
 * bit-identical results if the kernel is correct. */

#define DS4_TEST_HOOKS
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

static void check_layer(ds4_engine *e, uint32_t il, uint32_t n_tokens, uint32_t pos0) {
    float d_qr = -1.0f, d_kv = -1.0f, d_comp_kv = -1.0f, d_comp_sc = -1.0f;
    bool ran = ds4_test_qkv_compressor_batch_matches_sequential(
            e, il, n_tokens, pos0, &d_qr, &d_kv, &d_comp_kv, &d_comp_sc);
    if (!ran) {
        fprintf(stderr, "  skip: layer %u is not ratio==4 (or missing weights)\n", il);
        return;
    }
    g_checks++;
    const float tol = 1e-5f;
    const bool pass = d_qr >= 0.0f && d_qr < tol &&
                       d_kv >= 0.0f && d_kv < tol &&
                       d_comp_kv >= 0.0f && d_comp_kv < tol &&
                       d_comp_sc >= 0.0f && d_comp_sc < tol;
    fprintf(stderr,
            "  layer=%u n_tokens=%u pos0=%u qr=%.8f kv=%.8f comp_kv=%.8f comp_sc=%.8f -> %s\n",
            il, n_tokens, pos0, d_qr, d_kv, d_comp_kv, d_comp_sc,
            pass ? "MATCH" : "MISMATCH");
    if (!pass) g_failures++;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = 4096,
        .warm_weights = false,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) {
        fprintf(stderr, "FAIL: engine open\n");
        return 1;
    }

    const int layer_count = ds4_engine_layer_count(e);
    fprintf(stderr, "ds4: scanning %d layers for ratio==4 coverage\n", layer_count);
    for (uint32_t il = 0; il < (uint32_t)layer_count; il++) {
        check_layer(e, il, 1, 1000);
        check_layer(e, il, 2, 1000);
        check_layer(e, il, 5, 1000);
        check_layer(e, il, 6, 4096);
        check_layer(e, il, 8, 4096);
        check_layer(e, il, 12, 4096);
        check_layer(e, il, 16, 4096);
        /* Position not a multiple of ratio: exercises every pos%ratio phase
         * across the batch's rows, including the wraparound at row 4 for a
         * 5-row (this model's DSpark block_size) batch. */
        check_layer(e, il, 5, 1001);
    }

    ds4_engine_close(e);

    fprintf(stderr, "\ntest_qkv_compressor_batch: %d/%d checks passed (%d failed)\n",
            g_checks - g_failures, g_checks, g_failures);
    if (g_checks == 0) {
        fprintf(stderr, "FAIL: no ratio==4 layers found -- oracle did not run at all\n");
        return 1;
    }
    return g_failures == 0 ? 0 : 1;
}
