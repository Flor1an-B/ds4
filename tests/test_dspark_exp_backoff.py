#!/usr/bin/env python3
"""Exercise the real DSpark scheduler miss-streak backoff helpers.

Run with Python's standard library and a C compiler; no model or GPU is
needed.  Mirrors the extraction approach of test_ngram_propose.py so the
production scheduler, not a copy of it, is what gets checked: streak
growth on rejected proposals, the exponential pause ladder and its cap,
reset on accept, no-draft cycles never growing the streak, and the flat
windowed pause never shortening a pause the streak already earned.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'ds4.c'

FUNCTIONS = (
    'ds4_dspark_env_u32',
    'ds4_dspark_scheduler_exp_backoff_base',
    'ds4_dspark_scheduler_exp_backoff_cap',
    'ds4_dspark_scheduler_exp_backoff_min_streak',
    'ds4_dspark_scheduler_exp_backoff_credit_cap',
    'ds4_dspark_scheduler_exp_backoff_pause',
    'ds4_dspark_scheduler_enabled',
    'ds4_dspark_scheduler_window',
    'ds4_dspark_scheduler_skip_cycles',
    'ds4_dspark_scheduler_slow_skip_cycles',
    'ds4_dspark_scheduler_min_avg_milli',
    'ds4_dspark_scheduler_max_ms_per_accept_milli',
    'ds4_dspark_scheduler_max_extra_saved_ratio_milli',
    'ds4_dspark_scheduler_break_even_window',
    'ds4_dspark_scheduler_no_draft_skip_cycles',
    'ds4_dspark_scheduler_short_accept_no_draft_skip_cycles',
    'ds4_dspark_scheduler_cold_low_confidence_skip_cycles',
    'ds4_dspark_scheduler_tail_min_tokens',
    'ds4_dspark_scheduler_cold_low_confidence_threshold',
    'ds4_dspark_scheduler_timing_enabled',
    'ds4_session_dspark_scheduler_reset',
    'ds4_session_dspark_scheduler_begin_request',
    'ds4_session_dspark_scheduler_should_skip',
    'ds4_session_dspark_scheduler_note',
)


def extract_function(source, name):
    matches = list(re.finditer(r'static \w+ ' + re.escape(name) + r'\s*\(', source))
    if len(matches) != 1:
        raise ValueError('require exactly one definition of ' + name)
    start = matches[0].start()
    opening = source.index('{', start)
    token = re.compile(r'/\*.*?\*/|"[^"]*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, opening):
        if match.group() == '{':
            depth += 1
        elif match.group() == '}':
            depth -= 1
            if depth == 0:
                return source[start:match.end()]
    raise ValueError('unterminated function ' + name)


HARNESS = r'''
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reduced mirrors of the session/stats shapes: only the fields the
 * extracted scheduler code actually touches. */
typedef struct ds4_dspark_spec_stats {
    uint64_t scheduler_skips;
    uint64_t scheduler_rejections;
    uint64_t scheduler_exp_pauses;
    uint64_t scheduler_credit_misses;
    double saved_ms;
} ds4_dspark_spec_stats;

typedef struct ds4_session {
    uint32_t dspark_sched_skip;
    uint32_t dspark_sched_miss_streak;
    uint32_t dspark_sched_credit;
    uint32_t dspark_sched_cycles;
    uint32_t dspark_sched_accepted;
    uint32_t dspark_sched_no_draft;
    uint32_t dspark_sched_lifetime_accepted;
    double dspark_sched_extra_ms;
    double dspark_sched_saved_ms;
    double dspark_sched_life_extra_ms;
    double dspark_sched_life_saved_ms;
    double dspark_last_target_eval_ms;
    float dspark_last_confidence0;
    bool dspark_sched_skipped_cycle;
    bool dspark_sched_long_accept_seen;
    bool dspark_sched_bypass;
    bool dspark_last_confidence0_valid;
    ds4_dspark_spec_stats dspark_stats;
} ds4_session;

/* Stand-ins for engine/device helpers the scheduler consults; the Metal
 * M5 paths under test never take the ROCm/Spark branches. */
static bool ds4_dspark_rocm_gfx1151_fast_path(void) { return false; }
static bool ds4_session_dspark_rocm_gfx1151_fast_path(const ds4_session *s) {
    (void)s;
    return false;
}
static bool ds4_session_dspark_seed_batch_enabled(const ds4_session *s) {
    (void)s;
    return false;
}
static bool ds4_session_dspark_seed_batch_short_fallback(const ds4_session *s) {
    (void)s;
    return false;
}
static bool ds4_gpu_device_is_spark(void) { return false; }
static bool ds4_dspark_stats_enabled(void) {
    const char *env = getenv("DS4_DSPARK_STATS");
    return env && env[0] && strcmp(env, "0") != 0;
}

{functions}

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d %s\n", __func__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

/* The n-gram/DSpark cycle shape: propose only when the scheduler lets the
 * cycle run, then report the verification outcome.  Paused cycles are
 * consumed exactly like ds4_session_eval_ngram_spec_cycle does: each
 * should_skip() hit is closed with a no-draft note that only clears the
 * skipped-cycle marker. */
static void consume_pause(ds4_session *s) {
    while (ds4_session_dspark_scheduler_should_skip(s)) {
        ds4_session_dspark_scheduler_note(s, 0, true, 0.0);
    }
}

static uint32_t reject(ds4_session *s) {
    CHECK(!ds4_session_dspark_scheduler_should_skip(s));
    ds4_session_dspark_scheduler_note(s, 0, false, 0.5);
    return s->dspark_sched_skip;
}

static uint32_t accept_n(ds4_session *s, uint32_t n) {
    CHECK(!ds4_session_dspark_scheduler_should_skip(s));
    ds4_session_dspark_scheduler_note(s, n, false, 0.5);
    return s->dspark_sched_skip;
}

static uint32_t no_draft(ds4_session *s) {
    CHECK(!ds4_session_dspark_scheduler_should_skip(s));
    ds4_session_dspark_scheduler_note(s, 0, true, 0.1);
    return s->dspark_sched_skip;
}

static void test_pause_ladder_default(void) {
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    /* The first miss is free (a periodic echo re-locks on the next
     * proposal); from the second consecutive miss the pause doubles
     * 2,4,8,16,32,64 and stays capped.  Consuming each pause in between
     * does not grow the streak. */
    const uint32_t want[] = {0, 2, 4, 8, 16, 32, 64, 64, 64};
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); i++) {
        CHECK(reject(&s) == want[i]);
        CHECK(s.dspark_sched_miss_streak == (uint32_t)(i + 1));
        consume_pause(&s);
    }
    CHECK(s.dspark_stats.scheduler_rejections == 9);
    CHECK(s.dspark_stats.scheduler_exp_pauses == 8);
    CHECK(s.dspark_stats.scheduler_skips > 0);
}

static void test_credit_absorbs_misses(void) {
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_CREDIT", "16", 1);
    /* Credit from accepted cycles absorbs misses before the ladder grows:
     * a long high-acceptance run keeps its drafting phase through
     * isolated misses, and only credit exhaustion restarts the streak. */
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    /* A full-width accept earns 2 credits, capped at 16. */
    for (int i = 0; i < 10; i++) accept_n(&s, 5);
    CHECK(s.dspark_sched_credit == 16);
    CHECK(s.dspark_sched_miss_streak == 0);
    /* Isolated misses spend credit, never grow the streak, never pause. */
    for (int i = 0; i < 8; i++) {
        CHECK(reject(&s) == 0);
        CHECK(s.dspark_sched_miss_streak == 0);
    }
    CHECK(s.dspark_sched_credit == 8);
    CHECK(s.dspark_stats.scheduler_credit_misses == 8);
    /* An accept re-fills credit and still resets the streak. */
    accept_n(&s, 5);
    CHECK(s.dspark_sched_credit == 10);
    /* Draining credit fully hands misses back to the ladder; the spend
     * itself keeps the streak flat, so the ladder restarts from zero. */
    for (int i = 0; i < 10; i++) {
        CHECK(reject(&s) == 0);
    }
    CHECK(s.dspark_sched_credit == 0);
    CHECK(s.dspark_sched_miss_streak == 0);
    CHECK(reject(&s) == 0);
    CHECK(s.dspark_sched_miss_streak == 1);
    CHECK(reject(&s) == 2);
    CHECK(s.dspark_sched_miss_streak == 2);
    /* begin_request wipes credit along with the streak. */
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(s.dspark_sched_credit == 0 && s.dspark_sched_miss_streak == 0);
}

static void test_credit_disabled(void) {
    /* CREDIT=0 keeps the bare miss-streak ladder. */
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_CREDIT", "0", 1);
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    accept_n(&s, 5);
    CHECK(s.dspark_sched_credit == 0);
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 2);
    unsetenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_CREDIT");
}

static void test_pause_ladder_min_streak_one(void) {
    /* MIN_STREAK=1 restores the pause-after-every-miss ladder. */
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_MIN_STREAK", "1", 1);
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(ds4_dspark_scheduler_exp_backoff_pause(1) == 2);
    const uint32_t want[] = {2, 4, 8, 16, 32, 64};
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); i++) {
        CHECK(reject(&s) == want[i]);
        consume_pause(&s);
    }
    CHECK(ds4_dspark_scheduler_exp_backoff_pause(0) == 0);
    /* min_streak=0 disables the feature entirely. */
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_MIN_STREAK", "0", 1);
    CHECK(ds4_dspark_scheduler_exp_backoff_pause(7) == 0);
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_MIN_STREAK", "2", 1);
}

static void test_pause_ladder_env_cap_and_base(void) {
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_CAP", "16", 1);
    const uint32_t want16[] = {0, 2, 4, 8, 16, 16, 16};
    for (size_t i = 0; i < sizeof(want16) / sizeof(want16[0]); i++) {
        CHECK(reject(&s) == want16[i]);
        consume_pause(&s);
    }
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_CAP", "64", 1);
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "3", 1);
    ds4_session_dspark_scheduler_begin_request(&s);
    const uint32_t want3[] = {0, 3, 6, 12, 24, 48, 64};
    for (size_t i = 0; i < sizeof(want3) / sizeof(want3[0]); i++) {
        CHECK(reject(&s) == want3[i]);
        consume_pause(&s);
    }
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "1000", 1);
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 64);
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "2", 1);
}

static void test_base_zero_disables_backoff(void) {
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "0", 1);
    CHECK(ds4_dspark_scheduler_exp_backoff_pause(1) == 0);
    CHECK(ds4_dspark_scheduler_exp_backoff_pause(9) == 0);
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(ds4_dspark_scheduler_exp_backoff_base() == 0);
    /* Rejections count, but no immediate exponential pause appears. */
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 0);
    CHECK(s.dspark_sched_miss_streak == 2);
    CHECK(s.dspark_stats.scheduler_rejections == 2);
    CHECK(s.dspark_stats.scheduler_exp_pauses == 0);
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "2", 1);
}

static void test_accept_resets_streak(void) {
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 2);
    consume_pause(&s);
    CHECK(reject(&s) == 4);
    consume_pause(&s);
    CHECK(s.dspark_sched_miss_streak == 3);
    /* A productive cycle clears the streak: the ladder starts over. */
    CHECK(accept_n(&s, 3) == 0);
    CHECK(s.dspark_sched_miss_streak == 0);
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 2);
    consume_pause(&s);
    /* begin_request also clears the streak for the next request. */
    CHECK(reject(&s) == 4);
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(s.dspark_sched_miss_streak == 0);
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 2);
    consume_pause(&s);
}

static void test_no_draft_never_grows_streak(void) {
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(reject(&s) == 0);
    CHECK(s.dspark_sched_miss_streak == 1);
    /* No-draft cycles keep their fixed pause (default 3) and leave the
     * streak untouched: the next miss pays only the second rung. */
    CHECK(no_draft(&s) == 3);
    CHECK(s.dspark_sched_miss_streak == 1);
    consume_pause(&s);
    no_draft(&s);
    consume_pause(&s);
    CHECK(reject(&s) == 2);
    CHECK(s.dspark_sched_miss_streak == 2);
    consume_pause(&s);
    /* A no-draft pause cannot shorten the pause the streak already
     * earned: after a streak of 4 (floor 8) the fixed 3-cycle pause
     * rises to 8. */
    CHECK(reject(&s) == 4);
    consume_pause(&s);
    CHECK(reject(&s) == 8);
    consume_pause(&s);
    CHECK(no_draft(&s) == 8);
    consume_pause(&s);
}

static void test_exp_pause_never_lowers_existing_skip(void) {
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    /* A single miss never lowers an existing pause. */
    s.dspark_sched_skip = 5;
    ds4_session_dspark_scheduler_note(&s, 0, false, 0.5);
    CHECK(s.dspark_sched_miss_streak == 1);
    CHECK(s.dspark_sched_skip == 5);
    /* The earned floor is not lowered by later notes either: a
     * second miss raises a short remaining pause to the rung. */
    s.dspark_sched_skip = 1;
    ds4_session_dspark_scheduler_note(&s, 0, false, 0.5);
    CHECK(s.dspark_sched_miss_streak == 2);
    CHECK(s.dspark_sched_skip == 2);
}

static void test_windowed_pause_respects_exp_floor(void) {
    /* A one-cycle window forces the legacy flat pause (skip_cycles=2 on
     * the stubbed device) after every noted cycle; the exponential floor
     * must survive that assignment. */
    setenv("DS4_DSPARK_SCHEDULER_WINDOW", "1", 1);
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    /* First miss free (flat only), then the floor matches and overtakes
     * the flat pause as the streak grows. */
    const uint32_t want[] = {2, 2, 4, 8, 16, 32};
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); i++) {
        CHECK(reject(&s) == want[i]);
        consume_pause(&s);
    }
    /* Base 0 keeps the windowed flat pause exactly as before. */
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "0", 1);
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(reject(&s) == 2);
    consume_pause(&s);
    CHECK(reject(&s) == 2);
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_BASE", "2", 1);
    setenv("DS4_DSPARK_SCHEDULER_WINDOW", "32", 1);
}

static void test_should_skip_consumes_pause(void) {
    ds4_session s;
    memset(&s, 0, sizeof(s));
    ds4_session_dspark_scheduler_begin_request(&s);
    CHECK(reject(&s) == 0);
    CHECK(reject(&s) == 2);
    consume_pause(&s);
    CHECK(reject(&s) == 4);
    consume_pause(&s);
    CHECK(reject(&s) == 8);
    uint32_t seen = 0;
    while (ds4_session_dspark_scheduler_should_skip(&s)) {
        seen++;
        ds4_session_dspark_scheduler_note(&s, 0, true, 0.0);
        CHECK(s.dspark_sched_skipped_cycle == false);
    }
    CHECK(seen == 8);
    CHECK(s.dspark_stats.scheduler_skips == 2 + 4 + 8);
    /* The skipped cycles did not grow the streak: the window counter
     * still holds only the real noted rejections. */
    CHECK(s.dspark_sched_miss_streak == 4);
    CHECK(s.dspark_sched_cycles == 4);
}

int main(void) {
    (void)ds4_gpu_device_is_spark;
    (void)ds4_dspark_scheduler_tail_min_tokens;
    (void)ds4_dspark_scheduler_timing_enabled;
    /* Keep the windowed flat pause out of the way; its interaction with
     * the floor has its own dedicated test below. */
    setenv("DS4_DSPARK_SCHEDULER_WINDOW", "32", 1);
    setenv("DS4_DSPARK_STATS", "1", 1);
    /* The ladder tests below exercise the bare miss-streak logic; the
     * credit interaction has its own dedicated tests at the end. */
    setenv("DS4_DSPARK_SCHEDULER_EXP_BACKOFF_CREDIT", "0", 1);
    test_pause_ladder_default();
    test_pause_ladder_min_streak_one();
    test_pause_ladder_env_cap_and_base();
    test_base_zero_disables_backoff();
    test_accept_resets_streak();
    test_no_draft_never_grows_streak();
    test_exp_pause_never_lowers_existing_skip();
    test_windowed_pause_respects_exp_floor();
    test_should_skip_consumes_pause();
    test_credit_absorbs_misses();
    test_credit_disabled();
    if (failures) {
        printf("exp-backoff: %d failures\n", failures);
        return 1;
    }
    printf("exp-backoff: all checks passed\n");
    return 0;
}
'''


def main():
    source = SOURCE.read_text()
    functions = '\n\n'.join(extract_function(source, name)
                            for name in FUNCTIONS)
    if not functions.strip():
        raise ValueError('no functions extracted')
    with tempfile.TemporaryDirectory() as tmp:
        c_path = Path(tmp) / 'exp_backoff_harness.c'
        bin_path = Path(tmp) / 'exp_backoff_harness'
        c_path.write_text(HARNESS.replace('{functions}', functions))
        cc = os.environ.get('CC', 'cc')
        compile_cmd = [cc, '-O2', '-Wall', '-Wextra', '-std=c99',
                      '-o', str(bin_path), str(c_path), '-lm']
        compile = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile.returncode != 0:
            print(compile.stderr)
            return 1
        if compile.stderr.strip():
            print('note: compiler warnings:')
            print(compile.stderr)
        run = subprocess.run([str(bin_path)], capture_output=True, text=True)
        if run.stdout.strip():
            print(run.stdout.strip())
        return run.returncode


if __name__ == '__main__':
    sys.exit(main())
