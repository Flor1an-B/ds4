/* Model-free publication/matching regression. Only the engine token/session and
 * payload boundary is synthetic; the recorder, disk policy and store are real. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include <assert.h>

struct ds4_session { ds4_tokens tokens; };
static const unsigned char payload[] = { 'K', 'V', 0, 0xff, 7, 3 };

/* Byte tokens are not a Qwen BPE or GPU-state oracle. */
void ds4_tokens_push(ds4_tokens *t, int token) {
    if (t->len == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 64;
        t->v = xrealloc(t->v, (size_t)t->cap * sizeof(*t->v));
    }
    t->v[t->len++] = token;
}
void ds4_tokens_free(ds4_tokens *t) { free(t->v); memset(t, 0, sizeof(*t)); }
bool ds4_tokens_starts_with(const ds4_tokens *t, const ds4_tokens *p) {
    return t->len >= p->len && !memcmp(t->v, p->v, (size_t)p->len * sizeof(*p->v));
}
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *t) {
    (void)e;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++)
        ds4_tokens_push(t, *p);
}
char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e;
    assert(token > 0 && token <= 255);
    char *text = xmalloc(2);
    text[0] = (char)token; text[1] = 0; *len = 1;
    return text;
}
bool ds4_think_mode_enabled(ds4_think_mode mode) { return mode != DS4_THINK_NONE; }
bool ds4_think_mode_parse_level(const char *text, ds4_think_mode *out) {
    assert(!strcmp(text, "medium")); *out = DS4_THINK_MEDIUM; return true;
}
const char *ds4_qwen4_reasoning_effort_text(ds4_think_mode mode) {
    assert(mode == DS4_THINK_MEDIUM); return NULL;
}
const char *ds4_glm_reasoning_effort_text(ds4_think_mode mode) { (void)mode; abort(); }
const char *ds4_deepseek41_reasoning_effort_text(ds4_think_mode mode) { (void)mode; abort(); }
const char *ds4_think_max_prefix(void) { abort(); }
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    (void)fp; (void)type; (void)fmt;
}
int ds4_engine_model_id(ds4_engine *e) { (void)e; return 5; }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; return 4; }
bool ds4_engine_is_glm_dsa(ds4_engine *e) { (void)e; return false; }
bool ds4_engine_is_qwen4(ds4_engine *e) { (void)e; return true; }
bool ds4_engine_is_deepseek41(ds4_engine *e) { (void)e; return false; }
bool ds4_engine_is_glm53(ds4_engine *e) { (void)e; return false; }
int ds4_session_rewind_floor(ds4_session *s, int pos) { (void)s; return pos; }
uint64_t ds4_session_payload_bytes(ds4_session *s) { (void)s; return 0; }
void ds4_tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}
const char *ds4_token_bytes(ds4_engine *e, int token, size_t *len) {
    return ds4_token_text(e, token, len);
}
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx) {
    (void)ctx; return mode;
}
bool ds4_engine_has_vision(ds4_engine *e) { (void)e; return false; }
void ds4_vision_embedding_free(ds4_vision_embedding *embedding) {
    (void)embedding; abort(); /* No image boundary may execute in this fixture. */
}
const ds4_tokens *ds4_session_tokens(ds4_session *s) { return &s->tokens; }
int ds4_session_pos(ds4_session *s) { return s->tokens.len; }
int ds4_session_ctx(ds4_session *s) { (void)s; return 8192; }
bool ds4_session_checkpoint_valid(const ds4_session *s) { return s->tokens.len > 0; }
bool ds4_session_has_vision_state(const ds4_session *s) { (void)s; return false; }
bool ds4_session_vision_prefix_matches(const ds4_session *s,
                                      const ds4_vision_span *images, size_t n) {
    (void)s; (void)images; return n == 0;
}
bool ds4_session_vision_fingerprint_prefix_matches(const ds4_session *s,
                                                  const ds4_vision_span *images, size_t n) {
    (void)s; (void)images; return n == 0;
}
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *t) {
    int n = 0;
    while (n < s->tokens.len && n < t->len && s->tokens.v[n] == t->v[n]) n++;
    return n;
}
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    char path[] = "/tmp/ds4-test-payload-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    FILE *fp = fdopen(fd, "wb");
    assert(fp);
    bool ok = fwrite(payload, 1, sizeof(payload), fp) == sizeof(payload);
    if (fclose(fp)) ok = false;
    if (!ok) { unlink(path); return -1; }
    out->path = xstrdup(path); out->bytes = sizeof(payload);
    return 0;
}
int ds4_session_write_staged_payload(const ds4_session_payload_file *p,
                                     FILE *fp, char *err, size_t errlen) {
    (void)err; (void)errlen;
    FILE *src = fopen(p->path, "rb");
    if (!src) return -1;
    int c;
    bool ok = true;
    while ((c = fgetc(src)) != EOF) if (fputc(c, fp) == EOF) { ok = false; break; }
    if (ferror(src)) ok = false;
    fclose(src);
    return ok ? 0 : -1;
}
void ds4_session_payload_file_free(ds4_session_payload_file *p) {
    if (p->path) unlink(p->path);
    free(p->path); memset(p, 0, sizeof(*p));
}

#define ROOT "{\"role\":\"system\",\"content\":\"Stable parent\"}," \
             "{\"role\":\"user\",\"content\":\"Run it\"}"
#define CALL "\"tool_calls\":[{\"id\":\"call_test\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{}\"}}]"
#define TURN(reasoning) "{\"role\":\"assistant\",\"content\":\"Running.\"," reasoning CALL "}," \
                        "{\"role\":\"tool\",\"tool_call_id\":\"call_test\",\"content\":\"ok\"}"
static const char schemas[] = "[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{\"type\":\"object\"}}}]";
static const char raw_tool[] = "\n\n<tool_call>\n<function=bash>\n</function>\n</tool_call>";
static const char sampled_suffix[] = "new hidden reasoning\n</think>\n\nRunning.\n\n<tool_call>\n<function=bash>\n</function>\n</tool_call>";
/* Our fork's build_qwen_assistant_suffix() unconditionally closes the turn
 * with <|im_end|>, unlike fc80bd6 at the time this fixture (antirez/ds4#1212)
 * was authored upstream -- unrelated to this port, adjusted so this fixture
 * exercises the real renderer instead of silently matching around it. */
static const char visible_suffix[] = "\n</think>\n\nRunning.\n\n<tool_call>\n<function=bash>\n</function>\n</tool_call><|im_end|>";

static void append_turn(buf *wire, const char *turn) {
    assert(wire->len && wire->ptr[wire->len - 1] == ']');
    wire->ptr[--wire->len] = 0;
    buf_puts(wire, ","); buf_puts(wire, turn); buf_puts(wire, "]");
}
static request request_for(server *s, const char *wire, const char *tools) {
    buf body = {0};
    buf_puts(&body, "{\"reasoning_effort\":\"medium\",\"tools\":");
    buf_puts(&body, tools); buf_puts(&body, ",\"messages\":"); buf_puts(&body, wire);
    buf_puts(&body, "}");
    request r;
    char err[160] = {0};
    assert(parse_chat_request(NULL, s, body.ptr, 128, 8192, &r, err, sizeof(err)));
    buf_free(&body);
    return r;
}
static void set_sampled(ds4_session *session, const char *prefix, const char *tail) {
    ds4_tokens_free(&session->tokens);
    ds4_tokenize_rendered_chat(NULL, prefix, &session->tokens);
    ds4_tokenize_rendered_chat(NULL, tail, &session->tokens);
}
static void record(server *s, server_slot *slot, request *r, bool visible) {
    tool_calls calls = {0};
    tool_calls_push(&calls, (tool_call){ .name = xstrdup("bash"), .arguments = xstrdup("{}") });
    calls.raw_tool_text = xstrdup(raw_tool);
    job j = { .req = *r };
    assert(remember_tool_turn_visible_checkpoint(
        s, slot, &j, "test", "tool_calls", false, "Running.", &calls, visible));
    tool_calls_free(&calls);
}
static ds4_kvstore_entry read_published(server *s) {
    ds4_kvstore_find_text_prefix(&s->kv, "", 5, 4, 8192);
    assert(s->kv.len == 1);
    ds4_kvstore_entry e = {0};
    assert(ds4_kvstore_read_entry_file(s->kv.entry[0].path, s->kv.entry[0].sha, &e));
    TEST_ASSERT(e.model_id == 5 && e.quant_bits == 4 && e.ctx_size == 8192);
    TEST_ASSERT(e.reason == DS4_KVSTORE_REASON_EVICT);
    TEST_ASSERT(e.payload_bytes == sizeof(payload));
    FILE *fp = fopen(e.path, "rb");
    assert(fp && fseek(fp, DS4_KVSTORE_FIXED_HEADER + 4 + (long)e.text_bytes, SEEK_SET) == 0);
    unsigned char bytes[sizeof(payload)] = {0};
    TEST_ASSERT(fread(bytes, 1, sizeof(bytes), fp) == sizeof(bytes));
    TEST_ASSERT(!memcmp(bytes, payload, sizeof(bytes)));
    TEST_ASSERT(fgetc(fp) == EOF);
    fclose(fp);
    return e;
}

/* visible=1 comes from the actual live probe; visible=2 comes from the actual
 * saved entry/matcher. Both retain an older hidden turn the request omits. */
static void storage_case(const char *name, const char *input, bool echo_key, int visible) {
    char dir[] = "/tmp/ds4-test-key-XXXXXX";
    assert(mkdtemp(dir));
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.kv_mu, NULL);
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&s.inference_mu, &attr);
    pthread_mutexattr_destroy(&attr);
    ds4_kvstore_options opt = ds4_kvstore_default_options();
    opt.min_tokens = 1;
    assert(ds4_kvstore_open(&s.kv, dir, 1, true, opt, "test", NULL, NULL));
    ds4_session session = {0};
    server_slot slot = { .session = &session };
    buf wire = {0};
    buf_puts(&wire, input);
    char *hidden_prefix = NULL;
    size_t suffix_off = 0;
    bool selected_visible = false;
    if (visible) {
        request first = request_for(&s, wire.ptr, schemas);
        set_sampled(&session, first.prompt_text, sampled_suffix);
        record(&s, &slot, &first, false);
        hidden_prefix = ds4_kvstore_render_tokens_text(NULL, &session.tokens, NULL);
        append_turn(&wire, TURN("") "," TURN("\"reasoning_content\":\"previous echo\","));
        request next = request_for(&s, wire.ptr, schemas);
        if (visible == 1) {
            slot_reuse probe = slot_probe_reuse(&s, &slot, &next);
            assert(probe.kind == REUSE_THINKING_VISIBLE);
            suffix_off = probe.suffix_off;
            selected_visible = probe.kind == REUSE_THINKING_VISIBLE;
        } else {
            kv_cache_store_current(&s, &slot, "evict", NULL);
            ds4_kvstore_entry e = read_published(&s);
            assert(ds4_kvstore_find_text_prefix(&s.kv, next.prompt_text, 5, 4, 8192) == 0);
            suffix_off = (size_t)e.text_bytes;
            selected_visible = (e.ext_flags & KV_EXT_THINKING_VISIBLE) != 0;
            assert(selected_visible);
            unlink(e.path);
            ds4_kvstore_entry_free(&e);
            ds4_kvstore_clear(&s.kv);
            thinking_live_clear(&s, &slot);
        }
        request_free(&first); request_free(&next);
    }
    request r = request_for(&s, wire.ptr, schemas);
    buf sampled = {0};
    if (visible) {
        buf_puts(&sampled, hidden_prefix);
        buf_puts(&sampled, r.prompt_text + suffix_off);
    } else buf_puts(&sampled, r.prompt_text);
    set_sampled(&session, sampled.ptr, sampled_suffix);
    record(&s, &slot, &r, selected_visible);
    kv_cache_store_current(&s, &slot, "evict", NULL);
    /* Scan through the on-disk inventory, rather than relying on a fabricated
     * entry or a key selected by the fixture. */
    ds4_kvstore_entry e = read_published(&s);
    buf expected = {0};
    buf_puts(&expected, echo_key ? sampled.ptr : r.prompt_text);
    buf_puts(&expected, echo_key ? sampled_suffix : visible_suffix);
    char *stored = xmalloc((size_t)e.text_bytes + 1);
    FILE *fp = fopen(e.path, "rb");
    assert(fp && fseek(fp, DS4_KVSTORE_FIXED_HEADER + 4, SEEK_SET) == 0);
    assert(fread(stored, 1, (size_t)e.text_bytes, fp) == e.text_bytes);
    stored[e.text_bytes] = 0;
    fclose(fp);
    bool key_ok = !strcmp(stored, expected.ptr) &&
                  !!(e.ext_flags & KV_EXT_THINKING_VISIBLE) == !echo_key;
    TEST_ASSERT(key_ok);
    TEST_ASSERT(e.tokens == (uint32_t)session.tokens.len);
    append_turn(&wire, echo_key ? TURN("\"reasoning_content\":\"new hidden reasoning\",") : TURN(""));
    request replay = request_for(&s, wire.ptr, schemas);
    int hit = ds4_kvstore_find_text_prefix(&s.kv, replay.prompt_text, 5, 4, 8192);
    TEST_ASSERT(hit == 0);
    request changed_tools = request_for(&s, wire.ptr, "[]");
    TEST_ASSERT(ds4_kvstore_find_text_prefix(&s.kv, changed_tools.prompt_text, 5, 4, 8192) == -1);
    char *parent = strstr(wire.ptr, "Stable parent");
    assert(parent);
    *parent = 'X';
    request changed = request_for(&s, wire.ptr, schemas);
    TEST_ASSERT(ds4_kvstore_find_text_prefix(&s.kv, changed.prompt_text, 5, 4, 8192) == -1);
    printf("%s: key=%s lookup=%s payload=checked header-changes=MISS\n", name,
           ds4_kvstore_key_kind(e.ext_flags), hit >= 0 ? "HIT" : "MISS");
    request_free(&changed_tools); request_free(&changed); request_free(&replay); request_free(&r);
    free(stored); free(hidden_prefix); buf_free(&sampled); buf_free(&expected);
    buf_free(&wire); visible_live_free(&slot.thinking_live);
    ds4_tokens_free(&session.tokens);
    unlink(e.path); ds4_kvstore_entry_free(&e); ds4_kvstore_close(&s.kv);
    assert(rmdir(dir) == 0);
    pthread_mutex_destroy(&s.tool_mu); pthread_mutex_destroy(&s.kv_mu);
    pthread_mutex_destroy(&s.inference_mu);
}

int main(void) {
    storage_case("known latest echo", "[" ROOT "," TURN("\"reasoning_content\":\"previous echo\",") "]", true, 0);
    storage_case("first tool/schema only", "[" ROOT "]", false, 0);
    storage_case("older echo/latest omit", "[" ROOT "," TURN("\"reasoning_content\":\"older echo\",") "," TURN("") "]", false, 0);
    storage_case("selected visible live", "[" ROOT "]", false, 1);
    storage_case("selected visible disk", "[" ROOT "]", false, 2);
    const char *omitted[] = {
        "[" ROOT "," TURN("\"reasoning_content\":null,") "]",
        "[" ROOT "," TURN("\"reasoning_content\":\"\",") "]",
        "[" ROOT "," TURN("\"reasoning_content\":\" \\t\\r\\n\",") "]",
        ("[{\"role\":\"system\",\"content\":\"Stable parent\",\"reasoning_content\":\"not assistant\"},"
         "{\"role\":\"user\",\"content\":\"Run it\",\"reasoning_content\":\"not assistant\"}]")
    };
    for (size_t i = 0; i < sizeof(omitted) / sizeof(*omitted); i++)
        storage_case("non-echo parser row", omitted[i], false, 0);
    printf("qwen tool disk key: %s (%d failures)\n", test_failures ? "FAIL" : "PASS", test_failures);
    return test_failures ? 1 : 0;
}
