#!/usr/bin/env python3
"""Live check for the visible-key tier across transient client metadata.

A turn that ends without tool calls remembers a visible key for the next
request.  The key is the current transcript plus the assistant turn, normalized
to transient-stripped space, because agent clients wrap per-turn metadata in
<environment_details> / <system-reminder> spans and differ in whether they
replay those spans.

The key used to be stored stripped while the probe compared the incoming
prompt in raw space.  A client that replays the spans verbatim -- the DSH
client does -- then never matched: the key held the stripped bytes, the request
held the block, and the comparison broke at the first removed span.  The next
request fell through every tier and re-prefilled the whole conversation even
though the live checkpoint was intact (live=163524 prompt=163635 common=135132
reason=token-mismatch).

The exact token-prefix tiers cannot cover that case.  Once the history holds a
tool-call turn, the live frontier's tokens are the model's own segmentation and
stop matching the tokenization of the re-rendered prompt: the server reports
``common`` frozen at the tool turn's frontier while ``live`` has moved on, so
``reason=token-mismatch``.  Every continuation after that depends on the
visible-key tier.

So each conversation below drives a tool turn, then a turn that ends without
tool calls, then a follow-up that must reuse the live prefix.  Two shapes:

  block     the client replays the <system-reminder> span verbatim
  plain     no transient span at all (the key must still be kept)

Requires an idle server:

    ./ds4-server -m <model>.gguf --metal -c 8192 > /tmp/ds4-transient.log 2>&1

    python3 tests/test_server_transient_stop_boundary.py \
        --url http://127.0.0.1:8055 --log /tmp/ds4-transient.log

The assertions read usage counters and the server log, not generated text.
"""

import argparse
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Look up the weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

SYSTEM = "Call the get_weather tool when asked. Otherwise answer in text."

BLOCK = ("<system-reminder>workspace state: branch main, 3 files changed"
         "</system-reminder>")

ASK_TOOL = "Call get_weather for Paris."
FOLLOW_UP = "Now what is 2 + 2? Answer with just the number."


def post(url, path, body, timeout=900):
    request = urllib.request.Request(url.rstrip("/") + path,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw}


def cached_tokens(usage):
    details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" in details:
        return details["cached_tokens"]
    return usage.get("cache_read_input_tokens", 0) or 0


def chat(args, messages, label):
    status, reply = post(args.url, "/v1/chat/completions", {
        "model": args.model, "messages": messages, "max_tokens": 256,
        "stream": False, "tools": TOOLS})
    assert status == 200, (label, status, reply)
    choice = reply["choices"][0]
    usage = reply["usage"]
    print("  %-14s prompt=%-6d cached=%-6d out=%-4d total=%-6d finish=%s" % (
        label, usage["prompt_tokens"], cached_tokens(usage),
        usage["completion_tokens"], usage["total_tokens"],
        choice.get("finish_reason")), flush=True)
    return choice["message"], choice.get("finish_reason"), usage


def replay_assistant(message):
    """What an agent client sends back: content, reasoning and any tool call."""
    replayed = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        replayed["tool_calls"] = message["tool_calls"]
    if message.get("reasoning_content"):
        replayed["reasoning_content"] = message["reasoning_content"]
    return replayed


def log_since(path, offset):
    if not path:
        return ""
    return Path(path).read_text(errors="replace")[offset:]


def log_size(path):
    if not path:
        return 0
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def run_conversation(args, label, block):
    """Return True when the stop-finishing turn's continuation reused the live
    prefix.  Returns None when the model did not produce the expected shape, so
    the caller reports the shape as uncovered instead of passing."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ASK_TOOL + (block if block else "")}]

    answer, finish, _ = chat(args, messages, label + "-tool")
    calls = answer.get("tool_calls") or []
    if finish != "tool_calls" or not calls:
        print("    note: no tool call (finish=%r); shape not covered" % finish,
              flush=True)
        return None
    messages = messages + [replay_assistant(answer),
                           {"role": "tool", "tool_call_id": calls[0]["id"],
                            "content": "sunny, 21C"}]

    mark = log_size(args.log)
    answer2, finish2, usage2 = chat(args, messages, label + "-answer")
    if finish2 != "stop":
        print("    note: answer turn finished with %r, not a plain stop"
              % finish2, flush=True)
        return None
    frontier = usage2["total_tokens"]
    remembered = re.findall(r"transient live checkpoint remembered[^\n]*",
                            log_since(args.log, mark))
    assert remembered, (
        "%s: the stop turn did not arm a visible key:\n  %s"
        % (label, log_since(args.log, mark)[-2000:]))

    messages = messages + [replay_assistant(answer2),
                           {"role": "user", "content": FOLLOW_UP}]
    mark = log_size(args.log)
    _, _, usage3 = chat(args, messages, label + "-followup")
    tail = log_since(args.log, mark)
    miss = re.findall(r"live kv cache miss[^\n]*", tail)
    assert not miss, ("%s: the continuation re-prefilled instead of reusing "
                      "the live prefix:\n  %s" % (label, "\n  ".join(miss)))
    cached = cached_tokens(usage3)
    assert cached >= frontier, (
        "%s: continuation reused %d of the %d-token frontier"
        % (label, cached, frontier))
    # Which tier served it is informative, not a gate: the visible key is the
    # one this path exists for, but an exact token hit is also a correct reuse.
    tier = ("visible" if "thinking live continuation match=visible-prefix" in tail
            else "token")
    print("    reused %d of the %d-token frontier (tier=%s)" % (
        cached, frontier, tier), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8055")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--log", type=Path,
                        help="server stderr log, for the tier assertions")
    args = parser.parse_args()

    covered = []
    for label, block in (("block", BLOCK), ("plain", "")):
        if run_conversation(args, label, block) is not None:
            covered.append(label)

    assert covered, "no conversation produced the tool-then-stop shape"
    print("PASS: the continuation reused the live prefix for: %s"
          % ", ".join(covered), flush=True)


if __name__ == "__main__":
    main()
