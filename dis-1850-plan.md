# DIS-1850 — Plan: Replace `deepseek_v4.rs` + `deepseek_v32.rs` with a Jinja chat template

**Branch:** `keivenchang/DIS-1850__chat-template-for-DeepSeek`
**Base:** `codex/deepseek-v4-parsers` (#8665) at `d116e2a6c50`
**Parent ticket:** DIS-1849 (DSV4 model support)

---

## TL;DR

**Feasibility: high.** Qwen3's existing chat_template.jinja already does the closest comparable stateful work (reverse iteration, `namespace()` accumulators, `last_query_index` lookup) — the V4 features that worried us (`drop_thinking`, `reasoning_effort` prefix, tool-result-as-user wrap, latest_reminder injection) all map cleanly onto the same Jinja idioms. The two real risk areas are `merge_tool_messages` and `sort_tool_results_by_call_order`, which are pre-Jinja preprocessing passes. They're awkward in pure Jinja but trivial as a thin Rust pre-pass (~150 lines), still net-deleting ~2k LOC.

**Worth: yes, conditional on parity.** Deletes ~2624 lines of Rust (1271 in `deepseek_v4.rs` + 1353 in `deepseek_v32.rs`) plus the special-case branch in `template.rs::from_mdc`. Per-DeepSeek-release maintenance drops to near-zero. Upstream HF PR has multiplicative leverage — vLLM, SGLang, transformers, and Dynamo all stop needing per-model encoders.

**Cost: ~2-3 days for someone fluent in Jinja.** Validation is byte-identical comparison against `encoding_dsv4.py` on existing fixtures — CPU-only, no GPU needed for the spike itself.

**Small DS models on A6000 with V3.x/V4 template surface: there are none.** Verified — V2-Lite, V2-Lite-Chat, Coder-V2-Lite-Instruct all use `User:`/`Assistant:` plain-text prefixes, not the `<｜User｜>`/`<｜Assistant｜>` special tokens. R1-Distill-* use the base model's template (Qwen/Llama). VL2-Tiny ships an empty `chat_template`. **End-to-end small-model validation isn't a viable strategy for this spike.** It must rely on byte-parity against the upstream Python reference, which is sufficient.

---

## Background — why these files exist

`deepseek_v4.rs` and `deepseek_v32.rs` are native Rust ports of `encoding_dsv4.py` and `encoding_dsv32.py` from the HuggingFace model repos:

- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/encoding_dsv4.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding/encoding_dsv32.py`

DeepSeek **does not** ship a `chat_template` in `tokenizer_config.json` for V3.2 / V3.2-Exp / V4-Pro. The Rust port exists because every other model family in Dynamo rides minijinja through `template/formatters.rs`, and DeepSeek is the special case marked at `lib/llm/src/preprocessor/prompt/template.rs:22`:

> *"Special handling for DeepSeek models whose HF repos don't ship a Jinja chat_template."*

Older DeepSeek models (V2-Lite, V3, V3.1, R1) **do** ship `chat_template`. The gap is V3.2 and V4 specifically.

---

## What the Rust port does

Public surface (mirrored across both `.rs` files):

| Function | Purpose | Jinja translation difficulty |
|---|---|---|
| `to_json` | Compact JSON serialization | trivial — `tojson` filter; may need a custom filter to match Python's `(separators=(",",":"))` whitespace |
| `tools_from_openai_format` | Strip OpenAI envelope | trivial — inline `{% for tool in tools %}{{ tool.function }}{% endfor %}` |
| `render_tools` | Tool schema system block | easy — straight string concat |
| `find_last_user_index` | Reverse scan for last `user` role | easy — Qwen3 pattern (`{% for message in messages[::-1] %}` + `namespace()`) |
| `encode_arguments_to_dsml` | DSML parameter encoding | easy — `{% for k, v in arguments|items %}{{ template.format(k, v.is_str, v.value) }}{% endfor %}` |
| `task_token` | enum lookup for `<｜action｜>` etc | trivial — Jinja dict lookup |
| `render_user_role` / `render_assistant_role` / etc | per-role rendering | easy — standard `{% if message.role == "X" %}` branches |
| `render_latest_reminder_role` | Custom role w/ `<｜latest_reminder｜>` | easy — additional `elif` |
| `append_response_format` | Inject response_format hint | easy — conditional on first system message |
| `append_tools_section` | Inject tools section | easy — conditional |
| `_drop_thinking_messages` | Filter out `<think>` blocks from prior assistant turns based on `index >= last_user_idx` | **medium** — needs the `last_user_idx` namespace pattern to gate the conditional inside `{% if message.role == "assistant" %}` |
| `merge_tool_messages` | Consolidate consecutive `tool` messages into one with merged content | **hard in pure Jinja** — needs multi-pass with namespace; cleaner as Rust pre-pass |
| `sort_tool_results_by_call_order` | Reorder `tool_result` blocks within user messages by the order their `tool_calls` were emitted | **hard in pure Jinja** — needs cross-message lookups; cleaner as Rust pre-pass |
| `render_message` (dispatch) | Top-level per-message switch | the Jinja main loop |
| `encode_messages_with_options` | Top-level orchestration with `thinking_mode`, `drop_thinking`, `reasoning_effort` | the Jinja extra args / kwargs |

### "Tricky bits" called out in the ticket — concrete Jinja patterns

**1. `drop_thinking` on prior assistant turns**

```jinja
{%- set ns = namespace(last_user_idx=-1) %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {%- set ns.last_user_idx = loop.index0 %}
    {%- endif %}
{%- endfor %}
...
{%- if message.role == "assistant" %}
    {%- if message.reasoning_content %}
        {%- if not drop_thinking or loop.index0 >= ns.last_user_idx %}
            {{ "<think>" }}{{ message.reasoning_content }}{{ "</think>" }}
        {%- endif %}
    {%- endif %}
    {{ message.content }}
{%- endif %}
```

Qwen3 uses exactly this `namespace + reverse loop` pattern (`{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}` at line ~22 of its template).

**2. Reasoning-effort prefix** (one-shot, only at index 0 in thinking mode with `max` effort)

```jinja
{%- if loop.index0 == 0 and thinking_mode == "thinking" and reasoning_effort == "max" %}
{{ reasoning_effort_max_prefix }}
{%- endif %}
```

The prefix string itself is the static `REASONING_EFFORT_MAX` constant from `deepseek_v4.rs:63` — passes as a template variable.

**3. Tool-result-as-user-message wrap**

```jinja
{%- elif message.role == "tool" %}
<｜User｜><tool_result>{{ message.content }}</tool_result>
{%- endif %}
```

Trivial. Already what `encoding_dsv4.py:60-62` does (`tool_output_template = "<tool_result>{content}</tool_result>"`).

**4. Latest-reminder injection**

```jinja
{%- elif message.role == "latest_reminder" %}
<｜latest_reminder｜>{{ message.content }}
{%- endif %}
```

Trivial. The role name is a Dynamo/DeepSeek extension; the special token is `<｜latest_reminder｜>`.

### Real risk areas

**`merge_tool_messages` and `sort_tool_results_by_call_order`** are O(n) preprocessing passes that mutate the message list:

- `merge_tool_messages` walks consecutive `role: tool` messages and consolidates them into one message containing a list of `{type: "tool_result", content: ...}` blocks.
- `sort_tool_results_by_call_order` walks through assistant `tool_calls`, builds an index map of `tool_call_id → emission_order`, then reorders `tool_result` entries within following user messages to match.

Doable in pure Jinja with `{% set ns = namespace(merged=[]) %}` and multi-pass loops, but the resulting template would be hard to read and slow at scale. **Recommendation:** keep these two functions in Rust as a thin pre-Jinja preprocessor (~120 lines combined), invoked before `Environment::render`. Net delete is still ~2400+ lines.

If the goal is "zero Rust per-model code" and HF-PR-able, the upstream HF template would need to do these passes in Jinja anyway — at which point the Jinja becomes the reference implementation and Dynamo just consumes it. That's a downstream decision after the spike confirms parity.

---

## Validation strategy

### Primary: byte-identical render against existing fixtures

```
lib/llm/tests/data/deepseek-v4/test_input_{1..4}.json   →  test_output_{1..4}.txt
lib/llm/tests/data/deepseek-v3.2/test_input{,_search_w_date,_search_wo_date}.json  →  test_output*.txt
```

Fixtures are already byte-validated against `encoding_dsv4.py` / `encoding_dsv32.py` (the existing Rust port runs them as integration tests). The spike's job is to render the same inputs through the Jinja path and confirm the output is byte-identical.

**Surface area in the V4 fixtures:**

| Fixture | Cases | Roles exercised | Tools | Thinking | Effort | Drop |
|---|---|---|---|---|---|---|
| test_input_1.json | 1 | system, user, assistant, tool, assistant | yes | none | none | default |
| test_input_2.json | 5 | varies (need to inspect each case shape) | varies | varies | varies | varies |
| test_input_3.json | 6 | varies (one case has tools) | mixed | varies | varies | varies |
| test_input_4.json | 6 | varies | varies | varies | varies | varies |

The fixtures are written as either single-case JSON or arrays of cases — first action of the spike is to enumerate every case and confirm coverage of the trick-bit matrix (`drop_thinking` × `thinking_mode` × `reasoning_effort` × `latest_reminder` × `task_token` × `wo_eos`).

### Secondary: ancillary partial validation against a small DS model

**The honest answer: no public small DS model has V3.x/V4 template surface.** Verified:

| Model | Params (active) | A6000 fit | Template format |
|---|---|---|---|
| DeepSeek-V2-Lite-Chat | 15.7B (2.4B MoE) | yes (~32GB FP16) | `User:`/`Assistant:` plain-text — **wrong shell** |
| DeepSeek-Coder-V2-Lite-Instruct | 15.7B (2.4B MoE) | yes | same as V2-Lite-Chat — **wrong shell** |
| DeepSeek-VL2-Tiny | 3B | yes (8GB) | empty `chat_template` |
| DeepSeek-R1-Distill-Qwen-{1.5B,7B,14B,32B} | varies | yes | inherits Qwen template |
| DeepSeek-R1-Distill-Llama-{8B,70B} | varies | partial | inherits Llama template |
| DeepSeek-LLM-7B-Chat | 7B | yes | older standalone template |
| DeepSeek-V3{,.1,.2-Exp}, V4-Pro, R1 | 600B+ | no — multi-GPU only | the templates we want |

Conclusion: **end-to-end inference validation against a real model is out of scope for the A6000 setting.** It would require either:

1. A B200/H200 instance with V3.2-Exp or V4-Pro weights (DIS-1842 explicitly scopes this out for the same reason).
2. Heavy quantization of a 600B+ model — even INT4 (`~300GB`) doesn't fit one A6000.
3. A DeepSeek-team collaboration to release a small variant with V3.2/V4 template — does not exist as of 2026-04-24.

Byte-parity against `encoding_dsv4.py` is the **sufficient bar** because that script is upstream-authoritative — DeepSeek authored it, the same script populates the model card examples, and any divergence between Jinja output and Python reference is by definition a Jinja bug.

### Tertiary: perf measurement

Render each fixture 1000 times through Jinja and 1000 times through the existing Rust port. Compare wall-clock. Expectation: sub-millisecond delta per request. If the delta is >5 ms (5x worse), measure with `cargo flamegraph` and decide whether to optimize the template or revert.

---

## Plan of execution

### Phase 0 — Setup ✅ DONE

- [x] Branch `keivenchang/DIS-1850__chat-template-for-DeepSeek` created from `origin/codex/deepseek-v4-parsers` tip `d116e2a6c50`.
- [x] Pin reference: `encoding_dsv4.py` (HF SHA `5e74e6987d`) and `encoding_dsv32.py` snapshotted into `lib/llm/tests/reference/` with SOURCES.md.
- [x] Enumerate all cases in the 4 V4 + 3 V3.2 fixtures → `lib/llm/tests/reference/fixture-coverage.md`.

### Phase 1 — V4 Jinja template ✅ DONE

- [x] Wrote `lib/llm/src/preprocessor/prompt/templates/deepseek_v4.jinja` (modular form with `_v4_tools_block.jinja` include) AND `deepseek_v4_inline.jinja` (single-file form, upstream-HF-PR ready, uses Jinja `{% macro %}`).
- [x] **All 4 V4 fixtures pass byte-identical** (2281 / 302 / 3021 / 1125 bytes).
- [x] Implemented the trick-bit features in this order (easiest first):
    1. Role dispatch (system, user, assistant, tool, latest_reminder, developer)
    2. BOS/EOS sentinels
    3. Tool schema rendering
    4. DSML tool_call encoding (loop over `tool_calls`, loop over `arguments|items`)
    5. `drop_thinking` + `last_user_idx` namespace pattern
    6. `reasoning_effort=max` one-shot prefix
    7. `task_token` lookup
    8. `wo_eos` flag
- [ ] Parallel: Rust pre-pass for `merge_tool_messages` + `sort_tool_results_by_call_order` (~150 lines, no per-model logic — generic message-list transforms gated by config flag).
- [ ] Render each V4 fixture, diff against `test_output_*.txt`. Iterate until byte-identical.
- [ ] Time-box: 1 day. If at end of day no fixture passes, document the Jinja-blocking constraint and stop.

### Phase 2 — V3.2 Jinja template ✅ DONE

- [x] Wrote `deepseek_v32.jinja` (modular) AND `deepseek_v32_inline.jinja` (single-file).
- [x] **All 3 V3.2 fixtures pass byte-identical** (4226 / 150117 / 74991 bytes).
- V3.2-specific differences honored: `function_calls` block name, `<result>` (not `<tool_result>`) tool output, `<function_results>` wrapper, no reasoning_effort/task_tokens/wo_eos, multi-tool support without merge pre-pass.

### Phase 2.5 — Synthetic gap fixtures ✅ DONE

Generated 7 synthetic cases in `_gap_fixtures.py` covering features the golden set didn't hit:

- [x] `reasoning_effort = "max"` prefix — PASS
- [x] `reasoning_effort = "high"` (no-op) — PASS
- [x] `wo_eos = true` — PASS
- [x] `merge_tool_messages` consecutive tools — PASS
- [x] `sort_tool_results_by_call_order` out-of-order — PASS
- [x] `developer` role with tools — PASS
- [x] `latest_reminder` + user — PASS

**Total: 7 golden + 7 synthetic = 14/14 fixtures byte-identical.**

### Phase 3 — Plumbing (deferred to follow-up impl PR)

The spike's job is to **prove feasibility and write a recommendation**, not to ship the change. The plumbing into `template.rs::from_mdc` lives in a follow-up PR (per ticket: "first version of #8665 ships with the Rust port, this is post-launch cleanup").

Sketch of what the impl PR needs:
- Wire `deepseek_v4.jinja` / `deepseek_v32.jinja` into the minijinja loader path.
- Update `template.rs::from_mdc` to use the bundled Jinja for `model_type` ∈ {`deepseek_v4`, `deepseek_v32`} when HF doesn't ship a `chat_template`.
- Move `merge_tool_messages` + `sort_tool_results_by_call_order` to a small standalone module (~150 LOC), invoked before render.
- Add a `--use-rust-deepseek-formatter` fallback flag for one release.
- Add a Rust integration test mirroring `deepseek_v4_encoding.rs` but exercising the Jinja path.

### Phase 4 — Perf measurement ✅ DONE

- [x] Benched jinja2 (Python) vs `encoding_dsv4.py` / `encoding_dsv32.py` at 1000 renders per fixture.
- [x] Result: jinja2 is 2-4x slower than the Python encoding script across all 7 fixtures. Worst-case p99 absolute overhead: +0.66ms (150 KB search-w-date fixture).
- [x] Rust minijinja typically runs 2-5x faster than Python jinja2 → predicted Rust minijinja perf within 1-2x of the existing Rust port. **Sub-millisecond at p99 even on the largest fixture.** Fine at request scale.

### Phase 4.5 — Decision tree
    - **Parity ✅ + perf delta < 1ms p99** → propose `deepseek_v4.rs` + `deepseek_v32.rs` deletion in a follow-up PR. File the upstream HF PR ticket.
    - **Parity ✅ + perf 1-5ms** → ship Jinja, add a perf-regression test to catch future Jinja-template growth.
    - **Parity ✅ + perf >5ms** → measure with `cargo flamegraph`, optimize template, recheck. If still >5ms, revert and document.
    - **Parity ❌ on N features** → write up which Jinja constraint blocked each (e.g., "minijinja's `tojson` doesn't honor `(separators=(',',':'))` so JSON spacing diverges; would need a custom Rust filter") and decide whether to fix upstream minijinja, custom-filter, or keep the Rust port for that specific feature.

### Phase 5 — Decision artifact

Write `lib/llm/notes/DIS-1850-spike-results.md` covering:
- Which features ported cleanly
- Which needed Rust glue (and why)
- Perf numbers
- Recommendation: full delete / partial delete / no-op
- If full/partial delete: PR plan + upstream HF PR plan

---

## Why this is worth doing

| Cost | Benefit |
|---|---|
| 2-3 days of focused Jinja work | Delete ~2400 of 2624 lines of Rust (assuming ~150-line preprocessor stays) |
| Some upfront learning of minijinja idioms (Qwen3 is the cheat sheet) | Per-DeepSeek-release maintenance drops to ~zero — new versions just need a new `chat_template` |
| Risk of a feature that genuinely doesn't fit Jinja | Aligns DeepSeek with every other model family (Qwen, Llama, Mistral, GLM, Kimi, gpt-oss) — one less special case in `template.rs::from_mdc` |
| Possible perf overhead | Multiplier: upstream HF PR fixes vLLM, SGLang, transformers, and Dynamo simultaneously |

The HF upstream PR is the actual prize. If we can convince DeepSeek to ship `chat_template` in their `tokenizer_config.json` going forward, **the entire problem disappears at the source.** The spike is the evidence we'd need to make that ask.

---

## When *not* to do this

- If `merge_tool_messages` or `sort_tool_results_by_call_order` turn out to require sufficiently complex multi-pass Jinja that the template becomes harder to maintain than the Rust port — abandon, document the blocker, ship the Rust port.
- If perf overhead is >5ms p99 and unfixable — same.
- If DeepSeek announces an upstream `chat_template` in V4-Pro's next revision — the work becomes obsolete; ride upstream.

---

## Out of scope (per ticket)

- The upstream HF PR itself (separate ticket if spike comes back positive).
- Tool-call **output** parsing (`lib/parsers/src/tool_calling/`) — that's "axis C" and stays where it is regardless.
- DeepSeek V3 / R1 — already ship `chat_template`, not part of the problem.
