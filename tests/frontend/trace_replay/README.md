<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frontend trace replay

Replays recorded agent trajectories through `dynamo.frontend` with `dynamo.mocker` as the engine. The mocker emits each assistant turn's exact token ids (`--response-replay-trace-path` plus the `output_replay_id` request annotation), so the frontend's decoder, stop handling, reasoning and tool-call parsers, and response assembly run on realistic output. Each reply is compared with what the trajectory recorded.

## Pieces

| Module | Role |
|---|---|
| `fixtures.py` | Offline builder: dataset rows (e.g. `nvidia/Open-SWE-Traces`) to mocker replay rows plus expected turns. Needs `transformers`, `tokenizers`, `huggingface_hub`. |
| `endpoints.py` | Builds `/v1/chat/completions`, `/v1/responses` and `/v1/messages` requests for a case and normalizes streaming or unary replies; stream parsers record protocol violations. |
| `compare.py` | Grades a reply against the trace turn, with tolerance tiers. |
| `runner.py` | Drives a live frontend: trajectories replay their turns in order, concurrently; writes `report.json`, `report.md`, `results.jsonl` and raw replies of mismatches. |

Trajectories store only parsed fields, so the builder rebuilds each turn's raw completion with the teacher's own chat template: the text a turn adds after the generation prompt, up to end-of-turn. It tokenizes that with the teacher tokenizer and checks the round trip. Besides the full turn (ending in EOS), every Nth turn also yields truncated scripts (cut inside reasoning, content or tool-call arguments) and a stop-string case.

## Tolerance tiers

| Tier | Meaning |
|---|---|
| `exact` | reasoning, content, tool names and argument values identical |
| `ws_edges` | reasoning or content differ only in leading or trailing whitespace |
| `ws_args` | a string argument differs only at its edges; reported separately because it can change what a tool does |
| `ws_internal` | reasoning or content equal after collapsing whitespace runs |
| `trunc_ok` | truncated script: partial reply is a prefix of the full turn, leaks no raw markup, ends with `length` |
| `mismatch` | anything else, with categories (`tool_call_count`, `tool_args_value`, `markup_in_content`, `finish_reason`, `replay_tokens`, ...) |

`replay_tokens` means `usage.completion_tokens` does not match the script length, i.e. the mocker did not replay the script (for example because the annotation was stripped and it fell back to random tokens).

## Run

```bash
# Build fixtures (writes model/, trajectories.jsonl, cases.jsonl, replay.jsonl).
python -m tests.frontend.trace_replay build \
  --teacher qwen3.8 \
  --out /data/replay/qwen3.8 \
  --min-turns 1000

# Serve them.
python -m dynamo.frontend --http-port 8000 --discovery-backend file --enable-anthropic-api &
python -m dynamo.mocker \
  --model-path /data/replay/qwen3.8/model \
  --model-name Qwen/Qwen3.8-27B \
  --discovery-backend file \
  --response-replay-trace-path /data/replay/qwen3.8/replay.jsonl \
  --dyn-tool-call-parser qwen3_coder \
  --dyn-reasoning-parser qwen3 \
  --speedup-ratio 1000 \
  --num-gpu-blocks-override 600000 &

# Replay and grade.
python -m tests.frontend.trace_replay run \
  --fixtures /data/replay/qwen3.8 \
  --url http://localhost:8000 \
  --model Qwen/Qwen3.8-27B \
  --out /data/replay/qwen3.8/run \
  --concurrency 16
```

Leave the mocker's `--max-model-len` unset: it caps the script at `max_model_len - prompt_len` and turns full turns into truncated ones. Run the worker without parser flags and pass `--raw` to the runner to check the decoder alone: every reply must then equal the decoded script.

Teachers are defined in `fixtures.TEACHERS` (dataset config, tokenizer repo, chat-template renderer, parser names). `tests/frontend/test_mocker_trace_replay.py` replays the small hand-written fixture in `fixtures/handwritten_qwen35/` in CI.

## Cross-family replay

Only the parsed fields of a trajectory are used, so one family's trajectories can be rendered as another family's output. The target supplies the chat template, tokenizer and parsers; the source supplies the conversation:

```bash
# Qwen3.8 trajectories, replayed as DeepSeek-V4 output (DSML tool calls, DeepSeek tokenizer).
python -m tests.frontend.trace_replay build \
  --teacher deepseek-v4 \
  --source-teacher qwen3.8 \
  --out /data/replay/deepseek-v4@qwen3.8
```

The source's `reasoning_effort` is mapped to the nearest level the target template accepts. This exercises the target's parsers with shapes its own traces may lack, such as parallel tool calls or edge whitespace in arguments. The formatting is canonical for the target, but the behavior (how often it calls tools, what it writes) is the source's, so same-family traces remain the reference.
