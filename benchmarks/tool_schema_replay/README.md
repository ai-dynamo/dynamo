<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tool Schema Replay

Replays tool-calling requests against an OpenAI-compatible endpoint. It checks
the parsed tool-call arguments against each tool's JSON Schema (Draft 2020-12)
and compares them with the raw text the engine generated. Use it to compare two
deployments, for example the default structural-tag schema mode `auto` and
`--dyn-structural-tag '{"schema":"strict"}'`.

The cases in [`cases.py`](cases.py) were written for this replay; they are not
copied from an external suite. They cover three failure patterns seen with tools
declared `strict: false`:

- a required nullable field is missing;
- a scalar has the wrong type;
- a nested structure is wrong.

- **Suite A**: every tool is `strict: false`. It covers the three patterns,
  each as a single tool, among several tools, with parallel calls, and with
  `tool_choice: "required"`.
- **Suite B**: control cases. It has the same tools with `strict` omitted or
  `true`; a regex-lookahead schema and a `$ref` schema, which a grammar backend
  may fail to compile; `tool_choice: "none"`; a request with no tools; a
  question that should get a text answer; and a `/v1/responses` request.

## Run

Start a frontend and one worker, then run each arm against it:

```bash
python3 benchmarks/tool_schema_replay/run.py --base-url http://localhost:8000 \
  --suite A --suite B --repeats 5 --out /tmp/baseline.json
# Restart only the worker with the treatment flag, then:
python3 benchmarks/tool_schema_replay/run.py --base-url http://localhost:8000 \
  --suite A --suite B --repeats 5 --out /tmp/treatment.json
python3 benchmarks/tool_schema_replay/run.py --compare /tmp/baseline.json /tmp/treatment.json
```

Each chat case runs streaming and non-streaming, with `seed` 0 to `repeats - 1`
and the model's default temperature. Each run is classified as `pass`,
`schema_violation`, `invalid_json`, `wrong_tool`, `no_tool_call`,
`unexpected_tool_call`, `empty_text`, `rejected` (non-200 status or stream
error), or `error`. A case passes in an arm only when every run passes.
`--compare` reports which cases were recovered or regressed, and how many
rejections are new.

The runner requests `nvext.extra_fields: ["completion_token_ids"]` and decodes
the token IDs with special tokens kept. For GPT-OSS it extracts each Harmony
call's arguments, or for `required` choice the final-channel JSON array, and
compares them with the parsed `tool_calls`. A `mismatch` means the parser changed
what the model wrote. Pass `--no-raw` to skip this step.

Seeded output is not bit-identical across worker restarts. To measure that
noise, run the baseline arm twice before you attribute a run-level difference
to the treatment.

## Offline hosts

The GPT-OSS Harmony parser loads `o200k_base.tiktoken` from the network. If it
cannot, the frontend logs `Harmony encoding unavailable`, and the raw Harmony
text comes back in `content` with no `tool_calls`. Place the file in a directory
and start the frontend with `TIKTOKEN_ENCODINGS_BASE=<dir>/`. The parser checks
the file's SHA-256.
