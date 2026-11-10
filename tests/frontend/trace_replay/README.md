<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental frontend trace replay

Replay recorded requests through the installed `dynamo.frontend`, observe the actual `PreprocessedRequest` at a registered CPU token worker, and inject recorded raw worker output before frontend decoding and parsing. This exercises the real HTTP, renderer, tokenizer, stop filter, reasoning/tool parser and response aggregation paths without model weights, GPUs or an inference engine. The initial public corpus contains five DeepSeek V4 BFCL captures; the same runner accepts a pinned DeepSeek V4.1 profile.

The checks establish the covered frontend contracts, not model quality or backend correctness. They do not exercise real sampling, engine grammar enforcement, GPU kernels, multimodal processing or distributed KV transfer. Whole-output and one-token schedules are synthetic regroupings of real generated IDs; they make no timing claim.

## Run

Use the target image's Python interpreter. The helper checkout must supply `tests/` without putting `components/src` on `PYTHONPATH`, installing the checkout, or replacing the target package. The runner uses existing `ManagedProcess` and port/readiness helpers; helper-only archives must include `tests/__init__.py`, `tests/frontend/__init__.py`, this package, and `tests/utils/{managed_process,constants,http_checks,output_paths,port_utils}.py`.

```bash
python -m tests.frontend.trace_replay prepare-assets \
  --corpus tests/frontend/trace_replay/fixtures/deepseek_v4 \
  --model-dir /tmp/replay-model

python -m tests.frontend.trace_replay run \
  --corpus tests/frontend/trace_replay/fixtures/deepseek_v4 \
  --model-dir /tmp/replay-model \
  --output-dir /tmp/replay-results-unique \
  --modes unary,sse --schedules whole,one_token --concurrency 4
```

Choose unique absolute model/output paths for real runs. `prepare-assets` downloads only the exact, hash-verified model files declared at the manifest's immutable revision: tokenizer/config files and, when shipped, generation config. V4 requires all four files; V4.1 permits its native three-file bundle without generation config. It uses `HF_TOKEN` from the environment when supplied. `run` performs no downloads and verifies every file again. The frontend and worker use file discovery, loopback TCP and ZMQ, so they require neither etcd nor NATS. The source HF model may be large, but no weight file is requested or loaded.

The target environment needs its installed `ai-dynamo`/`ai-dynamo-runtime` and frontend dependencies (including `uvloop`). The replay helpers additionally use `requests` and `psutil`, already present in Dynamo test images. No stock vLLM/SGLang package, reference renderer or Rust compilation is needed at replay time. The runner itself does not import Dynamo; only its worker subprocess imports the target bindings. A frontend import from `components/src` is refused.

The default public run performs five cases × two modes × two schedules = 20 executions. `--limit N` selects the first N corpus records, while still validating all shard hashes, records and declared counts. Omit the limit for the complete corpus. Supported schedules are `whole`, `one_token`, `fixed32` and `recorded`; `recorded` preserves the supplied worker frames. A corpus argument can be a manifest or a directory containing `manifest.json`; `.jsonl.gz` shards are read transparently and their compressed bytes are hashed.

## Corpus contract

Manifest schema is `dynamo-frontend-replay/v1`:

```json
{"schema":"dynamo-frontend-replay/v1","profile":{"name":"deepseek-v4","model_name":"deepseek-v4-flash","model":{"repo_id":"org/model","revision":"<40-character commit>","assets":{"config.json":"<sha256>","generation_config.json":"<sha256>","tokenizer.json":"<sha256>","tokenizer_config.json":"<sha256>"}},"runtime_config":{"tool_call_parser":"deepseek_v4","reasoning_parser":"deepseek_v4","context_length":32768}},"traces":[{"path":"cases.jsonl","sha256":"<sha256>","count":5}]}
```

Each JSONL object has `id`, `source`, `request`, `replay.worker_events` and `expect.preprocessed`. `source.kind` is required and is exactly one of `benchmark`, `production`, `custom_report`, `agent_report`, `authored`. `source.context` is an optional object; its optional `ref`, `pr`, `issue` values are strings. Both `{"kind":"authored"}` and `{"kind":"benchmark","context":{}}` are valid. No URL or nonempty-string constraint is imposed on context values.

`expect.preprocessed.token_ids` is the independently established prompt. `expect.preprocessed.fields_equal` maps JSON Pointers to specific effective worker fields, such as `/stop_conditions/max_tokens`. Missing values differ from null. The whole observed IR is retained even when a field has no independent expectation.

`replay.worker_events` holds raw token-output frames, not parsed API responses. Every frame has `token_ids`, and the final frame carries the sole `stop` or `length` finish. Keep terminal EOS IDs in that frame. The worker leaves `tokens` and `text` null for frontend detokenization. A stock engine's parsed API `tool_calls` finish is normalized to worker `stop` only when the capture's EOS evidence establishes that mapping; the frontend must independently produce `tool_calls` again. These five captures have EOS 1, worker finish `stop` and null stop reason. The worker never injects captured usage as the answer to an accounting check.

Optional `expect.response` contains `http_status`, a canonical assistant `message`, `finish_reason` and independently expected usage fields. Canonical tool calls contain `type:function` and `function:{name,arguments}`, with typed JSON arguments rather than generated call IDs. `reference.message` is retained as diagnostic provenance, never promoted to a golden. A missing response expectation is reported as `unjudged` even if consistency checks pass.

The reviewed V4 goldens use the pinned vendor encoder, not candidate Dynamo output or unreviewed peer parsing. In particular, the vendor format owns the two newlines before a DSML call block, so those newlines are absent from the two parallel-call content goldens. Reference and capture provenance are recorded in the fixture manifest and [NOTICE.md](fixtures/NOTICE.md).

## Judgments and results

Prompt IDs and explicitly declared effective controls are compared at worker ingress. Response content, reasoning, tool names/order, typed arguments, finish reason and selected usage fields are judged independently. JSON number values are compared losslessly (`1` equals `1.0`); booleans and strings remain distinct. Object-key order and argument JSON formatting are irrelevant, while string whitespace and array order are preserved. Content null differs from an empty string. The optional wire `reasoning_content` and `tool_calls` fields normalize absent/null to no reasoning/no calls; emitting the wrong `reasoning` key is rejected. Generated tool IDs must be present, unique and stable across streaming deltas, but independent executions may choose different IDs.

Every case must also preserve its semantic IR/response across selected modes and chunk groupings. SSE is accumulated independently and checked for one terminal per choice, contiguous tool indexes, valid JSON arguments, stable IDs and a completed `[DONE]` event. Source request controls used only for capture are removed explicitly and listed with the actual sent request in each wire artifact.

Outputs are `summary.json`, `junit.xml`, `observations.jsonl`, complete `wire/` replies, `worker/` ingress/emission observations and process `logs/`. Output files use private permissions; choose a fresh output directory. Summary identity includes corpus/profile/asset hashes, selected IDs, schedules, modes and the runner policy/code fingerprint. Build identity records actual installed package paths/version and binding hash. `DYN_REPLAY_HARNESS_REVISION` and `DYN_REPLAY_IMAGE_DIGEST` add the published harness/image identities supplied by CI.

Use `--baseline <previous-run-directory>` to compare identical corpus, policy and delivery identities. It distinguishes new golden failures, fixes, unchanged failures and unjudged behavior changes. Missing baselines are `baseline_unavailable`; changed harness/corpus identities are `baseline_not_comparable`, not regressions. The baseline excludes only generated replay annotations, model-card checksums and selected worker instance IDs from IR comparisons; it excludes generated response IDs/timestamps and chunk-estimated reasoning-token accounting from semantic response comparisons.

Exit codes: **0** means the selected declared checks passed without new baseline attention; **1** means product failures or unjudged changes requiring review; **2** means invalid corpus/assets, missing dependencies, failed startup or incomplete replay instrumentation. Zero does not turn unjudged responses into verified answers. Live replay is experimental and runs outside required PR/release checks; results and coverage remain visible even when a nightly is nonblocking.

## Tests

```bash
python -m pytest tests/frontend/test_trace_replay_contract.py -q
DYN_RUN_FRONTEND_REPLAY=1 DYN_REPLAY_MODEL_DIR=/absolute/model/bundle \
  python -m pytest tests/frontend/test_trace_replay.py -q
```

The contract tests cover provenance validation, lossless typed comparisons, missing-versus-null JSON Pointers, gzip/count/limit handling, planted prompt/argument failures, EOS-preserving schedules, malformed SSE and baseline classification. The live test is skipped unless explicitly enabled and is excluded from the existing nightly selector. Once enabled, it requires the verified native bundle; unavailable assets are a failure. Its initial timeout is 300 seconds and should be tuned from measured execution times.
