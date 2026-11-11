<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental frontend trace replay

Replay recorded requests through the installed `dynamo.frontend`, observe the actual `PreprocessedRequest` at a registered CPU token worker, and inject recorded raw worker output before frontend decoding and parsing. This exercises the real HTTP, renderer, tokenizer, stop filter, reasoning/tool parser and response aggregation paths without model weights, GPUs or an inference engine. The public corpus contains five independently reviewed DeepSeek V4.1 Flash BFCL captures.

The checks establish the covered frontend contracts, not model quality or backend correctness. They do not exercise real sampling, engine grammar enforcement, GPU kernels, multimodal processing or distributed KV transfer. Whole-output and one-token schedules are synthetic regroupings of real generated IDs; they make no timing claim.

## Run

Use the target image's Python interpreter. The helper checkout must supply `tests/` without putting `components/src` on `PYTHONPATH`, installing the checkout, or replacing the target package. The runner uses existing `ManagedProcess` and port/readiness helpers; helper-only archives must include `tests/__init__.py`, `tests/frontend/__init__.py`, this package, and `tests/utils/{managed_process,constants,http_checks,output_paths,port_utils}.py`.

```bash
python -m tests.frontend.trace_replay prepare-assets \
  --corpus tests/frontend/trace_replay/fixtures/deepseek_v41 \
  --model-dir /tmp/replay-model

python -m tests.frontend.trace_replay run \
  --corpus tests/frontend/trace_replay/fixtures/deepseek_v41 \
  --model-dir /tmp/replay-model \
  --output-dir /tmp/replay-results-unique \
  --modes unary,sse --schedules whole,one_token --concurrency 4
```

Choose unique absolute model/output paths for real runs. `prepare-assets` downloads only the exact, hash-verified model files declared at the manifest's immutable revision: tokenizer/config files and, when shipped, generation config. V4 requires all four files; V4.1 permits its native three-file bundle without generation config. It uses `HF_TOKEN` from the environment when supplied. `run` performs no downloads and verifies every file again. The frontend and worker use file discovery, loopback TCP and ZMQ, so they require neither etcd nor NATS. The source HF model may be large, but no weight file is requested or loaded.

The target environment needs its installed `ai-dynamo`/`ai-dynamo-runtime` and frontend dependencies (including `uvloop`). The replay helpers additionally use `requests` and `psutil`, already present in Dynamo test images. No stock vLLM/SGLang package, reference renderer or Rust compilation is needed at replay time. The runner itself does not import Dynamo; only its worker subprocess imports the target bindings. A frontend import from `components/src` is refused.

The subprocess environment drops inherited `DYN_*` controls and supplies only the replay's discovery/transport settings. Frontend migration is explicitly disabled with `--migration-limit 0`. The report's `build.execution` records both commands and the runner-owned environment values; inherited credentials are excluded. CI metadata is read separately from the parent environment.

The default public run performs five cases × two modes × two schedules = 20 executions. `--limit N` selects the first N corpus records, while still validating all shard hashes, records and declared counts. Omit the limit for the complete corpus. Supported schedules are `whole`, `one_token`, `fixed32` and `recorded`; `recorded` preserves the supplied worker frames. A corpus argument can be a manifest or a directory containing `manifest.json`; `.jsonl.gz` shards are read transparently and their compressed bytes are hashed.

## Capture and prepare fixtures

Capture and preparation run separately from replay. Capture calls an **already running** reference server with `/version`, `/v1/models`, and `/v1/chat/completions`. It requires a response that returns `prompt_token_ids` and `choices[0].token_ids` when `return_token_ids` is true. It does not deploy an engine. Preparation uses a pinned vendor reference on CPU; replay does not import that reference or require its Python dependencies.

Start with prepared JSONL requests, each containing `id`, `source` and `request`, with optional diagnostic `reference`. Extract the public five without copying their expectations:

```bash
REPLAY_WORK=$(mktemp -d)
python - "$REPLAY_WORK/requests.jsonl" <<'PYTHON'
import sys
from pathlib import Path
from tests.frontend.trace_replay.contract import dumps, loads

source = Path("tests/frontend/trace_replay/fixtures/deepseek_v41/cases.jsonl")
with Path(sys.argv[1]).open("w", encoding="utf-8") as destination:
    for line in source.read_text(encoding="utf-8").splitlines():
        record = loads(line)
        destination.write(dumps({key: record[key] for key in ("id", "source", "request")}) + "\n")
PYTHON

python -m tests.frontend.trace_replay.capture \
  --requests "$REPLAY_WORK/requests.jsonl" \
  --profile tests/frontend/trace_replay/fixtures/deepseek_v41/manifest.json \
  --url http://127.0.0.1:8000 --output "$REPLAY_WORK/capture"
```

Each request must explicitly name the profile's model alias. The tool preflights the complete seed set before sending requests, uses unary transport and requests token IDs, and refuses conflicting capture selectors. Choose a fresh output directory: this MVP does not resume captures. Optional authentication uses `--api-key-env ENVIRONMENT_VARIABLE_NAME`; the token is excluded from artifacts and requires HTTPS outside loopback. Environment proxies and implicit credentials are disabled, and redirects fail.

`capture/requests.jsonl` preserves exact seed bytes. Hash-named records preserve the original request, exact sent bytes, exact received bytes, lossless parsed JSON and their hashes. `capture.json` is written last and identifies complete coverage or per-case failures. A partial capture exits nonzero and cannot be prepared as a smaller passing corpus. Reported server metadata is diagnostic; a declared model pin does not prove what weights the server loaded. Captured API replies and API finish reasons are not native Dynamo worker observations.

Prepare the native model assets as described above. Download the official encoder separately at the manifest's pinned revision; preparation checks its hash before executing its exact verified bytes:

```bash
curl --fail --location \
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/encoding/encoding.py \
  --output "$REPLAY_WORK/encoding.py"

uv run --no-project --python 3.12 --with tokenizers==0.21.4 \
  python -m tests.frontend.trace_replay.prepare \
  --profile tests/frontend/trace_replay/fixtures/deepseek_v41/manifest.json \
  --model-dir /absolute/verified/native/assets --encoder "$REPLAY_WORK/encoding.py" \
  --capture-dir "$REPLAY_WORK/capture" --output "$REPLAY_WORK/corpus"
```

The optional preparation dependency is `tokenizers==0.21.4`. The profile may be an existing corpus manifest, using `provenance.golden_review.encoder` and its sibling `tokenizers_version`, or a small document with `schema:dynamo-frontend-replay/v1`, `profile` and `reference`. The reference descriptor carries `repo_id`, `revision`, `path`, `sha256` and `tokenizers_version`; it must bind to the profile's immutable model revision. All native assets are checked before reference execution. Unsupported rendering inputs fail explicitly rather than acquiring guessed expectations.

Preparation independently renders and tokenizes each prompt with `add_special_tokens=False`, then requires exact equality with captured prompt IDs. It verifies request/response hashes and usage, retains EOS in output IDs, and derives worker termination from verified model assets and capture evidence. It refuses ambiguous terminal evidence and unsupported stop controls. The exported request retains its actual input shape. Array-valued reasoning is supported only with captured prompt evidence that verifies the documented reference adaptation; reference-only mode refuses that extension and array content.

Preparation declares these worker-ingress checks. It rejects both cap aliases in one request and requires `n` to be one when supplied.

| Input | Expected IR pointer |
| --- | --- |
| `temperature` | `/sampling_options/temperature` |
| `top_p` | `/sampling_options/top_p` |
| `seed` | `/sampling_options/seed` |
| `n` | `/sampling_options/n` |
| `max_tokens` or `max_completion_tokens` | `/stop_conditions/max_tokens` |
| EOS from verified config/tokenizer assets | `/eos_token_ids` and `/stop_conditions/stop_token_ids_hidden` |

Rendering inputs use explicit `chat_template_kwargs.thinking_mode` (`chat` or `thinking`), with optional consistent `thinking` and `reasoning_effort` in that same object. The supported protocol is text Chat Completions with OpenAI function tools and `tool_choice:auto`. Top-level effort, `chat_template_args`, response-format controls, parallel-tool controls, message names/prefixes, multimodal blocks and undeclared fields require a future explicit adapter. Seed sampling values remain exact JSON values.

The new corpus contains deterministic `cases.jsonl` and a checksummed manifest. Identical frozen evidence and tool code produce identical exports across directories. Fresh model inference may produce different tokens. Captured parsed responses stay diagnostic, so new captures have unjudged responses by default. `--reviewed-corpus <approved-corpus>` may reuse an explicitly approved response only when its model/reference, request, prompt IDs, output IDs and terminal binding match. Supplying expectations in request seeds never approves them.

For a renderer check without inference, replace `--capture-dir` with `--requests`:

```bash
uv run --no-project --python 3.12 --with tokenizers==0.21.4 \
  python -m tests.frontend.trace_replay.prepare \
  --profile tests/frontend/trace_replay/fixtures/deepseek_v41/manifest.json \
  --model-dir /absolute/verified/native/assets --encoder "$REPLAY_WORK/encoding.py" \
  --requests "$REPLAY_WORK/requests.jsonl" --output "$REPLAY_WORK/renderer-corpus"

python - "$REPLAY_WORK/renderer-corpus" <<'PYTHON'
import sys
from pathlib import Path
from tests.frontend.trace_replay.contract import load_corpus

approved = load_corpus(Path("tests/frontend/trace_replay/fixtures/deepseek_v41"))
prepared = load_corpus(Path(sys.argv[1]))
expected = {case.identifier: case.expected_preprocessed["token_ids"] for case in approved.cases}
actual = {case.identifier: case.expected_preprocessed["token_ids"] for case in prepared.cases}
if actual != expected:
    raise SystemExit("prepared prompts differ from the five independently verified fixtures")
print("All five prepared prompt ID arrays match")
PYTHON
```

This independently reproduces the five prompts and emits a vendor-rendered synthetic completion that fits the request cap, marked synthetic in each record and the manifest. It does not capture model inference or approve response semantics. Run either exported corpus with the installed target frontend using the existing `run` command. Unary/SSE and rechunking are execution choices, not corpus preparation choices.

## Corpus contract

Manifest schema is `dynamo-frontend-replay/v1`:

```json
{"schema":"dynamo-frontend-replay/v1","profile":{"name":"deepseek-v41","model_name":"deepseek-v41-flash","model":{"repo_id":"deepseek-ai/DeepSeek-V4.1-Flash","revision":"<40-character commit>","assets":{"config.json":"<sha256>","tokenizer.json":"<sha256>","tokenizer_config.json":"<sha256>"}},"runtime_config":{"tool_call_parser":"deepseek_v41","reasoning_parser":"deepseek_v41","context_length":32768}},"traces":[{"path":"cases.jsonl","sha256":"<sha256>","count":5}]}
```

Each JSONL object has `id`, `source`, `request`, `replay.worker_events` and `expect.preprocessed`. `source.kind` is required and is exactly one of `benchmark`, `production`, `custom_report`, `agent_report`, `authored`. `source.context` is an optional object; its optional `ref`, `pr`, `issue` values are strings. Both `{"kind":"authored"}` and `{"kind":"benchmark","context":{}}` are valid. No URL or nonempty-string constraint is imposed on context values.

`expect.preprocessed.token_ids` is the independently established prompt. `expect.preprocessed.fields_equal` maps JSON Pointers to specific effective worker fields, such as `/stop_conditions/max_tokens`. Missing values differ from null. The whole observed IR is retained even when a field has no independent expectation.

`replay.worker_events` holds raw token-output frames, not parsed API responses. Every frame has `token_ids`, and the final frame carries the sole `stop` or `length` finish. Keep terminal EOS IDs in that frame. The worker leaves `tokens` and `text` null for frontend detokenization. A stock engine's parsed API `tool_calls` finish is normalized to worker `stop` only when the capture's EOS evidence establishes that mapping; the frontend must independently produce `tool_calls` again. These five captures have EOS 1, worker finish `stop` and null stop reason. The worker never injects captured usage as the answer to an accounting check.

Optional `expect.response` contains `http_status`, a canonical assistant `message`, `finish_reason` and independently expected usage fields. Canonical tool calls contain `type:function` and `function:{name,arguments}`, with typed JSON arguments rather than generated call IDs. `reference.message` is retained as diagnostic provenance, never promoted to a golden. A missing response expectation is reported as `unjudged` even if consistency checks pass.

The reviewed V4.1 goldens use the pinned vendor encoder, not candidate Dynamo output or unreviewed peer parsing. The five original JSONL records are preserved byte-for-byte. The vendor format owns the two newlines before a DSML call block, so those structural newlines are excluded from visible content. A five-record prompt-audit receipt and approved request/prompt/output identities accompany the fixtures. Its request-set hash explicitly identifies the full upstream capture set, while its record count covers only these five cases. Reference and capture provenance are recorded in the fixture manifest and [NOTICE.md](fixtures/NOTICE.md).

## Judgments and results

Prompt IDs and explicitly declared effective controls are compared at worker ingress. Response content, reasoning, tool names/order, typed arguments, finish reason and selected usage fields are judged independently. JSON number values are compared losslessly (`1` equals `1.0`); booleans and strings remain distinct. Object-key order and argument JSON formatting are irrelevant, while string whitespace and array order are preserved. Content null differs from an empty string. The optional wire `reasoning_content` and `tool_calls` fields normalize absent/null to no reasoning/no calls; emitting the wrong `reasoning` key is rejected. Generated tool IDs must be present, unique and stable across streaming deltas, but independent executions may choose different IDs.

Every case must also preserve its semantic IR/response across selected modes and chunk groupings. SSE is accumulated independently and checked for one terminal per choice, contiguous tool indexes, valid JSON arguments, stable IDs and a completed `[DONE]` event. Source request controls used only for capture are removed explicitly and listed with the actual sent request in each wire artifact.

Outputs are `summary.json`, `junit.xml`, `observations.jsonl`, complete `wire/` replies, `worker/` ingress/emission observations and process `logs/`. Output files use private permissions; choose a fresh output directory. Summary identity includes corpus/profile/asset hashes, selected IDs, schedules, modes and the runner policy/code fingerprint. Build identity records actual installed package paths/version and binding hash. `DYN_REPLAY_HARNESS_REVISION` and `DYN_REPLAY_IMAGE_DIGEST` add the published harness/image identities supplied by CI.

Use `--baseline <previous-run-directory>` to compare identical corpus, policy and delivery identities. It distinguishes new golden failures, fixes, unchanged failures and unjudged behavior changes. Missing baselines are `baseline_unavailable`; changed harness/corpus identities are `baseline_not_comparable`, not regressions. A previous infrastructure failure or missing execution rows is `baseline_incomplete` and produces no product-review attention. Individual infrastructure observations are `infrastructure_unjudged`. The baseline excludes only generated replay annotations, model-card checksums and selected worker instance IDs from IR comparisons; it excludes generated response IDs/timestamps and chunk-estimated reasoning-token accounting from semantic response comparisons.

Exit codes: **0** means the selected declared checks passed without new baseline attention; **1** means product failures or unjudged changes requiring review; **2** means invalid corpus/assets, missing dependencies, failed startup or incomplete replay instrumentation. Zero does not turn unjudged responses into verified answers. Live replay is experimental and runs outside required PR/release checks; results and coverage remain visible even when a nightly is nonblocking.

The separate GitHub workflow verifies the originating nightly run and binds its publication descriptor to the source SHA and run attempt before pulling an immutable public image. Promotion runs separately from staging, so the follower allows two hours before reporting the target unavailable. It records the resolution duration and never substitutes another nightly when that publication is missing. Manual dispatch takes the originating nightly run ID.

## Tests

```bash
python -m pytest tests/frontend/test_trace_replay_contract.py \
  tests/frontend/test_trace_capture.py tests/frontend/test_trace_prepare.py -q
DYN_RUN_FRONTEND_REPLAY=1 DYN_REPLAY_MODEL_DIR=/absolute/model/bundle \
  python -m pytest tests/frontend/test_trace_replay.py -q
```

Capture/preparation tests exercise local HTTP evidence, completeness, independent source and asset pins, reviewed bindings, synthetic outputs, and deterministic exports without requiring the optional tokenizer package. The contract tests cover provenance validation, lossless typed comparisons, missing-versus-null JSON Pointers, gzip/count/limit handling, planted prompt/argument failures, EOS-preserving schedules, malformed SSE and baseline classification. The live test is skipped unless explicitly enabled and is excluded from the existing nightly selector. Once enabled, it requires the verified native bundle; unavailable assets are a failure. Its initial timeout is 300 seconds and should be tuned from measured execution times.
