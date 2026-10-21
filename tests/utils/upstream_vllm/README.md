<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Upstream vLLM chat behavior tests

Run the four nodes in `selection.json` against direct vLLM and Dynamo. The
upstream test bodies, requests, assertions, server fixtures and client fixtures
remain unchanged. This is the initial chat subset, not the full upstream suite.
Tracking: [#16192](https://github.com/ai-dynamo/dynamo/issues/16192).

## How it becomes a Dynamo test

`tests/frontend/test_upstream_vllm.py` exposes one Dynamo pytest item per upstream
node. Its module fixture runs two isolated upstream pytest processes, sequentially
on one GPU. Isolation is required because both repositories use a `tests` package.
Each child keeps the upstream module-scoped fixture and starts each model once.

```text
Dynamo pytest item: test_upstream_chat[<upstream test name>]
  module fixture → direct vLLM pytest → upstream JUnit
                 → Dynamo pytest      → upstream JUnit
  item assertion → require that named case to pass in both reports
```

The runner binds `tests.utils.RemoteOpenAIServer` before importing test modules.
For Dynamo, `RemoteDynamoServer` extends upstream `RemoteOpenAIServerCustom`:

| Reused code | Small adaptation |
| --- | --- |
| vLLM CLI parsing and model pre-download | Pin model/tokenizer revisions; cap KV cache at 1 GiB and sequences at eight on both endpoints |
| vLLM process creation, context manager, process-group cleanup and GPU-memory release check | Execute Dynamo's existing `examples/backends/vllm/launch/agg.sh` |
| vLLM port allocation and URL/client methods | Route `--port` to `DYN_HTTP_PORT`; translate `--reasoning-parser` to `--dyn-reasoning-parser` |
| Native upstream server/client fixtures | Wait for the expected model in `/v1/models`; use the generic shutdown grace period |
| Existing Dynamo `ManagedProcess` | Protect the outer child pytest invocation on timeout or interruption |
| Existing Dynamo `reserved_ports` | Reserve the worker system port within Rust's signed 16-bit range |

All other fixture arguments pass through to the worker. Invalid worker arguments
fail setup. The selected test/support files are hash-checked, and collection must
match the ordered node list exactly. Adding tests requires auditing their fixtures
and arguments; this adapter does not promise compatibility with every server option.

A private cache view maps `main` to each pinned snapshot because the upstream
vocabulary helper loads the model without a revision. Shared cache references
are not rewritten. Both endpoints disable prefix caching and use upstream's
default seed zero. The served model name stays equal to the upstream model ID,
even when offline loading resolves a local snapshot path. HTTP capture currently
supports these four non-streaming cases.

The older vLLM 0.28 runtime lacks a shutdown helper imported by the newer test
utilities. The runner supplies a fail-if-called import guard and uses the generic
upstream shutdown timeout for that reference profile. It records this exception
in `environment.json`; no request or response is patched.

## Run in the CI test image

Prepare a tests-only checkout at the manifest's vLLM commit. Keep the installed
runtime wheel on the Python path; copying upstream `vllm/` would shadow it.

```bash
git -C /path/to/pinned-vllm archive HEAD tests pyproject.toml | tar -x -C /path/to/tests-only
uv venv --python "$(command -v python)" --system-site-packages /path/to/client-env
uv pip install --python /path/to/client-env/bin/python --no-deps tblib==3.1.0
export VLLM_TESTS_DIR=/path/to/tests-only
export VLLM_TEST_PYTHON=/path/to/client-env/bin/python
# Use a fresh output directory for every invocation.
export DYN_TEST_OUTPUT_PATH=/path/to/new-results
python -m pytest tests/frontend/test_upstream_vllm.py -m upstream_vllm -v
```

The proposed `.github/workflows/upstream-vllm-chat.yml` is manually dispatched.
Its `source_ref` must be a full Dynamo commit SHA with this suite and an existing
matching `vllm-runtime-test` CI image. It reuses `shared-test.yml`, the existing
sequential GPU stage, standard JUnit/Allure reporting and artifact upload. It adds
only pinned upstream preparation and an always-uploaded HTTP/JUnit evidence folder.
The job is bounded to 30 minutes and disables flaky retries. Ordinary test lanes
skip these four cases unless `VLLM_TESTS_DIR` is configured.

Evidence is under `DYN_TEST_OUTPUT_PATH/upstream-vllm-chat/`: per-endpoint logs,
JUnit, collection/fixture mapping, installed versions, model pins, and raw HTTP
requests/responses. Model weights stay outside the evidence directory.

Known divergence remains a red test result. There is no `xfail`, response rewrite
or expected-error exception. Reference failures, setup errors and skips also fail
the corresponding Dynamo item; they must not be reported as protocol divergence.

The test-source SHA and runtime versions are separate. A runtime upgrade requires
a new passing reference run. The earlier Dynamo 1.5.1/vLLM 0.28 evidence does not
establish behavior on the current CI runtime. When expanding coverage, search
for chat-related tests outside `chat_completion/` as well, and audit each
selected test's fixtures before adding it to the manifest.
