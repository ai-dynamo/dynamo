<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run upstream vLLM chat tests against Dynamo

`selection.json` lists four upstream test nodes. The test bodies, assertions and
fixtures stay unchanged. `RemoteDynamoServer` replaces upstream
`RemoteOpenAIServer` and launches Dynamo through the existing `agg.sh` script.
It reuses vLLM's argument parsing, model download, clients and process cleanup.
Only launch arguments, model readiness and discovery isolation are adapted.

The Dynamo pytest wrapper starts one isolated upstream pytest process because
both repositories have a `tests` package. It reports each upstream case separately
and retains the child log and JUnit under `DYN_TEST_OUTPUT_PATH/upstream-vllm-chat/`.
There is no reference-server run or custom HTTP capture.

## Initial failures

The goal is for every selected test to pass. A `null` value in `selection.json`
requires a pass. A reason string marks a known test-body failure as `xfail`.
Passing known failures show as `XPASS`; replace their reason with `null` after
confirming the fix. Setup/teardown errors, missing results and upstream skips
remain failures. Use `--runxfail` on the outer pytest command to enforce all tests.
The two initial allowances track [#16192](https://github.com/ai-dynamo/dynamo/issues/16192):
`include_reasoning=false` rejection and acceptance of non-integer `logit_bias` keys.

## Run

Use Dynamo's vLLM test image and a tests-only checkout of vLLM commit
`817fa3518cd4bcc94b31e044c534c002e701c801`. Do not copy upstream `vllm/` onto the
Python path: the tests must use the image's installed runtime.

```bash
git -C /path/to/pinned-vllm archive HEAD tests pyproject.toml | tar -x -C /path/to/tests-only
uv venv --python "$(command -v python)" --system-site-packages /path/to/client-env
uv pip install --python /path/to/client-env/bin/python --no-deps tblib==3.1.0
export VLLM_TESTS_DIR=/path/to/tests-only
export VLLM_TEST_PYTHON=/path/to/client-env/bin/python
export DYN_TEST_OUTPUT_PATH=/path/to/new-results
python -m pytest tests/frontend/test_upstream_vllm.py -m upstream_vllm -v -rxX
```

The manual `upstream-vllm-chat.yml` workflow reuses Dynamo's shared GPU job.
Supply a full Dynamo commit SHA with an existing matching `vllm-runtime-test`
image. Other test lanes skip the suite unless `VLLM_TESTS_DIR` is set.

Models follow the upstream fixtures' default revisions; the adapter caps KV
cache at 1 GiB and sequences at eight. vLLM 0.28 needs an import guard for a newer
shutdown helper; Dynamo always uses the upstream generic shutdown timeout.

To expand coverage, add node IDs and audit their fixture arguments. Discover
chat tests in `chat_completion/`, elsewhere in `tests/entrypoints/openai/`, and
outside that directory. These four tests do not establish full compatibility.
