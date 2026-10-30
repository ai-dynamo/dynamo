<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Discover upstream vLLM chat tests

The suite uses tests from the release tag matching the vLLM package installed in
the Dynamo image, then discovers every pytest case under upstream
`tests/entrypoints/openai/chat_completion/`, including parametrized cases.
`allow_failures.json` controls only which known failures are tolerated; it never
selects tests. Newly added tests run without a manifest change.

The original tests and fixtures stay unchanged. `RemoteDynamoServer` replaces
upstream `RemoteOpenAIServer`, reusing its model download, clients and process
cleanup while launching Dynamo through the existing `agg.sh` script. Each server
reserves its own system port, so fixtures can start multiple servers.

The folder also contains vLLM-only unit tests. Each executed case records
`upstream_target=dynamo` when its fixtures include an adapted server, or
`vllm-only` otherwise. A vLLM-only pass is not evidence of Dynamo compatibility.
Keep these result groups separate when reviewing the report.

## Results and known failures

The goal is for every Dynamo case to pass. Unlisted failures fail the job.
A listed test-body failure reports `XFAIL`; a listed pass reports `XPASS`.
Remove an allowance after confirming a fix, or use `--runxfail` on the outer
pytest command to require every test to pass. The initial two allowances track
[#16192](https://github.com/ai-dynamo/dynamo/issues/16192): rejection of
`include_reasoning=false` and acceptance of non-integer `logit_bias` keys.

Collection errors, setup/teardown errors, missing results and upstream skips
remain failures. They are not protocol incompatibilities. Stale allowance IDs
fail collection rather than silently disappearing from the report.

An isolated child process prevents conflicts between the repositories' `tests`
packages. Collection runs first, then execution; the Dynamo wrapper reports each
full upstream node ID separately. Both phases retain logs, collection IDs and
JUnit under `DYN_TEST_OUTPUT_PATH/upstream-vllm-chat/`. The execution JUnit keeps
the original outcomes before the outer wrapper applies allowances.

## Run

Use tests from the same vLLM release as the Dynamo image and include
the upstream examples used by fixtures. Exclude upstream `vllm/` from the Python
path so the installed runtime remains authoritative.

```bash
# In the Dynamo image, select the matching upstream release.
version="$(python -c 'from importlib.metadata import version; print(version("vllm").split("+", 1)[0])')"
git -C /path/to/pinned-vllm checkout "v${version}"
git -C /path/to/pinned-vllm archive HEAD tests examples pyproject.toml | tar -x -C /path/to/tests-only
uv venv --python "$(command -v python)" --system-site-packages /path/to/client-env
uv pip install --python /path/to/client-env/bin/python --no-deps tblib==3.1.0 pytest-mock==3.15.1
export VLLM_TESTS_DIR=/path/to/tests-only
export VLLM_TEST_PYTHON=/path/to/client-env/bin/python
export DYN_TEST_OUTPUT_PATH=/path/to/new-results
python -m pytest tests/frontend/test_upstream_vllm.py -m upstream_vllm -v -rxX
```

The manual `upstream-vllm-chat.yml` workflow reuses Dynamo's shared single-GPU
job and retrieves the matching release's tests at job time. It records the resolved
upstream commit and rejects development-version wheels without a release tag.
Supply a full Dynamo SHA with
an existing matching `vllm-runtime-test` image. Other lanes skip this suite when
`VLLM_TESTS_DIR` is unset.

Full-directory discovery does not guarantee that all fixtures can execute on
that image and hardware. The folder includes a 7B LoRA model, gated models and
a 35B two-GPU case. Upstream hardware skips remain visible as failed coverage.
Models follow upstream default revisions; the adapter caps each server's KV
cache at 1 GiB and sequences at eight. Unsupported server options fail setup.
Do not combine tests from a newer development commit with an older runtime:
upstream tests also import vLLM internals. Earlier exploratory runs used that
combination; their import errors are setup failures, not Dynamo incompatibilities.

Chat tests elsewhere in `tests/entrypoints/openai/` and outside that directory
remain candidates for later discovery expansion.
