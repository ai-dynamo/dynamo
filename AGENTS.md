<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

- Keep changes focused and reviewable.
- Use Conventional Commit PR titles: `type(scope): summary`. Accepted types:
  `feat`, `fix`, `docs`, `test`, `ci`, `refactor`, `perf`, `chore`, `revert`,
  `style`, and `build`.
- PR descriptions must include `Summary` and `Validation`.
- Sign every commit with DCO: `git commit -s`.

## Cursor Cloud specific instructions

Scope: local dev of the Rust core + Python components. Full build/run docs live in
`.devcontainer/README.md` and `.devcontainer/post-create.sh`; the notes below only
capture what is non-obvious in this VM.

- **CPU-only VM (no GPU/CUDA).** The real backends (`dynamo.vllm`, `dynamo.sglang`,
  `dynamo.trtllm`) need NVIDIA GPUs and will not run here. Use the GPU-less
  `dynamo.mocker` backend for end-to-end work. GPU-marked tests
  (`gpu_1`/`gpu_2`/...) and backend-specific suites cannot run in this environment.
- **Python venv:** the environment is a plain venv at `/workspace/.venv` (not the
  container's `/opt/dynamo/venv`). Activate it (`source .venv/bin/activate`) or call
  `.venv/bin/python`. `uv` and `maturin` are installed inside it.
- **Rust → Python bindings** are prebuilt (`ai-dynamo-runtime` is installed editable
  via maturin). After changing Rust code, rebuild them with
  `CARGO_TARGET_DIR=/workspace/target maturin develop --uv` from `lib/bindings/python`
  (inside the venv). Always set `CARGO_TARGET_DIR=/workspace/target`. The bindings
  rebuild is NOT part of the startup update script (too heavy) — run it manually when
  Rust changes need to reach Python.
- **C/C++ compiler:** the `cc`/`c++` alternatives are pinned to GCC. The base image's
  default `c++` is clang, which cannot find libstdc++ headers and breaks the bundled
  libzmq build (`zmq-sys`). This is already fixed in the snapshot; don't revert it.
- **HuggingFace Hub is NOT reachable** from this VM. Anything that downloads from
  `huggingface.co` fails (e.g. the default `Qwen/Qwen3-0.6B` used by router tests via
  `ROUTER_MODEL_NAME`). Export `HF_HUB_OFFLINE=1` and point `--model-path` at a local
  model dir under `lib/llm/tests/data/sample-models/`. The frontend requires a
  `chat_template` in `tokenizer_config.json` to register a model (needed even for
  `/v1/completions`); of the checked-in samples only `mock-llama-3.1-8b-instruct` has
  one, but its 4.5 KB mock vocab detokenizes random mocker tokens to empty strings.
  `TinyLlama_v1.1` has a full real tokenizer but no chat template.
- **Running E2E without NATS/etcd:** pass `--discovery-backend file` to both the
  frontend and the worker (request-plane defaults to `tcp`, event-plane auto-selects
  `zmq`). The file discovery store defaults to `/tmp/dynamo_store_kv`; delete it
  between runs to clear stale worker registrations. Do NOT override `--model-name` to a
  bare name — the frontend then tries to resolve it from HF and fails; leave it so the
  model id derives from `--model-path`.
  - Frontend: `python -m dynamo.frontend --http-port 8000 --discovery-backend file`
  - Worker:   `python -m dynamo.mocker --model-path <local-model-dir> --discovery-backend file --speedup-ratio 0`
  - Then `POST /v1/chat/completions` with `"model": "<local-model-dir>"`.
- **Lint:** `ruff` (pinned `0.5.2`) and `cargo fmt --all --check`. The canonical
  entrypoint is `pre-commit`, which fetches hook environments over the network.
- **Tests:** the CPU-safe quick check is
  `python -m pytest components/src/dynamo/mocker/tests/unit`. Full test deps are in
  `container/deps/requirements.test.txt`.
