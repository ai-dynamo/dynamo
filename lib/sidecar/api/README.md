<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar API ownership experiment

Sidecars consume each engine's native gRPC server and expose a Dynamo worker.
The human review boundary is the engine wire contract, the launch interface,
and the shared Dynamo engine lifecycle. Request conversion, transport,
discovery, and backend-specific lifecycle implementation belong to
`@ai-dynamo/agent-sidecar-codeowners`.

## Contract map

Paths below are relative to `lib/sidecar/` unless stated otherwise.

| Contract | Human-owned source | Review team |
| --- | --- | --- |
| `dynamo-sidecar <vllm\|sglang\|trtllm> [OPTIONS]` dispatch and exit behavior | `api/dynamo-sidecar` | Runtime |
| Shared gRPC flags, environment variables, defaults, endpoint parsing, and structured startup errors | `common/api/` | Runtime |
| Backend flags, `from_args` / `try_from_args` (plus TRT-LLM's `from_env`), executable entrypoints, and Rust exports | `<backend>/api/{args,startup,main,lib}.rs` | Respective backend |
| SGLang RPC names and protobuf messages/tags | `sglang/api/proto/`, `sglang/api/{proto,build}.rs` | SGLang |
| TensorRT-LLM RPC names and protobuf messages/tags | `trtllm/api/proto/`, `trtllm/api/{proto,build}.rs` | TensorRT-LLM |
| vLLM's released protocol | `vllm/api/lib.rs` re-exports `vllm_proto`; the version is pinned in the root `Cargo.toml` | vLLM and the existing workspace owners, respectively |
| Library, executable, build-script, and dependency selection | `<crate>/Cargo.toml` | Respective backend, or Runtime for `common` |

The native RPC boundary includes streaming generation, model/server metadata,
health, cancellation, and supported engine controls. vLLM provides `Inference`
and `Control` services plus standard gRPC health; SGLang provides
`sglang.runtime.v1.SglangService`; TensorRT-LLM provides
`trtllm.TrtllmService`. Vendored schemas describe the upstream wire protocol;
not every upstream RPC is implemented by every Dynamo sidecar.

The runtime boundary already lives outside the sidecars, in human-owned
`lib/backend-common/`:

- `src/engine.rs`: `LLMEngine`, `EngineConfig`, generation input/output types,
  cancellation, health, controls/updates, KV events, and metrics hooks.
- `src/worker.rs`: `WorkerConfig`, `RuntimeConfig`, registration and lifecycle.
- `src/args.rs` and `src/disagg.rs`: common worker CLI and disaggregation roles.
- `src/run.rs`: standalone runtime entrypoints.

`lib/runtime/`, the Python launch modules
`components/src/dynamo/{vllm,sglang,trtllm}/sidecar.py`, and their PyO3 bridge
in `lib/bindings/python/rust/backend.rs` retain their existing human owners.
They are shared or externally visible boundaries outside this experiment.
The sidecar Dockerfile retains the repository-wide Ops ownership rule.

## Layout and review routing

Each crate's `api/lib.rs` is its actual Cargo library root. It selects the
implementation modules in `src/` with `#[path]`, preserving existing Rust
import paths. `api/main.rs` remains the same installed executable; the image
still installs the dispatcher as `/usr/local/bin/dynamo-sidecar`. The source
location of the dispatcher is now `lib/sidecar/api/dynamo-sidecar`.

The generated policy assigns `lib/sidecar/` to the agent team by default and
replaces that ownership for each `api/` directory and crate manifest with
its human team. New implementation files inherit agent ownership. The
`required_owners` declarations keep later nested policy rules from dropping
human ownership of the API directories or manifests.

Do not co-own API files with the agent team: GitHub accepts approval from
any owner listed on a line, so that would allow an agent approval to satisfy
the human API review requirement. CODEOWNERS routes file review; it does not
prove that an implementation change preserves behavior. Existing executable,
protocol, and engine-conformance tests remain necessary. Public constructors
are isolated here; their discovery and worker-configuration implementation
stays in `src/engine.rs`.

Before merging the experiment, provision the visible GitHub team
`@ai-dynamo/agent-sidecar-codeowners`, add the intended agent accounts, and
grant the team explicit write access to `ai-dynamo/dynamo`. A nonexistent or
ineligible team cannot enforce ownership. The existing
`dynamo-agents-codeowners` team owns agentic-inference features and is not the
agent reviewer identity for this experiment. Branch protection must also
require code-owner reviews; this source change does not alter repository
settings or provision accounts.

Change routing in `.github/codeowners/areas.yaml` and regenerate `CODEOWNERS`;
see `.github/codeowners/README.md`. Keep changes to public contracts in `api/`
and implementation work in `src/`.
