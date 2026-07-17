<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Use Cases: CLI → DGD Audit

Audit of every instance in the **Use Cases** nav tab (`docs/index.yml`, `- tab: use-cases`)
where the docs assume the local CLI (`python -m dynamo.*`, `docker ...`, `cargo run`, or
env vars set in a shell) instead of a Kubernetes **DynamoGraphDeployment (DGD)**. The goal is
to make the whole section assume you are on the Kubernetes platform.

Reference for the target form: [`docs/kubernetes/dgd-guide.md`](docs/kubernetes/dgd-guide.md)
(v1beta1 API, `spec.components` list, each component's container named `main` carries
`command`/`args`/`env`/`envFrom`).

## Classification legend

- **REWRITE** — a serving launch (`dynamo.frontend`, `dynamo.vllm`, `dynamo.sglang`,
  `dynamo.trtllm`, `dynamo.mocker`) that has a direct DGD equivalent. Show the DGD instead.
- **ENV** — an env var or credential shown in a shell (`export ...`, `DYN_TOKENIZER=`,
  `DYN_LORA_*`, AWS creds, `VLLM_EXTRA_ARGS`) that in Kubernetes belongs in a component's
  `env:` / `envFrom:` block. Show the placement.
- **KEEP** — genuinely CLI-only tooling with no DGD form: `dynamo.replay` (DynoSim runs,
  sweeps, planner benchmarking, agent-trace replay), `docker build` / `docker compose`
  (local image build), `cargo run` (trace conversion). Left as-is; may add a one-line K8s note.

## Frontend flag → env var map (used by the rewrites)

DGD Frontend components rarely take `command`/`args`; when a page needs a frontend flag, the
env var form (set via `env:`) is often cleaner and is what the operator supports. Confirmed in
[`docs/components/frontend/configuration.md`](docs/components/frontend/configuration.md):

| CLI flag | Env var | Default |
|---|---|---|
| `--dyn-chat-processor` | `DYN_CHAT_PROCESSOR` | `dynamo` |
| `--tokenizer` | `DYN_TOKENIZER` | `default` |
| `--migration-limit` | `DYN_MIGRATION_LIMIT` | `0` |
| `--admission-control` | `DYN_ADMISSION_CONTROL` | `none` |
| `--active-decode-blocks-threshold` | `DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD` | `1.0` |
| `--active-prefill-tokens-threshold` | `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD` | `10000000` |
| `--active-prefill-tokens-threshold-frac` | `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC` | `64.0` |

The `--tool-call-parser` / `--reasoning-parser` (engine-fallback) frontend flags have no env
var; they must go in the Frontend's `args:` list. The `--dyn-tool-call-parser` /
`--dyn-reasoning-parser` / `--dyn-enable-structural-tag` flags go on the **worker** `args:`.

---

## Tool Calling & Reasoning

### `docs/tool-calling/README.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 33-36 | `python -m dynamo.<backend> --help` | KEEP | Discovery command; reword to "the worker's `--help`" but keep — not a deployment. |
| 38-40 | `python -m dynamo.<backend> --custom-jinja-template ...` (TIP) | REWRITE | Show `--custom-jinja-template` as a worker `args:` entry. |
| 85-91 | Launch backend + frontend (`dynamo.sglang ... --dyn-tool-call-parser`; `dynamo.frontend`) | REWRITE | Convert to a DGD: Frontend + SGLangWorker with the parser flags in worker `args:`. |
| 96-116 | `curl http://localhost:8000/...` | ENV/KEEP | Keep the curl, but front it with `kubectl port-forward svc/<name>-frontend 8000:8000`. |

### `docs/reasoning/README.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 23-26 | `python -m dynamo.<backend> --help` | KEEP | Same as above. |
| 83-89 | Launch backend + frontend | REWRITE | Same DGD form as the tool-calling README (shared example). |
| 93-100 | `curl http://localhost:8000/...` | KEEP | Add `port-forward` preamble. |

### `docs/tool-calling/parser-configuration.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 45-57 | Default (Dynamo-native): `dynamo.frontend`, `dynamo.frontend --dyn-chat-processor dynamo`, three workers with `--dyn-tool-call-parser`/`--dyn-reasoning-parser` | REWRITE | One DGD showing default frontend (no args) + worker with `--dyn-*` parser flags in `args:`. Note the three backends via tabs or a note. |
| 61-68 | Engine fallback: `dynamo.frontend --dyn-chat-processor vllm --tool-call-parser ... --reasoning-parser ...` + worker | REWRITE | DGD where the **Frontend** carries `--dyn-chat-processor`/`--tool-call-parser`/`--reasoning-parser` in `args:` and the worker has no parser flags. |

### `docs/tool-calling/engine-fallback.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 26-33 | vLLM + SGLang engine-fallback launches | REWRITE | Same DGD form as parser-configuration engine-fallback (Frontend carries parser args). |

### `docs/tool-calling/structural-tag.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 34-43 | Quick Start: `dynamo.sglang ... --dyn-enable-structural-tag` + `dynamo.frontend` | REWRITE | DGD: worker `args:` gets `--dyn-tool-call-parser` + `--dyn-enable-structural-tag`; default Frontend. |
| 141-149 | Example: worker with `--dyn-enable-structural-tag --dyn-structural-tag-scope always --dyn-structural-tag-schema strict` | REWRITE | Show the same worker `args:` extended with scope/schema flags. |

### `docs/tool-calling/README.zh-CN.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 29, 34, 74-77 | Chinese mirror of README.md | REWRITE | Keep in sync with the English README rewrite (translate prose only; YAML stays verbatim). |

### `docs/tool-calling/parsing.md`, `troubleshooting.md`, `release-1.2-probe-snapshot.md`

- `parsing.md` — no CLI. **No change.**
- `troubleshooting.md:40` — one `curl localhost:8000`. **ENV** — add port-forward note.
- `release-1.2-probe-snapshot.md` — no CLI. **No change.**

---

## Fastokens Tokenizer — `docs/features/tokenizer/README.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 67-74 | Quick Start: `dynamo.frontend --tokenizer fastokens`; `export DYN_TOKENIZER=fastokens` + `dynamo.frontend` | REWRITE/ENV | Show DGD Frontend with `env: - name: DYN_TOKENIZER / value: fastokens` (preferred) and the `--tokenizer` `args:` alternative. |
| 78-80 | `dynamo.frontend --tokenizer default` | ENV | Show `DYN_TOKENIZER=default` env / omit. |
| 135-142 | `sweep_runner.py` benchmark | KEEP | Benchmark tooling, not a serving deploy. |
| 151, 166 | Already references "benchmark DGD templates" + "DGD frontend pod" with `DYN_TOKENIZER` | OK | Already K8s-aware; align wording with the rewritten Quick Start. |

Note: this page is already partly K8s-aware (mentions DGD frontend pod, `DYN_TOKENIZER`). The
Quick Start is the main CLI-first part to convert.

---

## LoRA Adapters — `docs/features/lora/README.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 67-73 | `DYN_SYSTEM_ENABLED=... DYN_SYSTEM_PORT=... python -m dynamo.vllm --model ... --enable-lora --max-lora-rank 64` | REWRITE/ENV | Convert to the DGD at `examples/backends/vllm/deploy/v1beta1/agg_lora.yaml`: worker `args:` gets `--enable-lora`/`--max-lora-rank`; `DYN_LORA_ENABLED`, `DYN_SYSTEM_ENABLED`, `DYN_SYSTEM_PORT` go in `env:`. |
| 77-85 | `curl http://localhost:8081/v1/loras` (load) | ENV | Point at the worker's system port via port-forward or `DynamoModel` CRD (already documented lower on the page). |
| 90-97 | `curl http://localhost:8000/...` (inference) | ENV | port-forward preamble. |
| 104-119 | `export AWS_*` + curl load from S3 | ENV | Move AWS creds into worker `env:` (`valueFrom.secretKeyRef` for keys, inline for endpoint/region), mirroring the deploy YAML. |
| 124-134 | Env var table | OK | Reference; add "set these in the worker `env:` block" pointer. |
| 212-250 | Kubernetes section (DynamoModel CRD, `kubectl`) | OK | Already K8s. This is the model for the whole page — pull it up / lead with it. |

The page already has a strong Kubernetes section. The fix is to make the **Quick Start** lead
with the DGD instead of the local `python -m dynamo.vllm` launch, and place all env vars/creds
in the pod spec.

---

## Fault Tolerance

### `docs/fault-tolerance/request-rejection.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 47-52 | `dynamo.frontend --admission-control token-capacity --active-decode-blocks-threshold ... --active-prefill-tokens-threshold ...` | REWRITE/ENV | DGD Frontend with these as `env:` (`DYN_ADMISSION_CONTROL`, `DYN_ACTIVE_*`) or `args:`. |
| 68-81 | `curl .../busy_threshold` (runtime API) | ENV | port-forward preamble; runtime API is fine as-is. |
| 223-238 | Tuning snippets (bare `--active-*` flags) | ENV | Reframe as env/args values to set on the Frontend component. |
| 249-251 | "don't set the threshold args" `dynamo.frontend` | REWRITE | Show a default Frontend (no admission-control env). |
| 262-267, 274-276, 289-293 | best-practice bare flags + `watch curl localhost:8000/metrics` | ENV | Same reframing; metrics via port-forward or ServiceMonitor. |
| 304-307 | Worker admission: `--engine-request-limit` / `DYN_ENGINE_REQUEST_LIMIT` | REWRITE/ENV | Show as worker `args:`/`env:`. |

### `docs/fault-tolerance/graceful-shutdown.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 221-223 | `python3 -m dynamo.frontend ... --migration-limit 3` | REWRITE/ENV | DGD Frontend with `DYN_MIGRATION_LIMIT: "3"` in `env:` (or `--migration-limit` in `args:`). The `terminationGracePeriodSeconds` guidance is already pod-spec-oriented. |

### `docs/fault-tolerance/request-migration.md`

- Content references `--migration-limit` on the frontend but no launch command. **ENV** — where
  it shows the flag, cross-link the DGD Frontend `env:` form. Minor.

### `README.md`, `request-cancellation.md`, `testing.md`

- No `python -m dynamo` launches. `request-cancellation.md` uses Python client `base_url=
  http://localhost:8000` + `curl localhost:8000/metrics`. **ENV/KEEP** — add port-forward note;
  client code is illustrative. No structural rewrite.

---

## Agents

### `docs/backends/sglang/agents.md` (SGLang for Agentic Workloads)

| Line | Snippet | Class | Action |
|---|---|---|---|
| 26-31 | `dynamo.sglang --model-path <model> --enable-priority-scheduling ...` | REWRITE | Show as worker `args:` fragment in a DGD. |
| 47-52 | `dynamo.sglang ... --radix-eviction-policy priority` | REWRITE | Worker `args:` fragment. |
| 184-189 | `dynamo.sglang ... --enable-streaming-session` | REWRITE | Worker `args:` fragment. |
| 197-201 | `dynamo.frontend --router-mode kv` | REWRITE | Frontend `args:` fragment (`--router-mode kv`). |
| 86-112 | Python OpenAI client (`base_url=localhost:8000`) | KEEP | Client example; add port-forward note. |
| 289-308 | `bash examples/.../agg_agent.sh`, `bun run` OpenCode | KEEP | Launch script + external agent harness; local by nature. Add a "for K8s, deploy the equivalent DGD" pointer. |

### `docs/agents/agent-tracing.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 282, 290-311 | `cargo run -p dynamo-bench ...`, `uv run ... python -m dynamo.replay` | KEEP | Trace-conversion + replay tooling; no DGD form. Leave. |

### `docs/agents/README.md`, `agent-hints.md`, `pi-mono.md`, `thunderagent-router.md`

- No `python -m dynamo` launches found. **No change** (spot-check `pi-mono.md` for any curl/env).

---

## Multimodal

### `docs/features/multimodal/multimodal-kv-routing.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 166, 225 | Env-var tables referencing `VLLM_EXTRA_ARGS` / `SGLANG_EXTRA_ARGS` "pass-through args to `python -m dynamo.vllm/sglang`" | ENV | Reword to "worker `args:`"; note these launch-script env vars map to worker `args:` in a DGD. |
| 170-173, 179-181, 199-209 | `bash examples/.../launch/*.sh`, `cd .../mm_router_worker && ./launch.sh` | KEEP | Local launch scripts. Add a DGD pointer; the scripts themselves are the documented interface here. |

This page is built around local launch scripts and their env knobs. Recommend a K8s framing
note + the `VLLM_EXTRA_ARGS → worker args:` mapping, rather than a full rewrite of every script.

### `docs/features/multimodal/README.md`, `embedding-cache.md`, `encoder-disaggregation.md`

- No `python -m dynamo` launches. **No change.**

---

## Diffusion (Preview)

### `docs/backends/vllm/vllm-omni.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 251 | "dedicated entrypoint: `python -m dynamo.vllm.omni`" | REWRITE (partial) | No DGD exists in-repo for omni (launch-script only). Show a DGD worker skeleton using `command: [python3, -m, dynamo.vllm.omni]` + the omni `args:`; keep the launch scripts as the local path. |
| 17-19, 43-56, 61-95, ... | `pip install`, `bash examples/.../agg_omni_*.sh`, curls | KEEP/ENV | Install + local launch scripts stay; curls get port-forward note. Storage/AWS env (line 290) → worker `env:` when in K8s. |

### `docs/backends/trtllm/trtllm-diffusion.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 42-46 | `dynamo.trtllm --modality video_diffusion --model-path ... --media-output-fs-url ...` | REWRITE | DGD worker with these `args:`. |
| 71-75 | `dynamo.trtllm --modality image_diffusion ...` | REWRITE | DGD worker variant. |
| 19-22 | `pip install` (outside-container path) | KEEP | Install note. |
| 53-64, 82-89 | curls | ENV | port-forward note. |

### `docs/features/diffusion/fastvideo.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 35-54, 83-88 | `docker build`, `docker compose up --build`, `COMPOSE_PROFILES=...` | KEEP | Local image build + Compose. No DGD form. |
| 92-118 | `./run_local.sh` + env vars | KEEP | Host-local script. |
| 128-198 | Kubernetes Deployment (`kubectl apply -f agg.yaml`, PVC, port-forward) | OK | Already K8s. This is the model; the `agg.yaml` uses the older `services:` schema — optionally modernize to v1beta1 `components:`. |

fastvideo is a custom `worker.py` example (not `dynamo.vllm/sglang/trtllm`) and already has a
Kubernetes section. Leave the Docker/local build; keep/extend the existing K8s deploy.

### `docs/backends/sglang/sglang-diffusion.md`

- No `python -m dynamo` matches in the grep. **Verify and no change** (uses launch scripts).

---

## Inference Simulation (DynoSim) — mostly KEEP

DynoSim is a simulation/benchmark surface. `dynamo.replay` (runs, sweeps, planner benchmarking)
is CLI-only with **no DGD form** — KEEP. The **mocker**, however, runs as a real Dynamo worker
and **does** have a DGD form (`examples/backends/mocker/deploy/v1beta1/{agg,disagg}.yaml`).

### `docs/dynosim/mocker.md`

| Line | Snippet | Class | Action |
|---|---|---|---|
| 30-69, 253-338 | `python -m dynamo.mocker ...` (basic, disagg, multi-worker, AIC) | REWRITE (add) | Add a DGD form for the live-deployment mocker (Frontend + `dynamo.mocker` worker, image `dynamo-planner`). Keep the CLI for local/no-GPU use, but lead the "live deployment" parts with the DGD since the page says mocker "runs as a Dynamo backend … exercises the real frontend/router path." |
| 132-338 (`dynamo.replay` blocks) | `python -m dynamo.replay ...` | KEEP | Replay is CLI-only. |

### `docs/dynosim/README.md`, `runs.md`, `sweeps.md`, `planner-benchmarking.md`

- All `dynamo.replay` / `replay_optimize` / `.venv/bin/python -m dynamo.replay`. **KEEP** —
  simulation tooling, no DGD equivalent. The README's component table (line 23) can note that
  the mocker also has a DGD form for live deployments; otherwise no change.

---

## Summary of recommended actions

**REWRITE to DGD (serving launches):**
- `tool-calling/README.md`, `reasoning/README.md` — shared frontend+worker launch → DGD.
- `tool-calling/parser-configuration.md`, `engine-fallback.md` — default + engine-fallback → DGD
  (worker `--dyn-*` args; Frontend `--dyn-chat-processor`/`--tool-call-parser` args).
- `tool-calling/structural-tag.md` — worker `--dyn-enable-structural-tag` args → DGD.
- `tool-calling/README.zh-CN.md` — mirror the English rewrite.
- `features/tokenizer/README.md` — Quick Start → DGD Frontend `env: DYN_TOKENIZER`.
- `features/lora/README.md` — Quick Start → DGD (`agg_lora.yaml`), env/creds in pod spec.
- `fault-tolerance/request-rejection.md` — admission-control → DGD Frontend env/args; worker
  `--engine-request-limit` → worker env/args.
- `fault-tolerance/graceful-shutdown.md` — `--migration-limit` → DGD Frontend env.
- `backends/sglang/agents.md` — engine flags → worker/Frontend `args:` fragments.
- `backends/trtllm/trtllm-diffusion.md` — `--modality` launches → DGD worker.
- `backends/vllm/vllm-omni.md` — add a `dynamo.vllm.omni` DGD worker skeleton.
- `dynosim/mocker.md` — add DGD form for the live mocker deployment.

**ENV placement (env vars / creds into `env:` / `envFrom:`):**
- LoRA `DYN_LORA_*`, `DYN_SYSTEM_*`, AWS creds; tokenizer `DYN_TOKENIZER`; admission-control
  `DYN_*`; migration `DYN_MIGRATION_LIMIT`; multimodal `VLLM_EXTRA_ARGS`/`SGLANG_EXTRA_ARGS`
  → worker `args:`.
- All `curl http://localhost:8000` / `:8081` → prefix with `kubectl port-forward svc/<name>-frontend`.

**KEEP (CLI-only, no DGD form):**
- `dynosim/{README,runs,sweeps,planner-benchmarking}.md` and `agents/agent-tracing.md`
  (`dynamo.replay`, `replay_optimize`, `cargo run` trace conversion).
- `features/diffusion/fastvideo.md` Docker/Compose/`run_local.sh` build+local paths.
- `multimodal-kv-routing.md` launch scripts (add K8s framing + `EXTRA_ARGS → args:` note).
- `pip install` / `--help` discovery commands.
